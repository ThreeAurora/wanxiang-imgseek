"""双 CLIP ONNX 会话：懒加载、图像/文本编码、攒批嵌入、向量落盘。

- 每个 model_key 一个独立向量空间（独立 .f16bin + vector_slot）；
- 只对「当前激活模型」做嵌入；切换模型后，其余图片凭
  embed_status.status=0 由流水线 backlog 自动补齐；
- ORT InferenceSession.run 线程安全：检索线程可直接编码查询文本，
  与 GPU 嵌入阶段并发无害。
- 输入/输出名自适应：兼容分塔导出（vision_model/text_model）与
  chinese-clip 整模（单 session 双塔，喂 dummy 配对输入）。
"""
import logging
import threading

import numpy as np

import config
from imgseek import vectors

log = logging.getLogger("clip")


def _l2(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(n, 1e-12)


class ClipSession:
    def __init__(self, key: str):
        self.key = key
        self.cfg = config.MODELS[key]
        self._vis = None
        self._txt = None
        self._tok = None
        self._img_out = ""
        self._txt_out = ""
        self.backend = "unloaded"

    @property
    def loaded(self) -> bool:
        return self._vis is not None

    def load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        base = config.MODEL_DIR / self.key
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._vis = ort.InferenceSession(str(base / self.cfg["vision_onnx"]),
                                         providers=providers)
        if self.cfg["arch"] == "clip_split":
            self._txt = ort.InferenceSession(
                str(base / self.cfg["text_onnx"]), providers=providers)
        else:  # chinese_clip 整模：同一 session 含双塔
            self._txt = self._vis

        tok = Tokenizer.from_file(str(base / "tokenizer.json"))
        tok.no_truncation()
        tok.no_padding()
        self._tok = tok

        self._img_out = self._pick_output(self._vis, "image")
        self._txt_out = self._pick_output(self._txt, "text")
        self.backend = self._vis.get_providers()[0]
        log.info("CLIP %s loaded backend=%s img_out=%s txt_out=%s",
                 self.key, self.backend, self._img_out, self._txt_out)

    @staticmethod
    def _pick_output(sess, side_keyword: str) -> str:
        names = [o.name for o in sess.get_outputs()]
        for n in names:
            ln = n.lower()
            if side_keyword in ln and ("embed" in ln or "feature" in ln):
                return n
        return names[0]

    def _feed(self, sess, px, ids, mask):
        """按输入名装配；整模推理时缺失一侧喂 dummy 零张量。"""
        s = self.cfg["size_px"]
        lt = self.cfg["max_text_tokens"]
        d_px = np.zeros((1, 3, s, s), dtype=np.float32)
        d_ids = np.zeros((1, lt), dtype=np.int64)
        feed = {}
        for i in sess.get_inputs():
            n = i.name.lower()
            if "pixel" in n:
                feed[i.name] = px if px is not None else d_px
            elif "input" in n:          # input_ids
                feed[i.name] = ids if ids is not None else d_ids
            elif "mask" in n:
                feed[i.name] = mask if mask is not None else d_ids * 0
        return feed

    def _prep_images(self, hwc_list) -> np.ndarray:
        mean = np.asarray(self.cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.cfg["std"], dtype=np.float32).reshape(1, 1, 3)
        arr = np.stack(hwc_list).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        return arr.transpose(0, 3, 1, 2).astype(np.float32)  # N,3,H,W

    def encode_images(self, hwc_list) -> np.ndarray:
        px = self._prep_images(hwc_list)
        out = self._vis.run([self._img_out],
                            self._feed(self._vis, px, None, None))[0]
        emb = np.asarray(out, dtype=np.float32).reshape(len(hwc_list), -1)
        if emb.shape[1] != self.cfg["dim"]:
            raise RuntimeError(
                f"{self.key}: vision output dim {emb.shape[1]} != "
                f"expected {self.cfg['dim']} - wrong ONNX export?")
        return _l2(emb)

    def _pad_id(self) -> int:
        for t in ("[PAD]", "<|endoftext|>", "<pad>"):
            v = self._tok.token_to_id(t)
            if v is not None:
                return v
        return 0

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        encs = self._tok.encode_batch(texts)
        lt = self.cfg["max_text_tokens"]
        pad = self._pad_id()
        ids = np.full((len(texts), lt), pad, dtype=np.int64)
        mask = np.zeros((len(texts), lt), dtype=np.int64)
        for i, e in enumerate(encs):
            seq = list(e.ids)[:lt]  # 截断保 BOS，丢尾部 EOS/SEP 影响可忽略
            ids[i, :len(seq)] = seq
            mask[i, :len(seq)] = 1
        out = self._txt.run([self._txt_out],
                            self._feed(self._txt, None, ids, mask))[0]
        emb = np.asarray(out, dtype=np.float32).reshape(len(texts), -1)
        if emb.shape[1] != self.cfg["dim"]:
            raise RuntimeError(
                f"{self.key}: text output dim {emb.shape[1]} != "
                f"expected {self.cfg['dim']} - wrong ONNX export?")
        return _l2(emb)


class ClipManager:
    def __init__(self) -> None:
        self.sessions: dict[str, ClipSession | None] = {
            k: None for k in config.MODELS}
        self.states: dict[str, str] = {k: "unloaded" for k in config.MODELS}
        self.errors: dict[str, str] = {}
        self.active_key = config.DEFAULT_MODEL
        self._buf: list = []

    # ---------- 会话 ----------
    def set_state(self, key: str, st: str, err: str = "") -> None:
        self.states[key] = st
        if err:
            self.errors[key] = err

    def session_ready(self, key: str | None = None) -> bool:
        s = self.sessions.get(key or self.active_key)
        return bool(s and s.loaded)

    def ensure_session(self, key: str) -> ClipSession:
        s = self.sessions.get(key)
        if s is None or not s.loaded:
            s = ClipSession(key)
            s.load()
            self.sessions[key] = s
            self.set_state(key, "ready")
        return s

    def activate(self, key: str) -> None:
        self.active_key = key

    def encode_query(self, text: str, model_key: str | None = None):
        key = model_key or self.active_key
        s = self.sessions.get(key)
        if s is None or not s.loaded:
            return None
        return s.encode_texts([text])[0]

    # ---------- 嵌入（流水线 GPU 阶段调用） ----------
    def embed_item(self, item, next_slot_fn) -> None:
        if item.clip is None:
            return
        self._buf.append(item)
        if len(self._buf) >= config.CLIP_BATCH:
            self.flush(next_slot_fn)

    def flush(self, next_slot_fn=None) -> None:
        if not self._buf:
            return
        items, self._buf = self._buf[:], []
        key = self.active_key
        try:
            sess = self.ensure_session(key)
        except Exception as e:  # noqa: BLE001 - 权重缺失等
            log.error("embed flush: cannot load %s: %s", key, str(e)[:200])
            return
        try:
            embs = sess.encode_images([it.clip for it in items])
        except Exception as e:  # noqa: BLE001
            log.error("embed flush inference failed: %s", str(e)[:200])
            return
        from imgseek import db
        conn = db.get_conn()
        vf = vectors.get_file(key)
        for it, vec in zip(items, embs):
            row = conn.execute(
                "SELECT slot FROM vector_slot WHERE model_key=? AND image_id=?",
                (key, it.image_id)).fetchone()
            slot = row["slot"] if row else (
                next_slot_fn or db.next_slot)(key)
            vf.write(conn, slot, vec)
            if row is None:
                conn.execute(
                    "INSERT INTO vector_slot(model_key, image_id, slot) "
                    "VALUES(?,?,?)", (key, it.image_id, slot))
            conn.execute(
                "UPDATE embed_status SET status=1 "
                "WHERE image_id=? AND model_key=?", (it.image_id, key))
        conn.commit()

    def unload_all(self) -> None:
        for k, s in list(self.sessions.items()):
            if s is not None:
                self.sessions[k] = None
        self.states = {k: "unloaded" for k in config.MODELS}


MANAGER = ClipManager()
