"""向量存储：<model>.f16bin 落盘 + fp32 常驻内存检索。

基准实测（2026-08，50 万 x 512）：
- memmap 逐块 fp16->fp32 点积：p95 ≈ 2s（磁盘 IO 与逐块转换双重瓶颈）
- fp32 常驻单次 matmul：p50 ≈ 72ms —— 采用此路线。
- 落盘保持 float16（省一半磁盘），常驻矩阵 float32；写路径双写。
- 向量已 L2 归一化 => 点积即余弦。padding 行为零向量，score=0，
  不影响 top-k 排序，且不会出现在 vector_slot 的 join 结果里。
"""
import logging
import threading

import numpy as np

import config

log = logging.getLogger("vectors")

_GROW_ROWS = 4096


class VectorFile:
    def __init__(self, model_key: str):
        self.key = model_key
        self.dim = config.MODELS[model_key]["dim"]
        self.path = config.VECTOR_DIR / f"{model_key}.f16bin"
        self._fh = None
        self._resident: np.ndarray | None = None
        self._cap_rows = 0
        self._lock = threading.Lock()

    # ---------- 容量与装载 ----------
    def _ensure_capacity(self, rows_needed: int) -> None:
        if rows_needed <= self._cap_rows and self._fh is not None:
            return
        target = ((max(rows_needed, 1) + _GROW_ROWS - 1) // _GROW_ROWS) * _GROW_ROWS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._fh is None:
            self._fh = open(self.path, "r+b" if self.path.exists() else "w+b")
        self._fh.truncate(target * self.dim * 2)
        if config.RESIDENT_VECTORS:
            old = self._resident
            new = np.zeros((target, self.dim), dtype=np.float32)
            if old is not None:
                new[:len(old)] = old
            self._resident = new
        self._cap_rows = target

    def preload(self, total: int) -> None:
        """把落盘文件前 total 行读入常驻矩阵（模型激活时调用）。"""
        self._ensure_capacity(max(total, 1))
        if not config.RESIDENT_VECTORS or total <= 0:
            return
        t0 = time.monotonic()
        with open(self.path, "rb") as f:
            raw = f.read(total * self.dim * 2)
        arr16 = np.frombuffer(raw, dtype=np.float16).reshape(total, self.dim)
        self._resident[:total] = arr16.astype(np.float32)
        log.info("vector preload %s: %d rows in %.1fs", self.key, total,
                 time.monotonic() - t0)

    # ---------- 写 / 检索 ----------
    def write(self, slot: int, vec_f32: np.ndarray) -> None:
        with self._lock:
            self._ensure_capacity(slot + 1)
            self._fh.seek(slot * self.dim * 2)
            self._fh.write(vec_f32.astype(np.float16).tobytes())
            if self._resident is not None:
                self._resident[slot] = vec_f32

    def search(self, qvec: np.ndarray, topk: int | None = None):
        """返回 [(slot, score)] 按 score 降序。"""
        with self._lock:
            if self._cap_rows == 0 and self._resident is None:
                self._recover_row_count()
        k = min(topk or config.VEC_TOPK, self._cap_rows)
        if k <= 0:
            return []
        q = np.ascontiguousarray(qvec, dtype=np.float32)
        with self._lock:
            if self._resident is not None:
                scores = self._resident @ q
                idx = np.argpartition(scores, -k)[-k:]
                order = idx[np.argsort(scores[idx])[::-1]]
                return [(int(j), float(scores[j])) for j in order]
            # 冷路径：非常驻模式下走 memmap 分块（慢，仅兜底）
            return self._search_cold(q, k)

    def _recover_row_count(self) -> None:
        """unload/close 后重建对象时的冷恢复：从 DB 的 slot 计数取行数。"""
        try:
            from imgseek import db
            row = db.get_conn().execute(
                "SELECT value FROM settings WHERE key=?",
                (f"next_slot_{self.key}",),
            ).fetchone()
            n = int(row["value"]) if row else 0
        except Exception:  # noqa: BLE001 - DB 不可用时按空处理
            n = 0
        if n > 0:
            self._cap_rows = n
            log.info("vector %s cold-recovered to %d rows", self.key, n)

    def _search_cold(self, q: np.ndarray, k: int):
        mm = np.memmap(self.path, dtype=np.float16, mode="r",
                       shape=(self._cap_rows, self.dim))
        best = []
        for start in range(0, self._cap_rows, config.VEC_CHUNK_ROWS):
            end = min(start + config.VEC_CHUNK_ROWS, self._cap_rows)
            block = np.asarray(mm[start:end], dtype=np.float32)
            scores = block @ q
            part = np.argpartition(scores, -min(k, len(scores)))[-k:]
            best.extend((float(scores[j]), start + int(j)) for j in part)
        del mm
        best.sort(reverse=True)
        out, seen = [], set()
        for s, slot in best:
            if slot in seen:
                continue
            seen.add(slot)
            out.append((slot, s))
            if len(out) >= k:
                break
        return out

    def drop_resident(self) -> None:
        """空闲卸载：释放常驻矩阵（落盘文件不受影响）。"""
        with self._lock:
            self._resident = None

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass
                self._fh.close()
                self._fh = None
            self._resident = None
            self._cap_rows = 0


import os  # noqa: E402
import time  # noqa: E402

_files: dict[str, VectorFile] = {}
_files_lock = threading.Lock()


def get_file(model_key: str) -> VectorFile:
    with _files_lock:
        if model_key not in _files:
            _files[model_key] = VectorFile(model_key)
        return _files[model_key]


def preload_model(model_key: str, total_rows: int) -> None:
    get_file(model_key).preload(total_rows)


def close_all() -> None:
    with _files_lock:
        for f in _files.values():
            f.close()
        _files.clear()
