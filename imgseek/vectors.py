"""向量存储：<model>.f16bin 裸文件 + slot 复用 + 分块暴力点积。

- 文件第 slot 行 = dim 维 float16；无文件头，行号由 DB vector_slot 表解释；
- 50 万条 x 512 维 fp16 = 512MB，memmap 分块 astype(fp32) 点积，
  页缓存接管后单次查询几十毫秒量级（tools/bench_vector.py 可实测）。
"""
import logging
import threading

import numpy as np

import config

log = logging.getLogger("vectors")

_GROW_ROWS = 4096  # 扩容粒度（行）


def _row_count(conn, model_key: str) -> int:
    row = conn.execute(
        "SELECT value FROM settings WHERE key=?", (f"next_slot_{model_key}",)
    ).fetchone()
    return int(row["value"]) if row else 0


class VectorFile:
    def __init__(self, model_key: str):
        self.key = model_key
        self.dim = config.MODELS[model_key]["dim"]
        self.path = config.VECTOR_DIR / f"{model_key}.f16bin"
        self._mm: np.memmap | None = None
        self._cap_rows = 0
        self._lock = threading.Lock()

    def _ensure_capacity(self, rows_needed: int) -> None:
        if rows_needed <= self._cap_rows and self._mm is not None:
            return
        target = ((rows_needed + _GROW_ROWS - 1) // _GROW_ROWS) * _GROW_ROWS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self.path.exists() else "w+"
        with open(self.path, mode) as f:
            f.truncate(target * self.dim * 2)
        self._mm = np.memmap(self.path, dtype=np.float16,
                             mode="r+", shape=(target, self.dim))
        self._cap_rows = target

    def write(self, conn, slot: int, vec_f32: np.ndarray) -> None:
        """把一条 L2 归一化向量写入 slot（原地覆写，永不碎片化）。"""
        with self._lock:
            self._ensure_capacity(slot + 1)
            self._mm[slot] = vec_f32.astype(np.float16)

    def search(self, conn, qvec: np.ndarray, topk: int | None = None):
        """分块暴力余弦（向量已 L2 归一化 => 点积即余弦）。

        返回 [(slot, score)] 按 score 降序，最多 topk 条。
        """
        k = topk or config.VEC_TOPK
        total = _row_count(conn, self.key)
        if total <= 0:
            return []
        q = np.ascontiguousarray(qvec, dtype=np.float32)
        chunk = config.VEC_CHUNK_ROWS
        best: list[tuple[float, int]] = []  # (score, slot) 小根堆语义用排序替代
        with self._lock:
            self._ensure_capacity(total)
            for start in range(0, total, chunk):
                end = min(start + chunk, total)
                block = np.asarray(self._mm[start:end], dtype=np.float32)
                scores = block @ q
                if len(scores) > k:
                    part = np.argpartition(scores, -k)[-k:]
                    for j in part:
                        best.append((float(scores[j]), start + int(j)))
                else:
                    for j, s in enumerate(scores):
                        best.append((float(s), start + j))
        best.sort(reverse=True)
        out, seen_slots = [], set()
        for s, slot in best:
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            out.append((slot, s))
            if len(out) >= k:
                break
        return out

    def close(self) -> None:
        with self._lock:
            if self._mm is not None:
                self._mm.flush()
                del self._mm
                self._mm = None
                self._cap_rows = 0


_files: dict[str, VectorFile] = {}


def get_file(model_key: str) -> VectorFile:
    if model_key not in _files:
        _files[model_key] = VectorFile(model_key)
    return _files[model_key]


def close_all() -> None:
    for f in _files.values():
        f.close()
