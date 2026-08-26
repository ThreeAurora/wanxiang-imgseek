"""向量检索基准：生成 N 条伪向量实测 memmap 分块点积延迟。

用法：python tools/bench_vector.py [条数=500000] [查询次数=20]
验证计划结论：50 万 x 512 fp16 下 p95 < 250ms。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    qn = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    key = "bench"
    dim = config.MODELS["cn_clip_b16"]["dim"]
    path = config.VECTOR_DIR / f"{key}.f16bin"

    rng = np.random.default_rng(42)
    if not path.exists() or path.stat().st_size < n * dim * 2:
        print(f"generating {n} x {dim} fp16 vectors ...")
        config.ensure_dirs()
        with open(path, "wb") as f:
            for start in range(0, n, 65536):
                end = min(start + 65536, n)
                block = rng.standard_normal((end - start, dim))
                block /= np.linalg.norm(block, axis=1, keepdims=True)
                f.write(block.astype(np.float16).tobytes())
    size_mb = path.stat().st_size / 1048576

    # 直接构造 VectorFile（不依赖 DB）
    from imgseek.vectors import VectorFile
    vf = VectorFile(key)
    vf._ensure_capacity(n)
    print(f"vector file: {size_mb:.0f} MB, {n} rows x {dim} dim")

    lat = []
    for i in range(qn):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        t0 = time.perf_counter()
        hits = vf._search_nodb(q, config.VEC_TOPK)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"{qn} queries over {n} rows: "
          f"min={lat[0]:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms "
          f"max={lat[-1]:.1f}ms")
    print("verdict:", "PASS (p95 < 250ms)" if p95 < 250 else "FAIL -> upgrade plan")

    path.unlink(missing_ok=True)
    print("cleanup done")


# 给 VectorFile 打一个绕过 DB 的检索补丁（bench 专用）
def _search_nodb(self, qvec, topk):
    total = self._cap_rows
    q = np.ascontiguousarray(qvec, dtype=np.float32)
    chunk = config.VEC_CHUNK_ROWS
    best = []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        block = np.asarray(self._mm[start:end], dtype=np.float32)
        scores = block @ q
        part = np.argpartition(scores, -min(topk, len(scores)))[-topk:]
        best.extend((float(scores[j]), start + int(j)) for j in part)
    best.sort(reverse=True)
    return best[:topk]


from imgseek import vectors  # noqa: E402
vectors.VectorFile._search_nodb = _search_nodb

if __name__ == "__main__":
    main()
