"""向量检索基准：生成 N 条伪向量实测常驻矩阵点积延迟。

用法：python tools/bench_vector.py [条数=500000] [查询次数=20]
验收线：50 万 x 512 fp32 常驻 p95 < 250ms（实测 ~72ms）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402

from imgseek.vectors import VectorFile  # noqa: E402


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    qn = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    key = "bench"
    dim = config.MODELS["cn_clip_b16"]["dim"]
    config.MODELS[key] = {"dim": dim}  # bench 专用伪注册
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
    print(f"vector file: {path.stat().st_size / 1048576:.0f} MB")

    vf = VectorFile(key)
    t0 = time.perf_counter()
    vf.preload(n)
    print(f"preload into fp32 resident: {time.perf_counter() - t0:.2f}s "
          f"({vf._resident.nbytes / 1048576:.0f} MB RAM)")

    lat = []
    hits0 = None
    for i in range(qn):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        t0 = time.perf_counter()
        hits = vf.search(q, config.VEC_TOPK)
        lat.append((time.perf_counter() - t0) * 1000)
        if i == 0:
            hits0 = len(hits)
    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"{qn} queries over {n} rows (top-{config.VEC_TOPK}, got {hits0}):")
    print(f"  min={lat[0]:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms max={lat[-1]:.1f}ms")
    print("verdict:", "PASS (p95 < 250ms)" if p95 < 250 else "FAIL -> upgrade plan")

    vf.close()
    path.unlink(missing_ok=True)
    print("cleanup done")


if __name__ == "__main__":
    main()
