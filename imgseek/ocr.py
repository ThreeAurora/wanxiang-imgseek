"""RapidOCR 封装：CUDA 探测 + CPU 回退 + 懒加载/卸载。

探测策略：优先以 CUDA 参数构造并用白图试跑；任何异常（典型为缺
CUDA13/cuDNN9 DLL）都回退 CPU 重建。回退是预期内的正常路径，
状态栏如实展示当前后端。
"""
import logging
import os
import threading
from pathlib import Path
import sysconfig

import numpy as np

log = logging.getLogger("ocr")

_MIN_SCORE = 0.5


def register_cuda_dll_dirs() -> int:
    """注册 pip 安装的 nvidia-*-cu13 wheel 内的 DLL 目录，返回注册数。

    onnxruntime-gpu 1.28 需要 CUDA 13 + cuDNN 9；系统未装时，
    pip wheel（nvidia-cublas-cu13 等）里的 DLL 即为运行时来源。
    """
    base = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    n = 0
    for sub in ("cuda_runtime", "cublas", "cudnn", "cufft", "curand"):
        d = base / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            n += 1
    return n


class OcrEngine:
    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()
        self.backend = "unloaded"

    def _create(self, use_cuda: bool):
        from rapidocr_onnxruntime import RapidOCR
        if use_cuda:
            eng = RapidOCR(det_use_cuda=True, cls_use_cuda=True,
                           rec_use_cuda=True,
                           intra_op_num_threads=2, inter_op_num_threads=2)
        else:
            eng = RapidOCR(intra_op_num_threads=max(2, os.cpu_count() // 2))
        return eng

    def _probe(self, eng) -> None:
        blank = np.full((64, 64, 3), 255, dtype=np.uint8)
        eng(blank)

    def ensure(self):
        with self._lock:
            if self._engine is not None:
                return self._engine
            register_cuda_dll_dirs()
            try:
                eng = self._create(use_cuda=True)
                self._probe(eng)
                self.backend = "cuda"
                log.info("OCR backend: CUDA")
            except Exception as e:  # noqa: BLE001 - 回退属正常路径
                log.warning("OCR CUDA init failed (%s), fallback to CPU",
                            str(e)[:200])
                eng = self._create(use_cuda=False)
                self._probe(eng)
                self.backend = "cpu"
            self._engine = eng
            return eng

    def run_text(self, bgr_image: np.ndarray) -> str:
        """识别 BGR 图，返回按换行拼接的文字（低分框丢弃）。"""
        result, _ = self.ensure()(bgr_image)
        lines = []
        for item in result or []:
            # RapidOCR 1.4.x: [box, text, score]
            try:
                _, text, score = item
            except (TypeError, ValueError):
                continue
            t = str(text).strip()
            if t and float(score) >= _MIN_SCORE:
                lines.append(t)
        return "\n".join(lines)

    def unload(self) -> None:
        with self._lock:
            self._engine = None
            self.backend = "unloaded"
