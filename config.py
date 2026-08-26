"""全局配置与 CLIP 模型注册表。

本文件位于项目根目录，通过脚本启动时根目录自动进入 sys.path，
包内模块以 ``import config`` 顶层引用。
"""
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "index.db"
THUMB_DIR = DATA_DIR / "thumbs"
VECTOR_DIR = DATA_DIR / "vectors"
MODEL_DIR = DATA_DIR / "models"
LOG_PATH = BASE_DIR / "logs" / "imgseek.log"
WEB_DIR = BASE_DIR / "web"

# ---------- 服务 ----------
HOST = "127.0.0.1"
PORT = 8747

# ---------- 扫描范围 ----------
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff", "heic", "heif"}

# ---------- 流水线参数 ----------
THUMB_SIZE = 256            # 缩略图长边像素
OCR_SKIP_MIN_BYTES = 8192   # 小于 8KB 的图跳过 OCR（图标/头像基本无文字）
OCR_MAX_EDGE = 960          # OCR 输入长边限制（加速 det）
CLIP_BATCH = 32             # CLIP 图像批大小（fp16 @ 6GB 显存安全）
COMMIT_EVERY = 256          # 流水线每处理 N 张提交一次事务（断点粒度）
CPU_WORKERS = 3             # 解码/缩略图线程池
QUEUE_MAXSIZE = 64          # CPU -> GPU 队列上限（控制内存）

# ---------- 检索 ----------
SEARCH_LIMIT = 500          # 返回结果上限（Everything 式截断提示）
FTS_LIMIT = 500
VEC_TOPK = 500
VEC_CHUNK_ROWS = 65536      # 冷路径 memmap 分块行数
RRF_K = 60                  # Reciprocal Rank Fusion 常数
# 语义命中下限（余弦）：中文 CLIP 的无关基线高达 0.3-0.4，须按模型分别设
SEM_MIN_SCORE = {"cn_clip_b16": 0.38, "clip_b32": 0.25}
SEM_MIN_SCORE_DEFAULT = 0.30
RESIDENT_VECTORS = True     # 向量 fp32 常驻内存（50 万条约 1GB/模型；
                            # 实测 p50 72ms vs 冷路径 ~2s，见 vectors.py）

# ---------- 维护 ----------
IDLE_UNLOAD_SECONDS = 1800  # 空闲多久后卸载模型会话（P4）
SCAN_INTERVAL_HOURS = 6     # 定时增量扫描间隔（P4）

# 图片解码防护
ImageMaxPixels = 300_000_000

# ---------- CLIP 模型注册表 ----------
# 每个 key 一个独立向量空间（独立 .f16bin 与 vector_slot），可一键切换。
# 权重来自 HuggingFace 现成 ONNX 导出；quantized 为可选 CPU 回退权重，
# 不存在时回退仍用 fp16 文件。
MODELS = {
    "cn_clip_b16": {
        "label": "中文 Chinese-CLIP B/16",
        "repo": "Xenova/chinese-clip-vit-base-patch16",
        "dim": 512,
        "max_text_tokens": 52,
        "arch": "chinese_clip",            # 整模：一次推理同时含双塔
        "vision_onnx": "onnx/model_fp16.onnx",
        "text_onnx": "onnx/model_fp16.onnx",
        "optional_files": ["onnx/model_quantized.onnx"],
        "files": [
            "onnx/model_fp16.onnx",
            "tokenizer.json",
            "vocab.txt",
            "preprocessor_config.json",
            "config.json",
        ],
        "size_px": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "clip_b32": {
        "label": "英文 CLIP B/32",
        "repo": "Xenova/clip-vit-base-patch32",
        "dim": 512,
        "max_text_tokens": 77,
        "arch": "clip_split",              # 分塔：视觉塔与文本塔各自独立 onnx
        "vision_onnx": "onnx/vision_model_fp16.onnx",
        "text_onnx": "onnx/text_model_fp16.onnx",
        "optional_files": [
            "onnx/vision_model_quantized.onnx",
            "onnx/text_model_quantized.onnx",
        ],
        "files": [
            "onnx/vision_model_fp16.onnx",
            "onnx/text_model_fp16.onnx",
            "tokenizer.json",
            "preprocessor_config.json",
            "config.json",
        ],
        "size_px": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}
DEFAULT_MODEL = "cn_clip_b16"


def ensure_dirs() -> None:
    for d in (DATA_DIR, THUMB_DIR, VECTOR_DIR, MODEL_DIR, LOG_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
