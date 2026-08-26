"""统一图片解码：EXIF 转正、损坏容错、一次解码派生三份输入。

derive() 返回：
- thumb PIL 图（256px，供 thumbs 存盘）
- clip 输入 uint8 HWC（224x224 中心裁剪，GPU 线程批量归一化）
- ocr 输入 BGR uint8（长边 960，None 表示跳过 OCR）
"""
import io

import numpy as np
from PIL import Image, ImageOps

import config

# pillow-heif 注册 HEIF/HEIC 解码器
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = config.ImageMaxPixels


class DecodeError(Exception):
    """无法解码（损坏/不支持/超尺寸）。"""


def load_rgb(data: bytes) -> Image.Image:
    """从原始字节解码为 EXIF 转正后的 RGB 图。"""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Image.DecompressionBombError:
        raise DecodeError("image too large")
    except Exception as e:  # noqa: BLE001 - 一切解码失败同权处理
        raise DecodeError(str(e)) from e


def center_crop(img: Image.Image, size: int) -> np.ndarray:
    """cover 缩放后中心裁剪，返回 uint8 HWC。"""
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    return np.asarray(resized.crop((left, top, left + size, top + size)),
                      dtype=np.uint8)


def long_edge_bgr(img: Image.Image, max_edge: int) -> np.ndarray:
    """长边压到 max_edge，返回 BGR uint8 HWC（OpenCV/RapidOCR 输入约定）。"""
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb[:, :, ::-1]  # RGB -> BGR


def derive(img: Image.Image, need_clip: bool, need_ocr: bool):
    """一次解码派生三份输入（按需裁剪，未需要的返回 None）。"""
    thumb = img.copy()
    thumb.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE), Image.LANCZOS)

    clip = None
    if need_clip:
        m = config.MODELS["cn_clip_b16"]
        clip = center_crop(img, m["size_px"])  # 两模型同为 224，共用一份裁剪

    ocr = long_edge_bgr(img, config.OCR_MAX_EDGE) if need_ocr else None
    return thumb, clip, ocr
