"""缩略图缓存：256px WebP，按内容 sha1 分桶存取。"""
import io
import logging

from PIL import Image

import config

log = logging.getLogger("thumbs")

_PLACEHOLDER: bytes | None = None


def path_for(content_hash: str):
    """hash 前 2 字符分桶，避免单目录几十万文件。"""
    safe = content_hash or "none"
    return config.THUMB_DIR / safe[:2] / f"{safe}.webp"


def exists(content_hash: str) -> bool:
    return path_for(content_hash).exists()


def save(content_hash: str, thumb_img: Image.Image) -> str:
    p = path_for(content_hash)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    thumb_img.save(tmp, "WEBP", quality=80, method=4)
    tmp.replace(p)  # 原子替换，防半写文件
    return str(p)


def placeholder_bytes() -> bytes:
    """灰底占位 PNG（坏图用），模块级缓存。"""
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (60, 60, 66)).save(buf, "PNG")
        _PLACEHOLDER = buf.getvalue()
    return _PLACEHOLDER


def make_now(data: bytes, content_hash: str) -> str:
    """API 兜底：现场解码生成缩略图（正常路径由流水线完成）。"""
    from imgseek import decoder
    img = decoder.load_rgb(data)
    thumb = img.copy()
    thumb.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE), Image.LANCZOS)
    return save(content_hash, thumb)
