"""生成端到端测试图片集：data/testset/

- 若干张白底黑字中文「截图」（词组来自 PHRASES，供 OCR 检索）
- 大量无字彩色渐变「风景照」（供语义检索）
- 一张故意损坏的 jpg（坏图容错）
用法：python tools/make_testset.py [数量]
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import config  # noqa: E402

PHRASES = [
    "会议记录 三月销售目标",
    "购物清单 牛奶 鸡蛋 面包",
    "登录页面 用户名 密码 验证码",
    "火车票 G1027 北京西到西安北",
    "药品说明书 每日三次 饭后服用",
    "课程表 高等数学 周二上午",
    "快递单号 SF1234567890",
]

FONT = Path(r"C:\Windows\Fonts\msyh.ttc")


def text_image(idx: int) -> Image.Image:
    img = Image.new("RGB", (800, 500), (250, 250, 248))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT), 40) if FONT.exists() else None
    phrase = PHRASES[idx % len(PHRASES)]
    d.text((60, 80), f"[{idx:03d}] {phrase}", fill=(20, 20, 20), font=font)
    d.text((60, 180), "第二行补充说明文字内容", fill=(60, 60, 60), font=font)
    d.rectangle([60, 300, 700, 420], outline=(120, 120, 120), width=3)
    return img


def scene_image(seed: int) -> Image.Image:
    rng = random.Random(seed)
    w, h = 640, 480
    top = (rng.randint(30, 200), rng.randint(80, 200), rng.randint(150, 255))
    bot = (rng.randint(100, 255), rng.randint(60, 160), rng.randint(20, 90))
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / h
        row = tuple(round(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = row
    return img


def main() -> None:
    n_scene = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = config.DATA_DIR / "testset"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(PHRASES)):
        text_image(i).save(out / f"截图_{i:02d}.png")
    for i in range(n_scene):
        scene_image(i).save(out / f"scene_{i:04d}.jpg", quality=88)
    # 故意损坏的文件
    (out / "broken_000.jpg").write_bytes(b"\xff\xd8\xff\xe0broken-not-jpeg")
    print(f"testset ready at {out}")


if __name__ == "__main__":
    main()
