"""抽样 10 张真实照片做效果演示：OCR 文字提取 + CLIP 语义匹配。

生成带缩略图的 HTML 报告并自动打开；纯离线，不碰索引库。
用法：python tools/demo_10.py [目录] [数量]
"""
import base64
import io
import random
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from imgseek import decoder  # noqa: E402
from imgseek.clip_models import ClipSession  # noqa: E402
from imgseek.ocr import OcrEngine, register_cuda_dll_dirs  # noqa: E402

DEFAULT_ROOT = r"G:\1-现实\1-手机拍照\2026\01\2222222"
QUERIES = ["日落 海边", "美食 食物", "夜景 灯光", "人物 人像",
           "风景 山水 树", "文字 屏幕 文档"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    n_pick = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    register_cuda_dll_dirs()
    files = [f for f in Path(root).rglob("*") if f.suffix.lower() in EXTS]
    if not files:
        print("no images under", root)
        return
    random.seed(20260826)
    picks = random.sample(files, min(n_pick, len(files)))

    eng = OcrEngine()
    eng.ensure()
    sess = ClipSession("cn_clip_b16")
    sess.load()

    items, crops = [], []
    for f in picks:
        try:
            img = decoder.load_rgb(f.read_bytes())
        except Exception as e:  # noqa: BLE001
            print("skip", f.name, str(e)[:60])
            continue
        t0 = time.perf_counter()
        txt = eng.run_text(decoder.long_edge_bgr(img, config.OCR_MAX_EDGE))
        ocr_ms = (time.perf_counter() - t0) * 1000
        crops.append(decoder.center_crop(img, 224))
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, "WEBP", quality=78)
        items.append({"name": f.name, "path": str(f),
                      "thumb": base64.b64encode(buf.getvalue()).decode(),
                      "ocr": txt.replace("\n", " / ")[:90],
                      "ocr_ms": ocr_ms})

    embs = sess.encode_images(crops)
    qvec = {q: sess.encode_texts([q])[0] for q in QUERIES}
    sims = {q: embs @ v for q, v in qvec.items()}

    cards = []
    for i, it in enumerate(items):
        tags = " ".join(
            f'<span class="tag">{q.split()[0]} {sims[q][i]:.2f}</span>'
            for q in QUERIES)
        ocr_html = it["ocr"] or "<i>(未识别到文字)</i>"
        cards.append(
            f'<div class="card"><img src="data:image/webp;base64,{it["thumb"]}">'
            f'<div class="info"><b>{it["name"]}</b>'
            f'<div class="ocr">OCR({it["ocr_ms"]:.0f}ms): {ocr_html}</div>'
            f'<div>{tags}</div></div></div>')
    ranks = ""
    for q in QUERIES:
        order = np.argsort(-sims[q])[:3]
        figs = "".join(
            f'<figure><img src="data:image/webp;base64,{items[i]["thumb"]}">'
            f'<figcaption>{sims[q][i]:.3f} · {items[i]["name"][:20]}</figcaption></figure>'
            for i in order)
        ranks += f"<h3>「{q}」Top3</h3><div class=\"row\">{figs}</div>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>万象图搜 - 抽样演示</title><style>
body{{font-family:"Segoe UI","Microsoft YaHei";background:#17171b;color:#ddd;
     max-width:1100px;margin:24px auto;padding:0 16px}}
.card{{display:flex;gap:12px;background:#222228;border-radius:8px;padding:10px;margin:8px 0}}
.card img{{width:170px;height:128px;object-fit:cover;border-radius:6px}}
.tag{{background:#2c3a55;color:#8ab8ff;border-radius:10px;padding:1px 8px;
     margin-right:6px;font-size:12px}}
.ocr{{color:#9a9aa4;font-size:13px;margin:4px 0}}
.row{{display:flex;gap:10px}} figure{{margin:0;text-align:center;width:160px}}
figure img{{width:150px;height:112px;object-fit:cover;border-radius:6px}}
figcaption{{font-size:12px;color:#9a9aa4;word-break:break-all}}
</style></head><body>
<h2>万象图搜 · 抽样 {len(items)} 张真实照片（RapidOCR-GPU + Chinese-CLIP B/16）</h2>
{''.join(cards)}
<h2>语义查询排行</h2>
{ranks}
</body></html>"""
    out = config.DATA_DIR / "demo_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"report written: {out}")
    webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
