# 万象图搜 (imgseek)

本地超轻量图片搜索引擎 —— Everything 的图片版：**一个搜索框，同时搜文件名、图片内文字（OCR）、画面内容（自然语言语义）**。

- 免 Docker，`python main.py` 即起即停；零侵入，不复制不移动你的照片
- 中英双 CLIP 模型一键切换（绕开单一模型的语义盲区）
- 几十万张量级：检索 p50 < 100ms（fp32 常驻向量），前端虚拟滚动
- 全部索引产物收在项目 `data/` 目录，删掉即完全卸载

## 快速开始

```bash
# 依赖（miniconda base 环境已含 onnxruntime-gpu/rapidocr/pillow 等）
pip install "fastapi>=0.115" "uvicorn>=0.30" "tokenizers>=0.20"

python main.py            # 自动打开浏览器 http://127.0.0.1:8747
```

首次使用：右上角「+目录」添加要索引的文件夹 → 自动扫描 → 几分钟后文件名与语义可搜，OCR 文字索引后台慢啃。

## 架构

| 组件 | 选型 |
|---|---|
| OCR | rapidocr-onnxruntime（CUDA 13 可用时自动 GPU，否则 CPU） |
| 语义 | Chinese-CLIP ViT-B/16 + OpenAI CLIP B/32（Xenova ONNX 导出），onnxruntime 推理 |
| 向量检索 | fp32 常驻矩阵单次 matmul（50 万条 p95 85ms）+ f16bin 落盘 |
| 全文 | SQLite FTS5 trigram（外部内容表 + 触发器同步） |
| 元数据 | SQLite WAL，状态字段即断点续传检查点 |
| 前端 | 单文件原生 JS：固定行高虚拟滚动 + 对象池 + 缩略图懒加载 |

## API 一览

```
GET  /api/search?q=&model=&sort=   融合搜索（name/ocr/sem 三路 RRF）
GET  /api/thumb/{id}               内容寻址缩略图（immutable 缓存）
GET  /api/file/{id}                原图预览     POST /api/open {id} 打开原图
GET/POST/DELETE /api/folders      监视目录管理
POST /api/scan/start               触发全量/增量扫描
POST /api/models/activate {key}    切换 cn_clip_b16 / clip_b32（自动下载权重）
POST /api/retry {stage}            重跑失败项（thumb/ocr/embed）
POST /api/unload                   释放 OCR 会话与常驻向量
GET  /api/status                   进度 / 速率 / 后端状态
```

## 权重下载

走本地代理 `127.0.0.1:7890` 直连 huggingface.co（hf-mirror 对大文件会 308 回源导致失败）。
已有代理环境变量时不覆盖。

## 已知边界

- HEIC 依赖 pillow-heif，个别 10bit HDR 图可能解码失败（标失败不出占位，不阻塞队列）
- 空闲 30 分钟自动释放 OCR 会话与常驻向量（查询降级为冷路径 ~2s，重新激活模型恢复）
- 中文 FTS 用 trigram：单词 ≥3 字符走 MATCH 索引，更短词退化为 LIKE 扫描
