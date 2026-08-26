# 任务计划：万象图搜（imgseek）

## 目标
本地超轻量图片搜索引擎：Everything 式交互（搜索框+即时列表），OCR 文字搜索 + 中英双 CLIP 模型可切换的语义搜索，几十万张量级，免 Docker，零侵入（不复制移动用户文件）。

计划全文（已批准）：`C:\Users\Administrator\.claude\plans\silly-inventing-dijkstra.md`

## 关键调研结论（2026-08）
- 无现成项目同时满足全部刚需。MaterialSearch(1947星) 最接近但无 OCR、锁死中文模型、核心 API 闭源、UI 为缩略图墙。
- Immich 只借思想不借代码：ML 推理串行排队、内容寻址缩略图、DB 即检查点续传、smart search 流程、坏图容错。

## Phase 0: 骨架
- [ ] 目录结构 + planning 三文件
- [ ] 安装 fastapi/uvicorn/tokenizers
- [ ] config.py（常量 + MODELS 注册表）
- [ ] db.py（完整 schema：settings/folder/image/embed_status/vector_slot/free_slot/stats/ocr_fts trigram）
- [ ] main.py + api.py 占位 + web/index.html 占位
- [ ] 验证：python main.py 起服务自动开浏览器，/api/status 通

## Phase 1: 扫描+浏览
- [ ] scanner.py：os.walk + mtime/size 增量 + epoch 软删除
- [ ] decoder.py：统一解码 EXIF 转正、损坏容错、sha1
- [ ] thumbs.py：256px WebP sha1 分桶
- [ ] pipeline.py 雏形：扫描线程 + CPU 池缩略图
- [ ] search.py 文件名过滤路 + api search 端点
- [ ] index.html 完整虚拟滚动（64px 行高、对象池、懒加载、debounce+seq）
- [ ] 验证：大目录扫描可中断续传；过滤 <100ms；滚动流畅

## Phase 2: OCR 管线
- [ ] ocr.py：RapidOCR 封装，白图试跑探测 CUDA，失败回退 CPU
- [ ] pipeline GPU 单线程：OCR → 入库 + FTS 同步
- [ ] search FTS 路（trigram MATCH + LIKE 回退）
- [ ] 验证：中文截图文字可搜；kill -9 续传；坏图不阻塞；状态栏如实显示后端

## Phase 3: 双 CLIP + 融合
- [ ] downloader.py：hf_hub_download 权重清单（HF_ENDPOINT 默认 hf-mirror）
- [ ] clip_models.py：双模型 ONNX 会话（CUDA→CPU）、图像/文本编码 L2 归一化
- [ ] vectors.py：f16bin memmap slot 复用、分块点积 top-k
- [ ] RRF(k=60) 融合 + sources 标注 + 模型切换 API/UI
- [ ] tools/bench_vector.py：50 万条延迟基准（p95<250ms 达标）
- [ ] 验证：「日落 海滩」命中无文字照片；切英文模型结果变化

## Phase 4: 打磨
- [ ] 空闲卸载模型会话（默认 30min）；向量常驻开关（默认关）
- [ ] 定时增量扫描（默认 6h）+ scan/start|stop + retry
- [ ] /api/status 进度速率完善（stats 计数器表）
- [ ] README.md
- [ ] 验证：空闲内存回落 ~100MB 级；所有端点异常返回 JSON

## Phase 5: 端到端验证
- [ ] 测试集：中文截图 + 无字风景照 + HEIC + 故意坏图
- [ ] 全剧本执行（见批准计划的验证剧本节）

## 决策
| # | 决策 | 理由 |
|---|---|---|
| D1 | 推理全走 onnxruntime-gpu，torch 不进运行时 | 已有 ort-gpu 1.28 + RTX 2060；tokenizers 直读 tokenizer.json |
| D2 | 向量检索 numpy memmap fp16 分块暴力点积，不引 FAISS/sqlite-vec | 50万×512 p95 预算 250ms 足够；保持绿色免安装 |
| D3 | 双模型各独立 .f16bin + vector_slot 表，UI 一键切换 | 用户对中文语料清洗导致敏感词盲区的顾虑 |
| D4 | DB 即断点续传检查点（status 字段），不做 job 框架 | 简化；中断重启只补 status=0 |
| D5 | GPU 所有 ORT 调用串行在单线程 | 防 6GB 显存争抢 |
| D6 | OCR <8KB 小图跳过、限边 960px；CLIP 先行 OCR 慢啃 | OCR 是吞吐大头（GPU 3-6 张/s） |
| D7 | 项目名「万象图搜」，包名 imgseek（ASCII） | 中文路径不影响 import（脚本目录进 sys.path） |
| D8 | 端口 8747 绑 127.0.0.1 单 worker | 本机单用户 |

## 错误记录
（空）

## Notes
- 根目录 e:\CCSpace\task_plan.md 属于旧项目「日记心理分析」，与本项目无关。
- HF 下载默认 HF_ENDPOINT=https://hf-mirror.com，失败可 HTTPS_PROXY=http://127.0.0.1:7890。
