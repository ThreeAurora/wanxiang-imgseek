# 任务计划：万象图搜（imgseek）

## 目标
本地超轻量图片搜索引擎：Everything 式交互（搜索框+即时列表），OCR 文字搜索 + 中英双 CLIP 模型可切换的语义搜索，几十万张量级，免 Docker，零侵入（不复制移动用户文件）。

计划全文（已批准）：`C:\Users\Administrator\.claude\plans\silly-inventing-dijkstra.md`

## 关键调研结论（2026-08）
- 无现成项目同时满足全部刚需。MaterialSearch(1947星) 最接近但无 OCR、锁死中文模型、核心 API 闭源、UI 为缩略图墙。
- AnyTXT(v1.3.3173) 仅 OCR 图片文字搜索，无 CLIP；Eagle/Excire 为商业闭源。
- Immich 只借思想不借代码：ML 推理串行排队、内容寻址缩略图、DB 即检查点续传、smart search 流程、坏图容错。

## Phase 0: 骨架 —— [X] 完成
- [X] 目录结构 + planning 三文件 + fastapi/uvicorn/tokenizers 安装
- [X] config.py / db.py 完整 schema / main.py / api.py / index.html 占位
- [X] 验证：服务启动、建库含 trigram FTS 触发器、folders 中文路径入库

## Phase 1: 扫描+浏览 —— [X] 完成
- [X] scanner.py（temp 表集合 SQL 增量 + epoch 软删除 + purge 回收 slot）
- [X] decoder.py（EXIF 转正/损坏容错/一次解码派生三份输入）thumbs.py（sha1 分桶 WebP）
- [X] pipeline.py（编排线程 + CPU 池，DB 写集中在编排线程）
- [X] search.py 文件名路 + api 全端点 + index.html 虚拟滚动完整版
- [X] 验证：68 张测试集 2 秒全处理；文件名搜索毫秒级；坏图出占位

## Phase 2: OCR 管线 —— [X] 完成
- [X] ocr.py（CUDA EP 实测探测 get_providers()，防静默回退假阳性）
- [X] pipeline GPU 阶段串行 + backlog 存量补捞（_backlog_skip 防重复捞取）
- [X] search FTS 路（trigram MATCH bm25 + LIKE 兜底 + FTS 安全转义）
- [X] 验证：「销售目标/火车票/验证码/高等数学」四词精确命中对应截图；36 秒全流程；解码失败级联终止不卡 pending

## Phase 3: 双 CLIP + 融合 —— [X] 完成
- [X] downloader.py（走代理直连 huggingface.co；hf-mirror 大文件 308 回源不可用）
- [X] clip_models.py（双模型独立会话、IO 名自适应 image_embeds/text_embeds、攒批嵌入）
- [X] vectors.py（fp32 常驻矩阵单次 matmul + f16bin 双写落盘 + unload 冷恢复）
- [X] RRF(k=60) 三路融合 + sources 标注 + models/activate 异步加载
- [X] bench_vector：50 万条 p95=85ms PASS（预算 250ms）
- [X] 验证：「日落」→ sunset.jpg 第一、「夜晚 月亮」→ night.jpg 第一；中英切换即时生效

## Phase 4: 打磨 —— [X] 完成
- [X] /api/retry（thumb/ocr/embed 失败重置）、/api/unload（OCR 会话+常驻向量）
- [X] 空闲卸载（默认 30min，drop_resident + ocr unload，查询降级冷路径）
- [X] status 增强：ocr_backend / failed 计数 / 模型 loading 态；前端状态栏显示
- [X] README.md

## Phase 5: 端到端验证 —— [ ] 进行中
- [ ] 十万级真实目录压测（等用户提供目录或合成数据）
- [ ] 断点续传显式验证（中途 kill 重启）
- [ ] 用户浏览器实际体验反馈
- [ ] CUDA13 pip 包装完后的 GPU 吞吐实测

## 决策
| # | 决策 | 理由 |
|---|---|---|
| D1 | 推理全走 onnxruntime-gpu，torch 不进运行时 | tokenizers 直读 tokenizer.json |
| D2 | ~~numpy memmap fp16 分块~~ → **fp32 常驻矩阵单次 matmul** | 实测冷路径 p95≈2s FAIL，常驻 p95=85ms PASS |
| D3 | 双模型各独立向量空间，UI 一键切换 | 用户敏感词盲区顾虑 |
| D4 | DB 即断点续传检查点（status 字段） | 中断重启只补 status=0 |
| D5 | GPU 所有 ORT 调用串行在编排线程 | 防 6GB 显存争抢；DB 写也集中此线程 |
| D6 | OCR <8KB 跳过、限边 960px；CLIP 先行 OCR 慢啃 | OCR 是吞吐大头 |
| D7 | 项目「万象图搜」包名 imgseek | 中文路径不影响 import |
| D8 | 端口 8747 绑 127.0.0.1 单 worker | 本机单用户 |
| D9 | HF 权重走代理 127.0.0.1:7890 直连官方站 | hf-mirror 大文件 308 回源不可用（实测） |
| D10 | ~~ORT 1.28 + pip nvidia-cu13~~ → **onnxruntime-gpu 降级 1.20.1 + 本机 E:\CUDA(12.3) + pip nvidia-cudnn-cu12** | PyPI 上 cu13 的 Windows wheel 未发布（runtime/cublas 等均为 0.0.1 占位包）；cu13 pypi 直下又遇 sha256 损坏；CUDA12 路径全部现成 |
| D11 | worker 线程绝不碰 DB，DB 写集中编排线程 | thread-local 连接未提交事务互锁教训 |

## 错误记录
| 错误 | 根因 | 修复 |
|---|---|---|
| logs 目录不存在即开 FileHandler | 初始化顺序 | ensure_dirs 提前 |
| no such function: reverse | SQLite 无 reverse 函数 | Python 端拆好 filename/ext 入 temp 表 |
| database is locked | 池线程连接未提交事务挂锁 | DB 写收拢编排线程统一 executemany+commit |
| 项目根被误加为监视目录 | P0 测试污染 | scanner 跳过 DATA_DIR；清库重来 |
| embed 卡 67 不动 | embed_item 多传 conn 参数 TypeError 循环炸 | 改签名 |
| backlog 同一行无限重复处理 | flush 前 embed_status 仍 0 | _backlog_skip 在飞集合 + flush 后清空 |
| 「OCR backend CUDA」谎报 | RapidOCR 静默回退不抛异常 | _cuda_really_works 用 get_providers() 实测 |
| unload 后搜索 0 结果 | close_all 后新对象 cap_rows=0 | search 冷恢复从 DB slot 计数取行数 |

## Notes
- HF 下载必须走代理；hf-mirror 只适合小文件。
- pip nvidia-cu13 包约 1.5GB 默认源下载慢，装完后 OCR/CLIP 才有真 CUDA。
- 渐变伪场景图区分度弱是合成数据固有局限，sunset/night 有圆形物体可正确命中。
