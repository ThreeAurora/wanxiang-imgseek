# 进度时间线

## 2026-08-26

### 规划（上午）
- 需求确认：Everything 式 UI + 缩略图 + OCR + 中英双模型切换（敏感词盲区顾虑）
- 环境探查：RTX 2060 6GB / onnxruntime-gpu 1.28.0 / rapidocr 1.4.4 / sqlite 3.53.4 trigram
- 现有项目调研：MaterialSearch 四缺口、AnyTXT 无 CLIP、商业品闭源 → 用户拍板自建
- 方案获批；向用户说明耗时预期（10 万张：名字几分钟、语义半小时、OCR 一晚）

### P0（09:00 前后）
- 目录/三文件/.gitignore；pip 装 fastapi/uvicorn/tokenizers
- config/db/main/api/index.html；修复 logs 目录初始化顺序
- 验证：status/schema/FTS trigram 触发器/folders 中文路径 全通

### P1（09:30 前后）
- scanner/decoder/thumbs/pipeline/search 五模块 + api 完整版 + 前端虚拟滚动完整版
- tools/make_testset.py 生成 68 张测试集（7 张中文截图 + 60 渐变 + 1 坏图）
- 修 3 bug：reverse 函数不存在、DB locked（池线程事务）、项目根误加监视目录
- 验证：68 张精确入库 2 秒缩略图全完成；搜索毫秒级；坏图占位

### P2（09:45 前后）
- RapidOCR 实测：返回 [box,text,score]；系统缺 CUDA13 DLL 自动落 CPU
- ocr.py（探测+回退）、pipeline OCR 阶段 + backlog 补捞、search FTS 路
- 修 2 bug：坏图未级联终止卡 pending=1；RateMeter 空闲不清窗
- 验证：36 秒全流程归零；四个中文关键词精确命中截图

### P3（10:00-10:40）
- downloader（改走代理）/ clip_models（双会话+攒批）/ vectors / RRF / activate API
- 修 3 bug：hf-mirror 308 回源改代理；embed_item 参数 TypeError；backlog 死循环（skip 集合）
- vectors 重写：bench FAIL(memmap p95≈2s) → fp32 常驻 matmul **p95=85ms PASS**
- 验证：「日落」→sunset.jpg、「夜晚 月亮」→night.jpg 均第一；中英切换生效

### P4（10:45-11:00）
- retry/unload 端点、空闲卸载、status 增强（ocr_backend/failed/loading）、README
- 修 2 bug：CUDA 探测假阳性（get_providers() 实测）；unload 冷恢复
- 验证：unload 后冷路径搜索结果一致；OCR 路不受影响

### 待办
- [ ] 十万级压测 + 断点续传显式验证（Phase 5）
- [ ] pip nvidia-cu13 装完后 GPU 吞吐实测（后台任务进行中，~1.5GB）
- [ ] 用户浏览器体验验收
