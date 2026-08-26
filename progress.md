# 进度时间线

## 2026-08-26

### 规划完成
- 需求确认：Everything 式 UI + 缩略图（几十万张）+ OCR + 中英双模型切换（用户敏感词顾虑）
- 环境探查：RTX 2060 6GB / onnxruntime-gpu 1.28.0 / rapidocr-onnxruntime 1.4.4 / pillow-heif / sqlite 3.53.4 trigram 可用 / E 盘余 197G
- 现有项目调研：确认无现成方案满足全部刚需，MaterialSearch 四缺口（无OCR/单模型/闭源/非列表UI），用户拍板自建
- 方案获批（EnterPlanMode → ExitPlanMode），计划文件 C:\Users\Administrator\.claude\plans\silly-inventing-dijkstra.md
- 耗时预期已向用户说明：10 万张 = 名字即时可搜几分钟、语义半小时内、OCR 挂一晚

### P0 进行中
- 创建 projects/2026/08/万象图搜/{imgseek,web,tools}
- 后台安装 fastapi/uvicorn/tokenizers
