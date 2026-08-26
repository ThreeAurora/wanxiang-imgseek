# 发现与踩坑记录

## 环境（探查确认的事实）
- onnxruntime 仅装 gpu 版 1.28.0（无双装冲突），providers 含 CUDAExecutionProvider
- RapidOCR 1.4.4 构造 kwargs 按 det_use_cuda/cls_use_cuda/rec_use_cuda/*_model_path/intra_op_num_threads 前缀解析（源码 utils/parse_parameters.py 核实）；自带模型在 site-packages/rapidocr_onnxruntime/models
- torch 2.2.2 是 CPU 版，勿走 torch 推理路径；huggingface_hub 1.4.1 已装不要动
- SQLite 3.53.4 支持 FTS5 trigram tokenizer（实测建表通过）
- WebFetch 对 github.com 被安全策略拦截 → curl 走代理 127.0.0.1:7890 可用；HF 下载优先 HF_ENDPOINT=https://hf-mirror.com 直连
- ONNX 现成权重实测存在：Xenova/clip-vit-base-patch32（vision_model_fp16.onnx 176MB + text_model_fp16.onnx 127MB 分塔）；Xenova/chinese-clip-vit-base-patch16（整模 model_fp16.onnx 377MB）；均带 tokenizer.json/preprocessor_config.json

## 外部项目情报
- MaterialSearch：主仓库只剩前端，核心拆到 materialsearch-core pip 包且 API 层不开源、部分代码故意混淆；仅 chinese-clip-vit-base-patch16；检索侧 J3455 上每秒匹配 3.1 万张
- Everything 不索引内容只索引文件名 —— 本项目差异化定位依据

## 待实测
- RTX 2060 上 ORT CUDA EP 是否可用（需系统 CUDA12/cuDNN9 DLL，pip 包不带）→ ocr.py 白图试跑探测
- Xenova 导出模型的实际输入输出名（实现时打印 IO 自适应：找 *image_embeds*/*text_embeds* 输出）
- chinese-clip 文本 max_length=52 vs clip B/32=77（从 config 自适应）
