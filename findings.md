# 发现与踩坑记录

## 环境硬事实（实测确认）
- onnxruntime-gpu 1.28.0 的 CUDA EP 需要 **CUDA 13 + cuDNN 9**（cublasLt64_13.dll）；系统 Toolkit 是 12.3/12.4 不匹配；驱动 591.59 支持 CUDA 13.1 ✓
- CUDA13 DLL 解决方案：pip 装 nvidia-cuda-runtime/cublas/cudnn/cufft/curand-cu13（~1.5GB）+ os.add_dll_directory 注册 site-packages/nvidia/*/bin
- **CUDA EP 可用性必须用 session.get_providers()[0] 实测**——RapidOCR 构造失败会静默回退 CPU 只打 WARNING，"没抛异常"是假阳性
- RapidOCR 1.4.4 返回 (result, elapse)，result=[[box四点, text, score],...]；构造 kwargs det_use_cuda 等前缀
- SQLite 无 reverse() 函数（MySQL 才有）；3.53.4 支持 UPDATE...FROM、FTS5 trigram
- hf-mirror.com 对大文件(LFS) 308 重定向回 huggingface.co（直连被墙即失败）；**走代理 127.0.0.1:7890 直连官方站稳定**
- Xenova ONNX 权重 IO：chinese_clip 整模输出 image_embeds/text_embeds 可按名自适应；输入 pixel_values/input_ids/attention_mask，缺侧喂 dummy 零张量即可
- chinese-clip 文本 max_len=52 vs CLIP B/32=77；pad token [PAD]/<|endoftext|> 自适应

## 向量检索基准结论（RTX 2060 + 这台机器的磁盘）
- memmap fp16 分块 astype 点积：50 万条 p95 ≈ **2s FAIL**（IO + fp16→fp32 转换双重瓶颈）
- fp32 常驻矩阵单次 matmul：p50=72ms / p95=85ms **PASS**；preload 9.2s（984MB RAM）
- 结论见 config.RESIDENT_VECTORS=True；unload 后冷路径靠 DB next_slot 计数恢复行数

## 工程教训（本项目踩过）
1. **thread-local 连接 + 线程池 = 未提交事务互锁**：worker 线程写 DB 不 commit，其他线程全部 locked。解法：DB 写收拢单一编排线程。
2. **backlog 补捞必须有"在飞集合"**：攒批 flush 前状态未更新，同一行会被无限重复捞取。
3. **上游阶段失败要级联终止下游状态**，否则永远挂在 pending 统计里。
4. **SQLite 大批量 IN 列表会撞参数上限**，用 TEMP 表 join 替代。
5. 索引目录绝不能包含程序自身的 data 目录（会把缩略图当图片吞进去），scanner 跳过 DATA_DIR。
6. RateMeter 惰性清理：rate() 里也要清过期样本，否则空闲后速率显示冻结。
7. Git Bash 向原生 curl 传中文 JSON body 会编码损坏，测试用 Python urllib。

## 外部项目情报
- MaterialSearch：主仓库只剩前端，核心在 materialsearch-core pip 包且 API 层不开源；仅 chinese-clip 单模型
- AnyTXT v1.3.3173（2026-06）：仅 OCR 图片文字搜索（图片慧眼），无任何 embedding 能力
- Everything 本身不索引内容只索引文件名 —— 本项目定位依据

## 待实测
- pip nvidia-cu13 装完后的 OCR/CLIP GPU 真实吞吐（当前 CPU：OCR ~2.5 张/s、B/16 编码 ~3 张/s、文本编码 ~300ms/次）
- 十万级真实目录扫描时长与断点续传表现
