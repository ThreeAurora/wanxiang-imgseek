# 发现与踩坑记录

## 环境硬事实（实测确认）
- onnxruntime-gpu 1.28 的 CUDA EP 要 CUDA 13；**PyPI 上 nvidia-*-cu13 的 Windows wheel 未发布**（runtime/cublas/cufft/curand 均为 0.0.1 占位包，仅 cudnn-cu13 有真身）；清华镜像只有 sdist 无法构建
- 用户本机 E:\CUDA = CUDA 12.3.0 全套 DLL（cudart64_12/cublas64_12/cublasLt64_12/cufft64_11）
- **最终 GPU 方案：onnxruntime-gpu==1.20.1（CUDA 12 世代）+ 本机 E:\CUDA\bin + pip nvidia-cudnn-cu12**；ocr.register_cuda_dll_dirs 注册这三处
- 驱动 591.59 支持 CUDA 13.1，向下兼容 12.x ✓
- **CUDA EP 可用性必须用 session.get_providers()[0] 实测**——RapidOCR 构造失败会静默回退 CPU 只打 WARNING，"没抛异常"是假阳性
- RapidOCR 1.4.4 返回 (result, elapse)，result=[[box四点, text, score],...]；构造 kwargs det_use_cuda 等前缀
- SQLite 无 reverse() 函数（MySQL 才有）；3.53.4 支持 UPDATE...FROM、FTS5 trigram
- hf-mirror.com 对大文件(LFS) 308 重定向回 huggingface.co（直连被墙即失败）；**走代理 127.0.0.1:7890 直连官方站稳定**
- pip 默认 pypi 源下载 ~GB 级 wheel 易损坏（sha256 校验失败且管道掩盖退出码），国内优先清华镜像
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

## GPU 吞吐实测（RTX 2060，2026-08-26 最终方案）
- ORT 1.28→**1.20.2** + E:\CUDA\bin(12.3) + nvidia-cudnn-cu12/cublas-cu12(pip)：CUDA EP 实测可用
- 关键坑:add_dll_directory 对 providers_cuda.dll 的隐式依赖无效(error 126),必须同时把 DLL 目录 prepend 进 os.environ['PATH']
- OCR(GPU):170ms/张 = 5.9 张/s(CPU 2.5);CLIP B/16 vision batch10:20 张/s(CPU ~3);文本编码 57ms(CPU ~500)
- G 盘真实照片目录(G:\1-现实\1-手机拍照\2026\01\2222222,776 张 4.9GB)已入库 848 总索引,断点冻结中(thumb 584/ocr 603/embed 未完),用户机器忙时勿自动续跑

## 待实测
- 用户空闲时重启服务,GPU 断点续跑剩余 pending
- 十万级真实目录扫描时长与断点续传表现
