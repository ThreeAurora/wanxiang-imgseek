# 模型权重分片说明 / Model Weights (Split Parts)

GitHub 单文件硬限 100MB，以下 3 个权重以 ≤90MB 分片入库；克隆后先重组为原文件再使用。
GitHub hard-limits a single file at 100 MB, so the 3 weights below are stored as ≤90 MB parts; reassemble them after cloning.

| 原文件 / original file | 分片 / parts | 原始字节数 / exact size (bytes) |
|---|---|---|
| `cn_clip_b16/onnx/model_fp16.onnx` | `.part-00` ~ `.part-03`（4 片） | 377377730 |
| `clip_b32/onnx/vision_model_fp16.onnx` | `.part-00` ~ `.part-01`（2 片） | 176080659 |
| `clip_b32/onnx/text_model_fp16.onnx` | `.part-00` ~ `.part-01`（2 片） | 127339794 |

重组（在仓库根目录执行）/ Reassemble (run from the repo root):

```bash
cat cn_clip_b16/onnx/model_fp16.onnx.part-*          > cn_clip_b16/onnx/model_fp16.onnx
cat clip_b32/onnx/vision_model_fp16.onnx.part-*      > clip_b32/onnx/vision_model_fp16.onnx
cat clip_b32/onnx/text_model_fp16.onnx.part-*        > clip_b32/onnx/text_model_fp16.onnx
```

重组后按上表字节数核对；不符则删除重组结果重试。分片由 `split -b 90M -d` 生成，顺序即文件名字典序。
Verify byte sizes against the table; delete and retry on mismatch. Parts were produced by `split -b 90M -d` — concatenation order is the lexicographic order of the part names.
