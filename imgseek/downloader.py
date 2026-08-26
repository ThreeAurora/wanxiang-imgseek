"""HuggingFace 权重下载：按 MODELS 注册表清单拉取现成 ONNX 导出。

网络策略（实测结论 2026-08）：
- hf-mirror.com 镜像会把大文件 308 重定向回 huggingface.co，
  官方站直连不可达 => 镜像路线不可用；
- 本地代理 http://127.0.0.1:7890 访问 huggingface.co 稳定可用，
  故默认走代理（可被既有 HTTPS_PROXY/HF_ENDPOINT 覆盖）。
"""
import logging
import os

import config

log = logging.getLogger("downloader")


def _prepare_env() -> None:
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def is_ready(model_key: str) -> bool:
    m = config.MODELS[model_key]
    target = config.MODEL_DIR / model_key
    return all((target / f).is_file() for f in m["files"])


def ensure_model(model_key: str) -> bool:
    """确保权重齐备（缺哪个下哪个），成功返回 True。"""
    if is_ready(model_key):
        return True
    m = config.MODELS[model_key]
    target = config.MODEL_DIR / model_key
    target.mkdir(parents=True, exist_ok=True)
    _prepare_env()
    from huggingface_hub import hf_hub_download
    for fname in m["files"]:
        dest = target / fname
        if dest.is_file():
            continue
        log.info("downloading %s :: %s ...", m["repo"], fname)
        hf_hub_download(repo_id=m["repo"], filename=fname,
                        local_dir=str(target))
        log.info("done %s", fname)
    log.info("model %s ready at %s", model_key, target)
    return True
