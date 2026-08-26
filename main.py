"""万象图搜唯一入口：python main.py [--port N] [--no-browser] [--rescan]"""
import argparse
import logging
import threading
import webbrowser

import uvicorn

import config
from imgseek import db
from imgseek.api import create_app


def setup_logging(verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # 终端输出保持 ASCII 安全文案；日志文件显式 utf-8
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(config.LOG_PATH), encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="imgseek local image search")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    setup_logging(args.verbose)
    log = logging.getLogger("main")

    db.init_db()
    log.info("starting imgseek at http://%s:%d", config.HOST, args.port)

    if not args.no_browser:
        url = f"http://{config.HOST}:{args.port}/"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app = create_app()
    uvicorn.run(app, host=config.HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
