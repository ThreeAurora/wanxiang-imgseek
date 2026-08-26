"""FastAPI 路由与服务装配。

所有阻塞型路由使用普通 def（FastAPI 自动丢线程池），
服务绑 127.0.0.1 单 worker，面向本机单人使用。
"""
import hashlib
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

import config
from imgseek import db, downloader, pipeline as pl, search, thumbs, vectors
from imgseek.clip_models import MANAGER

log = logging.getLogger("api")

_pipeline = pl.Pipeline()


def _load_model_async(key: str) -> None:
    """后台完成 权重下载 -> 会话加载 -> 向量常驻 -> 流水线开启嵌入。"""
    MANAGER.set_state(key, "loading")
    try:
        downloader.ensure_model(key)
        MANAGER.activate(key)
        MANAGER.ensure_session(key)
        # 已有向量读入常驻矩阵（新索引实时双写）
        n = int(db.get_setting(f"next_slot_{key}", "0"))
        if n > 0:
            vectors.preload_model(key, n)
        _pipeline.clip_manager = MANAGER
        _pipeline.clip_enabled = True
        log.info("model %s activated", key)
    except Exception as e:  # noqa: BLE001 - 状态栏如实展示错误
        log.exception("activate model %s failed", key)
        MANAGER.set_state(key, "error", str(e)[:300])


def _autostart_active_model() -> None:
    key = db.get_setting("active_model", config.DEFAULT_MODEL)
    if key in config.MODELS and downloader.is_ready(key):
        threading.Thread(target=_load_model_async, args=(key,),
                         daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    _pipeline.start()
    _autostart_active_model()
    yield
    try:
        _pipeline.stop()
    except Exception:  # noqa: BLE001 - 关停路径尽力而为
        pass


_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def create_app() -> FastAPI:
    app = FastAPI(title="imgseek", lifespan=lifespan)

    @app.get("/")
    def index():
        # 开发期页面频繁更新，禁缓存避免浏览器用旧版
        return FileResponse(str(config.WEB_DIR / "index.html"),
                            headers={"Cache-Control": "no-cache"})

    # ---------- 状态 ----------
    @app.get("/api/status")
    def status():
        conn = db.get_conn()
        total = conn.execute("SELECT COUNT(*) c FROM image").fetchone()["c"]
        pend_thumb = conn.execute(
            "SELECT COUNT(*) c FROM image WHERE thumb_status=0 AND dead=0"
        ).fetchone()["c"]
        pend_ocr = conn.execute(
            "SELECT COUNT(*) c FROM image WHERE ocr_status=0 AND dead=0"
        ).fetchone()["c"]
        embed_pending = {
            r["model_key"]: r["c"]
            for r in conn.execute(
                "SELECT model_key, COUNT(*) c FROM embed_status "
                "WHERE status=0 GROUP BY model_key"
            ).fetchall()
        }
        active = db.get_setting("active_model", config.DEFAULT_MODEL)
        failed = {
            "thumb": conn.execute(
                "SELECT COUNT(*) c FROM image WHERE thumb_status=2").fetchone()["c"],
            "ocr": conn.execute(
                "SELECT COUNT(*) c FROM image WHERE ocr_status=2").fetchone()["c"],
        }
        return {
            "active_model": active,
            "models": [
                {"key": k, "label": v["label"],
                 "session": MANAGER.states.get(k, "unloaded"),
                 "error": MANAGER.errors.get(k, ""),
                 "pending_embed": embed_pending.get(k, 0)}
                for k, v in config.MODELS.items()
            ],
            "images_total": total,
            "pending": {"thumb": pend_thumb, "ocr": pend_ocr},
            "failed": failed,
            "ocr_backend": _pipeline.ocr_engine.backend,
            "scanning": _pipeline.scanning,
            "paused": _pipeline.paused.is_set(),
            "rate_per_sec": round(_pipeline.rate.rate(), 1),
        }

    # ---------- 监视目录管理 ----------
    class FolderIn(BaseModel):
        path: str

    @app.get("/api/folders")
    def list_folders():
        rows = db.get_conn().execute("""
            SELECT f.id, f.path, f.enabled, f.paused, f.sort_order,
                   COUNT(i.id) AS images,
                   COALESCE(SUM(CASE WHEN i.thumb_status=1 AND i.dead=0
                               THEN 1 ELSE 0 END),0) AS processed,
                   COALESCE(SUM(CASE WHEN i.thumb_status=0 AND i.dead=0
                               THEN 1 ELSE 0 END),0) AS pending,
                   COALESCE(SUM(CASE WHEN i.dead=1 THEN 1 ELSE 0 END),0) AS dead
            FROM folder f LEFT JOIN image i ON i.folder_id = f.id
            GROUP BY f.id ORDER BY f.sort_order, f.id
        """).fetchall()
        return {"folders": [dict(r) for r in rows]}

    class ToggleIn(BaseModel):
        enabled: bool

    @app.post("/api/folders/{folder_id}/toggle")
    def toggle_folder(folder_id: int, body: ToggleIn):
        conn = db.get_conn()
        conn.execute("UPDATE folder SET enabled=? WHERE id=?",
                     (1 if body.enabled else 0, folder_id))
        conn.commit()
        return {"ok": True}

    class PauseIn(BaseModel):
        on: bool

    @app.post("/api/folders/{folder_id}/pause")
    def pause_folder(folder_id: int, body: PauseIn):
        """单目录暂停/继续索引处理（不影响是否纳入搜索结果）。"""
        conn = db.get_conn()
        conn.execute("UPDATE folder SET paused=? WHERE id=?",
                     (1 if body.on else 0, folder_id))
        conn.commit()
        return {"ok": True}

    @app.post("/api/open-data")
    def open_data_folder():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(config.DATA_DIR))  # noqa: S606 - 本机单用户工具
        return {"ok": True}

    class ReorderIn(BaseModel):
        ids: list[int]

    @app.post("/api/folders/reorder")
    def reorder_folders(body: ReorderIn):
        conn = db.get_conn()
        for order, fid in enumerate(body.ids):
            conn.execute("UPDATE folder SET sort_order=? WHERE id=?",
                         (order, fid))
        conn.commit()
        return {"ok": True}

    @app.post("/api/folders")
    def add_folder(body: FolderIn):
        path = os.path.abspath(os.path.normpath(body.path))
        if not os.path.isdir(path):
            raise HTTPException(400, "not a directory")
        conn = db.get_conn()
        cur = conn.execute(
            "INSERT OR IGNORE INTO folder(path) VALUES(?)", (path,))
        conn.commit()
        _pipeline.request_scan()
        return {"ok": True,
                "msg": "" if cur.rowcount else "already exists"}

    @app.delete("/api/folders/{folder_id}")
    def delete_folder(folder_id: int):
        conn = db.get_conn()
        conn.execute("DELETE FROM folder WHERE id=?", (folder_id,))
        conn.commit()
        _pipeline.request_scan()  # 触发 purge
        return {"ok": True}

    # ---------- 扫描控制 ----------
    @app.post("/api/scan/start")
    def scan_start():
        _pipeline.request_scan()
        return {"ok": True}

    # ---------- 失败重跑 / 资源释放 ----------
    class RetryIn(BaseModel):
        stage: str  # thumb | ocr | embed

    @app.post("/api/retry")
    def retry(body: RetryIn):
        col = {"thumb": "thumb_status", "ocr": "ocr_status"}.get(body.stage)
        conn = db.get_conn()
        if col:
            n = conn.execute(
                f"UPDATE image SET {col}=0 WHERE {col}=2 AND dead=0"
            ).rowcount
            conn.commit()
            return {"ok": True, "reset": n}
        if body.stage == "embed":
            if not MANAGER.session_ready():
                raise HTTPException(400, "no active model session")
            key = MANAGER.active_key
            n = conn.execute(
                "UPDATE embed_status SET status=0 "
                "WHERE status=2 AND model_key=?", (key,)).rowcount
            conn.commit()
            return {"ok": True, "reset": n}
        raise HTTPException(400, "unknown stage")

    @app.post("/api/unload")
    def unload():
        _pipeline.ocr_engine.unload()
        vectors.close_all()  # 释放常驻矩阵与句柄；查询自动走冷路径
        return {"ok": True}

    class PauseIn(BaseModel):
        on: bool

    @app.post("/api/pipeline/pause")
    def pipeline_pause(body: PauseIn):
        if body.on:
            _pipeline.paused.set()
        else:
            _pipeline.paused.clear()
        return {"ok": True, "paused": body.on}

    # ---------- 模型切换（P3 接管会话加载，当前持久化选择） ----------
    class ModelIn(BaseModel):
        key: str

    @app.post("/api/models/activate")
    def models_activate(body: ModelIn):
        if body.key not in config.MODELS:
            raise HTTPException(400, "unknown model")
        db.set_setting("active_model", body.key)
        MANAGER.activate(body.key)
        if not MANAGER.session_ready(body.key):
            threading.Thread(target=_load_model_async, args=(body.key,),
                             daemon=True).start()
        else:
            _pipeline.clip_manager = MANAGER
            _pipeline.clip_enabled = True
        return {"ok": True}

    # ---------- 搜索 ----------
    @app.get("/api/search")
    def do_search(q: str = "", model: str | None = None,
                  sort: str = "relevance"):
        return search.search(q=q, model=model, sort=sort)

    # ---------- 缩略图 / 原图 / 打开 ----------
    @app.get("/api/thumb/{image_id}")
    def get_thumb(image_id: int):
        row = db.get_conn().execute(
            "SELECT id, path, content_hash FROM image WHERE id=?",
            (image_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        h = row["content_hash"]
        if h and thumbs.exists(h):
            return FileResponse(str(thumbs.path_for(h)),
                                media_type="image/webp", headers=_IMMUTABLE)
        # 兜底：现场生成并回写 hash，下次即走缓存
        try:
            with open(row["path"], "rb") as fh:
                data = fh.read()
            h2 = h or hashlib.sha1(data).hexdigest()
            thumbs.make_now(data, h2)
            if not h:
                db.get_conn().execute(
                    "UPDATE image SET content_hash=? WHERE id=? AND "
                    "content_hash IS NULL", (h2, image_id))
                db.get_conn().commit()
            return FileResponse(str(thumbs.path_for(h2)),
                                media_type="image/webp", headers=_IMMUTABLE)
        except Exception:  # noqa: BLE001 - 坏图出占位
            return Response(content=thumbs.placeholder_bytes(),
                            media_type="image/png")

    @app.get("/api/file/{image_id}")
    def get_file(image_id: int):
        row = db.get_conn().execute(
            "SELECT path FROM image WHERE id=?", (image_id,)).fetchone()
        if row is None or not os.path.exists(row["path"]):
            raise HTTPException(404)
        return FileResponse(row["path"])

    class OpenIn(BaseModel):
        id: int

    @app.post("/api/open")
    def open_file(body: OpenIn):
        row = db.get_conn().execute(
            "SELECT path FROM image WHERE id=?", (body.id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        if not os.path.exists(row["path"]):
            raise HTTPException(404, "file missing on disk")
        os.startfile(row["path"])  # noqa: S606 - 本机单用户工具
        return {"ok": True}

    # ---------- 详情 ----------
    @app.get("/api/detail/{image_id}")
    def detail(image_id: int):
        row = db.get_conn().execute(
            "SELECT * FROM image WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        d = dict(row)
        return d

    @app.exception_handler(Exception)
    async def json_errors(request, exc):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
