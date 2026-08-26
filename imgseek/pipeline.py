"""后台流水线编排。

线程模型：
- T_work（编排线程，本文件所有逻辑都在它上面）：
    扫描 -> 缩略图批次(CPU 池) -> GPU 阶段串行消费(OCR/CLIP)
- CPU 池只做「读文件/解码/存缩略图」，绝不触碰 DB；
  所有 DB 写集中在编排线程（避免多连接未提交事务互锁）。
- GPU 阶段（ORT 推理）串行执行：RapidOCR 逐张 -> CLIP 攒批。
- DB 即检查点：各阶段 status 落库，中断重启自动从 status=0 续跑；
  「缩略图已完成但 OCR/嵌入待处理」的历史行由 backlog 补捞机制覆盖。
"""
import hashlib
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import config
from imgseek import db, decoder, ocr as ocr_mod, scanner, thumbs

log = logging.getLogger("pipeline")


class RateMeter:
    """滑动窗口速率表（张/秒）。"""

    def __init__(self, window: float = 10.0):
        self._win = window
        self._ts: deque[float] = deque()
        self._lock = threading.Lock()

    def mark(self, n: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            for _ in range(n):
                self._ts.append(now)
            while self._ts and self._ts[0] < now - self._win:
                self._ts.popleft()

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            while self._ts and self._ts[0] < now - self._win:
                self._ts.popleft()
            return len(self._ts) / self._win if self._ts else 0.0


class Item:
    """CPU 池产物 -> GPU 阶段的轻量载荷。"""

    __slots__ = ("image_id", "content_hash", "clip", "ocr")

    def __init__(self, image_id: int, content_hash: str, clip, ocr):
        self.image_id = image_id
        self.content_hash = content_hash
        self.clip = clip          # uint8 HWC 224x224 或 None
        self.ocr = ocr            # BGR uint8 或 None(小图跳过)


class Pipeline:
    def __init__(self) -> None:
        self.stop_evt = threading.Event()
        self.scan_req = threading.Event()
        self.q: queue.Queue[Item] = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.rate = RateMeter()
        self.scanning = False
        # P2 起 OCR 就绪（rapidocr 自带模型）；clip 待 P3 权重就绪后打开
        self.ocr_enabled = True
        self.clip_enabled = False
        self.ocr_engine = ocr_mod.OcrEngine()
        self.clip_manager = None          # P3 注入 ClipManager
        self._backlog_skip: set[int] = set()
        self.next_scan_ts = time.time() + config.SCAN_INTERVAL_HOURS * 3600
        self._last_busy_ts = time.monotonic()
        self._idle_unloaded = False
        self._pool: ThreadPoolExecutor | None = None
        self._t: threading.Thread | None = None

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self.stop_evt.clear()
        self._pool = ThreadPoolExecutor(max_workers=config.CPU_WORKERS,
                                        thread_name_prefix="dec")
        self._t = threading.Thread(target=self._run, name="pipeline",
                                   daemon=True)
        self._t.start()
        log.info("pipeline started (cpu=%d)", config.CPU_WORKERS)

    def stop(self) -> None:
        self.stop_evt.set()
        if self._t:
            self._t.join(timeout=5)

    def request_scan(self) -> None:
        self.scan_req.set()

    # ---------- 主循环 ----------
    def _run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                did = False
                if self._scan_due():
                    self._scan_all()
                    did = True
                if self._thumb_batch() > 0:
                    did = True
                if self._gpu_step() > 0:
                    did = True
                if did:
                    self._last_busy_ts = time.monotonic()
                    self._idle_unloaded = False
                else:
                    self._maybe_idle_unload()
                    self.stop_evt.wait(1.0)
            except Exception:  # noqa: BLE001 - 流水线永不因单次异常退出
                log.exception("pipeline loop error")
                self.stop_evt.wait(5.0)

    def _maybe_idle_unload(self) -> None:
        """空闲超时释放大块内存（OCR 会话 + 向量常驻矩阵）。

        落盘文件不受影响：查询自动走冷路径，新写入照常落盘，
        下次激活模型时 preload 恢复常驻。
        """
        if not config.IDLE_UNLOAD_SECONDS or self._idle_unloaded:
            return
        idle = time.monotonic() - self._last_busy_ts
        if idle < config.IDLE_UNLOAD_SECONDS:
            return
        from imgseek import vectors as vec_mod
        self.ocr_engine.unload()
        for f in list(vec_mod._files.values()):
            f.drop_resident()
        self._idle_unloaded = True
        log.info("idle %.0fs: unloaded ocr engine and resident vectors",
                 idle)

    def _scan_due(self) -> bool:
        if self.scan_req.is_set():
            self.scan_req.clear()
            return True
        if time.time() >= self.next_scan_ts:
            self.next_scan_ts = time.time() + config.SCAN_INTERVAL_HOURS * 3600
            return True
        return False

    def _scan_all(self) -> None:
        self.scanning = True
        try:
            folders = db.get_conn().execute(
                "SELECT id, path FROM folder WHERE enabled=1").fetchall()
            for f in folders:
                if self.stop_evt.is_set():
                    break
                scanner.scan_folder(f["id"], f["path"])
            scanner.purge_dead()
        finally:
            self.scanning = False

    # ---------- 缩略图阶段（CPU 池） ----------
    def _pending_thumb(self, limit: int):
        return db.get_conn().execute(
            "SELECT id, path, size, content_hash FROM image "
            "WHERE thumb_status=0 AND dead=0 ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()

    def _thumb_batch(self) -> int:
        rows = self._pending_thumb(config.CPU_WORKERS * 8)
        if not rows:
            return 0
        jobs = [(dict(r), self._pool.submit(self._decode_one, dict(r)))
                for r in rows]
        oks, fails = [], []
        for row, fut in jobs:
            try:
                res = fut.result()
            except decoder.DecodeError as e:
                fails.append((row["id"], f"decode: {e}"))
            except OSError as e:
                fails.append((row["id"], f"io: {e}"))
            except Exception as e:  # noqa: BLE001 - 单图意外错误同权处理
                fails.append((row["id"], f"unexpected: {e}"))
            else:
                oks.append(res)

        conn = db.get_conn()
        now = time.time()
        if fails:
            # 解码失败 => 下游阶段全部级联终止（否则永远挂在 pending 上）
            conn.executemany(
                "UPDATE image SET thumb_status=2, ocr_status=2, error=? "
                "WHERE id=?",
                [(msg[:500], iid) for iid, msg in fails])
            conn.executemany(
                "UPDATE embed_status SET status=2 WHERE image_id=?",
                [(iid,) for iid, _ in fails])
        if oks:
            conn.executemany(
                "UPDATE image SET thumb_status=1, width=?, height=?, "
                "content_hash=?, indexed_at=? WHERE id=?",
                [(w, h, hh, now, iid) for (_, iid, w, h, hh, _, _) in oks])
        conn.commit()
        for (_, iid, _, _, hh, clip, ocr_img) in oks:
            if clip is not None or ocr_img is not None:
                # 阻塞 put：GPU 消费速度即背压，队列上限即内存上限
                self.q.put(Item(iid, hh, clip, ocr_img))
            self.rate.mark()
        if fails:
            for iid, msg in fails:
                log.warning("image %d failed: %s", iid, msg)
        return len(rows)

    def _decode_one(self, row: dict):
        """读文件 -> 解码 -> 存缩略图；返回 (ok, id, w, h, hash, clip, ocr)。"""
        with open(row["path"], "rb") as fh:
            data = fh.read()
        h = row["content_hash"] or hashlib.sha1(data).hexdigest()
        img = decoder.load_rgb(data)
        w, hh = img.size
        need_clip = self.clip_enabled
        need_ocr = self.ocr_enabled and row["size"] >= config.OCR_SKIP_MIN_BYTES
        thumb_img, clip, ocr_img = decoder.derive(img, need_clip, need_ocr)
        if not thumbs.exists(h):  # 增量场景避免重写已有缩略图
            thumbs.save(h, thumb_img)
        return ("ok", row["id"], w, hh, h, clip, ocr_img)

    # ---------- GPU 阶段（串行；OCR 已接入，CLIP 由 P3 注入） ----------
    def _gpu_step(self) -> int:
        processed = 0
        while not self.stop_evt.is_set():
            item = None
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                item = self._take_backlog()
            if item is None:
                break
            self._process_item(item)
            processed += 1
            self.rate.mark()
            db.get_conn().commit()
        if processed:
            if self.clip_manager is not None:
                self.clip_manager.flush(db.next_slot)
            self._backlog_skip.clear()  # flush 后状态已更新，允许重新扫描
            db.get_conn().commit()
        return processed

    def _take_backlog(self) -> Item | None:
        """补捞存量：缩略图已完成但 OCR/嵌入仍待处理的行。

        关键：攒批 flush 之前 embed_status 仍为 0，同一行会被反复查出，
        用 _backlog_skip 保证「在飞」的行不重复进入处理。
        """
        if len(self._backlog_skip) > 4096:
            self._backlog_skip.clear()
        cond_extra = ""
        if self.clip_enabled:
            cond_extra = (" OR EXISTS(SELECT 1 FROM embed_status e "
                          "WHERE e.image_id=i.id AND e.status=0)")
        rows = db.get_conn().execute(
            f"SELECT id, path, size, content_hash, ocr_status FROM image i "
            f"WHERE thumb_status=1 AND ocr_status=0{cond_extra} "
            f"AND dead=0 ORDER BY id LIMIT 64").fetchall()
        for row in rows:
            iid = row["id"]
            if iid in self._backlog_skip:
                continue
            item = self._decode_for_gpu(dict(row))
            if item is None:
                continue
            self._backlog_skip.add(iid)
            return item
        return None

    def _decode_for_gpu(self, row: dict) -> Item | None:
        """backlog 行就地解码（编排线程内，量小不构成瓶颈）。"""
        try:
            with open(row["path"], "rb") as fh:
                data = fh.read()
            h = row["content_hash"] or hashlib.sha1(data).hexdigest()
            img = decoder.load_rgb(data)
            # 已完成 OCR 的 backlog 图只需补嵌入
            need_ocr = (row["ocr_status"] == 0
                        and row["size"] >= config.OCR_SKIP_MIN_BYTES)
            _, clip, ocr_img = decoder.derive(img, self.clip_enabled,
                                              need_ocr)
            return Item(row["id"], h, clip, ocr_img)
        except decoder.DecodeError as e:
            db.get_conn().execute(
                "UPDATE image SET ocr_status=2, error=? WHERE id=? AND ocr_status=0",
                (f"decode: {e}"[:500], row["id"]))
            db.get_conn().commit()
        except OSError:
            pass  # 文件暂时不可读（占用/网络盘），下轮再试
        return None

    def _process_item(self, item: Item) -> None:
        conn = db.get_conn()
        # ---- OCR ----
        if item.ocr is not None:
            try:
                text = self.ocr_engine.run_text(item.ocr)
                conn.execute(
                    "UPDATE image SET ocr_status=1, ocr_text=?, error=NULL "
                    "WHERE id=?", (text, item.image_id))
            except Exception as e:  # noqa: BLE001 - 单图失败不阻塞
                log.warning("ocr %d failed: %s", item.image_id, str(e)[:200])
                conn.execute(
                    "UPDATE image SET ocr_status=2, error=? WHERE id=?",
                    (str(e)[:500], item.image_id))
        else:
            conn.execute(
                "UPDATE image SET ocr_status=1 WHERE id=? AND ocr_status=0",
                (item.image_id,))
        # ---- CLIP 嵌入 ----
        if self.clip_manager is not None:
            self.clip_manager.embed_item(item, db.next_slot)
