"""目录扫描：os.walk + mtime/size 增量比对 + epoch 软删除。

策略：把本轮扫描结果与"待变更集合"写入 TEMP 表，全部差异用集合 SQL
完成，避免几十万行逐条往返，也规避大 IN 列表参数上限。
DB 即检查点——任何一步中断，重扫只会补差异。
"""
import logging
import os

import config
from imgseek import db

log = logging.getLogger("scanner")


def scan_folder(folder_id: int, path: str) -> dict:
    """扫描单个监视目录，返回统计信息。"""
    conn = db.get_conn()
    row = conn.execute(
        "SELECT current_epoch FROM folder WHERE id=?", (folder_id,)
    ).fetchone()
    epoch = (row["current_epoch"] if row else 0) + 1

    conn.execute("CREATE TEMP TABLE IF NOT EXISTS scan_tmp("
                 " path TEXT PRIMARY KEY, filename TEXT, ext TEXT,"
                 " size INTEGER, mtime REAL)")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS changed_ids(id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM scan_tmp")
    conn.execute("DELETE FROM changed_ids")

    exts = {e.lower() for e in config.IMAGE_EXTS}
    skip_self = os.path.abspath(config.DATA_DIR)
    found = 0
    batch = []
    for root, dirs, files in os.walk(path):
        # 跳过隐藏目录与自身数据目录（防索引吞噬自己的缓存产物）
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and os.path.abspath(os.path.join(root, d)) != skip_self
        ]
        for fn in files:
            if "." not in fn:
                continue
            ext = fn.rsplit(".", 1)[-1].lower()
            if ext not in exts:
                continue
            full = os.path.join(root, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            batch.append((full.replace("\\", "/"), fn, ext,
                          st.st_size, st.st_mtime))
            found += 1
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT OR IGNORE INTO scan_tmp VALUES(?,?,?,?,?)", batch)
                batch.clear()
    if batch:
        conn.executemany("INSERT OR IGNORE INTO scan_tmp VALUES(?,?,?,?,?)",
                         batch)

    # 0. 记录"内容可能已变"的既有文件（新文件不算）
    conn.execute("""
        INSERT OR IGNORE INTO changed_ids
        SELECT i.id FROM image i JOIN scan_tmp t ON i.path = t.path
        WHERE i.size <> t.size OR i.mtime <> t.mtime
    """)

    # 1. 新文件入库（filename/ext 已在扫描时拆好）
    cur = conn.execute("""
        INSERT INTO image(folder_id, path, filename, ext, size, mtime,
                          last_seen_epoch)
        SELECT ?, t.path, t.filename, t.ext, t.size, t.mtime, ?
        FROM scan_tmp t
        WHERE NOT EXISTS (SELECT 1 FROM image i WHERE i.path = t.path)
    """, (folder_id, epoch))
    new_count = cur.rowcount

    # 新文件 + 变更文件都需要双模型嵌入任务行（变更行重置为待处理）
    for mk in config.MODELS:
        conn.execute("""
            INSERT OR IGNORE INTO embed_status(image_id, model_key, status)
            SELECT id, ?, 0 FROM image
            WHERE last_seen_epoch = ? AND folder_id = ?
        """, (mk, epoch, folder_id))
        conn.execute("""
            UPDATE embed_status SET status = 0
            WHERE model_key = ?
              AND image_id IN (SELECT id FROM changed_ids)
        """, (mk,))
    # 变更行：重置所有阶段状态（保留 id 与 vector_slot 原地覆写）
    conn.execute("""
        UPDATE image SET size = (SELECT t.size FROM scan_tmp t WHERE t.path = image.path),
               mtime = (SELECT t.mtime FROM scan_tmp t WHERE t.path = image.path),
               thumb_status = 0, ocr_status = 0, ocr_text = '',
               content_hash = NULL, error = NULL, indexed_at = NULL, dead = 0,
               last_seen_epoch = ?
        WHERE id IN (SELECT id FROM changed_ids)
    """, (epoch,))
    changed = conn.execute("SELECT changes()").fetchone()[0]

    # 2. 未变但本轮见到的：刷新 epoch（防止被误标 dead）
    conn.execute("""
        UPDATE image SET last_seen_epoch = ?, dead = 0
        WHERE folder_id = ? AND last_seen_epoch < ? AND dead = 0
          AND path IN (SELECT path FROM scan_tmp)
    """, (epoch, folder_id, epoch))

    # 3. 本轮未见的 → 软删除标记
    conn.execute("""
        UPDATE image SET dead = 1
        WHERE folder_id = ? AND last_seen_epoch < ? AND dead = 0
    """, (folder_id, epoch))
    dead_count = conn.execute("SELECT changes()").fetchone()[0]

    conn.execute("UPDATE folder SET current_epoch=? WHERE id=?", (epoch, folder_id))
    conn.commit()
    log.info("scan %s: %d files, %d new, %d changed, %d dead",
             path, found, new_count, changed, dead_count)
    return {"files": found, "new": new_count, "changed": changed,
            "dead": dead_count}


def purge_dead() -> int:
    """物理清除软删除图片：FTS 由触发器同步、向量槽回收、缩略图删盘。"""
    import os as _os
    from imgseek import thumbs

    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, content_hash FROM image WHERE dead=1").fetchall()
    if not rows:
        return 0

    conn.execute("CREATE TEMP TABLE IF NOT EXISTS dead_ids(id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM dead_ids")
    conn.executemany("INSERT OR IGNORE INTO dead_ids VALUES(?)",
                     [(r["id"],) for r in rows])

    # 回收向量槽 → 删任务行 → 删主行（FTS 由触发器自动清理）
    conn.execute("""
        INSERT OR IGNORE INTO free_slot(model_key, slot)
        SELECT vs.model_key, vs.slot FROM vector_slot vs
        JOIN dead_ids d ON vs.image_id = d.id
    """)
    conn.execute("DELETE FROM vector_slot WHERE image_id IN (SELECT id FROM dead_ids)")
    conn.execute("DELETE FROM embed_status WHERE image_id IN (SELECT id FROM dead_ids)")
    conn.execute("DELETE FROM image WHERE id IN (SELECT id FROM dead_ids)")
    conn.commit()

    n = 0
    for r in rows:
        if r["content_hash"]:
            p = thumbs.path_for(r["content_hash"])
            if p.exists():
                try:
                    _os.remove(p)
                    n += 1
                except OSError:
                    pass
    log.info("purged %d dead images (%d thumbs removed)", len(rows), n)
    return len(rows)
