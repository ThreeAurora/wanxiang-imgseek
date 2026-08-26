"""SQLite 连接管理与完整 schema。

设计要点：
- per-thread 连接（sqlite3 连接不可跨线程），WAL 允许读写并行；
- DB 即断点续传检查点：各阶段状态落库，重启只补 status=0 的行；
- ocr_fts 为外部内容表（rowid == image.id），用触发器自动同步，
  避免“忘同步”类 bug；异常时可 rebuild 自愈。
"""
import sqlite3
import threading

import config


_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """取当前线程连接（无则创建）。调用方自行管理事务粒度。"""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(config.DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS folder(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  enabled INTEGER DEFAULT 1,
  current_epoch INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS image(
  id INTEGER PRIMARY KEY,
  folder_id INTEGER NOT NULL REFERENCES folder(id) ON DELETE CASCADE,
  path TEXT UNIQUE NOT NULL,
  filename TEXT NOT NULL,
  ext TEXT NOT NULL,
  size INTEGER DEFAULT 0,
  mtime REAL DEFAULT 0,
  width INTEGER,
  height INTEGER,
  content_hash TEXT,
  last_seen_epoch INTEGER DEFAULT 0,
  dead INTEGER DEFAULT 0,
  thumb_status INTEGER DEFAULT 0,   -- 0 待处理 1 完成 2 永久失败
  ocr_status INTEGER DEFAULT 0,     -- 0 待处理 1 完成(含"无文字") 2 永久失败
  ocr_text TEXT DEFAULT '',
  error TEXT,
  indexed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_image_dead ON image(dead);
CREATE INDEX IF NOT EXISTS idx_image_thumb ON image(thumb_status) WHERE thumb_status=0;
CREATE INDEX IF NOT EXISTS idx_image_ocr ON image(ocr_status) WHERE ocr_status=0;

-- 每模型的嵌入进度（双模型各自独立续传）
CREATE TABLE IF NOT EXISTS embed_status(
  image_id INTEGER NOT NULL REFERENCES image(id) ON DELETE CASCADE,
  model_key TEXT NOT NULL,
  status INTEGER NOT NULL DEFAULT 0,  -- 0 待处理 1 完成 2 永久失败
  PRIMARY KEY(image_id, model_key)
);
CREATE INDEX IF NOT EXISTS idx_embed_pending ON embed_status(model_key, status);

-- 向量文件行号分配：slot 可回收复用，使 .f16bin 保持紧凑追加
CREATE TABLE IF NOT EXISTS vector_slot(
  model_key TEXT NOT NULL,
  image_id INTEGER NOT NULL REFERENCES image(id) ON DELETE CASCADE,
  slot INTEGER NOT NULL,
  PRIMARY KEY(model_key, image_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_slot ON vector_slot(model_key, slot);

CREATE TABLE IF NOT EXISTS free_slot(
  model_key TEXT NOT NULL,
  slot INTEGER NOT NULL,
  PRIMARY KEY(model_key, slot)
);

-- OCR 全文索引：外部内容表，rowid == image.id
CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
  ocr_text,
  content='image',
  content_rowid='id',
  tokenize='trigram',
  detail='none'
);

CREATE TRIGGER IF NOT EXISTS image_ocr_ai AFTER INSERT ON image BEGIN
  INSERT INTO ocr_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS image_ocr_ad AFTER DELETE ON image BEGIN
  INSERT INTO ocr_fts(ocr_fts, rowid, ocr_text) VALUES('delete', old.id, old.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS image_ocr_au AFTER UPDATE OF ocr_text ON image BEGIN
  INSERT INTO ocr_fts(ocr_fts, rowid, ocr_text) VALUES('delete', old.id, old.ocr_text);
  INSERT INTO ocr_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;
"""


def init_db() -> None:
    """初始化数据目录与全部表结构（幂等，含轻量迁移）。"""
    config.ensure_dirs()
    conn = get_conn()
    conn.executescript(SCHEMA)
    # 轻量迁移：folder.sort_order（拖动排序）、folder.paused（单目录暂停）
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(folder)")}
    if "sort_order" not in cols:
        conn.execute(
            "ALTER TABLE folder ADD COLUMN sort_order INTEGER DEFAULT 0")
    if "paused" not in cols:
        conn.execute(
            "ALTER TABLE folder ADD COLUMN paused INTEGER DEFAULT 0")
    # 默认设置
    conn.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES('active_model', ?)",
        (config.DEFAULT_MODEL,),
    )
    conn.commit()


def get_setting(key: str, default: str = "") -> str:
    row = get_conn().execute(
        "SELECT value FROM settings WHERE key=?", (key,)
    ).fetchone()
    return row["value"] if row and row["value"] is not None else default


def set_setting(key: str, value: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def next_slot(model_key: str) -> int:
    """分配一个向量 slot：优先复用回收槽，否则递增计数器。

    注意：必须在持有写事务的同一线程内调用，调用方负责 commit。
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT slot FROM free_slot WHERE model_key=? LIMIT 1", (model_key,)
    ).fetchone()
    if row is not None:
        conn.execute(
            "DELETE FROM free_slot WHERE model_key=? AND slot=?",
            (model_key, row["slot"]),
        )
        return row["slot"]
    counter_key = f"next_slot_{model_key}"
    cur = int(get_setting(counter_key, "0"))
    set_setting(counter_key, str(cur + 1))
    return cur
