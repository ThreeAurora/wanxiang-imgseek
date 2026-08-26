"""检索：FTS 全文 + 向量语义两路融合排序。

P2：文件名 LIKE 路 + ocr_fts 全文路（trigram MATCH，bm25 排序）。
P3：再加当前激活模型的向量路与 RRF 融合。
results[].sources = [{type:'name'|'ocr'|'sem', ...}]
"""
import logging
import re
import sqlite3
import threading
import time

import config
from imgseek import db

log = logging.getLogger("search")

_COLS = ("image.id AS id, image.path AS path, "
         "image.filename AS filename, image.ext AS ext, "
         "image.size AS size, image.mtime AS mtime, "
         "image.width AS width, image.height AS height")

_JOIN_ON = ("FROM image JOIN folder ON folder.id = image.folder_id "
            "AND folder.enabled = 1 WHERE image.dead = 0")

_ORDER = {
    "name": "filename COLLATE NOCASE ASC",
    "mtime": "mtime DESC",
    "size": "size DESC",
}

_FTS_SAFE = re.compile(r"[^\w一-鿿 ]+")


def _fts_rows(conn, q: str):
    """OCR 全文路：>=3 字符走 trigram MATCH（bm25 序），否则 LIKE 兜底。"""
    tokens = [t for t in _FTS_SAFE.sub(" ", q).split() if t]
    if not tokens:
        return [], False
    joined = " ".join(tokens)
    if len(joined) >= 3:
        # 注意：detail='none' 下不支持带引号的短语查询，只能用裸词 AND
        expr = " AND ".join(tokens)
        try:
            rows = conn.execute(
                f"SELECT {_COLS} FROM ocr_fts "
                f"JOIN image ON image.id = ocr_fts.rowid "
                f"JOIN folder ON folder.id = image.folder_id "
                f"AND folder.enabled = 1 "
                f"WHERE ocr_fts MATCH ? AND image.dead = 0 "
                f"ORDER BY bm25(ocr_fts) LIMIT ?",
                (expr, config.SEARCH_LIMIT),
            ).fetchall()
            return rows, len(rows) >= config.SEARCH_LIMIT
        except sqlite3.OperationalError as e:
            log.warning("fts match failed (%s), fallback to LIKE", e)
    like = f"%{joined}%"
    rows = conn.execute(
        f"SELECT {_COLS} {_JOIN_ON} AND image.ocr_text LIKE ? "
        f"ORDER BY image.mtime DESC LIMIT ?", (like, config.SEARCH_LIMIT),
    ).fetchall()
    return rows, len(rows) >= config.SEARCH_LIMIT


def _name_rows(conn, q: str):
    like = f"%{q}%"
    rows = conn.execute(
        f"SELECT {_COLS} {_JOIN_ON} AND "
        f"(image.filename LIKE ? OR image.path LIKE ?) "
        f"ORDER BY image.filename COLLATE NOCASE ASC LIMIT ?",
        (like, like, config.SEARCH_LIMIT),
    ).fetchall()
    return rows, len(rows) >= config.SEARCH_LIMIT


def _vector_rows(conn, q: str, model_key: str):
    """向量路：文本编码 -> 常驻矩阵点积 -> join 元数据。

    返回 ([(row, score)], truncated)；已按相似度降序并应用阈值。
    """
    from imgseek import clip_models as cm, vectors
    if not cm.MANAGER.session_ready(model_key):
        return [], False
    qvec = cm.MANAGER.encode_query(q, model_key)
    if qvec is None:
        return [], False
    hits = vectors.get_file(model_key).search(qvec)
    if not hits:
        return [], False
    slot_score = dict(hits)
    placeholders = ",".join("?" * len(hits))
    rows = conn.execute(
        f"SELECT {_COLS}, vs.slot FROM image "
        f"JOIN folder ON folder.id = image.folder_id AND folder.enabled = 1 "
        f"JOIN vector_slot vs ON vs.image_id = image.id "
        f"WHERE vs.model_key=? AND vs.slot IN ({placeholders}) "
        f"AND image.dead = 0",
        [model_key, *slot_score.keys()],
    ).fetchall()
    floor = config.SEM_MIN_SCORE.get(
        model_key, config.SEM_MIN_SCORE_DEFAULT)
    out = [(r, slot_score[r["slot"]]) for r in rows
           if slot_score[r["slot"]] >= floor]
    out.sort(key=lambda x: x[1], reverse=True)
    return out, False


def search(q: str = "", model: str | None = None, sort: str = "relevance") -> dict:
    t0 = time.perf_counter()
    conn = db.get_conn()
    q = q.strip()

    merged: dict[int, dict] = {}
    order: list[int] = []
    truncated = False

    def put(row, source: dict | None, rank: int | None = None):
        iid = row["id"]
        if iid not in merged:
            merged[iid] = {
                "id": row["id"], "path": row["path"],
                "filename": row["filename"], "ext": row["ext"],
                "size": row["size"], "mtime": row["mtime"],
                "width": row["width"], "height": row["height"],
                "sources": [], "rrf": 0.0,
            }
            order.append(iid)
        if rank is not None:
            merged[iid]["rrf"] += 1.0 / (config.RRF_K + rank)
        if source is not None:
            merged[iid]["sources"].append(source)

    # 各路串行执行：均为毫秒级，thread-local 连接下避免额外复杂度
    if q:
        name_rows, t1 = _name_rows(conn, q)
        for i, r in enumerate(name_rows):
            put(r, {"type": "name"}, rank=i)
        truncated = truncated or t1
        fts_rows, t2 = _fts_rows(conn, q)
        for i, r in enumerate(fts_rows):
            put(r, {"type": "ocr", "matched": q}, rank=i)
        truncated = truncated or t2
        model_key = model or db.get_setting("active_model",
                                            config.DEFAULT_MODEL)
        vec_hits, _t3 = _vector_rows(conn, q, model_key)
        for i, (r, score) in enumerate(vec_hits):
            put(r, {"type": "sem", "score": round(float(score), 4)}, rank=i)
    else:
        rows = conn.execute(
            f"SELECT {_COLS} {_JOIN_ON} "
            f"ORDER BY image.mtime DESC LIMIT ?", (config.SEARCH_LIMIT,)
        ).fetchall()
        for r in rows:
            put(r, None)

    results = [merged[i] for i in order]

    # 排序：relevance = RRF 融合序；其余按列排
    if sort == "relevance" and q:
        results.sort(key=lambda x: -x.pop("rrf"))
    else:
        for x in results:
            x.pop("rrf", None)
    if sort == "name":
        results.sort(key=lambda x: x["filename"].lower())
    elif sort == "mtime":
        results.sort(key=lambda x: x["mtime"] or 0, reverse=True)
    elif sort == "size":
        results.sort(key=lambda x: x["size"] or 0, reverse=True)

    return {
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "total": len(results),
        "truncated": truncated,
        "results": results[:config.SEARCH_LIMIT],
    }
