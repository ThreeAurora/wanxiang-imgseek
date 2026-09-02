# Wanxiang Image Search (imgseek)

[中文](./README.md) | English

A super-lightweight local image semantic search service — the image edition of Everything: **one search box that simultaneously searches filenames, text inside images (OCR), and visual content (natural-language semantics)**.

- No Docker required — `python main.py` starts and stops it on the spot; zero intrusion — your photos are never copied or moved, and deleting `data/` removes it completely
- One-click switching between the Chinese and English CLIP models (Chinese-CLIP ViT-B/16 + OpenAI CLIP B/32), sidestepping the semantic blind spots of any single model
- Handles hundreds of thousands of images: retrieval p50 < 100 ms (fp32 resident vector matrix), fixed-row-height virtual scrolling on the frontend
- Fully automatic background pipeline: scan → decode → thumbnails → OCR → CLIP embedding, with each stage's status persisted to the database — restart after an interruption and it resumes from the checkpoint

## Architecture

| Component | Choice |
|---|---|
| Service | FastAPI + uvicorn, bound to 127.0.0.1:8747, single worker per machine, blocking routes dispatched to a thread pool |
| OCR | RapidOCR (PaddleOCR-family models, onnxruntime inference): GPU automatically when CUDA is available, otherwise CPU fallback |
| Semantics | Chinese-CLIP ViT-B/16 + OpenAI CLIP B/32 (Xenova ONNX exports), onnxruntime inference |
| Vector search | Single matmul over an fp32 resident matrix (p50 ≈ 72 ms at 500k entries) + `.f16bin` on-disk persistence + slot reuse |
| Full-text index | SQLite FTS5 trigram (external content table + trigger-based sync), with a LIKE fallback for short words |
| Metadata | SQLite WAL; tables such as `image/folder/vector_slot/embed_status` — the status fields double as resume-from-checkpoint markers |
| Fusion ranking | Three-way RRF (Reciprocal Rank Fusion) over filename LIKE / OCR full-text / semantic vectors |
| Frontend | Single-file vanilla JS (`web/index.html`): virtual scrolling + object pooling + lazy thumbnail loading + index management page |

Module layout: inside the `imgseek/` package — `scanner` (incremental scanning), `decoder` (one decode pass yields all three inputs), `thumbs`, `ocr`, `clip_models`, `vectors`, `pipeline` (CPU pool + serial GPU thread), `search`, `db`, `api`; the root-level `config.py` centralizes all parameters and the model registry.

## Installation & Running

```bash
# Python 3.10+; GPU inference requires a matching CUDA runtime on your end (or just fall back to CPU)
pip install "fastapi>=0.115" "uvicorn>=0.30" "onnxruntime-gpu" \
            "rapidocr-onnxruntime" "pillow" "numpy" "tokenizers>=0.20"

python main.py                # automatically opens http://127.0.0.1:8747 in your browser
python main.py --pause        # start but keep the pipeline paused (do not auto-resume indexing)
python main.py --rescan       # immediately trigger a full incremental scan
```

First use: click "+ Folder" in the top-right corner to add a folder to index → automatic scanning starts → after a short while filenames and semantics become searchable, while OCR text indexing keeps chewing through the backlog in the background. The index management page supports per-folder "indexing on/off / result inclusion & exclusion / delete index" plus drag-and-drop reordering.

## API Overview

```
GET  /api/search?q=&model=&sort=    Fused search (three-way RRF over name/ocr/sem)
GET  /api/thumb/{id}                Content-addressed thumbnail (immutable caching)
GET  /api/file/{id}                 Original-image preview; POST /api/open {id} opens it with the system default app
GET/POST/DELETE /api/folders        Watched-folder management (add/remove, enable/disable, reorder)
POST /api/scan/start                Trigger a full/incremental scan
POST /api/models/activate {key}     Switch cn_clip_b16 / clip_b32 (weights auto-download when missing)
POST /api/retry {stage}             Re-run failed items (thumb/ocr/embed)
POST /api/unload                    Release the OCR session and resident vectors
GET  /api/status                    Progress / throughput / backend status
```

## Relationship with the Gaze Viewer

This project is the backend for the "search images with text" feature of [Gaze](https://github.com/ThreeAurora/Gaze) (a local image viewer for Windows): the Gaze client submits natural-language queries over the HTTP API at `127.0.0.1:8747` and fetches matching thumbnails, while all retrieval and indexing capability is carried by this service. The two run independently — you can use this project on its own in a browser without Gaze.

## Model Weights & Data Notes

- Model weights ship with the repository as ≤90 MB split parts (GitHub caps a single file at 100 MB); after cloning, reassemble them per `data/models/REBUILD_MODELS.md`. The HuggingFace download path is kept as a fallback: the first time a model is activated it can be auto-downloaded into `data/models/` per the `config.py` registry (about 650 MB per model), via the local proxy `127.0.0.1:7890` straight to huggingface.co (hf-mirror 308-redirects large files back to the origin, which breaks them); existing proxy environment variables are left untouched.
- The `data/index.db*` index-database snapshot, `data/thumbs/` (thumbnail cache) and `data/vectors/` (fp16 vectors, expensive to recompute) are all backed up with the repository; `index.db-wal/-shm` are live files while the service runs — what gets committed is a snapshot taken at commit time.
- `data/demo_report.html` in the repository is a sampled demo report left over from development; it can be deleted at any time.

## Known Limitations

- HEIC depends on pillow-heif; a few 10-bit HDR images may fail to decode (marked as failed with no placeholder, never blocking the queue)
- After 30 minutes of idle time the OCR session and resident vectors are released automatically; queries degrade to the cold path at roughly 2 s until the models are re-activated
- Chinese FTS uses trigram: words of ≥3 characters go through the MATCH index, shorter words fall back to a LIKE scan

## Acknowledgements

- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) / [OpenAI CLIP](https://github.com/openai/CLIP) (Xenova's ONNX export versions)
- [RapidOCR](https://github.com/RapidAI/RapidOCR) (based on PaddleOCR models)
- [onnxruntime](https://github.com/microsoft/onnxruntime), [FastAPI](https://github.com/fastapi/fastapi), [SQLite FTS5](https://www.sqlite.org/fts5.html)

## License

No open-source license is attached (currently a private project). If you wish to reuse the code or model configuration, please contact the author first.
