"""
main.py — FastAPI backend for Medusa micro-expression recognition UI

Run:
    cd backend
    uvicorn main:app --reload --port 8000

Or from repo root:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure both the repo root and the backend/ dir are on the path
_HERE   = os.path.dirname(os.path.abspath(__file__))   # .../medusa/backend
_ROOT   = os.path.dirname(_HERE)                        # .../medusa
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from pipeline import load_model, process_video  # noqa: E402

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Medusa MER API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files at /
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ── Model loading (once at startup) ──────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_baseline.pth"),
)

_model = None
_device = None


@app.on_event("startup")
async def startup():
    global _model, _device
    _model, _device = load_model(CHECKPOINT_PATH)
    print(f"[startup] Model ready on {_device}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/api/process")
async def process(
    video: UploadFile = File(...),
    magnify_alpha: float = Form(default=20.0),
):
    """
    Accept a video file (mp4, webm, mov, avi), run the full pipeline,
    and return emotion classification + base64-encoded result images.
    """
    # Determine suffix from content type or filename
    filename  = video.filename or "upload.mp4"
    suffix    = os.path.splitext(filename)[-1].lower() or ".mp4"
    allowed   = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    if suffix not in allowed:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await video.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = process_video(tmp_path, _model, _device, magnify_alpha=magnify_alpha)
        return JSONResponse(result)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
