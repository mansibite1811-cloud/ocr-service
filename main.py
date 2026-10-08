"""
Production OCR Service — main.py
FastAPI backend powered by PaddleOCR (PP-OCRv4).

Standalone REST API for processing image OCR requests in production.
Works on local machine & Render cloud platform with dynamic $PORT support.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import time
import logging
from pathlib import Path
from functools import lru_cache
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

# ---------------------------------------------------------------------------
# Logging — suppress PaddleOCR's very chatty default output
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress PaddlePaddle and PaddleOCR internal logs
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

# Optional environment variable API key authentication
OCR_API_KEY = os.environ.get("OCR_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Supported image types and file size limit
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
MAX_FILE_SIZE_MB = 30

# ---------------------------------------------------------------------------
# PaddleOCR engine — cached per language to avoid reloading models
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_ocr_engine(lang: str = "en"):
    """
    Return a cached PaddleOCR engine for the given language.
    First call downloads/loads the models (~50-100 MB); subsequent calls reuse the cache.
    """
    from paddleocr import PaddleOCR   # imported lazily to keep startup fast
    logger.info(f"Initializing PaddleOCR engine for language: '{lang}'...")
    return PaddleOCR(
        use_angle_cls=True,   # Detect rotated / upside-down text
        lang=lang,
        use_gpu=False,        # CPU mode — works on standard container hardware
        show_log=False,       # Suppress PaddleOCR's internal verbose logging
        enable_mkldnn=False,  # Keep disabled for cross-platform stability
    )


def _run_ocr_sync(engine, img_array):
    """Synchronous worker function to perform OCR."""
    return engine.ocr(img_array, cls=True)


def reconstruct_text(ocr_result) -> str:
    """
    Convert PaddleOCR's raw output into clean, readable plain text.

    PaddleOCR returns lines sorted roughly by position, but the grouping
    can be imperfect. This function:
      1. Sorts all detected text blocks by their top-left Y coordinate.
      2. Groups blocks on the same visual line (within ~half a line height).
      3. Within each line, sorts blocks by X coordinate (left → right).
      4. Joins blocks on the same line with spaces, lines with newlines.
    """
    if not ocr_result or not ocr_result[0]:
        return ""

    # Flatten all pages (we always send one image, so one page)
    all_lines = []
    for page in ocr_result:
        if page:
            all_lines.extend(page)

    if not all_lines:
        return ""

    # Sort by top-left Y coordinate for reading order
    all_lines.sort(key=lambda x: x[0][0][1])

    # Group into visual rows
    rows = []
    current_row = [all_lines[0]]

    for line in all_lines[1:]:
        curr_y   = line[0][0][1]
        prev_y   = current_row[-1][0][0][1]
        # Estimate line height from bounding box
        line_h   = abs(line[0][2][1] - line[0][0][1]) if len(line[0]) >= 3 else 12
        threshold = max(line_h * 0.55, 8)

        if abs(curr_y - prev_y) <= threshold:
            current_row.append(line)
        else:
            rows.append(current_row)
            current_row = [line]

    if current_row:
        rows.append(current_row)

    # Build text — sort each row by X, join with spaces, rows with newlines
    text_lines = []
    for row in rows:
        row_sorted = sorted(row, key=lambda x: x[0][0][0])
        row_text = "  ".join(item[1][0] for item in row_sorted)
        text_lines.append(row_text)

    return "\n".join(text_lines)


def build_blocks(ocr_result) -> list:
    """
    Convert PaddleOCR output into a list of structured block dicts,
    each with text, confidence score, and bounding box.
    """
    blocks = []
    if not ocr_result or not ocr_result[0]:
        return blocks

    for page in ocr_result:
        if not page:
            continue
        for line in page:
            bbox_pts = line[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            text, conf = line[1]
            # Flatten bbox to [x_min, y_min, x_max, y_max]
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            blocks.append({
                "text":       text,
                "confidence": round(float(conf), 4),
                "bbox":       [min(xs), min(ys), max(xs), max(ys)],
            })

    return blocks


# ---------------------------------------------------------------------------
# API Key Authentication Security Helper
# ---------------------------------------------------------------------------
def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    api_key_query: str | None = Query(None, alias="api_key"),
):
    """
    If OCR_API_KEY environment variable is configured, verify incoming request authentication key.
    If OCR_API_KEY is not set (empty), allow request without key.
    """
    if OCR_API_KEY:
        provided_key = x_api_key or api_key_query
        if not provided_key or provided_key != OCR_API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing API key. Send 'X-API-Key' header.",
            )


# ---------------------------------------------------------------------------
# FastAPI Lifespan Handler (Pre-warm Engine)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm default PaddleOCR engine on container startup."""
    logger.info("Initializing OCR engine startup sequence...")
    try:
        # Pre-load default English model into memory cache
        get_ocr_engine("en")
        logger.info("PaddleOCR engine pre-warmed successfully.")
    except Exception as exc:
        logger.warning(f"Engine pre-warm on startup skipped or deferred: {exc}")
    yield


# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Standalone OCR Production API",
    description="Production-ready OCR service powered by PaddleOCR (PP-OCRv4).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ---------------------------------------------------------------------------
# Supported Languages Reference
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "en":    "English",
    "ch":    "Chinese (Simplified) + English",
    "ch_tra":"Chinese (Traditional)",
    "hi":    "Hindi (Devanagari — also covers Marathi, Sanskrit)",
    "gu":    "Gujarati",
    "ar":    "Arabic",
    "ta":    "Tamil",
    "te":    "Telugu",
    "kn":    "Kannada",
    "mr":    "Marathi (use lang=hi for Devanagari script)",
    "fr":    "French",
    "de":    "German",
    "es":    "Spanish",
    "pt":    "Portuguese",
    "it":    "Italian",
    "ru":    "Russian",
    "ja":    "Japanese",
    "ko":    "Korean",
    "latin": "Latin-script multilingual",
    "cyrillic": "Cyrillic-script multilingual",
    "devanagari": "Devanagari-script multilingual",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the web UI test interface."""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found (static/index.html is missing).")
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    """Render health check endpoint — verifies service availability & engine status."""
    try:
        import paddleocr  # noqa
        paddle_ok = True
    except ImportError:
        paddle_ok = False

    return {
        "status": "ok" if paddle_ok else "degraded",
        "service": "OCR Production Service",
        "engine": "PaddleOCR PP-OCRv4",
        "paddle_installed": paddle_ok,
        "api_key_protected": bool(OCR_API_KEY),
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.get("/languages")
async def list_languages():
    """List all supported OCR language codes."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/ocr", dependencies=[Depends(verify_api_key)])
async def perform_ocr(
    image: UploadFile = File(...),
    lang: str = Query(
        default="en",
        description=(
            "OCR language code. Examples: en, ch, hi, gu, ar, fr, de, ja, ko, latin. "
            "Use /languages to see the full list."
        ),
    ),
    include_blocks: bool = Query(
        default=True,
        description="Include per-line blocks with confidence scores and bounding boxes.",
    ),
):
    """
    Extract all text from an uploaded image using PaddleOCR (PP-OCRv4).

    - **image** — image file (JPEG, PNG, WEBP, BMP, TIFF, GIF)
    - **lang** — language code for OCR (default: `en`)
    - **include_blocks** — include per-line confidence + bbox in response

    Returns JSON with the complete extracted text and optional per-block metadata.
    """
    # ── File type check ─────────────────────────────────────────────────────
    filename  = image.filename or "upload"
    ext       = Path(filename).suffix.lower()
    mime_type = (image.content_type or "").lower()

    if ext not in SUPPORTED_EXTENSIONS and not mime_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: JPG, PNG, WEBP, BMP, TIFF, GIF.",
        )

    # ── Read bytes ──────────────────────────────────────────────────────────
    image_bytes = await image.read()
    file_size_kb = round(len(image_bytes) / 1024, 2)

    if file_size_kb > MAX_FILE_SIZE_MB * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_kb:.0f} KB). Maximum limit is {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Open and validate image ─────────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")   # Normalize to RGB (handles RGBA, palette, CMYK, etc.)
        width, height = pil_img.size
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image file: {exc}")

    # Convert to numpy array for PaddleOCR processing
    img_array = np.array(pil_img)

    # ── Validate language code ───────────────────────────────────────────────
    valid_langs = set(SUPPORTED_LANGUAGES.keys())
    if lang not in valid_langs:
        logger.warning(f"Language '{lang}' is not in the standard list. Attempting processing anyway.")

    # ── Run PaddleOCR safely in a threadpool to prevent blocking ASGI event loop
    start = time.time()
    try:
        ocr_engine = get_ocr_engine(lang)
        result = await run_in_threadpool(_run_ocr_sync, ocr_engine, img_array)
    except Exception as exc:
        logger.error(f"PaddleOCR processing error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {exc}",
        )

    elapsed = round(time.time() - start, 3)

    # ── Build response ───────────────────────────────────────────────────────
    full_text = reconstruct_text(result)
    blocks    = build_blocks(result) if include_blocks else []

    if not full_text.strip():
        full_text = ""   # Explicit empty string for images with no text detected

    response: dict = {
        "success":            True,
        "text":               full_text,
        "language":           lang,
        "file_name":          filename,
        "file_size_kb":       file_size_kb,
        "image_width_px":     width,
        "image_height_px":    height,
        "lines_detected":     len(blocks),
        "char_count":         len(full_text),
        "processing_time_s":  elapsed,
        "engine":             "PaddleOCR PP-OCRv4",
    }

    if include_blocks:
        response["blocks"] = blocks

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
