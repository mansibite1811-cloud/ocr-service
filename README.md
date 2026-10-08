# OCR Studio — Local OCR Service

A fully **offline** localhost OCR application powered by **PaddleOCR PP-OCRv4**.  
No API key. No cloud. No account. Everything runs on your machine.

---

## Model Selection: PaddleOCR PP-OCRv4

| Criterion | PaddleOCR PP-OCRv4 |
|---|---|
| OCR Accuracy | ~93.5% F1 (benchmark) |
| Multilingual | 80+ languages incl. Hindi, Gujarati, Arabic, CJK |
| Invoice/Table OCR | ⭐⭐⭐⭐⭐ (PP-Structure module, industry standard) |
| Reading order | ✅ Built-in |
| Confidence + BBox | ✅ Per-line |
| CPU compatible | ✅ Fully CPU — no GPU required |
| Mac/Apple Silicon | ✅ |
| Offline/No API Key | ✅ 100% local |
| Free/Open Source | ✅ Apache 2.0 |

PaddleOCR is the industry standard for document/invoice OCR. Its PP-Structure module is specifically designed for structured documents with tables and columns, which makes it the best practical choice for this use case.

---

## Project Structure

```
OCR/
├── main.py           # FastAPI backend with POST /ocr endpoint
├── requirements.txt  # Python dependencies
├── static/
│   └── index.html    # Web UI (served at http://localhost:8000)
└── README.md         # This file
```

> **No `.env` file needed.** PaddleOCR runs fully offline.

---

## Setup & Installation

### Step 1 — Python version
Requires Python 3.8–3.11. Check your version:
```bash
python3 --version
```

### Step 2 — Create a virtual environment (recommended)
```bash
cd /Users/mansibite/Desktop/OCR
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install PaddlePaddle (CPU)

**macOS (Intel or Apple Silicon):**
```bash
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

**Windows / Linux (CPU):**
```bash
pip install paddlepaddle==3.0.0
```

> If the above URL is slow, try:
> `pip install paddlepaddle`  (PyPI version, may be slightly older)

### Step 4 — Install remaining dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `paddleocr` is already in `requirements.txt` and will be installed in this step.

### Step 5 — Run the server
```bash
uvicorn main:app --reload --port 8000
```

### Step 6 — Open the UI
```
http://localhost:8000
```

> **First run:** PaddleOCR will automatically download the PP-OCRv4 model files (~50–100 MB).  
> You will see download progress in the terminal. Subsequent runs are instant.

---

## API Reference

### POST /ocr

Extract all text from an uploaded image.

**Request:**
```
POST http://localhost:8000/ocr?lang=en&include_blocks=true
Content-Type: multipart/form-data

image = <image file>
```

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `lang` | `en` | Language code for OCR (see `/languages` for all) |
| `include_blocks` | `true` | Include per-line confidence scores and bounding boxes |

**Example with curl:**
```bash
curl -X POST http://localhost:8000/ocr \
  -F "image=@invoice.jpg" \
  -F "" \
  "?lang=en&include_blocks=true"
```

Or without blocks:
```bash
curl -X POST "http://localhost:8000/ocr?lang=en&include_blocks=false" \
  -F "image=@invoice.jpg"
```

**Response (minimum):**
```json
{
  "success": true,
  "text": "Complete OCR text extracted from the image...",
  "language": "en",
  "file_name": "invoice.jpg",
  "file_size_kb": 245.3,
  "image_width_px": 1200,
  "image_height_px": 1600,
  "lines_detected": 42,
  "char_count": 1138,
  "processing_time_s": 3.21,
  "engine": "PaddleOCR PP-OCRv4"
}
```

**Response with blocks:**
```json
{
  "success": true,
  "text": "INVOICE\n\nABC Company Pvt. Ltd.\n\nInvoice Number: INV-12345\nDate: 07/10/2026\n\nProduct A    ₹500\nProduct B    ₹700\n\nTotal: ₹1,200",
  "language": "en",
  "lines_detected": 8,
  "char_count": 128,
  "processing_time_s": 2.87,
  "engine": "PaddleOCR PP-OCRv4",
  "blocks": [
    {
      "text": "INVOICE",
      "confidence": 0.9987,
      "bbox": [312, 45, 688, 92]
    },
    {
      "text": "ABC Company Pvt. Ltd.",
      "confidence": 0.9943,
      "bbox": [210, 120, 790, 155]
    }
  ]
}
```

### GET /health
```bash
curl http://localhost:8000/health
```
Returns server status and whether PaddleOCR is installed.

### GET /languages
```bash
curl http://localhost:8000/languages
```
Returns all supported language codes.

---

## Language Codes (Common)

| Code | Language |
|---|---|
| `en` | English |
| `ch` | Chinese Simplified + English |
| `hi` | Hindi / Marathi (Devanagari) |
| `gu` | Gujarati |
| `ta` | Tamil |
| `te` | Telugu |
| `kn` | Kannada |
| `ar` | Arabic |
| `ja` | Japanese |
| `ko` | Korean |
| `fr` | French |
| `de` | German |
| `ru` | Russian |
| `latin` | Latin-script multilingual |

> For Marathi: use `lang=hi` (same Devanagari script model).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB+ |
| CPU | Any modern CPU | Multi-core for faster inference |
| GPU | Not required | NVIDIA GPU (optional, faster) |
| Disk (models) | ~150 MB | — |

**Typical processing time on CPU:**
- Small image (< 500px): 1–3 seconds
- Medium invoice (A4 scan): 3–8 seconds
- Large/complex document: 8–20 seconds

---

## Troubleshooting

**`paddlepaddle` install fails on Mac:**
```bash
# Try installing from PaddlePaddle's own package index
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

**`ImportError: cannot import name 'PaddleOCR'`:**
```bash
pip install paddleocr --upgrade
```

**Slow first run / model download:**
The models download automatically on first use. Check terminal for progress.
If download is very slow, you can manually download them from:
https://paddlepaddle.github.io/PaddleOCR/ppocr/overview/

**`enable_mkldnn` warning on Mac:**
This is expected. The backend disables MKL-DNN for Mac compatibility. No action needed.

**Port 8000 already in use:**
```bash
uvicorn main:app --reload --port 8001
# Then open http://localhost:8001
```
