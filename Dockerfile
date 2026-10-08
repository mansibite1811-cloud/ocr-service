# ==============================================================================
# Standalone Production Dockerfile for FastAPI + PaddleOCR API on Render
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install Linux system dependencies required by OpenCV, PaddlePaddle, and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and bake default PaddleOCR model weights (~50-100 MB) into the image layer
# This guarantees models exist locally when Render boots/restarts the container.
COPY main.py /app/main.py
COPY static /app/static
RUN python -c "from main import get_ocr_engine; get_ocr_engine('en')"

# Copy remaining application code
COPY . /app

# Render passes dynamic PORT at runtime
EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 and dynamic $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
