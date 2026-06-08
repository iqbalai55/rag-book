# syntax=docker/dockerfile:1.7
# CPU image for local dev / CI repro. Modal uses Dockerfile.cuda instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps: OpenCV stack (docling / easyocr), poppler for PDF, ffmpeg for media
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        poppler-utils \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached separately from source)
COPY requirements/requirements_main.txt /app/requirements/requirements_main.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements/requirements_main.txt

# App source
COPY core/ /app/core/
COPY agents/ /app/agents/
COPY scripts/ /app/scripts/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "scripts.main_docker:app", "--host", "0.0.0.0", "--port", "8000"]
