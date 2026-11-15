FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TEXTSWAP_DATA=/tmp/textswap

# Tesseract runtime + language packs, and metric-compatible fonts so the
# raster renderer can match Arial/Times/Courier on a server with no GUI fonts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        tesseract-ocr-urd \
        fonts-liberation \
        fonts-dejavu-core \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /tmp/textswap \
    && chown -R appuser:appuser /app /tmp/textswap
USER appuser

EXPOSE 10000
CMD ["sh", "-c", "waitress-serve --listen=0.0.0.0:${PORT:-10000} --threads=4 --channel-timeout=300 app:app"]
