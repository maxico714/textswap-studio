#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "TextSwap Studio setup"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract found: $(command -v tesseract)"
  tesseract --version | head -n 1
else
  echo "Tesseract NOT found. Install it, e.g. sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin"
fi
echo "Next: ./run.sh  ->  http://127.0.0.1:5000"
