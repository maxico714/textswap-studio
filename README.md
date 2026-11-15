# TextSwap Studio — local web document text editor

A local Flask web app that turns the original Colab notebook into a page-based text editor for images, scans and PDFs. It finds text with OCR, lets you select and edit blocks, replaces text while matching the source font/colour/weight/width, and removes text without damaging the paper, watermark, logo, stamp or table lines.

Everything runs on your machine by default. No cloud OCR, no paid API, no model key, no document upload to a third party. The only optional network call is an explicit font-family download from Google Fonts. An optional Docker/Render deployment is included for self-hosting; a public instance is not appropriate for confidential documents (see below).

## What it does

- **Editor workflow** — upload, OCR, page canvas, selectable text blocks, text inspector, undo/redo, before/after toggle, export.
- **Full find & replace** — case-sensitive, whole-word, regex, fuzzy OCR matching, per-occurrence selection, replace selected or all.
- **Font recognition** — digital PDFs use the embedded font name/file; images and scans are ranked against installed and downloaded families using glyph-shape scoring. Download a missing family from the Fonts panel.
- **Faithful replacement** — source font, size, colour, weight, letter width, alignment, baseline, rotation and scan softness are measured and reused. The renderer draws a whole line so kerning and advances stay natural.
- **Background-preserving removal** — the old glyph strokes are rebuilt from the local paper; gradients, grain, table borders, underlines and neighbouring words are protected. Watermark/logo/stamp pixels are detected and preserved.
- **Native PDF editing** — a PDF with a real text layer is edited as text: redaction removes only the selected span, and the replacement is inserted with the original embedded font/size/colour/origin where possible. Images and vector art are not rasterised.
- **Scanned PDFs and images** — the proven pixel engine from the notebook is used instead.
- **Exports** — edited PDF/PNG/JPG/WEBP, OCR `.txt`, and a JSON report.

## Requirements

- Python 3.12 (3.10+ should work)
- Tesseract OCR 5.x
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin`
  - macOS: `brew install tesseract tesseract-lang`
- A venv is created by the setup script.

## Setup (Windows)

```powershell
.\setup.ps1
.\run.ps1
```

Open <http://127.0.0.1:5000>.

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

## Setup (Linux / macOS)

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Open <http://127.0.0.1:5000>.

## Editor guide

1. Drop one or more images/PDFs in the left panel. OCR runs on upload for images and scanned PDFs.
2. Click a text block on the page. The right-side **Edit** inspector opens.
3. Change the text, choose a font family, weight, colour, size scale, alignment or nudges, then **Apply**. Empty text removes the block.
4. Use **Detect font** to rank candidates. **Fonts** downloads a missing family; it becomes available immediately.
5. **Find** finds text across the document. Select occurrences and replace/remove only those; **Replace all** applies to every match.
6. **Remove** erases selected blocks or the whole page while protecting the background and watermark.
7. **Download** renders the current edit set and lists the output files. **Original** toggles the before image.

Keyboard: `Ctrl+Z` undo, `Ctrl+Shift+Z` redo, `Ctrl+F` find, `Delete` remove selected block(s), `Ctrl+Enter` apply the open block.

## How font matching works

- **Digital PDF:** the app reads the real text spans. If the font is embedded it is extracted and reused; attribute values (size, colour, origin, weight/italic flags) come from the PDF. This is the closest to "identical attributes" because the same font program is used.
- **Image / scanned PDF:** the app measures the source ink and ranks installed/downloaded font files by glyph shape, natural aspect ratio and stroke width. If the exact commercial font is not installed, a metric-compatible clone can be downloaded (Arial→Arimo, Times→Tinos, Calibri→Carlito, Cambria→Caladea, Courier→Cousine, Georgia→Gelasio).
- **Honest limit:** OCR, scan quality, damaged glyphs and anti-aliasing mean pixel-perfect identity cannot be guaranteed on arbitrary images. Digital PDFs edited natively are substantially more faithful. The report records the source-reconstruction IoU and chosen attributes.

Fonts are downloaded only when you press **Download**; the server requests the public Google Fonts CSS/files for the selected family and caches `.ttf` files in `data/fonts/google/`. No API key or account is used.

## Privacy and safety

- Files are written under `data/sessions/<session>/` and expire after 6 hours (or when the process/session is cleaned).
- Uploads are limited to 40 MB and the allowed image/PDF extensions. Filenames are sanitised.
- Font downloads are restricted to the curated catalogue and fixed Google Fonts hosts.
- No shell execution of uploaded content; Tesseract is invoked through `pytesseract`.
- The Flask secret key is random per process unless `TEXTSWAP_SECRET` is set. For a shared machine, set `TEXTSWAP_SECRET` and run behind a reverse proxy.

## Project layout

```
app.py                     Flask entry point + JSON API
textswap/engine.py         original pixel renderer (OCR, erase, font fit, draw)
textswap/fonts.py          discovery, catalogue, download, ranking
textswap/native_pdf.py     embedded-font PDF inspection + native replacement
textswap/documents.py      sessions, OCR blocks, find/replace, render pipeline
templates/editor.html      editor shell
static/editor.css          editor theme
static/editor.js           canvas editor, inspector, find/replace, fonts
tests/                     regression + API tests
setup.ps1 / setup.sh       venv + dependency setup
run.ps1 / run.sh           start the app
```

## Deploy on Render (Docker, free plan)

`Dockerfile` installs Tesseract (English, Hindi, Urdu) and metric-compatible fonts; `render.yaml` declares a free Docker web service with a `/healthz` check.

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, select the repo, apply `render.yaml` (or **New → Web Service**, runtime **Docker**, plan **Free**).
3. The blueprint sets `TEXTSWAP_ENV=production`, `TEXTSWAP_PROXY=1`, `TEXTSWAP_HTTPS=1`, `TEXTSWAP_DATA=/tmp/textswap`, and generates `TEXTSWAP_SECRET`.

Manual Docker run:

```bash
docker build -t textswap-studio .
docker run --rm -p 10000:10000 -e PORT=10000 -e TEXTSWAP_SECRET=change-me textswap-studio
```

Operational notes:

- **Single process required.** Sessions and documents are kept in memory, so the container runs one Waitress process with four threads. Do not add multiple workers without moving state to shared storage.
- **Free plan limits.** 512 MB RAM, 0.1 CPU, spin-down after 15 idle minutes, and an ephemeral filesystem. Uploads and downloaded fonts are lost on restart/redeploy; the app's session TTL is six hours. Use small, single-page documents on the free plan.
- **Privacy.** A public instance accepts documents from anyone with the URL and stores them temporarily on the server. Use the local app, a private Render service, or a paid instance with your own access controls for sensitive material.
- **Request time.** OCR of large scans can approach Render's proxy timeout; keep uploads modest.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The legacy renderer test (`tests/test_source_calibrated_renderer.py`) loads the notebook engine directly; it is kept as a regression suite. New tests cover the web API, font catalogue and native PDF path.

## Legacy Colab notebook

`TextSwap_Studio.ipynb` is still present and runnable in Colab. The web app is the new primary interface; the notebook remains as the original single-cell implementation and reference.

## Troubleshooting

- **`TesseractNotFoundError`** — install Tesseract and ensure `tesseract` is on PATH; on Windows the app also probes the standard installer paths.
- **Hindi/other language missing** — install the matching Tesseract language pack (e.g. `hin`) and restart.
- **Font download fails** — check internet access to `fonts.googleapis.com`; existing installed fonts still work.
- **PDF has a text layer but editing is disabled** — the app automatically chooses native mode. If a PDF is a scan with a tiny hidden text layer, use the raster path by re-uploading a flattened image/PDF.
