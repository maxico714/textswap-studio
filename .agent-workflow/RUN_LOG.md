# Run log

## 2026-11-09
- Read README and extracted notebook cell for inspection (4,233 lines; one Colab code cell with engine + HTML UI + callbacks).
- Confirmed Python 3.12.10, Tesseract 5.4.0, Flask/cv2/numpy/PIL/pytesseract/PyMuPDF present globally.
- Created mission board. Starting engine extraction.

## 2026-11-09 — implementation complete and verified

### Engine / backend
- Extracted the Colab engine into `textswap/` (`engine.py`, `documents.py`, `native_pdf.py`, `fonts.py`, config/settings).
- Raster path: word-level OCR blocks, per-block style analysis, background-preserving erase and re-render, styling controls.
- Native PDF path: span/chunk replacement via PyMuPDF redaction + font embedding/fallback.
- Added `engine.insert_text()`: draws brand-new text into a box with width/height fit, alignment, colour and size scale — no source analysis.
- Raster render accepts insert ops (`insert`, `page`, `bbox`, `text`, `style`) and draws them alongside edits.
- Native `replace_spans()` accepts insert edits, skips redaction for insert-only pages, emits real vector text (`insert_text` single line, `insert_textbox` multi-line) and reports `inserted`.

### API / UI
- Flask API and session store; editor with page canvas, block overlay, inspector, find/replace, fonts, undo/redo.
- Added Add text panel, Duplicate, Delete box buttons; custom blocks are synthetic `ins-N` with a “Custom” badge.
- History snapshots include synthetic blocks; moved/resized/deleted copies restore through Undo/Redo.
- Reset box returns inserted boxes to their creation bbox; area OCR merges custom blocks; bulk remove/Remove page/Clear all are synthetic-aware.
- Status count includes native replaced + inserted + removed.

### Verification
- Unit/API suite: `Ran 43 tests ... OK` (55 s). Includes raster insert (red-pixel + source-word preservation) and native insert (`"Custom note"` extracted from output PDF).
- JS `node --check` and Python `py_compile` clean.
- Browser: add raster (6→7), duplicate (+16 px), move/reset, delete custom → Undo, delete OCR block → Undo, area OCR preserving customs, Clear all → Undo, native add (7→8, 2,752 dark pixels in the inserted box).
- Cleanup: removed `test_native.pdf`, `check_chunks.py`, `inspect_rawdict.py`, `check_native_chunks.py`, `test_run_latest.txt`.

### Handoff
- Committed locally on 2026-11-09 (`main`); no push performed.
- Known risks tracked in `TASK_BOARD.md` (upload/render bounds, embedded-font growth, error leakage, CSRF/Host validation, regex ReDoS, `_FILES` cleanup).
