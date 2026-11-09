# TextSwap Studio mission

## Mission
Convert the existing Colab `TextSwap_Studio.ipynb` engine into a local, venv-installable web text editor that can:
- upload images and scanned/digital PDFs;
- OCR and edit text in a page-like editor (select block, edit, restyle, delete, undo/redo);
- recognize the source font, download missing families on demand, and render replacement text with source-matched attributes;
- find/replace with case, whole-word, regex and fuzzy options;
- remove text while preserving the background, watermark, logo, stamp and table/line pixels;
- export edited images/PDF plus OCR text/report.

## Constraints
- Free/local only. No paid APIs, no paid model keys, no cloud processing.
- Keep all document pixels local; font-family download must be explicit and use a fixed allowlisted provider.
- Preserve existing notebook/history; new web app lives beside it.
- Be honest: digital PDFs with embedded fonts can preserve exact font/size/colour; scans/images are best-fit and 100% pixel identity cannot be guaranteed.

## Acceptance
1. `setup.ps1` / `setup.sh` create `.venv` and install requirements; Tesseract is detected.
2. `run.ps1` starts the Flask app and the editor loads in a browser.
3. Upload image/PDF -> OCR blocks -> select/edit/remove -> preview -> export works.
4. Font panel lists installed/downloadable families; download adds a family and it becomes usable for matching.
5. Find/replace supports case, whole-word, regex, fuzzy and selected occurrences.
6. Legacy engine tests pass and new API/PDF tests pass.

## Owner additions (2026-11-09)
7. Select/edit/delete text at word/chunk level, not whole OCR lines.
8. Custom-area OCR: draw a rectangle on the page and OCR only that region.
9. Image-only zoom: Ctrl/Cmd + wheel over the page never zooms the browser/editor UI.
10. Rich controls: resize, move, colour, boldness ±, size ±, font family, italic, alignment.
11. Add brand-new custom text boxes, delete text boxes, and duplicate/copy text boxes.
12. Undo/Redo restores added, moved, restyled and deleted custom boxes.

## Status (2026-11-09) — COMPLETE, pending owner review
All items verified end-to-end:
- Word/chunk selection, editing and find/replace (case, whole-word, regex, fuzzy).
- Area OCR adds `p0-rN-l0-w*` blocks and preserves custom boxes.
- Ctrl/Cmd + wheel zoom verified 51% → 93% → 170% with `defaultPrevented: true`.
- Move (drag + arrow nudge), resize, Reset box, boldness ±, size ±, colour swatches,
  font family and italic verified live and at pixel level.
- Add text (raster + native PDF), Duplicate (+16 px offset), Delete box, Undo/Redo,
  bulk remove, Remove page and Clear all verified in the browser.
- Full suite: 43/43 tests pass; JS `node --check` and Python `py_compile` clean.

Deliverables were committed locally on 2026-11-09 (`main`); nothing was pushed.
Owner approval is still required before any push or external publication.
