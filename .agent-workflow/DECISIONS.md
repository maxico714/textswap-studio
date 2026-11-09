# Decisions

- 2026-11-09: Use Flask + vanilla JS (no build step) instead of FastAPI. Flask is already available, the workload is CPU-bound, and a single-process local app is simpler for venv setup. FastAPI would add uvicorn/async complexity without a benefit here.
- 2026-11-09: Keep the proven pixel engine from the notebook; wrap it rather than rewrite. Add native PDF editing for digital PDFs with a text layer, and fall back to raster editing for scans/images.
- 2026-11-09: Font download is explicit user action, from Google Fonts only, cached under `data/fonts`. No credentials or spend. Recognition uses embedded PDF font names where available; otherwise best-fit scoring against installed/downloaded fonts.
- 2026-11-09: "100% identical" is scoped: native PDF path reuses the embedded font file, size, colour and origin; raster path matches measured attributes best-effort. UI/README must state confidence and limitations.
