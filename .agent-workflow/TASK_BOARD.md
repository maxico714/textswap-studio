# Task board

Last reconciled: 2026-11-09

| ID | Task | Owner | Status | Evidence |
|---|---|---|---|---|
| T1 | Extract engine from notebook into importable package | CEO | done | `textswap/engine.py`, `native_pdf.py`, `documents.py` import cleanly |
| T2 | Font manager: metadata, discovery, Google Fonts download, recognition | CEO | done | `/api/fonts`, catalog UI, detection endpoint; tests pass |
| T3 | Native PDF text-layer inspection and replacement | CEO | done | span/chunk ids, `replace_spans`, native insert path; `test_native_pdf.py` |
| T4 | Document/session store, OCR blocks, find/replace, render pipeline | CEO | done | `textswap/documents.py`, 43/43 suite |
| T5 | Flask API and routes | CEO | done | `app.py`, browser end-to-end verified |
| T6 | Editor UI (canvas, inspector, find/replace, fonts, undo/redo) | CEO | done | `templates/editor.html`, `static/editor.js` |
| T7 | venv/setup/run scripts, requirements, README | CEO | done | `setup.ps1`, `run.ps1`, `requirements.txt`, README |
| T8 | Tests: engine regression, API smoke, native PDF, fonts | CEO + QA | done | `Ran 43 tests ... OK` |
| T9 | Independent QA and security review | QA/CISO | pending | risk list below; not yet formally reviewed |
| T10 | Final verification and owner handoff | CEO | done | this report; no commit/push performed |

## Owner additions (2026-11-09)

| ID | Task | Status | Evidence |
|---|---|---|---|
| T11 | Word/chunk-level selection and editing | done | raster `p0-l0-wN`, native `...-s0-wN` block ids |
| T12 | Custom-area OCR | done | 3 `p0-r1-l0-w*` blocks added; status/toast correct |
| T13 | Image-only Ctrl/Cmd + wheel zoom | done | 51% → 93% → 170%, `defaultPrevented: true` |
| T14 | Resize, move, colour, weight ±, size ±, fonts, italic, alignment | done | pixel-level colour check; live inspector changes |
| T15 | Add custom text box | done | raster 6→7 blocks, native 7→8 blocks, Custom badge, rendered |
| T16 | Duplicate text box | done | new `ins-N` at +16/+16 px, selected, rendered |
| T17 | Delete text box | done | custom box deletion + Undo restore; real block delete + Undo |
| T18 | Reset box / move on inserted boxes | done | [340,128] → [380,148] → reset [340,128] |
| T19 | Bulk remove / Remove page / Clear all synthetic-aware | done | Clear all 11→9 blocks; Undo restored 9→11 |

## Carried over
- Existing notebook remains runnable in Colab; not rewritten.
- `tests/test_source_calibrated_renderer.py` preserved and passing.
- Owner approval granted; committed locally on 2026-11-09. Not pushed.

## Known risks (not addressed in this iteration)
- Unbounded PDF/image expansion during upload/render.
- Persistent embedded fonts can accumulate in `data/fonts`.
- Raw exception text may leak into API errors.
- No CSRF/Host-header validation on state-changing routes.
- Regex find/replace has no ReDoS guard.
- `_FILES` session cleanup on exit is not guaranteed.
