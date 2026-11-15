"""Document store, OCR blocks, find/replace and render orchestration."""
from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from . import engine, fonts, native_pdf

ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = Path(
    os.environ.get("TEXTSWAP_DATA")
    or ("/tmp/textswap" if os.environ.get("RENDER") else ROOT / "data")
)
DATA_ROOT = DATA_BASE / "sessions"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD = 40 * 1024 * 1024
MAX_FILES = 12
SESSION_TTL = 6 * 3600
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".pdf"}
ALLOWED_FORMATS = {"png", "jpg", "webp", "pdf"}
LANG_OPTIONS = [
    ("eng", "English"), ("hin", "हिन्दी"), ("urd", "اردو"), ("ara", "العربية"),
    ("ben", "বাংলা"), ("tam", "தமிழ்"), ("tel", "తెలుగు"), ("mar", "मराठी"),
    ("guj", "ગુજરાતી"), ("pan", "ਪੰਜਾਬੀ"), ("san", "संस्कृतम्"), ("fra", "Français"),
    ("deu", "Deutsch"), ("spa", "Español"), ("por", "Português"), ("ita", "Italiano"),
    ("nld", "Nederlands"), ("rus", "Русский"), ("tur", "Türkçe"), ("chi_sim", "中文简体"),
    ("chi_tra", "中文繁體"), ("jpn", "日本語"), ("kor", "한국어"),
]
ALLOWED_LANGS = {c for c, _ in LANG_OPTIONS}

_LOCK = threading.RLock()
_SESSIONS: dict[str, dict] = {}
_DOCS: dict[str, dict] = {}
_FILES: dict[str, dict] = {}


def _cleanup() -> None:
    now = time.time()
    with _LOCK:
        for sid, sess in list(_SESSIONS.items()):
            if now - sess.get("created", now) > SESSION_TTL:
                for did in sess.get("docs", []):
                    _DOCS.pop(did, None)
                shutil.rmtree(sess.get("dir", ""), ignore_errors=True)
                _SESSIONS.pop(sid, None)
        if len(_SESSIONS) > 30:
            oldest = sorted(_SESSIONS.items(), key=lambda kv: kv[1].get("created", 0))
            for sid, sess in oldest[: len(_SESSIONS) - 30]:
                for did in sess.get("docs", []):
                    _DOCS.pop(did, None)
                shutil.rmtree(sess.get("dir", ""), ignore_errors=True)
                _SESSIONS.pop(sid, None)


def read_session(sid: str | None) -> dict:
    _cleanup()
    if not sid or not re.fullmatch(r"[0-9a-f]{32}", sid):
        sid = uuid.uuid4().hex
    with _LOCK:
        sess = _SESSIONS.get(sid)
        if sess is None:
            root = Path(tempfile.mkdtemp(prefix="run_", dir=str(DATA_ROOT)))
            sess = {"id": sid, "dir": root, "created": time.time(), "docs": []}
            _SESSIONS[sid] = sess
    return sess


def safe_name(name: str, fallback: str = "upload") -> str:
    name = Path(str(name or "")).name
    name = re.sub(r"[^\w.\- ()\[\]]+", "_", name, flags=re.UNICODE).strip(" .")[:160]
    return name or fallback


def safe_stem(name: str, fallback: str = "file") -> str:
    stem = Path(str(name or "")).stem
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE).strip(" ._")
    return (stem or fallback)[:80]


def _hex_to_bgr(value) -> tuple[int, int, int] | None:
    if not value:
        return None
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        return None
    return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))


def available_languages() -> set[str]:
    try:
        return set(pytesseract.get_languages(config="")) & ALLOWED_LANGS
    except Exception:
        return {"eng"}


def _ensure_langs(codes, warnings: list[str]) -> str:
    codes = [c for c in (codes or []) if c in ALLOWED_LANGS] or ["eng"]
    codes = list(dict.fromkeys(codes))
    avail = available_languages()
    usable = [c for c in codes if c in avail]
    missing = [c for c in codes if c not in avail]
    if missing:
        warnings.append("OCR language pack missing: " + ", ".join(missing) + ". Tesseract install me pack add karein.")
    if not usable:
        usable = ["eng"] if "eng" in avail else sorted(avail)[:1] or ["eng"]
        warnings.append("Selected OCR language available nahi thi; fallback use hua.")
    return "+".join(usable)


def _preprocess(bgr: np.ndarray, opts: dict) -> np.ndarray:
    img = bgr.copy()
    if opts.get("auto_contrast"):
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except Exception:
            pass
    if opts.get("denoise"):
        try:
            img = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 21)
        except Exception:
            pass
    if opts.get("grayscale"):
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if opts.get("binarize"):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if opts.get("binarize_mode") == "adaptive":
            g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        else:
            _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if opts.get("invert"):
        img = cv2.bitwise_not(img)
    return img


def _page_text(lines: list[list[engine.Word]]) -> str:
    return "\n".join(" ".join(w.text for w in ln).strip() for ln in lines if ln)


def _preview_b64(bgr: np.ndarray, quality: int = 82, max_w: int = 1600) -> str:
    h, w = bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        bgr = cv2.resize(bgr, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _register(doc: dict, path: Path, label: str, mime: str | None = None) -> dict:
    fid = uuid.uuid4().hex[:14]
    rec = {
        "id": fid, "path": str(path), "filename": path.name, "label": label,
        "size": path.stat().st_size, "mime": mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }
    _FILES[fid] = rec
    doc.setdefault("files", []).append(rec)
    return {"id": fid, "label": label, "filename": path.name, "size": rec["size"], "mime": rec["mime"]}


def get_file(sid: str, fid: str) -> dict | None:
    sess = _SESSIONS.get(sid)
    if not sess:
        return None
    rec = _FILES.get(fid)
    if not rec:
        return None
    if not any(f["id"] == fid for d in sess.get("docs", []) for f in _DOCS.get(d, {}).get("files", [])):
        return None
    return rec


def _block_bbox(line: list[engine.Word]) -> list[float]:
    return [
        float(min(w.x for w in line)), float(min(w.y for w in line)),
        float(max(w.x + w.w for w in line)), float(max(w.y + w.h for w in line)),
    ]


def _ocr_blocks(pages: list[engine.Page], settings: dict, warnings: list[str]) -> tuple[list[list[engine.Word]], list[dict]]:
    lang = _ensure_langs(settings.get("langs"), warnings)
    psm = int(settings.get("psm", 3))
    enhance = bool(settings.get("enhance", True))
    min_conf = float(settings.get("min_conf", 0))
    all_lines: list[list[list[engine.Word]]] = []
    blocks: list[dict] = []
    for pno, page in enumerate(pages):
        proc = _preprocess(page.bgr, settings.get("preprocess") or {})
        try:
            lines = engine.ocr_lines(proc, lang=lang, psm=psm, enhance=enhance)
        except Exception as exc:
            if len(lang.split("+")) > 1:
                warnings.append("Multi-language OCR fail hua; pehli language se retry.")
                lines = engine.ocr_lines(proc, lang=lang.split("+")[0], psm=psm, enhance=enhance)
            else:
                raise RuntimeError(f"OCR failed: {exc}") from exc
        if min_conf > 0:
            lines = [[w for w in ln if w.conf >= min_conf] for ln in lines]
            lines = [ln for ln in lines if ln]
        all_lines.append(lines)
        for li, line in enumerate(lines):
            line_text = " ".join(w.text for w in line).strip()
            if not line_text:
                continue
            line_bbox = _block_bbox(line)
            # One editable block per OCR word/chunk.  A whole OCR line often
            # contains several unrelated fields ("Name: Rehan Khan"), so making
            # the line the selection unit forces users to rewrite everything on
            # it.  Chunk blocks keep their line metadata for context/skew while
            # each word gets its own bbox, text and stable id.
            for wi, word in enumerate(line):
                text = str(word.text).strip()
                if not text:
                    continue
                blocks.append({
                    "id": f"p{pno}-l{li}-w{wi}",
                    "page": pno,
                    "text": text,
                    "bbox": [float(word.x), float(word.y), float(word.x + word.w), float(word.y + word.h)],
                    "words": [{"text": word.text, "x": word.x, "y": word.y, "w": word.w, "h": word.h, "conf": round(float(word.conf), 1)}],
                    "source": "ocr",
                    "confidence": round(float(word.conf) / 100.0, 3),
                    "style": None,
                    "line": li,
                    "word_index": wi,
                    "line_bbox": line_bbox,
                })
    return all_lines, blocks


def _native_blocks(pages: list[list[dict]], scales: list[float]) -> list[dict]:
    out = []
    for pno, spans in enumerate(pages):
        sx = scales[pno] if pno < len(scales) else 1.0
        for span in spans:
            style = {
                "font": span["style"]["font"],
                "font_family": span["style"]["font"],
                "size": span["style"]["size"],
                "color": span["style"]["color"],
                "weight": span["style"]["weight"],
                "italic": span["style"]["italic"],
                "serif": span["style"]["serif"],
                "mono": span["style"]["mono"],
                "source": "embedded",
                "confidence": 1.0,
            }
            chunks = span.get("chunks") or []
            if not chunks:
                # Defensive fallback: keep the whole span editable if chunk
                # extraction produced nothing for it.
                bbox = [span["bbox"][0] * sx, span["bbox"][1] * sx, span["bbox"][2] * sx, span["bbox"][3] * sx]
                out.append({
                    "id": span["id"],
                    "page": pno,
                    "text": span["text"],
                    "bbox": [round(float(v), 2) for v in bbox],
                    "words": [],
                    "source": "pdf",
                    "confidence": 1.0,
                    "style": style,
                })
                continue
            for chunk in chunks:
                bbox = [
                    chunk["bbox"][0] * sx, chunk["bbox"][1] * sx,
                    chunk["bbox"][2] * sx, chunk["bbox"][3] * sx,
                ]
                out.append({
                    "id": f"{span['id']}-w{chunk['index']}",
                    "page": pno,
                    "text": chunk["text"],
                    "bbox": [round(float(v), 2) for v in bbox],
                    "words": [],
                    "source": "pdf",
                    "confidence": 1.0,
                    "style": dict(style),
                    "span_id": span["id"],
                    "chunk_index": chunk["index"],
                    "char_start": chunk["start"],
                    "char_end": chunk["end"],
                })
    return out


def ocr_region(sid: str, did: str, payload: dict) -> dict:
    """OCR only a user-selected rectangle and append its chunks to the page."""
    doc = get_doc(sid, did)
    pages = doc.get("pages") or []
    pno = int(payload.get("page") or 0)
    if pno < 0 or pno >= len(pages):
        raise ValueError("Page nahi mila.")
    box = _clamped_bbox(doc, pno, payload.get("bbox"), None)
    if box is None:
        raise ValueError("OCR area bbox valid nahi hai; page par rectangle banayein.")
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    page = pages[pno]
    crop = page.bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Selected area empty hai.")
    alpha = page.alpha[y0:y1, x0:x1] if page.alpha is not None else None
    settings = {
        "langs": payload.get("langs"),
        "psm": payload.get("psm", 6),
        "enhance": bool(payload.get("enhance", True)),
        "min_conf": payload.get("min_conf", 0),
        "preprocess": payload.get("preprocess") or {},
    }
    warnings: list[str] = []
    region_page = engine.Page(crop, alpha, page.meta)
    all_lines, blocks = _ocr_blocks([region_page], settings, warnings)
    region_no = int(doc.get("region_seq", 0)) + 1
    doc["region_seq"] = region_no

    offset_lines: list[list[engine.Word]] = []
    for line in (all_lines[0] if all_lines else []):
        offset_lines.append([
            engine.Word(w.text, w.x + x0, w.y + y0, w.w, w.h, w.conf) for w in line
        ])
    lines_by_page = doc.setdefault("all_lines", [])
    while len(lines_by_page) <= pno:
        lines_by_page.append([])
    lines_by_page[pno].extend(offset_lines)

    new_blocks: list[dict] = []
    for block in blocks:
        block["id"] = f"p{pno}-r{region_no}-l{block['line']}-w{block['word_index']}"
        block["page"] = pno
        block["region"] = region_no
        block["bbox"] = [
            block["bbox"][0] + x0, block["bbox"][1] + y0,
            block["bbox"][2] + x0, block["bbox"][3] + y0,
        ]
        if block.get("line_bbox"):
            block["line_bbox"] = [
                block["line_bbox"][0] + x0, block["line_bbox"][1] + y0,
                block["line_bbox"][2] + x0, block["line_bbox"][3] + y0,
            ]
        for word in block.get("words") or []:
            word["x"] += x0
            word["y"] += y0
        new_blocks.append(block)
    doc.setdefault("blocks", []).extend(new_blocks)

    region_text = _page_text(all_lines[0]) if all_lines else ""
    if region_text:
        texts = doc.setdefault("page_texts", [])
        while len(texts) <= pno:
            texts.append("")
        texts[pno] = (texts[pno] + "\n" + region_text).strip() if texts[pno] else region_text
    if doc.get("kind") == "pdf_native":
        warnings.append("Native PDF area OCR: mixed OCR/PDF edits render from the preview-resolution page.")
    return {
        "doc": doc_summary(doc),
        "blocks": new_blocks,
        "count": len(new_blocks),
        "text": region_text,
        "warnings": warnings,
    }


def create_document(sid: str, filename: str, raw: bytes, settings: dict | None = None) -> dict:
    settings = settings or {}
    if not raw:
        raise ValueError("File empty hai.")
    if len(raw) > MAX_UPLOAD:
        raise ValueError(f"File {MAX_UPLOAD // (1024 * 1024)} MB se badi hai.")
    filename = safe_name(filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError("Sirf image (PNG/JPG/WEBP/TIFF) ya PDF upload karein.")
    sess = _get_session(sid)
    if len(sess["docs"]) >= MAX_FILES:
        raise ValueError(f"Ek session me maximum {MAX_FILES} files allowed hain.")
    did = uuid.uuid4().hex[:14]
    work = Path(sess["dir"]) / did
    work.mkdir(parents=True, exist_ok=True)
    src = work / ("source" + ext)
    src.write_bytes(raw)
    warnings: list[str] = []
    dpi = max(120, min(500, int(settings.get("dpi", 220) or 220)))
    stem = safe_stem(filename)
    doc: dict = {
        "id": did, "session": sid, "filename": filename, "ext": ext, "path": str(src),
        "dir": work, "kind": "image", "dpi": dpi, "created": time.time(),
        "files": [], "warnings": warnings, "edited": {},
    }

    if ext == ".pdf":
        if native_pdf.has_text_layer(src):
            spans = native_pdf.extract_spans(src)
            doc["kind"] = "pdf_native"
            doc["native_spans"] = spans
            doc["pages"] = []
            doc["native_scales"] = []
            for pno in range(len(spans)):
                bgr = native_pdf.render_page(src, pno, dpi=140)
                doc["pages"].append(engine.Page(bgr, meta={"dpi": (140, 140), "mode": "RGB", "pdf_page": pno}))
                page_size = spans[pno][0]["page_size"] if spans[pno] else [0, 0]
                scale = (bgr.shape[1] / page_size[0]) if page_size and page_size[0] else 1.0
                doc["native_scales"].append(scale)
            doc["blocks"] = _native_blocks(spans, doc["native_scales"])
            doc["all_lines"] = [[] for _ in spans]
            doc["page_texts"] = ["\n".join(s["text"] for s in page) for page in spans]
            warnings.append("Digital PDF detected: native text-layer editing enabled (embedded font reused where possible).")
        else:
            pages = engine.load_pages(str(src), dpi=dpi)
            doc["kind"] = "pdf_scan"
            doc["pages"] = pages
            all_lines, blocks = _ocr_blocks(pages, settings, warnings)
            doc["all_lines"] = all_lines
            doc["blocks"] = blocks
            doc["page_texts"] = [_page_text(ln) for ln in all_lines]
    else:
        pages = engine.load_pages(str(src), dpi=dpi)
        doc["kind"] = "image"
        doc["pages"] = pages
        all_lines, blocks = _ocr_blocks(pages, settings, warnings)
        doc["all_lines"] = all_lines
        doc["blocks"] = blocks
        doc["page_texts"] = [_page_text(ln) for ln in all_lines]

    with _LOCK:
        _DOCS[did] = doc
        sess["docs"].append(did)
    return doc_summary(doc)


def _get_session(sid: str) -> dict:
    sess = _SESSIONS.get(sid)
    if not sess:
        raise ValueError("Session invalid ya expired hai. Page reload karein.")
    return sess


def get_doc(sid: str, did: str) -> dict:
    sess = _SESSIONS.get(sid)
    doc = _DOCS.get(did)
    if not sess or not doc or doc.get("session") != sid:
        raise ValueError("Document session me nahi mila. Dobara upload karein.")
    return doc


def doc_summary(doc: dict) -> dict:
    pages = doc.get("pages", [])
    previews = []
    for pno, page in enumerate(pages):
        previews.append({"page": pno, "width": int(page.bgr.shape[1]), "height": int(page.bgr.shape[0])})
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "kind": doc["kind"],
        "page_count": len(pages),
        "dpi": doc.get("dpi"),
        "blocks": doc.get("blocks", []),
        "pages": previews,
        "warnings": doc.get("warnings", []),
        "text": "\n\n".join(f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(doc.get("page_texts", []))).strip(),
        "native": doc.get("kind") == "pdf_native",
    }


def page_preview(sid: str, did: str, page: int = 0, *, dpi: int | None = None) -> str:
    doc = get_doc(sid, did)
    pages = doc.get("pages", [])
    if not pages:
        raise ValueError("Page nahi mila.")
    pno = max(0, min(int(page), len(pages) - 1))
    bgr = pages[pno].bgr
    if doc.get("kind") == "pdf_native" and dpi:
        bgr = native_pdf.render_page(doc["path"], pno, dpi=dpi)
    return _preview_b64(bgr)


def block_by_id(doc: dict, bid: str) -> dict | None:
    return next((b for b in doc.get("blocks", []) if b.get("id") == bid), None)


def _block_context(doc: dict, block: dict) -> str:
    pno = int(block["page"])
    text = block.get("text", "")
    return f"page {pno + 1}: {text[:180]}"


def recognize_block(sid: str, did: str, bid: str) -> dict:
    doc = get_doc(sid, did)
    block = block_by_id(doc, bid)
    if not block:
        raise ValueError("Text block nahi mila.")
    style = block.get("style")
    if doc.get("kind") == "pdf_native" and style:
        family = style.get("font_family") or style.get("font") or ""
        rec = fonts.find_family(family, int(style.get("weight") or 400), bool(style.get("italic")))
        candidates = []
        if rec:
            candidates.append({**rec.to_dict(), "score": 1.0, "confidence": 1.0, "rank": 1})
        return {
            "best": candidates[0] if candidates else None,
            "candidates": candidates,
            "method": "embedded-pdf-font",
            "note": "PDF ke embedded font name se exact family mili." if candidates else "Embedded font file nahi mila; fallback available.",
        }
    page = doc["pages"][int(block["page"])]
    reg = engine.analyze_region(page.bgr, block["bbox"])
    if reg is None:
        return {"best": None, "candidates": [], "method": "none", "note": "Is block me ink nahi mili."}
    cx0, cy0, _, _ = reg.ctx
    x0, y0, x1, y1 = reg.core_bbox
    obs = reg.core[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
    result = fonts.recognize(block.get("text", ""), obs)
    result["method"] = "glyph-shape-match"
    result["note"] = "Shape score se best installed/downloaded family. Confidence 1.0 ka matlab exact match nahi."
    return result


def _native_chunk_hits(span: dict, start: int, end: int) -> tuple[list[dict], list[list[float]]]:
    """Map a character range in a PDF span to independently editable chunks."""
    edits: list[dict] = []
    boxes: list[list[float]] = []
    for chunk in span.get("chunks") or []:
        c_start = int(chunk["start"])
        c_end = int(chunk["end"])
        if c_end <= start or c_start >= end:
            continue
        edits.append({
            "block_id": f"{span['id']}-w{chunk['index']}",
            "start": max(start, c_start) - c_start,
            "end": min(end, c_end) - c_start,
        })
        boxes.append(chunk["bbox"])
    if not edits:
        # Defensive fallback for a span with no chunk metadata.
        edits = [{"block_id": span["id"], "start": start, "end": end}]
        boxes = [span["bbox"]]
    return edits, boxes


def find_text(sid: str, did: str, payload: dict) -> list[dict]:
    doc = get_doc(sid, did)
    query = str(payload.get("query") or "")
    if not query:
        return []
    case = bool(payload.get("case_sensitive"))
    whole = bool(payload.get("whole_word"))
    regex = bool(payload.get("regex"))
    fuzzy = float(payload.get("fuzzy", 0.8) or 0)
    page_filter = payload.get("page")
    matches: list[dict] = []

    if doc.get("kind") == "pdf_native":
        flags = 0 if case else re.IGNORECASE
        try:
            pat = re.compile(query if regex else re.escape(query), flags)
        except re.error as exc:
            raise ValueError(f"Regex invalid: {exc}") from exc
        for pno, spans in enumerate(doc.get("native_spans", [])):
            if page_filter is not None and int(page_filter) != pno:
                continue
            scale = (doc.get("native_scales") or [1.0] * (pno + 1))[pno] if pno < len(doc.get("native_scales") or []) else 1.0
            for span in spans:
                text = span["text"]
                hay = text if case else text.lower()
                q = query if case else query.lower()
                for m in pat.finditer(hay if not regex else text):
                    start, end = m.start(), m.end()
                    if whole:
                        if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
                            continue
                        if end < len(text) and (text[end].isalnum() or text[end] == "_"):
                            continue
                    if not regex and hay[start:end] != q:
                        continue
                    chunk_edits, boxes = _native_chunk_hits(span, start, end)
                    if not chunk_edits:
                        continue
                    bbox = [
                        min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes),
                    ]
                    bbox = [v * scale for v in bbox]
                    first = chunk_edits[0]
                    matches.append({
                        "id": uuid.uuid4().hex[:12],
                        "page": pno,
                        "block_id": first["block_id"],
                        "span_id": span["id"],
                        "chunk_edits": chunk_edits,
                        "text": text[start:end],
                        "matched_text": text[start:end],
                        "start": first["start"],
                        "end": first["end"],
                        "bbox": [round(float(v), 2) for v in bbox],
                        "score": 1.0,
                        "context": text[:start] + "[[" + text[start:end] + "]]" + text[end:],
                        "source": "pdf",
                    })
        return matches[:1000]

    for pno, lines in enumerate(doc.get("all_lines", [])):
        if page_filter is not None and int(page_filter) != pno:
            continue
        if regex:
            flags = 0 if case else re.IGNORECASE
            try:
                pat = re.compile(query, flags)
            except re.error as exc:
                raise ValueError(f"Regex invalid: {exc}") from exc
            for line in lines:
                raw, s, spans = engine._index_line(line, case)
                for m in pat.finditer(s):
                    if whole:
                        a, b = m.start(), m.end()
                        if a > 0 and (s[a - 1].isalnum() or s[a - 1] == "_"):
                            continue
                        if b < len(s) and (s[b].isalnum() or s[b] == "_"):
                            continue
                    match = engine._make_match(pno, line, raw, spans, m.start(), m.end(), 1.0)
                    matches.append(_match_payload(doc, match, pno))
        else:
            found = engine.find_matches([lines], query, case_sensitive=case, whole_word=whole, fuzzy=fuzzy)
            for match in found:
                match.page = pno
                matches.append(_match_payload(doc, match, pno))
    matches.sort(key=lambda m: (m["page"], m["bbox"][1], m["bbox"][0]))
    return matches[:1000]


def _match_payload(doc: dict, match: engine.Match, pno: int) -> dict:
    block = next((b for b in doc.get("blocks", []) if b.get("page") == pno and _overlap(b.get("bbox"), match.box)), None)
    return {
        "id": uuid.uuid4().hex[:12],
        "page": pno,
        "block_id": block.get("id") if block else None,
        "text": match.text,
        "matched_text": match.text,
        "bbox": [round(float(v), 2) for v in match.box],
        "score": round(float(match.score), 4),
        "context": match.context,
        "source": "ocr",
    }


def _overlap(a, b) -> bool:
    if not a or not b:
        return False
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _clamped_bbox(doc: dict, pno: int, raw, fallback) -> tuple[float, float, float, float] | None:
    """Validate an optional edit bbox and keep it inside page bounds."""
    def parse(value):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x0, y0, x1, y1 = (float(v) for v in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return None
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    box = parse(raw) or parse(fallback)
    if box is None:
        return None
    pages = doc.get("pages") or []
    if pno < 0 or pno >= len(pages):
        return None
    h, w = pages[pno].bgr.shape[:2]
    x0 = max(0.0, min(float(w), box[0]))
    y0 = max(0.0, min(float(h), box[1]))
    x1 = max(0.0, min(float(w), box[2]))
    y1 = max(0.0, min(float(h), box[3]))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _match_from_op(doc: dict, op: dict) -> tuple[int, engine.Match] | None:
    bid = op.get("block_id")
    block = block_by_id(doc, bid) if bid else None
    if block:
        pno = int(block["page"])
        bbox = _clamped_bbox(doc, pno, op.get("bbox"), block["bbox"])
        if bbox is None:
            return None
        text = block.get("text", "")
        return pno, engine.Match(pno, text, bbox, None, 1.0, _block_context(doc, block))
    if op.get("bbox") and op.get("matched_text") is not None:
        pno = int(op.get("page") or 0)
        bbox = _clamped_bbox(doc, pno, op.get("bbox"), op.get("bbox"))
        if bbox is None:
            return None
        text = str(op.get("matched_text") or "")
        return pno, engine.Match(pno, text, bbox, None, 1.0, str(op.get("context") or text))
    return None


def _style_path(style: dict) -> str | None:
    if not style:
        return None
    # Render payloads are untrusted.  Only font files that live inside the
    # app's font cache may be referenced by path; never open an arbitrary
    # local file supplied in JSON.
    candidate = style.get("font_file")
    if candidate:
        try:
            path = Path(str(candidate)).resolve()
            path.relative_to(fonts.FONT_ROOT.resolve())
        except (ValueError, OSError):
            path = None
        if path and path.is_file() and path.suffix.lower() in (".ttf", ".otf"):
            return str(path)
    family = style.get("font_family") or style.get("font")
    if family:
        rec = fonts.find_family(str(family), int(style.get("weight") or 400), bool(style.get("italic")))
        if rec:
            return rec.path
    return None


def _raster_settings(payload: dict) -> dict:
    s = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    method = str(s.get("method", "auto")).lower()
    if method not in {"auto", "fill", "inpaint"}:
        method = "auto"
    wm = str(s.get("watermark_mode", "auto")).lower()
    if wm not in {"auto", "under", "over", "off"}:
        wm = "auto"
    return {
        "method": method,
        "align": str(s.get("align", "left")).lower() if str(s.get("align", "left")).lower() in {"left", "center", "right"} else "left",
        "shrink": bool(s.get("shrink", False)),
        "watermark_mode": wm,
        "watermark_sensitivity": max(0.0, min(1.0, float(s.get("watermark_sensitivity", 0.5) or 0.5))),
        "case_match": bool(s.get("case_match", True)),
        "weight_match": bool(s.get("weight_match", True)),
        "stretch_match": bool(s.get("stretch_match", True)),
        "pixel_lock": bool(s.get("pixel_lock", True)),
        "output_format": str(s.get("output_format", "pdf" if payload.get("native") else "png")).lower(),
        "quality": max(60, min(100, int(s.get("quality", 92) or 92))),
    }


def render(sid: str, did: str, payload: dict) -> dict:
    doc = get_doc(sid, did)
    ops = payload.get("ops") if isinstance(payload.get("ops"), list) else []
    if not ops:
        raise ValueError("Koi edit operation nahi mila.")
    settings = _raster_settings(payload)
    if settings["output_format"] not in ALLOWED_FORMATS:
        settings["output_format"] = "png"
    warnings: list[str] = []
    styles: list[dict] = []

    if doc.get("kind") == "pdf_native" and all(
        op.get("insert") or str(op.get("block_id", "")).startswith("pdf-") for op in ops
    ):
        edits = []
        for op in ops:
            edits.append({
                "block_id": op.get("block_id"),
                "text": op.get("text", ""),
                "remove": bool(op.get("remove")),
                "style": op.get("style") or {},
                "bbox": op.get("bbox"),
                "page": op.get("page"),
                "insert": bool(op.get("insert")),
            })
        out = Path(doc["dir"]) / f"{safe_stem(doc['filename'])}_textswap.pdf"
        report = native_pdf.replace_spans(
            doc["path"], out, edits,
            font_lookup=lambda family, weight, italic: (
                fonts.find_family(family, weight, italic).path if fonts.find_family(family, weight, italic) else None
            ),
            scales=doc.get("native_scales"),
        )
        warnings.extend(report.get("warnings", []))
        previews = []
        for pno in range(len(doc.get("pages", []))):
            after = native_pdf.render_page(out, pno, dpi=140)
            before = native_pdf.render_page(doc["path"], pno, dpi=140)
            previews.append({"page": pno, "before": _preview_b64(before), "after": _preview_b64(after)})
        dl = _register(doc, out, "Edited PDF", "application/pdf")
        return {
            "kind": "native",
            "previews": previews,
            "downloads": [dl],
            "styles": [],
            "warnings": warnings,
            "changed": True,
            "native_report": report,
        }

    # Raster path: rebuild every edited page from its original source so undo /
    # redo and repeated edits stay deterministic.
    fonts_list = engine.discover_fonts(None)
    if not fonts_list:
        raise RuntimeError("Koi usable font nahi mila. System font install karein ya Fonts panel se family download karein.")
    edited_pages = list(doc["pages"])
    by_page: dict[int, list[dict]] = {}
    for op in ops:
        if op.get("insert"):
            pno = int(op.get("page") or 0)
            bbox = _clamped_bbox(doc, pno, op.get("bbox"), None)
            if bbox is None or not str(op.get("text") or "").strip():
                warnings.append(f"Insert skipped (invalid box/text): {op.get('block_id')}")
                continue
            by_page.setdefault(pno, []).append({**op, "_insert": True, "_bbox": bbox})
            continue
        resolved = _match_from_op(doc, op)
        if not resolved:
            warnings.append(f"Block not found for op: {op.get('block_id') or op.get('bbox')}")
            continue
        pno, match = resolved
        # A block's annotation box is the destination.  The source ink must be
        # erased at the block's original box, then the replacement is drawn at
        # the destination.  compute the difference as a draw nudge.
        nudge = (0.0, 0.0)
        block = block_by_id(doc, op.get("block_id")) if op.get("block_id") else None
        if block is not None and op.get("bbox"):
            src = tuple(float(v) for v in block["bbox"])
            dest = _clamped_bbox(doc, pno, op.get("bbox"), src)
            if dest is not None:
                nudge = (dest[0] - src[0], dest[1] - src[1])
                match = engine.Match(pno, match.text, src, match.next_x, match.score, match.context)
        by_page.setdefault(pno, []).append({**op, "_match": match, "_nudge": nudge})
    for pno, page_ops in sorted(by_page.items()):
        page = doc["pages"][pno]
        cur = page
        page_underlay = None
        if settings["watermark_mode"] != "off":
            try:
                page_underlay = engine.detect_underlay_page(page.bgr, settings["watermark_sensitivity"])
            except Exception:
                page_underlay = None
        skew = 0.0
        if any(not op.get("remove") for op in page_ops):
            try:
                skew = engine.estimate_skew(doc.get("all_lines", [[]])[pno] if pno < len(doc.get("all_lines", [])) else [])
            except Exception:
                skew = 0.0
        page_ops.sort(key=lambda o: (
            o["_match"].box[1] if o.get("_match") else o["_bbox"][1],
            o["_match"].box[0] if o.get("_match") else o["_bbox"][0],
        ))
        for op in page_ops:
            style = op.get("style") or {}
            color = _hex_to_bgr(style.get("color"))
            font_path = _style_path(style)
            style_info = {"block_id": op.get("block_id"), "page": pno + 1}
            if op.get("_insert"):
                if op.get("remove"):
                    continue
                try:
                    new_bgr = engine.insert_text(
                        cur, str(op.get("text", "")), op["_bbox"],
                        fonts=fonts_list, font_override=font_path, color=color,
                        size_scale=float(style.get("size_scale") or 1.0),
                        align=str(style.get("align") or settings["align"]),
                        style_info=style_info,
                    )
                    cur = engine.Page(new_bgr, page.alpha, page.meta)
                    styles.append(style_info)
                except Exception as exc:
                    warnings.append(f"'{str(op.get('text', ''))[:40]}' insert failed: {exc}")
                continue
            match = op["_match"]
            move_x, move_y = op.get("_nudge", (0.0, 0.0))
            try:
                new_bgr = engine.edit_page(
                    cur, [match], None if op.get("remove") else str(op.get("text", "")),
                    method=settings["method"], align=str(style.get("align") or settings["align"]),
                    shrink=settings["shrink"], fonts=fonts_list, font_override=font_path,
                    skew=skew, color=color,
                    size_scale=float(style.get("size_scale") or 1.0),
                    preserve_size=True,
                    weight_match=settings["weight_match"], stretch_match=settings["stretch_match"],
                    case_match=bool(style.get("case_match", settings["case_match"])),
                    nudge_x=float(style.get("nudge_x") or 0) + float(move_x),
                    nudge_y=float(style.get("nudge_y") or 0) + float(move_y),
                    style_info=style_info, underlay_mask=page_underlay,
                    watermark_mode=settings["watermark_mode"],
                    watermark_sensitivity=settings["watermark_sensitivity"],
                    pixel_lock=settings["pixel_lock"], source_bgr=page.bgr,
                    log=lambda *_: None,
                )
                cur = engine.Page(new_bgr, page.alpha, page.meta)
                styles.append(style_info)
            except Exception as exc:
                warnings.append(f"'{match.text[:40]}' edit failed: {exc}")
        edited_pages[pno] = cur

    # Save outputs.
    outdir = Path(doc["dir"]) / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(doc["filename"])
    downloads = []
    fmt = settings["output_format"]
    if fmt == "pdf":
        out = outdir / f"{stem}_textswap.pdf"
        engine.save_pages(edited_pages, str(out))
        downloads.append(_register(doc, out, "Edited PDF", "application/pdf"))
    else:
        ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}[fmt]
        multi = len(edited_pages) > 1
        for i, page in enumerate(edited_pages):
            path = outdir / (f"{stem}_textswap" + (f"_p{i + 1}" if multi else "") + ext)
            pil = engine._to_pil(page)
            if fmt == "jpg":
                if pil.mode in ("RGBA", "LA"):
                    from PIL import Image
                    base = Image.new("RGB", pil.size, (255, 255, 255))
                    base.paste(pil, mask=pil.split()[-1])
                    pil = base
                else:
                    pil = pil.convert("RGB")
                pil.save(path, "JPEG", quality=settings["quality"], subsampling=0)
            elif fmt == "webp":
                pil.save(path, "WEBP", quality=settings["quality"], method=6)
            else:
                pil.save(path, "PNG", optimize=True)
            downloads.append(_register(doc, path, "Edited image" if not multi else f"Edited page {i + 1}"))

    # OCR text + report.
    text_path = outdir / f"{stem}_ocr.txt"
    text_path.write_text("\n\n".join(f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(doc.get("page_texts", []))), encoding="utf-8")
    downloads.append(_register(doc, text_path, "OCR text (.txt)", "text/plain"))

    report = {
        "app": "TextSwap Studio Web", "version": "3.0", "filename": doc["filename"],
        "kind": doc["kind"], "ops": len(ops), "styles": styles, "warnings": warnings,
        "settings": settings,
    }
    report_path = outdir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    downloads.append(_register(doc, report_path, "Report (.json)", "application/json"))

    previews = []
    for pno, page in enumerate(doc["pages"]):
        before = page.bgr
        after = edited_pages[pno].bgr
        previews.append({"page": pno, "before": _preview_b64(before), "after": _preview_b64(after)})
    changed = any(np.any(edited_pages[i].bgr != doc["pages"][i].bgr) for i in range(len(edited_pages)))
    return {
        "kind": "raster",
        "previews": previews,
        "downloads": downloads,
        "styles": styles,
        "warnings": warnings,
        "changed": bool(changed),
    }


def export_text(sid: str, did: str) -> dict:
    doc = get_doc(sid, did)
    stem = safe_stem(doc["filename"])
    out = Path(doc["dir"]) / f"{stem}_ocr.txt"
    text = "\n\n".join(f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(doc.get("page_texts", [])))
    out.write_text(text, encoding="utf-8")
    return _register(doc, out, "OCR text (.txt)", "text/plain")


def delete_document(sid: str, did: str) -> None:
    doc = get_doc(sid, did)
    with _LOCK:
        _DOCS.pop(did, None)
        sess = _SESSIONS.get(sid)
        if sess and did in sess.get("docs", []):
            sess["docs"].remove(did)
    shutil.rmtree(doc["dir"], ignore_errors=True)
