"""Native PDF text-layer editing.

For digital PDFs we should not rasterise and repaint glyphs.  PyMuPDF can read
the real text spans (font name, size, colour, origin), remove only the selected
text with a redaction that leaves images/vector art untouched, and insert the
replacement with the original embedded font file when it is available.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF
    import fitz  # type: ignore

from .fonts import EMBEDDED_ROOT, _strip_subset

_BASE14 = {
    "helvetica": "helv", "helvetica-bold": "hebo", "helvetica-oblique": "heit",
    "helvetica-boldoblique": "hebi", "times-roman": "tiro", "times-bold": "tibo",
    "times-italic": "tiit", "times-bolditalic": "tibi", "courier": "cour",
    "courier-bold": "cobo", "courier-oblique": "coit", "courier-boldoblique": "cobi",
    "symbol": "symb", "zapfdingbats": "zadb",
}


def _font_file_name(basefont: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", _strip_subset(basefont))[:80] or "font"


def extract_embedded_font(doc, page, basefont: str) -> str | None:
    """Write the embedded font for ``basefont`` to disk and return its path."""
    target = _strip_subset(basefont).lower()
    try:
        fonts = page.get_fonts(full=True)
    except Exception:
        return None
    for item in fonts:
        try:
            xref, ext, ftype, name = item[0], item[1], item[2], item[3]
        except Exception:
            continue
        if _strip_subset(name).lower() != target:
            continue
        try:
            info = doc.extract_font(xref)
        except Exception:
            continue
        if not info or len(info) < 4 or not info[3]:
            continue
        suffix = (ext or info[1] or "ttf").lower()
        suffix = {"ttf": "ttf", "otf": "otf", "cff": "cff", "pfb": "pfb", "type1": "pfb"}.get(suffix, suffix)
        digest = hashlib.sha1(info[3]).hexdigest()[:16]
        out = EMBEDDED_ROOT / f"{_font_file_name(basefont)}-{digest}.{suffix}"
        if not out.exists():
            EMBEDDED_ROOT.mkdir(parents=True, exist_ok=True)
            out.write_bytes(info[3])
        if suffix in ("ttf", "otf"):
            return str(out)
    return None


def _color_tuple(value) -> tuple[float, float, float]:
    try:
        n = int(value)
    except Exception:
        return (0.0, 0.0, 0.0)
    return (((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0)


def _color_hex(value) -> str:
    try:
        return f"#{int(value) & 0xFFFFFF:06x}"
    except Exception:
        return "#000000"


def _chunk_record(text: str, chars: list[dict], start: int, end: int, index: int) -> dict:
    """Build one word/chunk record from a rawdict span character range."""
    group = chars[start:end]
    boxes = [ch.get("bbox") for ch in group if ch.get("bbox")]
    if boxes:
        x0 = min(float(b[0]) for b in boxes)
        y0 = min(float(b[1]) for b in boxes)
        x1 = max(float(b[2]) for b in boxes)
        y1 = max(float(b[3]) for b in boxes)
    else:
        x0 = y0 = x1 = y1 = 0.0
    origin = group[0].get("origin") if group else None
    if not origin:
        origin = (x0, y1)
    return {
        "index": index,
        "text": text[start:end],
        "bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
        "origin": [round(float(origin[0]), 3), round(float(origin[1]), 3)],
        "start": start,
        "end": end,
    }


def _span_text_and_chunks(span: dict) -> tuple[str, list[dict]]:
    """Return span text and per-word chunk boxes/origins.

    ``rawdict`` gives character boxes and origins, which lets us split a PDF
    span into independently editable words without guessing coordinates from
    character widths.  Whitespace separates chunks; punctuation stays with the
    word it touches.
    """
    chars = span.get("chars") or []
    text = "".join(str(ch.get("c") or "") for ch in chars)
    chunks: list[dict] = []
    if not chars:
        # Defensive fallback for a dict-mode span without characters.
        fallback = str(span.get("text") or "")
        bbox = span.get("bbox") or (0, 0, 0, 0)
        for index, match in enumerate(re.finditer(r"\S+", fallback)):
            span_w = max(0.0, float(bbox[2]) - float(bbox[0]))
            total = max(1, len(fallback))
            x0 = float(bbox[0]) + span_w * match.start() / total
            x1 = float(bbox[0]) + span_w * match.end() / total
            chunks.append({
                "index": index,
                "text": match.group(0),
                "bbox": [round(x0, 3), round(float(bbox[1]), 3), round(x1, 3), round(float(bbox[3]), 3)],
                "origin": [round(x0, 3), round(float(bbox[3]), 3)],
                "start": match.start(),
                "end": match.end(),
            })
        return fallback, chunks

    start: int | None = None
    for index, ch in enumerate(chars):
        if str(ch.get("c") or "").isspace():
            if start is not None:
                chunks.append(_chunk_record(text, chars, start, index, len(chunks)))
                start = None
            continue
        if start is None:
            start = index
    if start is not None:
        chunks.append(_chunk_record(text, chars, start, len(chars), len(chunks)))
    return text, chunks


def extract_spans(path: str | Path) -> list[list[dict]]:
    """Return editable text spans for every PDF page.

    Each span keeps its original geometry/style and also carries ``chunks``:
    per-word boxes, origins and character ranges used for chunk-level editing.
    """
    pages: list[list[dict]] = []
    with fitz.open(str(path)) as doc:
        for pno, page in enumerate(doc):
            page_spans: list[dict] = []
            try:
                data = page.get_text("rawdict")
            except Exception:
                pages.append(page_spans)
                continue
            for bi, block in enumerate(data.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                for li, line in enumerate(block.get("lines", [])):
                    for si, span in enumerate(line.get("spans", [])):
                        text, chunks = _span_text_and_chunks(span)
                        if not text.strip():
                            continue
                        bbox = [round(float(v), 3) for v in span.get("bbox", (0, 0, 0, 0))]
                        origin = [round(float(v), 3) for v in span.get("origin", (bbox[0], bbox[3]))]
                        flags = int(span.get("flags", 0))
                        font = _strip_subset(span.get("font") or "")
                        style = {
                            "font": font,
                            "size": round(float(span.get("size") or 12.0), 3),
                            "color": _color_hex(span.get("color", 0)),
                            "weight": 700 if (flags & 16) else 400,
                            "italic": bool(flags & 2),
                            "serif": bool(flags & 4),
                            "mono": bool(flags & 8),
                            "direction": [round(float(v), 4) for v in span.get("dir", (1, 0))],
                        }
                        page_spans.append({
                            "id": f"pdf-p{pno}-b{bi}-l{li}-s{si}",
                            "page": pno,
                            "text": text,
                            "bbox": bbox,
                            "origin": origin,
                            "page_size": [round(float(page.rect.width), 3), round(float(page.rect.height), 3)],
                            "style": style,
                            "chunks": chunks,
                            "source": "pdf",
                            "confidence": 1.0,
                        })
            pages.append(page_spans)
    return pages


def has_text_layer(path: str | Path, min_chars: int = 24) -> bool:
    try:
        with fitz.open(str(path)) as doc:
            chars = 0
            for page in doc:
                chars += len("".join(page.get_text().split()))
                if chars >= min_chars:
                    return True
    except Exception:
        return False
    return False


def _base14_for(font_name: str, bold: bool, italic: bool) -> str | None:
    key = _strip_subset(font_name).lower().replace(" ", "-")
    if bold and italic:
        key += "-boldoblique"
    elif bold:
        key += "-bold"
    elif italic:
        key += "-oblique"
    if key in _BASE14:
        return _BASE14[key]
    # Strip common suffixes before retrying.
    plain = re.sub(r"[-_ ]?(bold|italic|oblique|regular|roman|mt|ps).*$", "", key)
    return _BASE14.get(plain) or _BASE14.get(plain + "-roman")


def _page_redaction_kwargs():
    images = getattr(fitz, "PDF_REDACT_IMAGE_NONE", 0)
    graphics = getattr(fitz, "PDF_REDACT_LINE_ART_NONE", 0)
    text = getattr(fitz, "PDF_REDACT_TEXT_REMOVE", 0)
    return {"images": images, "graphics": graphics, "text": text}


_SPAN_ID_RE = re.compile(r"^pdf-p(\d+)-b(\d+)-l(\d+)-s(\d+)(?:-w(\d+))?$")


def _preview_bbox_to_pdf(raw, scale: float):
    """Convert an editor pixel bbox back to PDF points; return None if invalid."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) / scale for v in raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if x1 - x0 < 0.5 or y1 - y0 < 0.5:
        return None
    return fitz.Rect(x0, y0, x1, y1)


def replace_spans(
    input_path: str | Path,
    output_path: str | Path,
    edits: list[dict],
    *,
    font_lookup: Callable[[str, int, bool], str | None] | None = None,
    preserve_width: bool = True,
    scales: list[float] | None = None,
) -> dict:
    """Replace PDF text spans or chunks by block id.

    ``edits`` items: ``{block_id, text, remove?, style?, bbox?}``.  A block id
    ending in ``-wN`` targets one word/chunk inside its parent span; the rest
    of the span is left untouched.  ``bbox`` is an optional editor-preview box
    (pixels) that overrides the redaction/insertion area after a user resizes
    the annotation.  The original span/chunk geometry and font are used unless
    a style or bbox override is supplied.
    """
    input_path, output_path = str(input_path), str(output_path)
    wanted: dict[int, dict[str, dict]] = {}
    inserts: dict[int, list[dict]] = {}
    for edit in edits:
        if edit.get("insert"):
            pno = int(edit.get("page") or 0)
            inserts.setdefault(pno, []).append(edit)
            continue
        bid = str(edit.get("block_id") or edit.get("id") or "")
        if not bid:
            continue
        m = _SPAN_ID_RE.match(bid)
        if not m:
            continue
        pno = int(m.group(1))
        base = f"pdf-p{m.group(1)}-b{m.group(2)}-l{m.group(3)}-s{m.group(4)}"
        chunk = int(m.group(5)) if m.group(5) is not None else None
        wanted.setdefault(pno, {})[bid] = {**edit, "_base": base, "_chunk": chunk}

    report = {"replaced": 0, "removed": 0, "inserted": 0, "warnings": [], "fonts": []}
    with fitz.open(input_path) as doc:
        for pno, page in enumerate(doc):
            page_edits = wanted.get(pno)
            page_inserts = inserts.get(pno)
            if not page_edits and not page_inserts:
                continue
            try:
                data = page.get_text("rawdict") if page_edits else {}
            except Exception:
                data = {}
            span_map: dict[str, dict] = {}
            for bi, block in enumerate(data.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                for li, line in enumerate(block.get("lines", [])):
                    for si, span in enumerate(line.get("spans", [])):
                        span_map[f"pdf-p{pno}-b{bi}-l{li}-s{si}"] = span
            if page_edits and not span_map:
                report["warnings"].append(f"Page {pno + 1}: selected spans PDF me nahi mile.")

            chunk_cache: dict[str, list[dict]] = {}
            prepared = []
            for bid, edit in (page_edits or {}).items():
                span = span_map.get(str(edit["_base"]))
                if not span:
                    report["warnings"].append(f"{bid}: span missing; skip.")
                    continue
                grid_bbox = span.get("bbox") or (0, 0, 0, 0)
                grid_origin = span.get("origin") or (grid_bbox[0], grid_bbox[3])
                chunk_index = edit["_chunk"]
                if chunk_index is not None:
                    base = str(edit["_base"])
                    if base not in chunk_cache:
                        _, chunk_cache[base] = _span_text_and_chunks(span)
                    chunks = chunk_cache[base]
                    if chunk_index < 0 or chunk_index >= len(chunks):
                        report["warnings"].append(f"{bid}: chunk missing; skip.")
                        continue
                    target = chunks[chunk_index]
                    grid_bbox = target["bbox"]
                    grid_origin = target["origin"]
                base_rect = fitz.Rect(grid_bbox)
                if base_rect.width <= 0 or base_rect.height <= 0:
                    continue
                override = None
                if scales:
                    sx = float(scales[pno]) if pno < len(scales) and scales[pno] else 1.0
                    override = _preview_bbox_to_pdf(edit.get("bbox"), sx)
                if override is not None:
                    # Keep the original text baseline; move it only when the
                    # bottom edge of the annotation was dragged.
                    origin = (
                        max(page.rect.x0, min(page.rect.x1, override.x0)),
                        max(page.rect.y0, min(page.rect.y1, float(grid_origin[1]) + (override.y1 - base_rect.y1))),
                    )
                    bbox = override
                else:
                    origin = (float(grid_origin[0]), float(grid_origin[1]))
                    bbox = base_rect
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                # Keep redaction tight so neighbouring glyphs and watermarks
                # outside this span/chunk survive.  When the annotation box was
                # moved/resized, redact both the original source box and the new
                # destination so a move cannot leave the old glyphs behind.
                pad = 0.35
                rects = [bbox]
                if override is not None and (abs(base_rect.x0 - bbox.x0) > 0.01 or
                                             abs(base_rect.y0 - bbox.y0) > 0.01 or
                                             abs(base_rect.x1 - bbox.x1) > 0.01 or
                                             abs(base_rect.y1 - bbox.y1) > 0.01):
                    rects.append(base_rect)
                for rect in rects:
                    page.add_redact_annot(fitz.Rect(rect.x0 - pad, rect.y0 - pad,
                                                    rect.x1 + pad, rect.y1 + pad))
                style = dict(edit.get("style") or {})
                font_name = str(style.get("font") or span.get("font") or "")
                base_font = extract_embedded_font(doc, page, font_name)
                builtin = _base14_for(
                    font_name,
                    bool(style.get("weight", 0) and int(style.get("weight", 0)) >= 600) or bool(int(span.get("flags", 0)) & 16),
                    bool(style.get("italic")) or bool(int(span.get("flags", 0)) & 2),
                )
                chosen = None if builtin else base_font
                if not chosen and not builtin and font_lookup is not None:
                    try:
                        chosen = font_lookup(
                            str(style.get("font_family") or font_name),
                            int(style.get("weight") or (700 if int(span.get("flags", 0)) & 16 else 400)),
                            bool(style.get("italic") or (int(span.get("flags", 0)) & 2)),
                        )
                    except Exception:
                        chosen = None
                prepared.append({
                    "bid": bid,
                    "span": span,
                    "edit": edit,
                    "style": style,
                    "fontfile": chosen,
                    "builtin": builtin,
                    "remove": bool(edit.get("remove")),
                    "bbox": bbox,
                    "origin": origin,
                })

            for edit in page_inserts or []:
                bid = str(edit.get("block_id") or edit.get("id") or "insert")
                style = dict(edit.get("style") or {})
                text = str(edit.get("text") or "")
                if not text.strip() or edit.get("remove"):
                    continue
                sx = float(scales[pno]) if scales and pno < len(scales) and scales[pno] else 1.0
                rect = _preview_bbox_to_pdf(edit.get("bbox"), sx)
                if rect is None:
                    report["warnings"].append(f"{bid}: insert bbox invalid; skip.")
                    continue
                family = str(style.get("font_family") or style.get("font") or "")
                weight = int(style.get("weight") or 400)
                italic = bool(style.get("italic"))
                builtin = _base14_for(family, weight >= 600, italic)
                chosen = None
                if not builtin and font_lookup is not None and family:
                    try:
                        chosen = font_lookup(family, weight, italic)
                    except Exception:
                        chosen = None
                prepared.append({
                    "bid": bid,
                    "kind": "insert",
                    "edit": edit,
                    "style": style,
                    "fontfile": chosen,
                    "builtin": builtin,
                    "remove": False,
                    "bbox": rect,
                    "origin": (rect.x0, rect.y1),
                })

            if not prepared:
                continue
            if any(item.get("kind") != "insert" for item in prepared):
                try:
                    page.apply_redactions(**_page_redaction_kwargs())
                except Exception:
                    page.apply_redactions()

            alias_cache: dict[str, str] = {}
            for item in prepared:
                if item["remove"]:
                    report["removed"] += 1
                    continue
                style = item["style"]
                text = str(item["edit"].get("text") or "")
                if not text:
                    report["removed"] += 1
                    continue
                if item.get("kind") == "insert":
                    color = style.get("color")
                    if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                        rgb = (r / 255.0, g / 255.0, b / 255.0)
                    else:
                        rgb = (0.0, 0.0, 0.0)
                    fontfile = item["fontfile"]
                    builtin = item["builtin"]
                    alias = builtin
                    if fontfile:
                        key = fontfile
                        alias = alias_cache.get(key)
                        if not alias:
                            alias = f"TS{len(alias_cache)}"
                            try:
                                page.insert_font(fontname=alias, fontfile=fontfile)
                                alias_cache[key] = alias
                            except Exception as exc:
                                report["warnings"].append(f"{item['bid']}: font embed failed ({exc}); fallback.")
                                alias = builtin or "helv"
                        if fontfile not in report["fonts"]:
                            report["fonts"].append(fontfile)
                    if not alias:
                        alias = "helv"
                    align = str(style.get("align") or "left")
                    text_align = {
                        "center": fitz.TEXT_ALIGN_CENTER,
                        "right": fitz.TEXT_ALIGN_RIGHT,
                    }.get(align, fitz.TEXT_ALIGN_LEFT)
                    rect = item["bbox"]
                    explicit = float(style.get("size") or 0)
                    try:
                        if "\n" not in text.strip():
                            size = explicit or max(6.0, rect.height * 0.78)
                            try:
                                text_w = (fitz.Font(fontfile=fontfile).text_length(text, fontsize=size)
                                          if fontfile else fitz.get_text_length(text, fontname=alias, fontsize=size))
                            except Exception:
                                text_w = 0.0
                            if text_w > rect.width > 0:
                                size *= rect.width / text_w
                                text_w = rect.width
                            if align == "center":
                                origin_x = rect.x0 + (rect.width - text_w) / 2
                            elif align == "right":
                                origin_x = rect.x1 - text_w
                            else:
                                origin_x = rect.x0
                            page.insert_text(
                                (origin_x, rect.y1 - max(1.5, size * 0.18)),
                                text, fontsize=size, fontname=alias, color=rgb, render_mode=0,
                            )
                        else:
                            size = explicit or min(rect.height, 24.0)
                            for _ in range(6):
                                if page.insert_textbox(
                                    rect, text, fontsize=size, fontname=alias,
                                    color=rgb, align=text_align,
                                ) >= 0:
                                    break
                                size *= 0.82
                            else:
                                raise RuntimeError("text did not fit in the box")
                        report["inserted"] += 1
                    except Exception as exc:
                        report["warnings"].append(f"{item['bid']}: insert failed ({exc})")
                    continue
                span = item["span"]
                size = float(style.get("size") or span.get("size") or 12.0)
                color = style.get("color")
                if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    rgb = (r / 255.0, g / 255.0, b / 255.0)
                else:
                    rgb = _color_tuple(span.get("color", 0))
                fontfile = item["fontfile"]
                builtin = item["builtin"]
                alias = builtin
                if fontfile:
                    key = fontfile
                    alias = alias_cache.get(key)
                    if not alias:
                        alias = f"TS{len(alias_cache)}"
                        try:
                            page.insert_font(fontname=alias, fontfile=fontfile)
                            alias_cache[key] = alias
                        except Exception as exc:
                            report["warnings"].append(f"{item['bid']}: font embed failed ({exc}); fallback.")
                            alias = builtin or "helv"
                    if fontfile not in report["fonts"]:
                        report["fonts"].append(fontfile)
                if not alias:
                    alias = "helv"
                origin = fitz.Point(item["origin"])
                morph = None
                if preserve_width:
                    try:
                        base_w = float(item["bbox"].width)
                        if fontfile:
                            probe = fitz.Font(fontfile=fontfile)
                            new_w = probe.text_length(text, fontsize=size)
                        else:
                            new_w = fitz.get_text_length(text, fontname=alias, fontsize=size)
                        if new_w > 0.5 and base_w > 0.5:
                            ratio = base_w / new_w
                            if 0.55 <= ratio <= 1.8 and abs(ratio - 1.0) > 0.015:
                                morph = (origin, fitz.Matrix(ratio, 1))
                    except Exception:
                        morph = None
                try:
                    page.insert_text(
                        origin,
                        text,
                        fontsize=size,
                        fontname=alias,
                        color=rgb,
                        render_mode=0,
                        morph=morph,
                    )
                    report["replaced"] += 1
                except Exception as exc:
                    report["warnings"].append(f"{item['bid']}: insert failed ({exc})")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path, garbage=3, deflate=True)
    return report


def render_page(path: str | Path, page_index: int = 0, dpi: int = 130) -> np.ndarray:
    """Render a PDF page to a BGR numpy image for the editor canvas."""
    with fitz.open(str(path)) as doc:
        page = doc.load_page(max(0, min(page_index, doc.page_count - 1)))
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        return rgb[:, :, ::-1].copy()
