"""Pixel-level text replacement engine extracted from TextSwap_Studio.ipynb.

The code in this module is the original, tested renderer: OCR boxes become
pixel regions, old ink is rebuilt from the local background (with underlay /
watermark protection), and replacement text is fitted to the source glyphs.
"""
from __future__ import annotations

import argparse
import difflib
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except Exception as _import_exc:  # pragma: no cover - environment check
    raise ImportError(
        "OCR packages load nahi hue. Run `pip install -r requirements.txt` in the venv. "
        f"Original error: {_import_exc}"
    ) from _import_exc


def _configure_tesseract() -> None:
    """Make pytesseract find the Windows installer path when PATH is bare."""
    import shutil

    if shutil.which("tesseract"):
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


_configure_tesseract()

# =============================================================================
# Image / PDF I/O
# =============================================================================


@dataclass
class Page:
    bgr: np.ndarray
    alpha: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def load_pages(path: str, dpi: int = 300) -> list[Page]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24
        except ImportError:
            import fitz

        pages = []
        with fitz.open(path) as doc:
            for p in doc:
                pix = p.get_pixmap(dpi=dpi, alpha=False)
                rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                pages.append(Page(rgb[:, :, ::-1].copy(), meta={"dpi": (dpi, dpi), "mode": "RGB"}))
        return pages

    im = Image.open(path)
    meta = {"dpi": im.info.get("dpi"), "icc": im.info.get("icc_profile"), "mode": im.mode}
    im = ImageOps.exif_transpose(im)  # pixels upright, so OCR coordinates make sense
    exif = im.getexif()
    meta["exif"] = exif if len(exif) else None
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = np.asarray(im)[:, :, 3].copy()
    rgb = np.asarray(im.convert("RGB"))
    return [Page(rgb[:, :, ::-1].copy(), alpha, meta)]


def _to_pil(p: Page) -> Image.Image:
    im = Image.fromarray(p.bgr[:, :, ::-1])
    if p.alpha is not None:
        im = im.convert("RGBA")
        im.putalpha(Image.fromarray(p.alpha))
    elif p.meta.get("mode") == "L":
        im = im.convert("L")
    return im


def save_pages(pages: list[Page], out_path: str) -> None:
    ext = Path(out_path).suffix.lower()
    pils = [_to_pil(p) for p in pages]
    meta = pages[0].meta
    kw: dict = {}
    if ext == ".pdf":
        dpi = meta.get("dpi") or (300, 300)
        pils = [p.convert("RGB") for p in pils]
        pils[0].save(out_path, "PDF", save_all=True, append_images=pils[1:],
                     resolution=float(dpi[0]), quality=95)
        return
    if meta.get("dpi"):
        kw["dpi"] = meta["dpi"]
    if meta.get("icc"):
        kw["icc_profile"] = meta["icc"]
    if ext in (".jpg", ".jpeg"):
        kw.update(quality=97, subsampling=0)
        pils[0] = pils[0].convert("RGB")
    if meta.get("exif") is not None and ext in (".jpg", ".jpeg", ".png", ".webp"):
        kw["exif"] = meta["exif"]
    pils[0].save(out_path, **kw)


# =============================================================================
# OCR + text search
# =============================================================================


def _odd(n) -> int:
    n = int(n)
    return n if n % 2 else n + 1


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float


@dataclass
class Match:
    page: int
    text: str            # exact OCR substring that matched
    box: tuple           # (x0, y0, x1, y1) approximate text box
    next_x: float | None  # left edge of the following word on the line
    score: float         # 1.0 = exact, <1 = fuzzy
    context: str         # whole OCR line with the match in [[ ]]


def _ocr_variants(gray: np.ndarray) -> list[np.ndarray]:
    """Extra views of the page that help OCR on textured / uneven backgrounds."""
    k = _odd(max(15, min(gray.shape) // 40))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def binarise(x):
        x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        _, b = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return 255 - b                                  # dark text on white
    dark_on_light = binarise(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker))
    light_on_dark = binarise(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker))
    return [dark_on_light, light_on_dark]


def _tess(gray: np.ndarray, lang: str, psm: int, scale: float) -> list[list[Word]]:
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(gray, lang=lang, config=f"--psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    lines: dict = {}
    for i, raw in enumerate(d["text"]):
        t = raw.strip()
        if not t or float(d["conf"][i]) < 0:
            continue
        key = (d["page_num"][i], d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append(Word(
            t, int(d["left"][i] / scale), int(d["top"][i] / scale),
            max(1, int(round(d["width"][i] / scale))), max(1, int(round(d["height"][i] / scale))),
            float(d["conf"][i])))
    out = [sorted(v, key=lambda k: k.x) for v in lines.values()]
    out.sort(key=lambda l: (l[0].y, l[0].x))
    return out


def ocr_lines(bgr: np.ndarray, lang: str = "eng", psm: int = 3, enhance: bool = False) -> list[list[Word]]:
    h, w = bgr.shape[:2]
    scale = 1.0 if max(h, w) >= 2000 else min(3.0, 2000.0 / max(h, w))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if not enhance:
        return _tess(gray, lang, psm, scale)
    best, best_score = [], -1.0
    for g in [gray] + _ocr_variants(gray):               # keep the view the OCR understands best
        lines = _tess(g, lang, psm, scale)
        score = sum(wd.conf for ln in lines for wd in ln if wd.conf > 40)
        if score > best_score:
            best, best_score = lines, score
    return best


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _index_line(line: list[Word], case_sensitive: bool):
    raw = " ".join(_nfkc(w.text) for w in line)
    s = raw
    if not case_sensitive:
        low = raw.lower()
        s = low if len(low) == len(raw) else raw
    spans, pos = [], 0
    for w in line:
        n = len(_nfkc(w.text))
        spans.append((pos, pos + n))
        pos += n + 1
    return raw, s, spans


def _make_match(page_no, line, raw, spans, s0, e0, score) -> Match:
    idx = [i for i, (a, b) in enumerate(spans) if b > s0 and a < e0]
    fi, li = idx[0], idx[-1]
    wf, wl = line[fi], line[li]
    a, b = spans[fi]
    x0 = wf.x + wf.w * max(0, s0 - a) / max(1, b - a)
    a, b = spans[li]
    x1 = wl.x + wl.w * min(1.0, (e0 - a) / max(1, b - a))
    y0 = min(w.y for w in line[fi:li + 1])
    y1 = max(w.y + w.h for w in line[fi:li + 1])
    ends_word = e0 >= spans[li][1]
    if ends_word:
        next_x = line[li + 1].x if li + 1 < len(line) else None
    else:
        next_x = x1
    ctx = raw[:s0] + "[[" + raw[s0:e0] + "]]" + raw[e0:]
    return Match(page_no, raw[s0:e0], (x0, y0, x1, y1), next_x, score, ctx)


def find_matches(all_lines: list[list[list[Word]]], query: str, case_sensitive=False,
                 whole_word=False, fuzzy=0.8) -> list[Match]:
    q = " ".join(_nfkc(query).split())
    if not q:
        return []
    qs = q if case_sensitive else q.lower()
    pat = re.escape(qs)
    if whole_word:
        pat = r"(?<!\w)" + pat + r"(?!\w)"
    found: list[Match] = []
    for pno, lines in enumerate(all_lines):
        for line in lines:
            raw, s, spans = _index_line(line, case_sensitive)
            for m in re.finditer(pat, s):
                found.append(_make_match(pno, line, raw, spans, m.start(), m.end(), 1.0))
    if found or fuzzy <= 0:
        return found

    # fuzzy fallback: tolerate OCR slips such as O/0, l/1, rn/m
    nw = len(qs.split())
    for pno, lines in enumerate(all_lines):
        for line in lines:
            raw, s, spans = _index_line(line, case_sensitive)
            toks = [s[a:b] for a, b in spans]
            cands = []
            for i in range(len(toks)):
                for j in range(i, min(len(toks), i + nw + 1)):
                    if j - i + 1 < max(1, nw - 1):
                        continue
                    cand = " ".join(toks[i:j + 1]).strip(".,;:!?()[]\"'")
                    r = difflib.SequenceMatcher(None, cand, qs).ratio()
                    if r >= fuzzy:
                        cands.append((r, i, j))
            used: set = set()
            for r, i, j in sorted(cands, key=lambda c: -c[0]):
                if used & set(range(i, j + 1)):
                    continue
                used |= set(range(i, j + 1))
                found.append(_make_match(pno, line, raw, spans, spans[i][0], spans[j][1], r))
    return found


# =============================================================================
# Watermark / underlay detection
# =============================================================================

@dataclass
class UnderlayProfile:
    """A watermark, logo or stamp that shares the page with the matched text.

    ``mask`` is the visible underlay ink inside the current region, ``bridge``
    is an estimate of the same structure continued across the old glyphs and
    ``alpha`` is the per-pixel veil strength used when the watermark sits on
    top of the text.  All arrays are in region (ctx) coordinates.
    """
    present: bool = False
    order: str = "none"          # "under" | "over" | "none"
    confidence: float = 0.0
    coverage: float = 0.0
    mask: np.ndarray | None = None
    bridge: np.ndarray | None = None
    alpha: np.ndarray | None = None
    color: tuple = (0.0, 0.0, 0.0)
    kind: str = "watermark"


def detect_underlay_page(img: np.ndarray, sensitivity: float = 0.5,
                         max_side: int = 1400) -> np.ndarray:
    """Find low-contrast structured ink (watermark / logo / stamp) on a page.

    Normal text is high contrast and is deliberately excluded.  What remains
    is mid-tone, image-like ink: a grey crest, a coloured seal, a repeated
    diagonal word, a bank stamp.  Colour is as important as brightness -- on
    a tinted scan the watermark is often *neutral* while the paper is pink --
    so the score combines local luminance and local chroma deviation.
    """
    h0, w0 = img.shape[:2]
    if min(h0, w0) < 48:
        return np.zeros((h0, w0), bool)
    scale = min(1.0, float(max_side) / float(max(h0, w0)))
    if scale < 0.999:
        small = cv2.resize(img, (max(2, int(round(w0 * scale))), max(2, int(round(h0 * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # A wide median keeps illumination gradients but rejects strokes thinner
    # than the kernel, so a crest/logo survives as a positive residual.
    k = _odd(np.clip(min(h, w) * 0.07, 25, 101))
    bg = cv2.medianBlur(lab.astype(np.uint8), int(k)).astype(np.float32)
    dl = bg[..., 0] - lab[..., 0]                  # >0: darker than local paper
    da = lab[..., 1] - bg[..., 1]
    db = lab[..., 2] - bg[..., 2]
    chroma = np.sqrt(da * da + db * db)
    score = 1.8 * chroma + 0.55 * np.maximum(dl, 0)
    score = cv2.GaussianBlur(score, (0, 0), 1.15)

    # Strong ink is content text, not underlay.  The absolute fallback catches
    # inverted scans where the local residual is not positive.
    strong = (np.abs(dl) > 58) | (gray < 105)
    near_text = cv2.dilate(strong.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vals = score[~near_text]
    if vals.size < 50:
        return np.zeros((h0, w0), bool)
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    base = max(4.0, med + 2.6 * mad)
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    thr = max(3.2, base * (1.30 - 0.60 * sens))
    cand = (score > thr) & ~near_text
    cand &= (dl > 2.0) | (chroma > 2.0)
    u8 = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros_like(u8)
    min_area = max(8, int(round(h * w * 0.000006)))
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area >= min_area and (max(ww, hh) >= 7 or area >= 30):
            keep[labels == i] = 1
    if scale < 0.999:
        keep = cv2.resize(keep, (w0, h0), interpolation=cv2.INTER_NEAREST)
        keep = cv2.dilate(keep, np.ones((3, 3), np.uint8)) > 0
    return keep


def _build_underlay_profile(ctx: np.ndarray, target: np.ndarray, core: np.ndarray,
                            mask: np.ndarray, roi: tuple, bh: int, stroke: float,
                            underlay: np.ndarray, color: tuple,
                            forced: str = "auto") -> UnderlayProfile:
    """Decide whether a local underlay is present and how it layers with text."""
    prof = UnderlayProfile()
    H, W = target.shape
    m = underlay.astype(bool).copy()
    if not m.any():
        return prof
    # Text anti-alias fringes look like low-contrast ink; never let them become
    # "watermark" pixels that would survive the erase and leave a ghost.
    near = cv2.dilate(target.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    m &= ~near
    if m.sum() < 6:
        return prof

    rl, rt, rr, rb = roi
    ex = max(4, int(round(bh * 1.1)))
    er0, er1 = max(0, rt - ex), min(H, rb + ex)
    ec0, ec1 = max(0, rl - ex), min(W, rr + ex)
    zone = np.zeros((H, W), bool)
    zone[er0:er1, ec0:ec1] = True
    hits = m & zone
    zone_area = max(1, int(zone.sum()))
    cov = float(hits.sum()) / float(zone_area)
    if hits.sum() < max(8, int(round(zone_area * 0.004))):
        return prof

    n, labels, stats, _ = cv2.connectedComponentsWithStats(hits.astype(np.uint8), 8)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if not comps:
        return prof

    prof.present = True
    prof.coverage = round(min(1.0, cov * 8.0), 3)
    # Continue the visible watermark across the text only where the old glyphs
    # are.  Closing with a kernel wider than a stem bridges the interruption.
    k = _odd(int(np.clip(stroke * 2.0 + 5.0, 5, 31)))
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    hidden = closed & mask

    g = cv2.cvtColor(ctx, cv2.COLOR_BGR2GRAY)
    ink_lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    probe = hidden & target
    light_frac = 0.0
    if probe.sum() >= 5:
        light_frac = float(np.mean(g[probe] > ink_lum + 24.0))

    order = "under"
    z_conf = 0.58
    if probe.sum() >= 5:
        if light_frac >= 0.34:
            order = "over"
            z_conf = float(np.clip(0.58 + light_frac, 0.0, 0.98))
        else:
            z_conf = float(np.clip(0.66 + (0.30 - min(0.30, light_frac)), 0.0, 0.95))
    if forced in ("under", "over"):
        order = forced
        z_conf = max(z_conf, 0.62)

    # Estimate the watermark colour and veil alpha from visible pixels only.
    vals = ctx[m].astype(np.float32)
    gv = g[m].astype(np.float32)
    if vals.size:
        cut = float(np.percentile(gv, 35))
        chosen = vals[gv <= cut]
        wcol = np.median(chosen if chosen.size else vals, axis=0).astype(np.float32)
    else:
        wcol = np.array(color, np.float32)
    known_paper = ~(mask | m | near)
    wgt = known_paper.astype(np.float32)
    paper = _norm_blur(ctx.astype(np.float32), wgt, max(2.0, bh * 0.6)).astype(np.float32)
    den = paper - wcol
    den2 = np.sum(den * den, axis=2)
    alpha = np.zeros((H, W), np.float32)
    if float(den2[m].mean()) > 1.0:
        a_raw = np.sum((paper - ctx.astype(np.float32)) * den, axis=2) / np.maximum(den2, 1.0)
        alpha[m] = np.clip(a_raw[m], 0.0, 1.0)
        if hidden.any():
            a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            fill8 = cv2.inpaint(a8, mask.astype(np.uint8) * 255, 2, cv2.INPAINT_TELEA)
            alpha[hidden] = (fill8[hidden].astype(np.float32) / 255.0)

    conf = 0.45 * min(1.0, cov / 0.06) + 0.25 * min(1.0, len(comps) / 3.0) + 0.30 * z_conf
    prof.order = order
    prof.confidence = round(float(np.clip(conf, 0.25, 0.99)), 3)
    prof.mask = m
    prof.bridge = hidden
    prof.alpha = alpha
    prof.color = tuple(float(v) for v in wcol)
    big = any(stats[i, cv2.CC_STAT_WIDTH] > W * 0.45 or stats[i, cv2.CC_STAT_AREA] > H * W * 0.02
              for i in comps)
    prof.kind = "logo" if big else "watermark"
    return prof


def _underlay_report(prof: UnderlayProfile | None, mode: str = "auto") -> dict:
    if prof is None or not prof.present:
        return {"present": False, "order": "none", "confidence": 0.0,
                "coverage": 0.0, "kind": "none", "mode": mode}
    return {
        "present": True,
        "order": prof.order,
        "confidence": round(float(prof.confidence), 3),
        "coverage": round(float(prof.coverage), 3),
        "kind": prof.kind,
        "mode": mode,
    }


# =============================================================================
# Pixel-level analysis: find the ink of the matched text
# =============================================================================


@dataclass
class Region:
    ctx: tuple             # (x0, y0, x1, y1) working window in page coordinates
    target: np.ndarray     # bool, ink of the matched text
    mask: np.ndarray       # bool, pixels that will be rebuilt (dilated target)
    protect: np.ndarray    # bool, ink that must NOT be touched (neighbours, lines)
    ink_bbox: tuple        # tight bbox of the target ink incl. anti-alias fringe
    core: np.ndarray       # bool, half-coverage contour (used for font/size/baseline fitting)
    core_bbox: tuple       # tight bbox of the core (page coordinates)
    color: tuple           # BGR ink colour
    bh: int                # nominal text height
    roi: tuple = (0, 0, 0, 0)  # padded match box in ctx coordinates (l, t, r, b)
    soften: float = 0.5    # edge softness estimated from the source ink
    stroke: float = 2.0    # typical full stroke width in pixels
    underlay: UnderlayProfile | None = None


def analyze_region(img: np.ndarray, box: tuple, underlay_mask: np.ndarray | None = None,
                   watermark_mode: str = "auto", watermark_sensitivity: float = 0.5) -> Region | None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    bh = max(4, y1 - y0)
    pad = max(2, int(round(bh * 0.12)))
    margin = max(6, int(round(bh * 1.2)))
    rx0, ry0, rx1, ry1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    cx0, cy0, cx1, cy1 = max(0, rx0 - margin), max(0, ry0 - margin), min(W, rx1 + margin), min(H, ry1 + margin)
    ctx = img[cy0:cy1, cx0:cx1]

    # local background estimate (handles gradients, shadows, coloured paper)
    bg = cv2.medianBlur(ctx, _odd(np.clip(bh * 3, 21, 201)))
    lab = cv2.cvtColor(ctx, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - lab_bg, axis=2)

    roi = (slice(ry0 - cy0, ry1 - cy0), slice(rx0 - cx0, rx1 - cx0))
    d8 = np.clip(dist[roi], 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(18.0, 0.45 * otsu)
    ink = dist > thr

    # long thin structures (underlines, table borders) are not part of the text
    L = max(25, int(bh * 3))
    u8 = ink.astype(np.uint8)
    hl = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vl = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8)) > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink & ~lines).astype(np.uint8), connectivity=8)
    rl, rt, rr, rb = rx0 - cx0, ry0 - cy0, rx1 - cx0, ry1 - cy0
    # Separate high-contrast content text from mid-tone underlay ink.  A
    # watermark arc crossing the OCR box must not be treated as a glyph: that
    # is what used to erase the crest behind the name.
    level = float(np.percentile(dist[ink], 95)) if ink.any() else float(thr)
    text_thr = max(float(thr), 0.55 * float(otsu), 0.45 * level)
    target = np.zeros(ink.shape, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        ix = max(0, min(x + w, rr) - max(x, rl))
        iy = max(0, min(y + h, rb) - max(y, rt))
        if area < 2 or ix * iy < 0.6 * w * h:
            continue
        comp = labels == i
        strength = float(np.percentile(dist[comp], 75))
        if strength >= text_thr:               # glyph, not a faint crest stroke
            target |= comp
    # A dotted underline is not part of the word, but its dots are strong ink
    # and can sit inside a wide OCR box.  Left in, they inflate the measured
    # glyph height and the replacement comes out far too large.
    tiny_h = max(3, int(round(bh * 0.22)))
    row = {}
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h > tiny_h or w > max(7, int(round(bh * 0.40))) or area > max(20, bh * bh * 0.10):
            continue
        yc = int(round((y + h * 0.5) / 2.0)) * 2
        row.setdefault(yc, []).append(i)
    dot_line = np.zeros(ink.shape, bool)
    for yc, ids in row.items():
        if len(ids) < 6:
            continue
        xs0 = min(stats[i, cv2.CC_STAT_LEFT] for i in ids)
        xs1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in ids)
        if xs1 - xs0 >= 0.5 * max(1, x1 - x0):
            for i in ids:
                dot_line |= labels == i
    if dot_line.any():
        target &= ~dot_line
        lines = lines | dot_line
    # Strong ink inside the OCR box itself belongs to the match.  Faint ink
    # inside the same box is left alone, not blanket-added.
    bl, bt, br, bb = x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0
    inner = np.zeros(ink.shape, bool)
    inner[max(0, bt):max(0, bb), max(0, bl):max(0, br)] = True
    target |= (ink & ~lines) & inner & (dist >= text_thr)
    if not target.any():
        return None

    r = max(1, int(round(bh / 20)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    other = (ink & ~target) | lines
    mask = (cv2.dilate(target.astype(np.uint8), k) > 0) & ~other

    ys, xs = np.nonzero(target)
    ink_bbox = (cx0 + int(xs.min()), cy0 + int(ys.min()), cx0 + int(xs.max()) + 1, cy0 + int(ys.max()) + 1)
    dt = dist[target]
    level = float(np.percentile(dt, 90))                  # typical full-ink contrast
    core = target & (dist >= 0.5 * level)                 # ~50% coverage contour
    if core.sum() < 4:
        core = target
    cys, cxs = np.nonzero(core)
    core_bbox = (cx0 + int(cxs.min()), cy0 + int(cys.min()), cx0 + int(cxs.max()) + 1, cy0 + int(cys.max()) + 1)
    # Use only the strongest ink pixels for colour.  The 80th percentile still
    # contains anti-aliased fringe and makes dark text look grey.
    strong = target & (dist >= np.percentile(dt, 92))
    if strong.sum() < 6:
        strong = core
    color = tuple(float(c) for c in np.median(ctx[strong], axis=0))

    # Estimate how soft the source glyph edges are, so freshly rendered text
    # gets the same amount of scan / anti-alias blur instead of looking pasted.
    fringe = target & ~core
    soften = 0.45
    if fringe.sum() > 8:
        from_core = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
        edge_w = float(np.percentile(from_core[fringe], 65))
        soften = float(np.clip(edge_w * 0.55, 0.25, 1.6))
    # A fixed pixel blur that is invisible on a 90 px heading destroys a 10 px
    # caption: every stem becomes a half-covered edge and the ink reads grey.
    # Scale the blur by glyph height so small replacements stay as dark and
    # crisp as the source they replace.
    soften *= min(1.0, max(0.0, bh / 60.0))

    # Typical full stroke width.  Distance transform on the core gives the
    # half-width at each interior pixel; the 85th percentile tracks the stem.
    dt_core = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    stroke = float(max(1.0, np.percentile(dt_core[core], 85) * 2.0))

    # Underlay-aware protection.  Visible watermark pixels are added to
    # ``protect`` so the erase mask never covers them, and the local profile
    # tells erase/draw how to rebuild the hidden continuation.
    underlay_prof = None
    if (underlay_mask is not None and watermark_mode != "off"
            and getattr(underlay_mask, "shape", None) == img.shape[:2]):
        crop = underlay_mask[cy0:cy1, cx0:cx1].astype(bool)
        if crop.any():
            near_t = cv2.dilate(target.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            safe = crop & ~near_t
            if safe.any():
                other = other | safe
                mask = (cv2.dilate(target.astype(np.uint8), k) > 0) & ~other
                underlay_prof = _build_underlay_profile(
                    ctx, target, core, mask, (rl, rt, rr, rb), bh, stroke,
                    safe, color, forced=watermark_mode,
                )

    return Region((cx0, cy0, cx1, cy1), target, mask, other, ink_bbox, core, core_bbox,
                  color, bh, (rl, rt, rr, rb), soften, stroke, underlay_prof)


# =============================================================================
# Erase: rebuild the background
# =============================================================================


def _norm_blur(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian average of only the 'known' pixels (weights w), multi-scale."""
    out = np.zeros_like(img)
    done = np.zeros(img.shape[:2], bool)
    for s in (sigma, sigma * 2, sigma * 4, sigma * 8):
        num = cv2.GaussianBlur(img * w[..., None], (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)
        ok = (den > 0.06) & ~done
        out[ok] = num[ok] / den[ok][:, None]
        done |= ok
    if not done.all():
        out[~done] = img[w > 0].mean(0) if (w > 0).any() else 255
    return out


def _robust_std(res: np.ndarray) -> np.ndarray:
    if res.size == 0:
        return np.zeros(3, np.float32)
    med = np.median(res, axis=0)
    return 1.4826 * np.median(np.abs(res - med), axis=0)


def erase(out: np.ndarray, src: np.ndarray, reg: Region, method: str = "auto", seed: int = 0) -> str:
    cx0, cy0, cx1, cy1 = reg.ctx
    ctx = src[cy0:cy1, cx0:cx1].astype(np.float32)
    prot = cv2.dilate(reg.protect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    known = ~reg.mask & ~prot
    # sample only pixels of the SAME surface as the text sits on (not the paper next to a coloured band)
    rl, rt, rr, rb = reg.roi
    labc = cv2.cvtColor(ctx.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    rk = known[rt:rb, rl:rr]
    if rk.sum() > 20:
        ref = np.median(labc[rt:rb, rl:rr][rk], axis=0)
        known &= np.linalg.norm(labc - ref, axis=2) < 45
    wgt = known.astype(np.float32)

    under = reg.underlay
    if under is not None and under.present and under.order == "under":
        # Flat fill creates a rectangular smear over a crest/stamp.  Rebuild
        # only the old glyph strokes and let the surrounding watermark ink
        # continue through them, then add a conservative structural hint from
        # the underlay alpha map.
        used = "watermark"
        rad = int(np.clip(round(reg.stroke * 0.45), 2, 4))
        fill = cv2.inpaint(ctx.astype(np.uint8), reg.mask.astype(np.uint8) * 255, rad,
                           cv2.INPAINT_TELEA).astype(np.float32)
        if under.bridge is not None and under.alpha is not None and under.bridge.any():
            paper_known = ~(reg.mask | under.mask | prot)
            paper = _norm_blur(ctx, paper_known.astype(np.float32), max(2.0, reg.bh * 0.6))
            wcol = np.array(under.color, np.float32)
            au = np.clip(under.alpha, 0.0, 1.0)
            struct = paper * (1 - au[..., None]) + wcol * au[..., None]
            wstruct = under.bridge.astype(np.float32) * 0.55 * np.clip(au * 3.0, 0.0, 1.0)
            wstruct *= float(np.clip(under.confidence, 0.3, 0.95))
            fill = fill * (1 - wstruct[..., None]) + struct * wstruct[..., None]
        m = reg.mask.astype(np.float32)
        grow = cv2.dilate(reg.mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        a = np.maximum(m, cv2.GaussianBlur(m, (0, 0), 0.7) * grow)
        a[reg.protect] = 0
        base = out[cy0:cy1, cx0:cx1].astype(np.float32)
        out[cy0:cy1, cx0:cx1] = np.clip(base * (1 - a[..., None]) + fill * a[..., None], 0, 255).astype(np.uint8)
        return used

    used = method
    if known.sum() < 20:
        used = "inpaint"
    if used == "auto":
        low = _norm_blur(ctx, wgt, max(3.0, reg.bh * 0.6))
        std = _robust_std((ctx - low)[known])
        used = "inpaint" if float(std.mean()) > 9.0 else "fill"

    if used == "fill":
        low = _norm_blur(ctx, wgt, max(3.0, reg.bh * 0.6))
        std = _robust_std((ctx - low)[known])             # paper grain per channel
        fill = _norm_blur(ctx, wgt, max(2.0, reg.bh * 0.30))
        rng = np.random.default_rng(seed)
        noise = cv2.GaussianBlur(rng.standard_normal(ctx.shape).astype(np.float32), (0, 0), 0.6)
        noise /= max(float(noise.std()), 1e-6)
        fill = np.clip(fill + noise * std, 0, 255)
    else:
        fill = cv2.inpaint(ctx.astype(np.uint8), reg.mask.astype(np.uint8) * 255, 3,
                           cv2.INPAINT_TELEA).astype(np.float32)

    m = reg.mask.astype(np.float32)
    grow = cv2.dilate(reg.mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    a = np.maximum(m, cv2.GaussianBlur(m, (0, 0), 0.7) * grow)
    a[reg.protect] = 0
    base = out[cy0:cy1, cx0:cx1].astype(np.float32)
    out[cy0:cy1, cx0:cx1] = np.clip(base * (1 - a[..., None]) + fill * a[..., None], 0, 255).astype(np.uint8)
    return used


# =============================================================================
# Replace: font matching + rendering
# =============================================================================

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts",
             "/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
             "~/Library/Fonts", "C:/Windows/Fonts"]
DATA_FONT_DIR = Path(__file__).resolve().parents[1] / "data" / "fonts"
PREFERRED = ("arial", "helvet", "liberation", "dejavu", "times", "georgia", "calibri", "carlito",
             "cambria", "caladea", "verdana", "tahoma", "trebuc", "cour", "roboto", "opensans",
             "notosans", "notoserif", "segoe", "garamond", "palatino", "bookman", "century")
_FONT_CACHE: tuple[float, list[str]] | None = None


def clear_font_cache() -> None:
    global _FONT_CACHE
    _FONT_CACHE = None


def discover_fonts(user_fonts: list[str] | None, *, limit: int = 120, refresh: bool = False) -> list[str]:
    global _FONT_CACHE
    if user_fonts:
        found = []
        for f in user_fonts:
            f = os.path.expanduser(f)
            if os.path.isdir(f):
                for root, _, files in os.walk(f):
                    found += [os.path.join(root, x) for x in files if x.lower().endswith((".ttf", ".otf", ".ttc"))]
            else:
                found.append(f)
        return sorted(found)
    stamp = 0.0
    try:
        stamp = DATA_FONT_DIR.stat().st_mtime
    except OSError:
        pass
    if _FONT_CACHE is not None and not refresh and _FONT_CACHE[0] == stamp:
        return _FONT_CACHE[1][:limit]
    allf = []
    for d in [*FONT_DIRS, str(DATA_FONT_DIR)]:
        d = os.path.expanduser(d)
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                allf += [os.path.join(root, x) for x in files if x.lower().endswith((".ttf", ".otf"))]

    def rank(p):
        n = os.path.basename(p).lower()
        for i, k in enumerate(PREFERRED):
            if k in n:
                return i
        return 999

    ranked = sorted(allf, key=lambda p: (rank(p), p))
    pref = [p for p in ranked if rank(p) < 999]
    result = (pref or ranked)[:120]
    _FONT_CACHE = (stamp, result)
    return result[:limit]


def has_glyphs(font_path: str, text: str) -> bool:
    try:
        f = ImageFont.truetype(font_path, 32)
        notdef = f.getmask("\uffff")
        nd = (notdef.size, bytes(notdef))
        for ch in set(text):
            if ch.isspace():
                continue
            m = f.getmask(ch)
            if (m.size, bytes(m)) == nd:
                return False
        return True
    except Exception:
        return False


def render_alpha(text: str, font_path: str, size: float, fx: float = 0.0, fy: float = 0.0, ss: int = 4):
    """Anti-aliased coverage map. Returns (alpha[0..1], L, T): alpha[0,0] sits at
    (origin_x + L, origin_y + T) where origin = left end of the baseline."""
    font = ImageFont.truetype(font_path, max(2, int(round(size * ss))))
    l, t, r, b = font.getbbox(text, anchor="ls")
    L, T = math.floor(l / ss) - 2, math.floor(t / ss) - 2
    R, B = math.ceil(r / ss) + 3, math.ceil(b / ss) + 3
    Wc, Hc = R - L, B - T
    big = Image.new("L", (Wc * ss, Hc * ss), 0)
    ImageDraw.Draw(big).text(((-L + fx) * ss, (-T + fy) * ss), text, font=font, fill=255, anchor="ls")
    a = np.asarray(big, dtype=np.float32)
    if ss > 1:
        a = cv2.resize(a, (Wc, Hc), interpolation=cv2.INTER_AREA)
    return a / 255.0, L, T


def _extent(a: np.ndarray, thr: float = 0.5):
    ys, xs = np.nonzero(a > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _align_to_mask(alpha: np.ndarray, L: float, T: float, ix: float, iy: float,
                   obs: np.ndarray, page_x0: int, page_y0: int,
                   max_shift: float = 2.0, step: float = 0.5) -> tuple[float, float]:
    """Find a small sub-pixel translation that best overlaps the source ink.

    Font hinting can move a glyph by a pixel even when its size is correct.
    Aligning the rendered source word to the observed core mask before placing
    the replacement removes that residual drift.  The search is deliberately
    tiny (a couple of pixels); it is a calibration, not a free-form layout.
    """
    oh, ow = obs.shape
    m = int(math.ceil(max_shift)) + 4
    bh, bw = oh + 2 * m, ow + 2 * m
    canvas = np.zeros((bh, bw), np.float32)
    # Keep both the baseline-origin and scaled-L fractions in the warp.  Array
    # placement is integer, but source calibration must not truncate subpixels.
    base_ix, base_iy = math.floor(ix), math.floor(iy)
    px = base_ix + L - page_x0 + m
    py = base_iy + T - page_y0 + m
    canvas_px, canvas_py = math.floor(px), math.floor(py)
    frac_x = (ix - base_ix) + (px - canvas_px)
    frac_y = (iy - base_iy) + (py - canvas_py)
    sx0, sy0 = max(0, -canvas_px), max(0, -canvas_py)
    dx0, dy0 = max(0, canvas_px), max(0, canvas_py)
    sw = min(alpha.shape[1] - sx0, bw - dx0)
    sh = min(alpha.shape[0] - sy0, bh - dy0)
    if sw <= 0 or sh <= 0:
        return 0.0, 0.0
    canvas[dy0:dy0 + sh, dx0:dx0 + sw] = alpha[sy0:sy0 + sh, sx0:sx0 + sw]

    best_dx = best_dy = 0.0
    best = -1.0
    steps = np.arange(-max_shift, max_shift + 1e-6, step)
    for dy in steps:
        for dx in steps:
            warp_x, warp_y = dx + frac_x, dy + frac_y
            if abs(warp_x) < 1e-9 and abs(warp_y) < 1e-9:
                shifted = canvas
            else:
                M = np.float32([[1, 0, warp_x], [0, 1, warp_y]])
                shifted = cv2.warpAffine(canvas, M, (bw, bh), flags=cv2.INTER_LINEAR)
            crop = shifted[m:m + oh, m:m + ow] > 0.5
            inter = np.logical_and(crop, obs).sum()
            union = np.logical_or(crop, obs).sum()
            iou = inter / float(max(1, union))
            if iou > best:
                best = iou
                best_dx, best_dy = float(dx), float(dy)
    return best_dx, best_dy



@dataclass
class FontFit:
    path: str
    size: float
    score: float
    weight: float = 0.0
    stretch: float = 1.0


_FONT_NAMES = {
    "arial": "Arial", "arialbd": "Arial Bold", "ariali": "Arial Italic",
    "arialbi": "Arial Bold Italic", "arialn": "Arial Narrow",
    "arialnb": "Arial Narrow Bold", "arialni": "Arial Narrow Italic",
    "arialnbi": "Arial Narrow Bold Italic", "arialuni": "Arial Unicode MS",
    "times": "Times New Roman", "timesbd": "Times New Roman Bold",
    "timesi": "Times New Roman Italic", "timesbi": "Times New Roman Bold Italic",
    "cour": "Courier New", "courbd": "Courier New Bold",
    "couri": "Courier New Italic", "courbi": "Courier New Bold Italic",
    "georgia": "Georgia", "georgiab": "Georgia Bold",
    "georgiai": "Georgia Italic", "georgiaz": "Georgia Bold Italic",
    "calibri": "Calibri", "calibrib": "Calibri Bold",
    "calibrii": "Calibri Italic", "calibriz": "Calibri Bold Italic",
    "calibril": "Calibri Light",
    "cambria": "Cambria", "cambriab": "Cambria Bold",
    "cambriai": "Cambria Italic", "cambriaz": "Cambria Bold Italic",
    "verdana": "Verdana", "verdanab": "Verdana Bold",
    "verdanai": "Verdana Italic", "verdanaz": "Verdana Bold Italic",
    "tahoma": "Tahoma", "tahomabd": "Tahoma Bold",
    "trebuc": "Trebuchet MS", "trebucbd": "Trebuchet MS Bold",
    "trebucit": "Trebuchet MS Italic", "trebucbi": "Trebuchet MS Bold Italic",
    "segoeui": "Segoe UI", "segoeuib": "Segoe UI Bold",
    "segoeuii": "Segoe UI Italic", "segoeuiz": "Segoe UI Bold Italic",
    "roboto-regular": "Roboto", "roboto-bold": "Roboto Bold",
    "roboto-italic": "Roboto Italic", "roboto-bolditalic": "Roboto Bold Italic",
    "liberationsans-regular": "Liberation Sans",
    "liberationsans-bold": "Liberation Sans Bold",
    "liberationsans-italic": "Liberation Sans Italic",
    "liberationsans-bolditalic": "Liberation Sans Bold Italic",
    "liberationserif-regular": "Liberation Serif",
    "liberationserif-bold": "Liberation Serif Bold",
    "liberationserif-italic": "Liberation Serif Italic",
    "liberationserif-bolditalic": "Liberation Serif Bold Italic",
    "dejavusans": "DejaVu Sans", "dejavusans-bold": "DejaVu Sans Bold",
    "dejavusans-oblique": "DejaVu Sans Oblique",
    "dejavusans-boldoblique": "DejaVu Sans Bold Oblique",
    "dejavuserif": "DejaVu Serif", "dejavuserif-bold": "DejaVu Serif Bold",
    "notosans-regular": "Noto Sans", "notosans-bold": "Noto Sans Bold",
    "notoserif-regular": "Noto Serif", "notoserif-bold": "Noto Serif Bold",
    "carlito-regular": "Carlito", "carlito-bold": "Carlito Bold",
    "carlito-italic": "Carlito Italic", "carlito-bolditalic": "Carlito Bold Italic",
    "caladea-regular": "Caladea", "caladea-bold": "Caladea Bold",
    "caladea-italic": "Caladea Italic", "caladea-bolditalic": "Caladea Bold Italic",
}


def font_label(font_path: str) -> str:
    stem = Path(font_path).stem.lower()
    if stem in _FONT_NAMES:
        return _FONT_NAMES[stem]
    # Common GNU/Linux naming with hyphens and weights.
    clean = re.sub(r"[_-]+", " ", stem)
    clean = re.sub(r"\b(regular|book|roman)\b", "", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean.title() or Path(font_path).name


def _stroke_width(mask: np.ndarray) -> float:
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        # Alpha coverage maps are float 0..1; binary masks may be 0/255.
        m = (m > 0.5) if (m.size and float(m.max()) <= 1.0) else (m > 0)
    if not m.any():
        return 0.0
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return float(max(0.0, np.percentile(dt[m], 85) * 2.0))


def _mask_score(rendered: np.ndarray, obs: np.ndarray) -> float:
    """Shape similarity plus a natural-aspect-ratio penalty.

    Both masks are compared at the same box size (so a single-pixel OCR height
    difference does not shift every later glyph), while the font's natural
    aspect ratio must still match the source.  That keeps Calibri from being
    beaten by a visually similar but wider/narrower face.
    """
    oh, ow = obs.shape
    rh, rw = rendered.shape
    if rh < 1 or rw < 1:
        return -1.0
    res = cv2.resize(rendered.astype(np.float32), (ow, oh),
                     interpolation=cv2.INTER_AREA) > 0.5
    inter = np.logical_and(res, obs).sum()
    union = np.logical_or(res, obs).sum()
    if not union:
        return -1.0
    iou = inter / union
    aspect_penalty = math.exp(-2.0 * abs(math.log((rw / rh) / (ow / oh))))
    return float(iou * aspect_penalty)


def _size_for_height(text: str, font_path: str, target_h: float, target_w: float | None = None) -> float:
    """Solve the raster size from observed ink height, then natural line width.

    Several nearby point sizes can have the same hinted height but different
    glyph advances.  When width is available, choose the size in that plateau
    whose natural whole-line width is closest instead of immediately hiding
    the discrepancy with horizontal stretching.  Measure with the same
    supersampling used by the final renderer so their hinting decisions agree.
    """
    a, _, _ = render_alpha(text, font_path, 100, ss=1)
    e = _extent(a)
    if not e or e[3] - e[1] < 1:
        return max(4.0, target_h * 0.9)
    size = max(4.0, 100.0 * target_h / (e[3] - e[1]))
    best_size = size
    best_error = (float("inf"), float("inf"))
    for _ in range(7):
        a, _, _ = render_alpha(text, font_path, size, ss=1 if size < 15 else 4)
        e = _extent(a)
        if not e:
            break
        h, w = e[3] - e[1], e[2] - e[0]
        error = (abs(h - target_h), abs(w - target_w) if target_w is not None else 0.0)
        if error < best_error:
            best_size, best_error = size, error
        height_done = abs(h - target_h) <= 0.35
        width_done = target_w is None or abs(w - target_w) <= 0.35
        if height_done and width_done:
            best_size = size
            break
        ratio = target_w / max(0.5, w) if height_done and target_w is not None else target_h / max(0.5, h)
        size = max(4.0, size * ratio)
    return best_size


def _font_prior(font_path: str) -> float:
    """Tiny tie-breaker that prefers the canonical face over lookalikes.

    Arial Unicode MS and Arial have nearly identical Latin glyphs.  When their
    scores are within noise, the plain Arial file is the better explanation of
    a normal document.  Symbol/emoji fonts should never win on glyph overlap.
    A clean scan also makes regular text look slightly heavier, so bold faces
    need a clearly better shape score before they are selected.
    """
    stem = Path(font_path).stem.lower()
    if any(k in stem for k in ("symbol", "wingding", "webding", "emoji", "icon", "dingbat")):
        return 0.55
    prior = 1.0
    if "unicode" in stem or stem == "arialuni":
        prior *= 0.95
    if any(k in stem for k in ("narrow", "condensed")):
        prior *= 0.96
    # A blurred/low-resolution regular glyph often has the same ink thickness
    # as a clean bold glyph.  Keep the regular face unless bold is clearly
    # better at reproducing the actual letter shapes.  Some Windows faces use
    # terse suffixes (calibrib, georgiab, calibriz); the label table knows them.
    label = _FONT_NAMES.get(stem, "").lower()
    if "bold" in label or any(
        k in stem for k in ("bold", "black", "heavy", "semibold", "demi")
    ):
        prior *= 0.92
    return prior



def _weighted_crop(rendered: np.ndarray, delta: float) -> np.ndarray:
    if abs(delta) < 0.04:
        return rendered
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if delta > 0:
        changed = cv2.dilate(rendered, k)
    else:
        changed = cv2.erode(rendered, k)
    f = float(min(1.0, abs(delta)))
    return rendered * (1.0 - f) + changed * f


def fit_font(candidates: list[str], text: str, obs: np.ndarray) -> FontFit:
    """Pick the installed font that reproduces the observed glyph shapes.

    The renderer keeps the source aspect ratio: only height is normalised when
    scoring, so a narrow or wide font cannot win merely by being stretched into
    the OCR box.  A small residual stroke-weight correction is measured after
    the family/style has been chosen.
    """
    oh, ow = obs.shape
    best = FontFit(candidates[0] if candidates else "", max(4.0, oh * 0.9), 0.0)
    for path in candidates:
        try:
            a, _, _ = render_alpha(text, path, 100, ss=1)
            e = _extent(a)
            if not e or e[3] - e[1] < 1:
                continue
            size = _size_for_height(text, path, oh, ow)
            a, _, _ = render_alpha(text, path, size, ss=1 if size < 15 else 4)
            e = _extent(a)
            if not e:
                continue
            crop = a[e[1]:e[3], e[0]:e[2]]
            score = _mask_score(crop, obs) * _font_prior(path)
            if score > best.score:
                best = FontFit(path, size, score)
        except Exception:
            continue
    if not best.path:
        return best

    # Height is locked; width is matched with a pure horizontal stretch below.
    # Changing the point size here (the old behaviour) also changed glyph
    # height by a pixel, which is exactly what pixel-lock must never do.

    # Estimate the residual stroke difference.  This catches a scanned / photocopied
    # original whose ink is slightly heavier than the clean digital font.
    try:
        a, _, _ = render_alpha(text, best.path, best.size, ss=1 if best.size < 15 else 4)
        e = _extent(a)
        crop = a[e[1]:e[3], e[0]:e[2]]
        delta = (_stroke_width(obs) - _stroke_width(crop)) / 2.0
        best.weight = float(np.clip(delta, -1.4, 1.8))
    except Exception:
        best.weight = 0.0

    # If the source itself is condensed/expanded (e.g. a stamped label), mirror
    # that width without changing the vertical font size.
    try:
        a, _, _ = render_alpha(text, best.path, best.size, ss=1 if best.size < 15 else 4)
        e = _extent(a)
        rw = max(1, e[2] - e[0])
        best.stretch = float(np.clip(ow / rw, 0.88, 1.12))
    except Exception:
        best.stretch = 1.0
    return best


def rank_fonts(candidates: list[str], text: str, obs: np.ndarray, limit: int = 8) -> list[FontFit]:
    """Return the best-scoring font candidates for an observed source mask.

    This is the recognition side of :func:`fit_font`: the same shape score and
    natural-aspect penalty, but it keeps the ranked list so the UI can explain
    the match instead of silently picking one lookalike.
    """
    results: list[FontFit] = []
    oh, ow = obs.shape
    for path in candidates:
        try:
            a, _, _ = render_alpha(text, path, 100, ss=1)
            e = _extent(a)
            if not e or e[3] - e[1] < 1:
                continue
            size = _size_for_height(text, path, oh, ow)
            a, _, _ = render_alpha(text, path, size, ss=1 if size < 15 else 4)
            e = _extent(a)
            if not e:
                continue
            crop = a[e[1]:e[3], e[0]:e[2]]
            score = _mask_score(crop, obs) * _font_prior(path)
            results.append(FontFit(path, size, float(score)))
        except Exception:
            continue
    results.sort(key=lambda f: (-f.score, font_label(f.path).lower()))
    # Collapse duplicate family/style labels so the list is readable.
    seen: set[str] = set()
    unique: list[FontFit] = []
    for fit in results:
        key = font_label(fit.path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(fit)
        if len(unique) >= limit:
            break
    return unique


def estimate_skew(lines: list[list[Word]]) -> float:
    """Median tilt (degrees, image coords: + = text slopes down to the right) of the OCR lines."""
    slopes = []
    for ln in lines:
        if len(ln) >= 3:
            xs = np.array([w.x + w.w / 2 for w in ln])
            ys = np.array([w.y + w.h / 2 for w in ln])
            if xs.max() - xs.min() >= 250:
                slopes.append(np.polyfit(xs, ys, 1)[0])
    if len(slopes) < 2:
        return 0.0
    deg = math.degrees(math.atan(float(np.median(slopes))))
    return deg if 0.3 <= abs(deg) <= 8 else 0.0


@dataclass
class StyleProfile:
    """Every visual attribute the replacement inherits from the source ink.

    Resolving size, colour, weight, width, softness and case in one object
    keeps the renderer and the report in sync: what is measured is exactly
    what gets drawn.
    """
    path: str
    font: str
    size: float
    score: float = 0.0
    weight: float = 0.0          # raw stroke correction measured on the source
    stretch: float = 1.0
    color: tuple = (0.0, 0.0, 0.0)   # BGR, 0..255
    soften: float = 0.45
    case: str = "as-typed"


def _case_kind(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    if words and all(w[0].isupper() and w[1:].islower() for w in words):
        return "title"
    return "mixed"


def match_source_case(source: str, replacement: str) -> str:
    """Give ``replacement`` the same letter case as the text it replaces.

    Digits and punctuation are untouched.  Deliberately mixed spellings such
    as ``iPhone`` or ``McDonald`` are left exactly as typed, because forcing
    a pattern there would damage a name rather than copy a style.
    """
    kind = _case_kind(source)
    if not replacement or kind in ("none", "mixed"):
        return replacement
    if kind == "upper":
        return replacement.upper()
    if kind == "lower":
        return replacement.lower()
    # Title Case: capitalise every word and lower-case the rest of that word.
    return " ".join(
        p[:1].upper() + p[1:].lower() if p else p
        for p in replacement.split(" ")
    )


def resolve_style(reg: Region, obs: np.ndarray, old_text: str, usable: list[str], *,
                  font_override=None, color=None, size_scale=1.0, soften=None,
                  weight_match=True, stretch_match=True) -> StyleProfile:
    """Resolve font, size, weight, width, colour and softness from the source."""
    if font_override:
        fit = FontFit(font_override, 0.0, 1.0)
        fit.size = _size_for_height(old_text, font_override, obs.shape[0], obs.shape[1])
        try:
            a, _, _ = render_alpha(old_text, font_override, fit.size, ss=1 if fit.size < 15 else 4)
            e = _extent(a)
            crop = a[e[1]:e[3], e[0]:e[2]]
            fit.weight = float(np.clip((_stroke_width(obs) - _stroke_width(crop)) / 2.0, -1.4, 1.8))
            fit.stretch = float(np.clip(obs.shape[1] / max(1, crop.shape[1]), 0.88, 1.12))
        except Exception:
            pass
    else:
        fit = fit_font(usable, old_text, obs)

    path = fit.path or (usable[0] if usable else "")
    size = fit.size or max(4.0, obs.shape[0] * 0.9)
    if size_scale and float(size_scale) != 1.0:
        size = max(4.0, size * float(size_scale))

    ink = color if color is not None else reg.color
    col = tuple(int(round(float(v))) for v in ink)
    soft = float(reg.soften if soften is None else soften)
    return StyleProfile(
        path=path,
        font=font_label(path),
        size=size,
        score=float(fit.score),
        weight=float(fit.weight) if weight_match else 0.0,
        stretch=float(fit.stretch) if stretch_match else 1.0,
        color=col,
        soften=soft,
        case=_case_kind(old_text),
    )


def _prepare_line_alpha(alpha: np.ndarray, sx: float, weight: float, soften: float) -> np.ndarray:
    """Apply line-wide source style without changing individual glyph geometry."""
    a = alpha
    if abs(sx - 1.0) > 0.002:
        a = cv2.resize(a, (max(1, int(round(a.shape[1] * sx))), a.shape[0]),
                       interpolation=cv2.INTER_AREA)
    if abs(weight) >= 0.3:
        a = _weighted_crop(a, float(weight) * 0.55)
    if soften > 0:
        a = cv2.GaussianBlur(a, (0, 0), soften)
    return a


def _source_reconstruction(alpha: np.ndarray, L: float, T: int,
                           origin_x: float, baseline_y: float,
                           obs: np.ndarray, page_x0: int, page_y0: int):
    """Return the calibrated source ink extent and IoU in the level OCR box."""
    oh, ow = obs.shape
    ix, iy = math.floor(origin_x), math.floor(baseline_y)
    origin_fx, origin_fy = origin_x - ix, baseline_y - iy
    margin = 8
    bh, bw = oh + 2 * margin, ow + 2 * margin
    canvas = np.zeros((bh, bw), np.float32)
    px = ix + L - page_x0 + margin
    py = iy + T - page_y0 + margin
    canvas_px, canvas_py = math.floor(px), math.floor(py)
    dx = origin_fx + px - canvas_px
    dy = origin_fy + py - canvas_py
    sx0, sy0 = max(0, -canvas_px), max(0, -canvas_py)
    dx0, dy0 = max(0, canvas_px), max(0, canvas_py)
    sw = min(alpha.shape[1] - sx0, bw - dx0)
    sh = min(alpha.shape[0] - sy0, bh - dy0)
    if sw <= 0 or sh <= 0:
        return None, None
    canvas[dy0:dy0 + sh, dx0:dx0 + sw] = alpha[sy0:sy0 + sh, sx0:sx0 + sw]
    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, M, (bw, bh), flags=cv2.INTER_LINEAR)
    reconstructed = canvas[margin:margin + oh, margin:margin + ow] > 0.5
    union = np.logical_or(reconstructed, obs).sum()
    iou = float(np.logical_and(reconstructed, obs).sum() / max(1, union))
    e = _extent(reconstructed.astype(np.float32))
    bbox = None if e is None else [int(page_x0 + e[0]), int(page_y0 + e[1]),
                                   int(page_x0 + e[2]), int(page_y0 + e[3])]
    return bbox, iou


def draw_replacement(out, reg: Region, old_text: str, new_text: str, fonts: list[str], align="left",
                     shrink=False, min_scale=0.6, soften=None, next_x=None, font_override=None, skew=0.0,
                     color=None, size_scale=1.0, preserve_size=True, weight_match=True,
                     stretch_match=True, style_info=None, nudge_x=0.0, nudge_y=0.0, case_match=True,
                     pixel_lock=True):
    cx0, cy0, _, _ = reg.ctx
    core = reg.core
    R = C = None
    if abs(skew) >= 0.3:
        # Measure one level source line; rotate only the finished replacement.
        th = math.radians(skew)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        ys_, xs_ = np.nonzero(core)
        cc = (float(xs_.mean()), float(ys_.mean()))
        Mi = cv2.getRotationMatrix2D(cc, skew, 1.0)
        core = cv2.warpAffine(core.astype(np.uint8), Mi, (core.shape[1], core.shape[0]),
                              flags=cv2.INTER_NEAREST) > 0
        C = np.array([cx0 + cc[0], cy0 + cc[1]])
        ys_, xs_ = np.nonzero(core)
        x0, y0 = cx0 + xs_.min(), cy0 + ys_.min()
        x1, y1 = cx0 + xs_.max() + 1, cy0 + ys_.max() + 1
    else:
        x0, y0, x1, y1 = reg.core_bbox
    obs = core[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]

    applied = match_source_case(old_text, new_text) if case_match else new_text
    usable = [f for f in fonts if has_glyphs(f, old_text + applied)]
    if not usable:
        raise RuntimeError("no available font has all the needed characters; pass --font <file.ttf>")
    profile = resolve_style(
        reg, obs, old_text, usable,
        font_override=font_override,
        color=color,
        size_scale=size_scale,
        soften=soften,
        weight_match=weight_match,
        stretch_match=stretch_match,
    )
    path = profile.path or usable[0]
    size = profile.size or max(4.0, reg.bh * 0.9)
    ss = 1 if size < 15 else 4

    # Calibrate the complete source line once.  The same font rasterisation,
    # line-wide horizontal scale, stroke correction and softness are used for
    # both source reconstruction and replacement.  No per-glyph resizing or
    # slot snapping occurs, and vertical glyph proportions remain natural.
    style_xs = float(profile.stretch) if stretch_match else 1.0
    style_xs = float(np.clip(style_xs, 0.88, 1.12) if pixel_lock else np.clip(style_xs, 0.94, 1.06))
    if abs(style_xs - 1.0) <= 0.002:
        style_xs = 1.0
    applied_weight = float(profile.weight) if weight_match else 0.0
    # Hinted fonts have several raster sizes on the same height/width plateau.
    # A tiny bounded calibration chooses the one that actually reconstructs the
    # source ink; it never scales individual glyphs or changes the user's
    # explicit size_scale override.
    size_offsets = (-0.5, -0.25, 0.0, 0.25, 0.5) if pixel_lock and abs(float(size_scale) - 1.0) < 1e-6 else (0.0,)
    calibrated = None
    for offset in size_offsets:
        candidate_size = max(4.0, size + float(offset))
        candidate_ss = 1 if candidate_size < 15 else 4
        old_a0, old_L, old_T = render_alpha(old_text, path, candidate_size, ss=candidate_ss)
        old_e0 = _extent(old_a0)
        if old_e0 is None:
            continue
        old_a = _prepare_line_alpha(old_a0, style_xs, applied_weight, profile.soften)
        source_ox = x0 - (old_L + old_e0[0]) * style_xs
        source_baseline = y0 - (old_T + old_e0[1])
        cal_dx, cal_dy = _align_to_mask(
            old_a, old_L * style_xs, old_T,
            source_ox, source_baseline, obs, x0, y0,
        )
        source_bbox, source_iou = _source_reconstruction(
            old_a, old_L * style_xs, old_T,
            source_ox + cal_dx, source_baseline + cal_dy, obs, x0, y0,
        )
        if source_iou is None:
            continue
        score = source_iou - abs(float(offset)) * 1e-4
        if calibrated is None or score > calibrated[0]:
            calibrated = (score, candidate_size, candidate_ss, source_ox, source_baseline,
                          cal_dx, cal_dy, source_bbox, source_iou)
    if calibrated is None:
        raise RuntimeError("source text produced no visible glyphs; choose a font with the required characters")
    _, size, ss, source_ox, source_baseline, cal_dx, cal_dy, source_bbox, source_iou = calibrated
    shift_x = cal_dx + float(nudge_x or 0.0)
    shift_y = cal_dy + float(nudge_y or 0.0)

    render_size = size
    xs = style_xs

    def measure(sz):
        alpha, L_, T_ = render_alpha(applied, path, sz, ss=1 if sz < 15 else ss)
        return L_, T_, _extent(alpha)

    new_L, new_T, new_e0 = measure(render_size)
    new_w = 0.0 if new_e0 is None else (new_e0[2] - new_e0[0]) * xs
    old_w = x1 - x0
    note = "whole-line calibrated" if pixel_lock else "whole-line"
    if new_e0 is not None and shrink and not preserve_size and new_w > old_w:
        gap = render_size * 0.3
        if align == "left":
            avail = old_w if next_x is None else max(old_w, next_x - x0 - gap)
            if next_x is None:
                avail = max(old_w, reg.ctx[2] - x0 + reg.bh * 4)
        else:
            avail = old_w
        if new_w > avail:
            need = avail / new_w
            fit_xs = max(need, 0.88)
            us = max(need / fit_xs, min_scale)
            render_size *= us
            xs = style_xs * fit_xs
            new_L, new_T, new_e0 = measure(render_size)
            new_w = 0.0 if new_e0 is None else (new_e0[2] - new_e0[0]) * xs
            note = f"fit {xs:.0%}" + (f", size {us:.0%}" if us < 1 else "")
            if need / fit_xs < min_scale:
                note += " (gap se lamba)"
    elif new_e0 is not None and preserve_size and new_w > old_w * 1.01:
        note = f"source size ({new_w / max(1.0, old_w):.0%} width)"

    # Recover a baseline-relative origin from the source L/T metrics.  Alignment
    # affects the whole line and is bounded by _align_to_mask; tracking remains
    # zero so font kerning and advances are preserved exactly.
    if new_e0 is not None:
        new_rel_l = (new_L + new_e0[0]) * xs
        new_rel_r = (new_L + new_e0[2]) * xs
        if align == "center":
            ox = (x0 + x1) / 2.0 + shift_x - (new_rel_l + new_rel_r) / 2.0
        elif align == "right":
            ox = x1 + shift_x - new_rel_r
        else:
            ox = x0 + shift_x - new_rel_l
    else:
        ox = source_ox + shift_x
    oy = source_baseline + shift_y
    if reg.bh <= 20:
        ox = float(round(ox))
        oy = float(round(oy))
    # PIL positions glyphs on a sub-pixel vertical origin.  A fractional
    # baseline (especially .5) adds a partially covered row at the baseline,
    # so the same font at the same size can measure one pixel taller than the
    # calibrated source and look "slightly bigger".  Pixel-lock snaps the
    # baseline to the nearest row (tie -> the source row) and renders with
    # fy=0, keeping ink extents identical; the positional error is <= 0.5 px,
    # which is below visible and OCR precision.
    if pixel_lock:
        oy = float(math.floor(oy + 0.5 - 1e-6))
    ix, iy = math.floor(ox), math.floor(oy)
    final_ss = 1 if render_size < 15 else ss
    fy = 0.0 if pixel_lock else (oy - iy)
    a, L, T = render_alpha(
        applied, path, render_size,
        fx=(ox - ix) if abs(xs - 1.0) <= 0.002 else 0.0,
        fy=fy,
        ss=final_ss,
    )
    a = _prepare_line_alpha(a, xs, applied_weight, profile.soften)
    actual_xs = xs if abs(xs - 1.0) > 0.002 else 1.0
    if abs(xs - 1.0) > 0.002:
        px = int(round(ox + L * xs))
    else:
        px = ix + L
    py = iy + T
    derotated_e = _extent(a)
    derotated_bbox = None if derotated_e is None else [
        px + derotated_e[0], py + derotated_e[1],
        px + derotated_e[2], py + derotated_e[3],
    ]

    if R is not None:
        # Every replacement, including pixel-lock, uses this common rotation.
        ah, aw = a.shape
        t = R @ np.array([px - C[0], py - C[1]]) + C
        corners = np.array([[0, 0], [aw, 0], [0, ah], [aw, ah]], float) @ R.T + t
        mn = np.floor(corners.min(0)).astype(int) - 1
        mx = np.ceil(corners.max(0)).astype(int) + 1
        M = np.hstack([R, (t - mn).reshape(2, 1)]).astype(np.float32)
        a = cv2.warpAffine(a, M, (int(mx[0] - mn[0]), int(mx[1] - mn[1])), flags=cv2.INTER_LINEAR)
        px, py = int(mn[0]), int(mn[1])

    H, W = out.shape[:2]
    sx0, sy0 = max(0, px), max(0, py)
    sx1, sy1 = min(W, px + a.shape[1]), min(H, py + a.shape[0])
    visible = a[sy0 - py:sy1 - py, sx0 - px:sx1 - px] if sx1 > sx0 and sy1 > sy0 else a[:0, :0]
    visible_e = _extent(visible)
    replacement_bbox = None if visible_e is None else [
        sx0 + visible_e[0], sy0 + visible_e[1],
        sx0 + visible_e[2], sy0 + visible_e[3],
    ]
    pixel_info = {
        "method": "whole-line-calibrated" if pixel_lock else "whole-line",
        "source_ink_bbox": [int(v) for v in reg.ink_bbox],
        "source_core_bbox": [int(v) for v in reg.core_bbox],
        "source_reconstruction_bbox": source_bbox,
        "source_reconstruction_frame": "de-rotated" if R is not None else "page",
        "source_reconstruction_iou": None if source_iou is None else round(float(source_iou), 4),
        "replacement_ink_bbox_derotated": derotated_bbox,
        "replacement_ink_bbox": replacement_bbox,
        "baseline_y_derotated": round(float(oy), 3),
        "baseline_offset": round(float(oy - y1), 3),
        "calibration_shift": [round(float(cal_dx), 3), round(float(cal_dy), 3)],
        "nudge": [round(float(nudge_x or 0.0), 3), round(float(nudge_y or 0.0), 3)],
        "source_scale_x": round(float(style_xs), 4),
        "replacement_scale_x": round(float(actual_xs), 4),
        "replacement_scale_y": 1.0,
        "tracking": 0.0,
        "rotation": round(float(skew if R is not None else 0.0), 3),
    }
    if style_info is not None:
        b, g, r = [int(round(v)) for v in profile.color]
        style_info.update({
            "font": font_label(path),
            "file": os.path.basename(path),
            "size": round(float(render_size), 2),
            "score": round(float(profile.score), 3),
            "weight": round(float(applied_weight * 0.55 if abs(applied_weight) >= 0.3 else 0.0), 2),
            "stretch": round(float(actual_xs), 3),
            "color": "#{:02x}{:02x}{:02x}".format(r, g, b),
            "soften": round(float(profile.soften), 2),
            "shift": [round(float(shift_x), 2), round(float(shift_y), 2)],
            "case": profile.case,
            "applied": applied,
            "pixel": pixel_info,
        })
        if source_iou is not None:
            note += f" · source IoU {source_iou:.3f}"

    if sx1 <= sx0 or sy1 <= sy0:
        return path, note
    al = np.clip(visible, 0, 1)[..., None]
    reg_px = out[sy0:sy1, sx0:sx1].astype(np.float32)
    col = np.array(profile.color, np.float32)
    out[sy0:sy1, sx0:sx1] = np.clip(reg_px * (1 - al) + col * al, 0, 255).astype(np.uint8)

    # If the watermark sits ON TOP of the text, the new glyphs need the same
    # veil as the old ones.  All renderer modes reach this common compositor.
    u = reg.underlay
    if u is not None and u.present and u.order == "over" and u.alpha is not None:
        aa = u.alpha[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0]
        if aa.shape == al.shape[:2] and aa.size:
            veil = np.clip(aa, 0, 1) * np.clip(al[..., 0] * 1.6, 0, 1) * 0.92
            wcol = np.array(u.color, np.float32)
            seg = out[sy0:sy1, sx0:sx1].astype(np.float32)
            out[sy0:sy1, sx0:sx1] = np.clip(
                seg * (1 - veil[..., None]) + wcol * veil[..., None], 0, 255
            ).astype(np.uint8)
    return path, note


def insert_text(page: Page, text: str, bbox, *, fonts=None, font_override=None,
                color=None, size_scale=1.0, align="left", style_info=None) -> np.ndarray:
    """Draw brand-new text inside ``bbox`` without analysing any source ink.

    Unlike :func:`edit_page` there is no original glyph to erase or match, so
    the box is treated as a text frame: the glyphs are fitted to its width and
    height, then aligned inside it.  The returned array is a new page image;
    the caller's page object is left untouched.
    """
    text = str(text)
    if not text.strip():
        return page.bgr.copy()
    x0, y0, x1, y1 = (float(v) for v in bbox)
    box_w, box_h = max(1.0, x1 - x0), max(1.0, y1 - y0)

    candidates = list(fonts if fonts is not None else discover_fonts(None))
    usable = [f for f in candidates if has_glyphs(f, text)]
    if font_override and has_glyphs(font_override, text):
        path = font_override
    elif usable:
        path = usable[0]
    else:
        raise RuntimeError("Koi usable font nahi mila jo is text ko render kar sake.")

    color = tuple(color) if color is not None else (0, 0, 0)
    scale = max(0.2, min(5.0, float(size_scale or 1.0)))
    size = max(4.0, box_h * 0.78 * scale)

    # Fit the full string inside the frame, never stretching individual glyphs.
    alpha = ext = None
    for _ in range(5):
        ss = 1 if size < 15 else 4
        alpha, _, _ = render_alpha(text, path, size, ss=ss)
        ext = _extent(alpha)
        if ext is None:
            raise RuntimeError("Is text ke liye glyphs render nahi hue.")
        ink_w = ext[2] - ext[0]
        ink_h = ext[3] - ext[1]
        fit = min(1.0, box_w / max(1.0, ink_w), (box_h * 0.94) / max(1.0, ink_h))
        if fit >= 0.995:
            break
        size = max(4.0, size * fit)

    ss = 1 if size < 15 else 4
    alpha, _, _ = render_alpha(text, path, size, ss=ss)
    ext = _extent(alpha)
    if ext is None:
        raise RuntimeError("Is text ke liye glyphs render nahi hue.")
    ink_w = ext[2] - ext[0]
    ink_h = ext[3] - ext[1]
    if align == "center":
        left = x0 + (box_w - ink_w) / 2.0
    elif align == "right":
        left = x1 - ink_w
    else:
        left = x0
    top = y0 + (box_h - ink_h) / 2.0
    px = int(round(left - ext[0]))
    py = int(round(top - ext[1]))

    out = page.bgr.copy()
    H, W = out.shape[:2]
    sx0, sy0 = max(0, px), max(0, py)
    sx1, sy1 = min(W, px + alpha.shape[1]), min(H, py + alpha.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        a = np.clip(alpha[sy0 - py:sy1 - py, sx0 - px:sx1 - px], 0, 1)[..., None]
        region = out[sy0:sy1, sx0:sx1].astype(np.float32)
        col = np.array(color, np.float32)
        out[sy0:sy1, sx0:sx1] = np.clip(region * (1 - a) + col * a, 0, 255).astype(np.uint8)

    if style_info is not None:
        b, g, r = [int(round(v)) for v in color]
        style_info.update({
            "font": font_label(path),
            "file": os.path.basename(path),
            "size": round(float(size), 2),
            "color": "#{:02x}{:02x}{:02x}".format(r, g, b),
            "inserted": True,
        })
    return out


# =============================================================================
# High-level edit
# =============================================================================


def edit_page(page: Page, matches: list[Match], new_text: str | None, *, method="auto", align="left",
              shrink=False, soften=None, fonts=None, font_override=None, skew=0.0, log=print,
              color=None, size_scale=1.0, preserve_size=True, weight_match=True,
              stretch_match=True, style_info=None, nudge_x=0.0, nudge_y=0.0,
              case_match=True, underlay_mask=None, watermark_mode="auto",
              watermark_sensitivity=0.5, pixel_lock=True, source_bgr=None) -> np.ndarray:
    src = page.bgr if source_bgr is None else source_bgr
    out = page.bgr.copy()
    fonts = fonts if fonts is not None else (discover_fonts(None) if new_text else [])
    if underlay_mask is None and watermark_mode != "off":
        try:
            underlay_mask = detect_underlay_page(src, watermark_sensitivity)
        except Exception:
            underlay_mask = None
    for m in matches:
        reg = analyze_region(
            src, m.box, underlay_mask=underlay_mask,
            watermark_mode=watermark_mode,
            watermark_sensitivity=watermark_sensitivity,
        )
        if reg is None:
            log(f"  ! skipped '{m.text}': no ink found at that spot (OCR false positive?)")
            continue
        if style_info is not None:
            style_info["watermark"] = _underlay_report(reg.underlay, watermark_mode)
        used = erase(out, src, reg, method)
        msg = f"  - '{m.text}' erased ({used})"
        if reg.underlay is not None and reg.underlay.present:
            msg += (f" [watermark {reg.underlay.order} "
                    f"{int(round(reg.underlay.confidence * 100))}%]")
        if new_text:
            fpath, note = draw_replacement(
                out, reg, m.text, new_text, fonts, align, shrink,
                soften=soften, next_x=m.next_x, font_override=font_override, skew=skew,
                color=color, size_scale=size_scale, preserve_size=preserve_size,
                weight_match=weight_match, stretch_match=stretch_match,
                style_info=style_info, nudge_x=nudge_x, nudge_y=nudge_y,
                case_match=case_match, pixel_lock=pixel_lock,
            )
            msg += f" -> '{new_text}' in {font_label(fpath)} ({os.path.basename(fpath)}) {note}"
        log(msg)
    return out


def preview(page: Page, matches: list[tuple[int, Match]]) -> np.ndarray:
    img = page.bgr.copy()
    for idx, m in matches:
        x0, y0, x1, y1 = [int(v) for v in m.box]
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 0, 255), 2)
        cv2.putText(img, str(idx), (x0 - 3, max(14, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


# =============================================================================
# CLI
# =============================================================================


def _ask(prompt: str, default: str | None = None) -> str:
    s = input(prompt).strip()
    return s if s or default is None else default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find text in a document image/PDF and remove or replace it.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--find", help="text to look for")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--remove", action="store_true", help="erase the text, keep the background")
    g.add_argument("--replace", metavar="NEW", help="replace the text with NEW")
    ap.add_argument("--all", action="store_true", help="apply to every match without asking")
    ap.add_argument("--only", help="comma-separated match numbers to apply, e.g. 1,3")
    ap.add_argument("--lang", default="eng", help="tesseract language(s), e.g. eng+hin")
    ap.add_argument("--psm", type=int, default=3, help="tesseract page segmentation mode (try 6 or 11)")
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--whole-word", action="store_true")
    ap.add_argument("--fuzzy", type=float, default=0.8, help="OCR-error tolerance when no exact hit (0 = off)")
    ap.add_argument("--method", choices=["auto", "fill", "inpaint"], default="auto")
    ap.add_argument("--font", action="append", help="font file/dir to use for replacement (repeatable)")
    ap.add_argument("--align", choices=["left", "center", "right"], default="left")
    ap.add_argument("--no-shrink", action="store_true", help="never shrink longer replacement text")
    ap.add_argument("--soften", type=float, default=0.5, help="blur on new text edges (match scan softness)")
    ap.add_argument("--dpi", type=int, default=300, help="rasterising DPI for PDFs")
    ap.add_argument("--enhance", action="store_true", help="always pre-process the page for OCR")
    ap.add_argument("--dump-ocr", action="store_true", help="print what OCR reads, then exit")
    a = ap.parse_args(argv)

    pages = load_pages(a.input, a.dpi)
    print(f"Loaded {len(pages)} page(s). Running OCR ...")
    all_lines = [ocr_lines(p.bgr, a.lang, a.psm, enhance=a.enhance) for p in pages]

    if a.dump_ocr:
        for i, lines in enumerate(all_lines):
            print(f"--- page {i + 1}")
            for ln in lines:
                print(" ".join(w.text for w in ln), f"  (conf {np.mean([w.conf for w in ln]):.0f})")
        return 0

    query = a.find or _ask("Text to find: ")
    matches = find_matches(all_lines, query, a.case_sensitive, a.whole_word, a.fuzzy)
    if not matches:
        print("No match - retrying OCR with image enhancement (textured/uneven backgrounds) ...")
        all_lines = [ocr_lines(p.bgr, a.lang, a.psm, enhance=True) for p in pages]
        matches = find_matches(all_lines, query, a.case_sensitive, a.whole_word, a.fuzzy)
    if not matches:
        print("No match. Run with --dump-ocr to see what the OCR reads; try --psm 6/11 or another --lang.")
        return 1

    print(f"\n{len(matches)} match(es):")
    for i, m in enumerate(matches, 1):
        tag = "" if m.score >= 1 else f"  (fuzzy {m.score:.0%})"
        print(f"  [{i}] page {m.page + 1}: ...{m.context}...{tag}")
    stem = Path(a.input).with_suffix("")
    for pno, p in enumerate(pages):
        pm = [(i, m) for i, m in enumerate(matches, 1) if m.page == pno]
        if pm:
            pv = f"{stem}_preview{'' if len(pages) == 1 else f'_p{pno + 1}'}.png"
            cv2.imwrite(pv, preview(p, pm))
            print(f"  preview with numbered boxes -> {pv}")

    new_text = a.replace
    if not a.remove and a.replace is None:
        act = _ask("\nAction  [r] remove   [p] replace : ").lower()
        if act.startswith("p"):
            new_text = _ask("Replacement text: ")
        else:
            new_text = None
    chosen = list(range(1, len(matches) + 1))
    if a.only:
        chosen = [int(x) for x in a.only.split(",") if x.strip().isdigit()]
    elif len(matches) > 1 and not a.all and sys.stdin.isatty():
        s = _ask("Apply to which matches? [Enter = all, or e.g. 1,3] : ")
        if s:
            chosen = [int(x) for x in s.split(",") if x.strip().isdigit()]

    print("\nEditing ...")
    fonts = discover_fonts(a.font) if new_text else None
    results = []
    for pno, p in enumerate(pages):
        pm = [matches[i - 1] for i in chosen if 1 <= i <= len(matches) and matches[i - 1].page == pno]
        if not pm:
            results.append(p)
            continue
        skew = estimate_skew(all_lines[pno]) if new_text else 0.0
        if skew:
            print(f"  page {pno + 1}: detected text tilt {skew:+.2f} deg - new text will follow it")
        new_bgr = edit_page(p, pm, new_text, skew=skew, method=a.method, align=a.align, shrink=not a.no_shrink,
                            soften=a.soften, fonts=fonts, font_override=(a.font[0] if a.font and len(a.font) == 1 and os.path.isfile(os.path.expanduser(a.font[0])) else None))
        changed = np.any(new_bgr != p.bgr, axis=2).mean() * 100
        print(f"  page {pno + 1}: {changed:.3f}% of pixels changed, the rest is bit-identical")
        results.append(Page(new_bgr, p.alpha, p.meta))

    out = a.output or f"{stem}_edited{Path(a.input).suffix}"
    save_pages(results, out)
    print(f"\nSaved -> {out}")
    if Path(out).suffix.lower() in (".jpg", ".jpeg"):
        print("Note: JPEG output is re-encoded (q97, 4:4:4) so every pixel shifts very slightly. "
              "Use a .png output for a truly lossless result.")
    return 0
