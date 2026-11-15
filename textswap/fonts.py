"""Font discovery, catalogue, on-demand family download and recognition.

The renderer needs real font files on disk.  This module keeps three sources in
one place:

* fonts already installed on the machine (Windows/macOS/Linux);
* fonts downloaded by the user from Google Fonts into ``data/fonts``;
* embedded PDF fonts extracted by :mod:`textswap.native_pdf`.

Nothing is downloaded implicitly.  The browser asks for a family, the server
fetches it once, and the renderer picks it up through ``discover_fonts``.
"""
from __future__ import annotations

import io
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import engine

try:  # fontTools is optional at import time; the app degrades to filename data.
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - environment dependent
    TTFont = None

ROOT = Path(__file__).resolve().parents[1]
FONT_ROOT = Path(os.environ.get("TEXTSWAP_DATA") or (ROOT / "data")) / "fonts"
GOOGLE_ROOT = FONT_ROOT / "google"
EMBEDDED_ROOT = FONT_ROOT / "embedded"
for _p in (FONT_ROOT, GOOGLE_ROOT, EMBEDDED_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

_GOOGLE_CSS = "https://fonts.googleapis.com/css2"
_UA_MODERN = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_UA_LEGACY = "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"
_ALLOWED_WEIGHTS = {100, 200, 300, 400, 500, 600, 700, 800, 900}


@dataclass
class FontRecord:
    path: str
    family: str
    style: str = "Regular"
    weight: int = 400
    italic: bool = False
    source: str = "system"
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = engine.font_label(self.path)

    @property
    def key(self) -> str:
        return f"{self.family.strip().lower()}|{self.weight}|{int(self.italic)}"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "family": self.family,
            "style": self.style,
            "weight": self.weight,
            "italic": self.italic,
            "source": self.source,
            "label": self.label,
        }


# Curated families.  ``google`` is the family name used by the Google Fonts
# API; ``aliases`` lets recognition fall back to a metric-compatible clone
# when the original commercial font is not installed.
CATALOG: list[dict] = [
    {"family": "Arial", "google": "Arimo", "aliases": ["Arial", "Helvetica", "Liberation Sans"], "kind": "sans", "note": "Metric-compatible clone (Arimo)."},
    {"family": "Times New Roman", "google": "Tinos", "aliases": ["Times New Roman", "Times", "Liberation Serif"], "kind": "serif", "note": "Metric-compatible clone (Tinos)."},
    {"family": "Calibri", "google": "Carlito", "aliases": ["Calibri", "Carlito"], "kind": "sans", "note": "Metric-compatible clone (Carlito)."},
    {"family": "Cambria", "google": "Caladea", "aliases": ["Cambria", "Caladea"], "kind": "serif", "note": "Metric-compatible clone (Caladea)."},
    {"family": "Courier New", "google": "Cousine", "aliases": ["Courier New", "Courier", "Liberation Mono"], "kind": "mono", "note": "Metric-compatible clone (Cousine)."},
    {"family": "Georgia", "google": "Gelasio", "aliases": ["Georgia", "Gelasio"], "kind": "serif", "note": "Metric-compatible clone (Gelasio)."},
    {"family": "Verdana", "google": "Noto Sans", "aliases": ["Verdana", "DejaVu Sans"], "kind": "sans", "note": "Approximate substitute."},
    {"family": "Tahoma", "google": "Noto Sans", "aliases": ["Tahoma", "DejaVu Sans"], "kind": "sans", "note": "Approximate substitute."},
    {"family": "Trebuchet MS", "google": "Nunito Sans", "aliases": ["Trebuchet MS", "Trebuchet"], "kind": "sans", "note": "Approximate substitute."},
    {"family": "Segoe UI", "google": "Open Sans", "aliases": ["Segoe UI", "Selawik"], "kind": "sans", "note": "Approximate substitute."},
    {"family": "Roboto", "google": "Roboto", "aliases": ["Roboto"], "kind": "sans"},
    {"family": "Open Sans", "google": "Open Sans", "aliases": ["Open Sans", "OpenSans"], "kind": "sans"},
    {"family": "Lato", "google": "Lato", "aliases": ["Lato"], "kind": "sans"},
    {"family": "Montserrat", "google": "Montserrat", "aliases": ["Montserrat"], "kind": "sans"},
    {"family": "Poppins", "google": "Poppins", "aliases": ["Poppins"], "kind": "sans"},
    {"family": "Inter", "google": "Inter", "aliases": ["Inter"], "kind": "sans"},
    {"family": "Source Sans 3", "google": "Source Sans 3", "aliases": ["Source Sans Pro", "Source Sans 3"], "kind": "sans"},
    {"family": "Noto Sans", "google": "Noto Sans", "aliases": ["Noto Sans"], "kind": "sans"},
    {"family": "Noto Serif", "google": "Noto Serif", "aliases": ["Noto Serif"], "kind": "serif"},
    {"family": "Merriweather", "google": "Merriweather", "aliases": ["Merriweather"], "kind": "serif"},
    {"family": "Playfair Display", "google": "Playfair Display", "aliases": ["Playfair Display"], "kind": "serif"},
    {"family": "Libre Baskerville", "google": "Libre Baskerville", "aliases": ["Libre Baskerville"], "kind": "serif"},
    {"family": "IBM Plex Sans", "google": "IBM Plex Sans", "aliases": ["IBM Plex Sans"], "kind": "sans"},
    {"family": "IBM Plex Serif", "google": "IBM Plex Serif", "aliases": ["IBM Plex Serif"], "kind": "serif"},
    {"family": "Work Sans", "google": "Work Sans", "aliases": ["Work Sans"], "kind": "sans"},
    {"family": "DM Sans", "google": "DM Sans", "aliases": ["DM Sans"], "kind": "sans"},
    {"family": "Manrope", "google": "Manrope", "aliases": ["Manrope"], "kind": "sans"},
    {"family": "Rubik", "google": "Rubik", "aliases": ["Rubik"], "kind": "sans"},
    {"family": "Mukta", "google": "Mukta", "aliases": ["Mukta"], "kind": "sans"},
    {"family": "Hind", "google": "Hind", "aliases": ["Hind"], "kind": "sans"},
    {"family": "Noto Sans Devanagari", "google": "Noto Sans Devanagari", "aliases": ["Noto Sans Devanagari", "Mangal", "Nirmala UI"], "kind": "sans", "note": "Devanagari / Hindi."},
    {"family": "Noto Serif Devanagari", "google": "Noto Serif Devanagari", "aliases": ["Noto Serif Devanagari"], "kind": "serif", "note": "Devanagari / Hindi."},
    {"family": "Tiro Devanagari Hindi", "google": "Tiro Devanagari Hindi", "aliases": ["Tiro Devanagari Hindi"], "kind": "serif", "note": "Devanagari / Hindi."},
]

_CATALOG_BY_KEY: dict[str, dict] = {}
for _entry in CATALOG:
    for _name in [_entry["family"], _entry.get("google", ""), *_entry.get("aliases", [])]:
        if _name:
            _CATALOG_BY_KEY.setdefault(_name.strip().lower(), _entry)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "font"


def _clean_family(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    return re.sub(r"[^A-Za-z0-9 +\-]", "", text)[:80]


def _strip_subset(name: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", str(name or ""))


def _norm_style(name: str) -> str:
    n = str(name or "").lower()
    if any(k in n for k in ("bold", "black", "heavy", "semibold", "demibold", "demi")):
        return "Bold"
    if "italic" in n or "oblique" in n:
        return "Italic"
    return "Regular"


def parse_font(path: str | Path, source: str = "system") -> FontRecord:
    path = str(path)
    family = style = ""
    weight = 400
    italic = False
    if TTFont is not None:
        try:
            font = TTFont(path, fontNumber=0, lazy=True)
            names = {}
            for rec in font["name"].names:
                try:
                    text = rec.toUnicode().strip()
                except Exception:
                    continue
                if rec.nameID in (1, 2, 16, 17) and text:
                    names.setdefault(rec.nameID, text)
            family = names.get(16) or names.get(1) or ""
            style = names.get(17) or names.get(2) or ""
            if "OS/2" in font:
                weight = int(getattr(font["OS/2"], "usWeightClass", 400) or 400)
            if "head" in font:
                mac_style = int(getattr(font["head"], "macStyle", 0) or 0)
                italic = bool(mac_style & 0x02)
            if "post" in font:
                italic = italic or bool(getattr(font["post"], "italicAngle", 0))
            font.close()
        except Exception:
            pass
    if not family:
        stem = Path(path).stem
        stem = re.sub(r"[-_](regular|book|roman|bold|italic|oblique|light|medium|semibold|black).*$", "", stem, flags=re.I)
        family = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem).strip() or Path(path).stem
    if not style:
        n = Path(path).stem.lower()
        if "bold" in n or "bd" in n:
            style = "Bold"
        if "italic" in n or n.endswith("i") or "oblique" in n:
            style = "Italic" if style == "Regular" else "Bold Italic"
        if style == "":
            style = "Regular"
    # Google Fonts files are saved as family-400.ttf / family-700.ttf and some
    # variable-font instances report usWeightClass=400 even for the bold file.
    suffix = re.search(r"-(\d{3})(i?)$", Path(path).stem)
    if suffix:
        weight = int(suffix.group(1))
        if suffix.group(2):
            italic = True
        if weight >= 600 and "bold" not in style.lower():
            style = "Bold" + (" Italic" if italic else "")
    if "italic" in style.lower() or "oblique" in style.lower():
        italic = True
    if "bold" in style.lower():
        weight = max(weight, 700)
    label = f"{family.strip()} {style.strip()}".strip()
    if style.strip().lower() in ("regular", "book", "roman"):
        label = family.strip()
    return FontRecord(path, family.strip(), style.strip(), weight, italic, source, label)


def scan_fonts(*, include_system: bool = True, refresh: bool = False) -> list[FontRecord]:
    paths: list[str] = []
    if include_system:
        paths.extend(engine.discover_fonts(None, limit=200, refresh=refresh))
    # Always include the app font tree, even when the system list is capped.
    for root in (GOOGLE_ROOT, EMBEDDED_ROOT):
        if root.is_dir():
            for p in root.rglob("*"):
                if p.suffix.lower() in (".ttf", ".otf"):
                    paths.append(str(p))
    seen: set[str] = set()
    records: list[FontRecord] = []
    for p in paths:
        if p in seen or not Path(p).is_file():
            continue
        seen.add(p)
        rec = parse_font(p, "google" if GOOGLE_ROOT in Path(p).parents else ("embedded" if EMBEDDED_ROOT in Path(p).parents else "system"))
        records.append(rec)
    records.sort(key=lambda r: (r.family.lower(), r.weight, r.italic, r.label.lower()))
    return records


def catalog_payload() -> list[dict]:
    installed = {(r.family.lower(), r.weight, r.italic): r for r in scan_fonts()}
    out = []
    for entry in CATALOG:
        aliases = [entry["family"], *entry.get("aliases", [])]
        matches = [
            r for r in installed.values()
            if r.family.lower() in {a.lower() for a in aliases}
            or r.label.lower() in {a.lower() for a in aliases}
        ]
        out.append({
            **entry,
            "installed": bool(matches),
            "installed_files": len(matches),
            "downloaded": any(r.source == "google" for r in matches),
        })
    return out


def find_family(family: str, weight: int = 400, italic: bool = False) -> FontRecord | None:
    query = _clean_family(family).lower()
    if not query:
        return None
    entries = [e for e in CATALOG if query in {e["family"].lower(), e.get("google", "").lower(), *[a.lower() for a in e.get("aliases", [])]}]
    aliases = {query}
    for e in entries:
        aliases |= {e["family"].lower(), e.get("google", "").lower(), *[a.lower() for a in e.get("aliases", [])]}
    records = [r for r in scan_fonts() if r.family.lower() in aliases or r.label.lower() in aliases]
    if not records:
        return None
    # Exact weight/style first, then nearest weight with matching italic flag.
    exact = [r for r in records if r.weight == int(weight) and r.italic == bool(italic)]
    if exact:
        return exact[0]
    same_italic = [r for r in records if r.italic == bool(italic)] or records
    return min(same_italic, key=lambda r: (abs(r.weight - int(weight)), not r.italic if italic else r.italic))


def _fetch(url: str, *, timeout: int = 25, max_bytes: int = 12 * 1024 * 1024, ua: str = _UA_MODERN) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Font file too large.")
    return data


def _css_url(family: str, weights: Iterable[int]) -> str:
    clean = _clean_family(family)
    if not clean:
        raise ValueError("Font family name is empty.")
    q = urllib.parse.quote(clean.replace(" ", "+"), safe="+")
    ws = sorted({w for w in weights if w in _ALLOWED_WEIGHTS}) or [400]
    if ws == [400]:
        return f"{_GOOGLE_CSS}?family={q}"
    return f"{_GOOGLE_CSS}?family={q}:wght@{';'.join(str(w) for w in ws)}"


def _parse_css(css: str) -> list[dict]:
    faces = []
    for block in re.findall(r"@font-face\s*\{(.*?)\}", css, flags=re.S):
        family = re.search(r"font-family\s*:\s*['\"]?([^;'\"]+)", block)
        weight = re.search(r"font-weight\s*:\s*([0-9]+)", block)
        style = re.search(r"font-style\s*:\s*([a-z]+)", block)
        urls = re.findall(r"url\((https://[^)]+)\)", block)
        urange = re.search(r"unicode-range\s*:\s*([^;]+)", block)
        if not family or not urls:
            continue
        faces.append({
            "family": family.group(1).strip(),
            "weight": int(weight.group(1)) if weight else 400,
            "italic": bool(style and style.group(1).lower() in ("italic", "oblique")),
            "url": urls[0].strip("'\""),
            "range": (urange.group(1).strip() if urange else ""),
        })
    return faces


def _range_score(urange: str) -> int:
    """Prefer the Latin subset; Google serves one @font-face per unicode block."""
    r = (urange or "").upper()
    if "U+0000-00FF" in r or "U+0000-00A0" in r or "U+0020" in r:
        return 100
    if "U+0100-02AF" in r or "U+0100-024F" in r:
        return 60
    if not r:
        return 50
    return 10


def _save_font_bytes(data: bytes, out_path: Path) -> str:
    """Save TTF/OTF bytes, converting WOFF/WOFF2 when needed."""
    if data[:4] == b"wOF2" or data[:4] == b"wOFF":
        if TTFont is None:
            raise ValueError("WOFF font received but fonttools is not installed.")
        font = TTFont(io.BytesIO(data))
        font.flavor = None
        font.save(str(out_path))
        font.close()
    else:
        out_path.write_bytes(data)
    return str(out_path)


def download_family(family: str, weights: Iterable[int] | None = None, *, refresh: bool = False) -> dict:
    """Download a Google Fonts family into ``data/fonts/google``.

    The call is explicit and allowlisted by the catalogue.  No API key is used;
    the public CSS endpoint serves the font files directly.
    """
    clean = _clean_family(family)
    entry = _CATALOG_BY_KEY.get(clean.lower())
    if not entry:
        raise ValueError("Yeh family catalogue me nahi hai. Pehle listed family chunein.")
    google_family = entry.get("google") or entry["family"]
    weights = [int(w) for w in (weights or [400, 700]) if int(w) in _ALLOWED_WEIGHTS] or [400]
    outdir = GOOGLE_ROOT / _slug(google_family)
    outdir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        existing = [p for p in outdir.rglob("*.ttf") if p.is_file()]
        if existing and len(existing) >= max(1, len(set(weights))):
            return {"family": entry["family"], "google": google_family, "cached": True,
                    "files": [str(p) for p in existing], "warnings": []}

    warnings: list[str] = []
    files: list[str] = []
    try:
        css = _fetch(_css_url(google_family, weights), ua=_UA_MODERN).decode("utf-8", "replace")
        faces = _parse_css(css)
    except Exception as exc:
        faces = []
        warnings.append(f"Google Fonts CSS fetch failed: {exc}")
    if not faces:
        # Legacy user agent makes Google serve plain TTF instead of WOFF2.
        try:
            css = _fetch(_css_url(google_family, weights), ua=_UA_LEGACY).decode("utf-8", "replace")
            faces = _parse_css(css)
        except Exception as exc:
            warnings.append(f"TTF fallback failed: {exc}")

    if not faces:
        # Last resort: a direct TTF from the google/fonts GitHub mirror.
        slug = google_family.replace(" ", "")
        url = f"https://raw.githubusercontent.com/google/fonts/main/ofl/{_slug(google_family)}/{slug}-Regular.ttf"
        try:
            data = _fetch(url, ua=_UA_MODERN)
            path = outdir / f"{_slug(google_family)}-400.ttf"
            files.append(_save_font_bytes(data, path))
        except Exception as exc:
            raise ValueError(
                f"'{entry['family']}' download nahi hua. Internet check karein. Detail: {exc}"
            ) from exc
    else:
        # Google serves several unicode-range faces per weight (Latin,
        # Cyrillic, Greek...).  Pick the Latin one for each style so we do not
        # download the same file name repeatedly or bloat the cache.
        chosen: dict[tuple[int, bool], dict] = {}
        for face in faces:
            key = (face["weight"], face["italic"])
            prev = chosen.get(key)
            if prev is None or _range_score(face.get("range")) > _range_score(prev.get("range")):
                chosen[key] = face
        for face in chosen.values():
            weight = face["weight"]
            italic = face["italic"]
            suffix = f"{weight}{'i' if italic else ''}"
            path = outdir / f"{_slug(google_family)}-{suffix}.ttf"
            try:
                data = _fetch(face["url"], ua=_UA_MODERN)
                files.append(_save_font_bytes(data, path))
            except Exception as exc:
                warnings.append(f"{face['family']} {weight}{' italic' if italic else ''}: {exc}")
        files = list(dict.fromkeys(files))

    if not files:
        raise ValueError(f"'{entry['family']}' ke liye koi font file download nahi hui.")
    engine.clear_font_cache()
    return {
        "family": entry["family"],
        "google": google_family,
        "cached": False,
        "files": files,
        "warnings": warnings,
        "records": [parse_font(p, "google").to_dict() for p in files],
    }


def recognize(text: str, obs, limit: int = 6) -> dict:
    """Rank installed/downloaded fonts against an observed source ink mask."""
    candidates = engine.discover_fonts(None, limit=200)
    fits = engine.rank_fonts(candidates, text, obs, limit=limit)
    ranked = []
    for i, fit in enumerate(fits):
        rec = parse_font(fit.path)
        ranked.append({
            **rec.to_dict(),
            "size": round(float(fit.size), 2),
            "score": round(float(fit.score), 4),
            "confidence": round(float(min(1.0, max(0.0, fit.score))), 3),
            "rank": i + 1,
        })
    return {
        "best": ranked[0] if ranked else None,
        "candidates": ranked,
        "margin": round(float(ranked[0]["score"] - ranked[1]["score"]), 4) if len(ranked) > 1 else 1.0,
    }
