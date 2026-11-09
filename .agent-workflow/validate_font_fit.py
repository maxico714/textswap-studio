"""Validate auto font selection and ink identity after replacement.

Acceptance: the font family is recognised from the source, the replacement ink
height matches the source within 1 px, top/bottom rows within 1 px, and a
same-length replacement matches width within 2 px.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont
import textswap.engine as engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".agent-workflow" / "font_fit_cases"
OUT.mkdir(parents=True, exist_ok=True)

cases = [
    ("arial_40", "C:/Windows/Fonts/arial.ttf", 40, "Name: Rehan Khan", "Rehan", "Amit", "Arial"),
    ("arial_40_same_len", "C:/Windows/Fonts/arial.ttf", 40, "Name: Rehan Khan", "Rehan", "Rohan", "Arial"),
    ("arial_18", "C:/Windows/Fonts/arial.ttf", 18, "Name: Rehan Khan", "Rehan", "Amit", "Arial"),
    ("arial_12", "C:/Windows/Fonts/arial.ttf", 12, "Name: Rehan Khan", "Rehan", "Amit", "Arial"),
]
roboto = ROOT / "data" / "fonts" / "google" / "roboto" / "roboto-400.ttf"
if roboto.is_file():
    cases.append(("roboto_36", str(roboto), 36, "Name: Rehan Khan", "Rehan", "Amit", "Roboto 400"))
    cases.append(("roboto_36_same_len", str(roboto), 36, "Name: Rehan Khan", "Rehan", "Rohan", "Roboto 400"))

failures = []
for name, font_path, px, line, old, new, expected in cases:
    img = Image.new("RGB", (1100, 160), (250, 248, 240))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(font_path, px)
    d.text((60, 60), line, font=f, fill=(20, 20, 20))
    path = OUT / f"{name}.png"
    img.save(path)

    pages = engine.load_pages(str(path), 300)
    page = pages[0]
    lines = engine.ocr_lines(page.bgr, "eng", 6, enhance=True)
    matches = engine.find_matches([lines], old, fuzzy=0.8)
    if not matches:
        failures.append((name, "no OCR match"))
        continue
    style = {}
    engine.edit_page(page, [matches[0]], new, fonts=engine.discover_fonts(None),
                     method="auto", align="left", style_info=style,
                     log=lambda *_: None, source_bgr=page.bgr)
    pix = style.get("pixel", {})
    src, rep = pix.get("source_core_bbox"), pix.get("replacement_ink_bbox")
    if not src or not rep:
        failures.append((name, "no ink bbox"))
        continue
    sh, rh = src[3] - src[1], rep[3] - rep[1]
    sw, rw = src[2] - src[0], rep[2] - rep[0]
    rows_ok = abs(src[1] - rep[1]) <= 1 and abs(src[3] - rep[3]) <= 1
    height_ok = abs(sh - rh) <= 1
    width_ok = (abs(sw - rw) <= 2) if len(old) == len(new) else True
    font_ok = style.get("font") == expected
    ok = height_ok and rows_ok and width_ok and font_ok
    print(f"{name:22s} src={sw:3d}x{sh:2d} rep={rw:3d}x{rh:2d} "
          f"font={style.get('font')!r} size={style.get('size')} "
          f"stretch={style.get('stretch')} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append((name, src, rep, style.get("font")))

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
