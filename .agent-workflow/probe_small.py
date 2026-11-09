import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from textswap import engine

for name, size in (("arial_18", 18), ("arial_40", 40)):
    img = Image.new("RGB", (1100, 160), (250, 248, 240))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    d.text((60, 60), "Name: Rehan Khan", font=f, fill=(20, 20, 20))
    path = f".agent-workflow/font_fit_cases/{name}.png"
    img.save(path)
    p = engine.load_pages(path, 300)[0]
    ls = engine.ocr_lines(p.bgr, "eng", 6, enhance=True)
    m = engine.find_matches([ls], "Rehan", fuzzy=0.8)[0]
    st = {}
    engine.edit_page(p, [m], "Amit", fonts=engine.discover_fonts(None), method="auto",
                     align="left", style_info=st, log=lambda *_: None, source_bgr=p.bgr)
    pix = st["pixel"]
    print(name, "size", st["size"], "stretch", st["stretch"])
    for k in ("source_core_bbox", "source_reconstruction_bbox", "replacement_ink_bbox",
              "baseline_y_derotated", "baseline_offset", "calibration_shift", "shift"):
        print("  ", k, pix.get(k) if k in pix else st.get(k))
