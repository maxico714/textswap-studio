import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from textswap import engine

pages = engine.load_pages("test_page.png", 300)
p = pages[0]
lines = engine.ocr_lines(p.bgr, "eng", 6, enhance=True)
for ln in lines:
    print("LINE", [(w.text, w.x, w.y, w.w, w.h) for w in ln])
ms = engine.find_matches([lines], "Rehan", fuzzy=0.8)
print("matches", ms)
m = ms[0]
reg = engine.analyze_region(p.bgr, m.box)
print("reg core_bbox", reg.core_bbox, "ink_bbox", reg.ink_bbox, "bh", reg.bh)
obs = reg.core[reg.core_bbox[1]-reg.ctx[1]:reg.core_bbox[3]-reg.ctx[1], reg.core_bbox[0]-reg.ctx[0]:reg.core_bbox[2]-reg.ctx[0]]
print("obs h,w", obs.shape)
fonts = engine.discover_fonts(None)
style = {}
out = engine.edit_page(p, [m], "Amit", fonts=fonts, method="auto", align="left", style_info=style, log=print, source_bgr=p.bgr)
print("style", style)
cv2 = engine.cv2
cv2.imwrite("test_page_replaced.png", out)
# measure ink diff bbox: changed pixels
diff = np.any(out != p.bgr, axis=2)
ys, xs = np.nonzero(diff)
print("changed bbox", xs.min(), ys.min(), xs.max()+1, ys.max()+1)
# source ink bbox and replacement bbox from style
print("source ink", style.get("pixel", {}).get("source_ink_bbox"))
print("replacement ink", style.get("pixel", {}).get("replacement_ink_bbox"))
