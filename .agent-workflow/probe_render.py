import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from textswap import engine

path = "C:/Windows/Fonts/arial.ttf"
size = 41.28
print("--- raw render extents at size", size)
for text in ("Rehan", "Amit"):
    a, L, T = engine.render_alpha(text, path, size, ss=4)
    e = engine._extent(a)
    print(text, "extent", e, "h", e[3] - e[1], "w", e[2] - e[0], "L,T", L, T)
    for fy in (0.0, 0.5):
        b, Lb, Tb = engine.render_alpha(text, path, size, fy=fy, ss=4)
        eb = engine._extent(b)
        print("  fy", fy, "extent", eb, "h", eb[3] - eb[1])
    a2 = engine._prepare_line_alpha(a, 0.9652, 0.0, 0.25)
    e2 = engine._extent(a2)
    print("  after xs+blur", e2, "h", e2[3] - e2[1], "w", e2[2] - e2[0])
    a3 = engine._prepare_line_alpha(a, 0.9652, 0.0, 0.0)
    e3 = engine._extent(a3)
    print("  after xs no blur", e3, "h", e3[3] - e3[1])

print()
print("--- scan sizes for matching height")
H = 29
for s in [x / 4 for x in range(160, 172)]:
    row = []
    for text in ("Rehan", "Amit"):
        a, L, T = engine.render_alpha(text, path, s, ss=4)
        e = engine._extent(a)
        row.append((text, e[3] - e[1], e[2] - e[0]))
    print(s, row)
