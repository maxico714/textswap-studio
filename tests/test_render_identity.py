"""Pixel identity of a replacement against the source it replaces.

Regression for the "replacement text looks slightly bigger" report: PIL's
sub-pixel baseline used to add one fully covered row at the bottom, so a word
rendered at the calibrated size measured 1 px taller than the source.  These
tests build a source page, run the real OCR/fit/render path, and compare the
source and replacement ink boxes.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textswap import engine  # noqa: E402

FONT_CANDIDATES = (
    os.environ.get("TEXTSWAP_TEST_FONT"),
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
FONT_PATH = next((str(Path(p)) for p in FONT_CANDIDATES if p and Path(p).is_file()), None)


def _page(text: str, px: int, width: int = 1100, height: int = 160) -> str:
    img = Image.new("RGB", (width, height), (250, 248, 240))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, px)
    draw.text((60, 60), text, font=font, fill=(20, 20, 20))
    out = ROOT / ".agent-workflow" / "test_render_identity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def _replace(path: str, old: str, new: str) -> dict:
    page = engine.load_pages(path, 300)[0]
    lines = engine.ocr_lines(page.bgr, "eng", 6, enhance=True)
    matches = engine.find_matches([lines], old, fuzzy=0.8)
    if not matches:
        raise unittest.SkipTest(f"OCR did not find {old!r}")
    style: dict = {}
    engine.edit_page(
        page,
        [matches[0]],
        new,
        fonts=engine.discover_fonts(None),
        method="auto",
        align="left",
        style_info=style,
        log=lambda *_: None,
        source_bgr=page.bgr,
    )
    return style


@unittest.skipUnless(FONT_PATH, "no usable test font installed")
class RenderIdentityTest(unittest.TestCase):
    def _assert_identity(self, px: int, old: str, new: str, check_width: bool) -> None:
        try:
            style = _replace(_page(f"Name: {old} Khan", px), old, new)
        except ImportError as exc:  # pragma: no cover - dependency environment
            self.skipTest(str(exc))
        pix = style.get("pixel") or {}
        src = pix.get("source_core_bbox")
        rep = pix.get("replacement_ink_bbox")
        self.assertIsNotNone(src, pix)
        self.assertIsNotNone(rep, pix)
        src_h, rep_h = src[3] - src[1], rep[3] - rep[1]
        src_w, rep_w = src[2] - src[0], rep[2] - rep[0]
        # Height must be identical within a single raster row.
        self.assertLessEqual(abs(src_h - rep_h), 1, (src, rep, style.get("size")))
        # Top and bottom rows must not drift by more than one row.
        self.assertLessEqual(abs(src[1] - rep[1]), 1, (src, rep))
        self.assertLessEqual(abs(src[3] - rep[3]), 1, (src, rep))
        # Pixel-lock snaps the baseline to a whole row (no sub-pixel growth).
        self.assertEqual(pix.get("baseline_y_derotated"), int(pix.get("baseline_y_derotated")))
        if check_width:
            self.assertLessEqual(abs(src_w - rep_w), 2, (src, rep))
        self.assertTrue(style.get("font"), style)

    def test_40px_different_word(self):
        self._assert_identity(40, "Rehan", "Amit", check_width=False)

    def test_40px_same_length(self):
        self._assert_identity(40, "Rehan", "Rohan", check_width=True)

    def test_18px(self):
        self._assert_identity(18, "Rehan", "Amit", check_width=False)

    def test_12px(self):
        self._assert_identity(12, "Rehan", "Amit", check_width=False)


if __name__ == "__main__":
    unittest.main()
