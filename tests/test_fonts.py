"""Font catalogue, metadata and CSS parsing tests (no network)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textswap import engine, fonts  # noqa: E402


class FontCssTest(unittest.TestCase):
    def test_parse_css_keeps_unicode_range(self):
        css = """
        @font-face {
          font-family: 'Roboto';
          font-style: normal;
          font-weight: 400;
          src: url(https://fonts.gstatic.com/s/roboto/latin.woff2) format('woff2');
          unicode-range: U+0000-00FF, U+0131, U+0152-0153;
        }
        @font-face {
          font-family: 'Roboto';
          font-style: normal;
          font-weight: 700;
          src: url(https://fonts.gstatic.com/s/roboto/latin-bold.woff2) format('woff2');
          unicode-range: U+0400-045F, U+0490-0491;
        }
        """
        faces = fonts._parse_css(css)
        self.assertEqual(len(faces), 2)
        self.assertEqual(faces[0]["weight"], 400)
        self.assertIn("U+0000-00FF", faces[0]["range"])
        scores = [fonts._range_score(f["range"]) for f in faces]
        self.assertGreater(scores[0], scores[1])

    def test_parse_css_ignores_faces_without_url(self):
        css = "@font-face { font-family: 'X'; font-weight: 400; }"
        self.assertEqual(fonts._parse_css(css), [])


class FontCatalogueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = fonts.scan_fonts()

    def test_scan_returns_records(self):
        self.assertTrue(self.records)
        for rec in self.records[:20]:
            self.assertTrue(rec.family)
            self.assertIn(rec.weight, {100, 200, 300, 400, 500, 600, 700, 800, 900})
            self.assertTrue(Path(rec.path).is_file())

    def test_catalog_payload_shape(self):
        payload = fonts.catalog_payload()
        self.assertTrue(payload)
        families = {entry["family"] for entry in payload}
        self.assertIn("Arial", families)
        self.assertIn("Roboto", families)
        for entry in payload:
            self.assertIn("installed", entry)
            self.assertIn("aliases", entry)

    def test_find_family_prefers_requested_weight(self):
        if not any(r.family.lower() == "arial" for r in self.records):
            self.skipTest("Arial is not installed on this machine")
        regular = fonts.find_family("Arial", 400, False)
        bold = fonts.find_family("Arial", 700, False)
        self.assertIsNotNone(regular)
        self.assertIsNotNone(bold)
        self.assertLessEqual(abs(regular.weight - 400), abs(regular.weight - 700))
        self.assertLessEqual(abs(bold.weight - 700), abs(bold.weight - 400))

    def test_parse_google_suffix_weight(self):
        rec = next((r for r in self.records if "roboto" in r.path.lower()), None)
        if rec is None:
            self.skipTest("Roboto has not been downloaded")
        self.assertIn(rec.weight, {400, 700})
        self.assertTrue(rec.family)

    def test_font_label_fallback(self):
        self.assertTrue(engine.font_label("C:/fonts/SomeFont-Bold.ttf"))


if __name__ == "__main__":
    unittest.main()
