"""Flask API tests for the raster (image/scan) editor path."""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402
from textswap import documents  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None


FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).is_file()), None)


def png_bytes(text: str = "Name: Rehan Khan") -> bytes:
    img = Image.new("RGB", (900, 180), (250, 248, 240))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 40)
    draw.text((60, 60), text, font=font, fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def ink_fraction(img, box, pad: int = 4, threshold: int = 140) -> float:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                     min(img.width, x1 + pad), min(img.height, y1 + pad)))
    dark = sum(crop.histogram()[:threshold])
    return dark / max(1, crop.width * crop.height)


class RasterApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Image is None or not FONT_PATH:
            raise unittest.SkipTest("Pillow or a test font is unavailable")
        if "eng" not in documents.available_languages():
            raise unittest.SkipTest("Tesseract English language data is unavailable")
        cls.client = app.test_client()

    def upload(self, name: str = "page.png", text: str = "Name: Rehan Khan"):
        resp = self.client.post(
            "/api/upload",
            data={
                "file": (io.BytesIO(png_bytes(text)), name),
                "settings": json.dumps({"langs": ["eng"], "psm": 6, "dpi": 300}),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        return resp.get_json()["doc"]

    def test_config_and_fonts(self):
        cfg = self.client.get("/api/config")
        self.assertEqual(cfg.status_code, 200)
        self.assertIn("languages", cfg.get_json())

        fonts = self.client.get("/api/fonts")
        self.assertEqual(fonts.status_code, 200)
        payload = fonts.get_json()
        self.assertTrue(payload["catalog"])
        self.assertTrue(all("family" in item for item in payload["catalog"]))

    def test_upload_find_replace_export(self):
        doc = self.upload()
        self.assertEqual(doc["kind"], "image")
        self.assertTrue(doc["blocks"])

        found = self.client.post(
            f"/api/docs/{doc['id']}/find",
            json={"query": "Rehan", "fuzzy": 0.8},
        )
        self.assertEqual(found.status_code, 200)
        matches = found.get_json()["matches"]
        self.assertTrue(matches)
        match = matches[0]

        rendered = self.client.post(
            f"/api/docs/{doc['id']}/render",
            json={
                "ops": [{
                    "page": match["page"],
                    "bbox": match["bbox"],
                    "matched_text": match["text"],
                    "text": "Amit",
                    "style": {},
                }],
                "settings": {"output_format": "png"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertEqual(result["kind"], "raster")
        self.assertTrue(result["changed"])
        labels = {item["label"] for item in result["downloads"]}
        self.assertIn("Edited image", labels)
        self.assertIn("OCR text (.txt)", labels)

        image = next(item for item in result["downloads"] if item["label"] == "Edited image")
        download = self.client.get(f"/api/download/{image['id']}")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.data[:8], b"\x89PNG\r\n\x1a\n")

        report = next(item for item in result["downloads"] if item["label"] == "Report (.json)")
        payload = json.loads(self.client.get(f"/api/download/{report['id']}").data)
        self.assertEqual(payload["kind"], "image")
        self.assertGreaterEqual(payload["ops"], 1)

    def test_chunks_are_word_level(self):
        doc = self.upload()
        blocks = doc["blocks"]
        self.assertEqual([b["text"] for b in blocks], ["Name:", "Rehan", "Khan"])
        self.assertEqual([b["id"] for b in blocks], ["p0-l0-w0", "p0-l0-w1", "p0-l0-w2"])
        self.assertTrue(all(b["bbox"][2] > b["bbox"][0] for b in blocks))
        self.assertEqual(len({tuple(b["bbox"]) for b in blocks}), 3)
        self.assertEqual({b["line"] for b in blocks}, {0})

        found = self.client.post(
            f"/api/docs/{doc['id']}/find",
            json={"query": "Rehan", "whole_word": True},
        )
        self.assertEqual(found.status_code, 200)
        match = found.get_json()["matches"][0]
        self.assertEqual(match["block_id"], "p0-l0-w1")

        rendered = self.client.post(
            f"/api/docs/{doc['id']}/render",
            json={
                "ops": [{"block_id": "p0-l0-w1", "text": "Amit", "style": {}}],
                "settings": {"output_format": "png"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertTrue(result["changed"])
        image = next(item for item in result["downloads"] if item["label"] == "Edited image")
        data = self.client.get(f"/api/download/{image['id']}").data

        import pytesseract

        after = pytesseract.image_to_string(Image.open(io.BytesIO(data)))
        self.assertIn("Name: Amit Khan", after)
        self.assertNotIn("Rehan", after)

    def test_area_ocr_offsets_and_joins_find(self):
        doc = self.upload()
        found = self.client.post(
            f"/api/docs/{doc['id']}/find",
            json={"query": "Rehan", "whole_word": True},
        )
        self.assertEqual(found.status_code, 200)
        match = found.get_json()["matches"][0]
        x0, y0, x1, y1 = match["bbox"]
        area = [max(0.0, x0 - 8), max(0.0, y0 - 8), x1 + 8, y1 + 8]

        resp = self.client.post(
            f"/api/docs/{doc['id']}/ocr-area",
            json={"page": 0, "bbox": area, "langs": ["eng"], "psm": 6},
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        payload = resp.get_json()
        self.assertGreaterEqual(payload["count"], 1)
        self.assertIn("Rehan", payload["text"])
        self.assertTrue(all(b["id"].startswith("p0-r1-") for b in payload["blocks"]))
        self.assertEqual(len(payload["doc"]["blocks"]), len(doc["blocks"]) + payload["count"])
        for block in payload["blocks"]:
            self.assertGreaterEqual(block["bbox"][0], area[0] - 2)
            self.assertGreaterEqual(block["bbox"][1], area[1] - 2)
            self.assertLessEqual(block["bbox"][2], area[2] + 2)
            self.assertLessEqual(block["bbox"][3], area[3] + 2)

        again = self.client.post(
            f"/api/docs/{doc['id']}/find",
            json={"query": "Rehan", "whole_word": True},
        )
        self.assertGreaterEqual(len(again.get_json()["matches"]), 2)

    def test_annotation_bbox_override_is_used(self):
        doc = self.upload()
        block = next(b for b in doc["blocks"] if b["text"] == "Khan")
        box = [block["bbox"][0], block["bbox"][1], block["bbox"][2] + 30, block["bbox"][3] + 6]
        parsed = documents._DOCS[doc["id"]]
        pno, match = documents._match_from_op(parsed, {"block_id": block["id"], "bbox": box})
        self.assertEqual(pno, 0)
        self.assertEqual(list(match.box), box)

        rendered = self.client.post(
            f"/api/docs/{doc['id']}/render",
            json={
                "ops": [{
                    "block_id": block["id"],
                    "text": "Khan",
                    "style": {"size_scale": 1.2},
                    "bbox": box,
                }],
                "settings": {"output_format": "png"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        self.assertTrue(rendered.get_json()["changed"])

    def test_move_block_erases_source_and_draws_destination(self):
        doc = self.upload(text="Move")
        block = next(b for b in doc["blocks"] if b["text"].strip().lower() == "move")
        x0, y0, x1, y1 = block["bbox"]
        dest = [x0, y0 + 70, x1, y1 + 70]

        rendered = self.client.post(
            f"/api/docs/{doc['id']}/render",
            json={
                "ops": [{
                    "block_id": block["id"],
                    "text": "Move",
                    "style": {},
                    "bbox": dest,
                }],
                "settings": {"output_format": "png"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertTrue(result["changed"])
        self.assertFalse(result["warnings"], result["warnings"])
        image = next(item for item in result["downloads"] if item["label"] == "Edited image")
        edited = Image.open(io.BytesIO(self.client.get(f"/api/download/{image['id']}").data)).convert("L")
        original = Image.open(io.BytesIO(png_bytes("Move"))).convert("L")

        src_before = ink_fraction(original, block["bbox"])
        dest_before = ink_fraction(original, dest)
        src_after = ink_fraction(edited, block["bbox"])
        dest_after = ink_fraction(edited, dest)
        self.assertGreater(src_before, 0.01)
        self.assertLess(dest_before, 0.004)
        self.assertLess(src_after, src_before * 0.3)
        self.assertGreater(dest_after, src_before * 0.3)

    def test_insert_custom_text(self):
        doc = self.upload(text="Base")
        block = doc["blocks"][0]
        box = [40.0, 115.0, 340.0, 160.0]
        rendered = self.client.post(
            f"/api/docs/{doc['id']}/render",
            json={
                "ops": [{
                    "block_id": "ins-1",
                    "insert": True,
                    "page": 0,
                    "text": "Custom",
                    "style": {"color": "#c0392b"},
                    "bbox": box,
                }],
                "settings": {"output_format": "png"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertTrue(result["changed"])
        self.assertFalse(result["warnings"], result["warnings"])
        self.assertTrue(result["styles"] and result["styles"][0].get("inserted"))
        image = next(item for item in result["downloads"] if item["label"] == "Edited image")
        edited = Image.open(io.BytesIO(self.client.get(f"/api/download/{image['id']}").data)).convert("RGB")

        crop = edited.crop(tuple(int(v) for v in box))
        red = sum(1 for px in crop.getdata()
                  if px[0] > 110 and px[0] - px[1] > 35 and px[0] - px[2] > 35)
        self.assertGreater(red, 50)
        self.assertGreater(ink_fraction(edited.convert("L"), block["bbox"]), 0.01)

    def test_upload_validation(self):
        resp = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"not an image"), "evil.exe")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_render_requires_ops(self):
        doc = self.upload()
        resp = self.client.post(f"/api/docs/{doc['id']}/render", json={"ops": []})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
