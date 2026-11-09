"""Native PDF text-layer replacement tests."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None


def sample_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "Invoice for Rehan Khan", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text((72, 170), "Account: 4821 7733", fontsize=14, fontname="helv", color=(0.1, 0.1, 0.1))
    data = doc.tobytes()
    doc.close()
    return data


@unittest.skipUnless(fitz is not None, "PyMuPDF is not installed")
class NativePdfTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        resp = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(sample_pdf()), "invoice.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        self.doc = resp.get_json()["doc"]

    def test_native_mode_and_spans(self):
        self.assertEqual(self.doc["kind"], "pdf_native")
        self.assertTrue(self.doc["native"])
        blocks = self.doc["blocks"]
        texts = {block["text"] for block in blocks}
        self.assertIn("Invoice", texts)
        self.assertIn("Rehan", texts)
        self.assertIn("Khan", texts)
        self.assertNotIn("Invoice for Rehan Khan", texts)
        self.assertEqual(len({b["id"] for b in blocks}), len(blocks))
        block = next(b for b in blocks if b["text"] == "Rehan")
        self.assertEqual(block["id"], "pdf-p0-b0-l0-s0-w2")
        self.assertEqual(block["span_id"], "pdf-p0-b0-l0-s0")
        self.assertGreater(block["bbox"][2], block["bbox"][0])
        self.assertEqual(block["style"]["font"], "Helvetica")
        same_line = [b for b in blocks if b["id"].startswith("pdf-p0-b0-l0-s0-w")]
        self.assertGreaterEqual(len(same_line), 4)

    def test_replace_chunk_preserves_neighbours(self):
        found = self.client.post(
            f"/api/docs/{self.doc['id']}/find",
            json={"query": "Rehan", "case_sensitive": True, "whole_word": True},
        )
        self.assertEqual(found.status_code, 200)
        matches = found.get_json()["matches"]
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match["block_id"], "pdf-p0-b0-l0-s0-w2")
        block = next(b for b in self.doc["blocks"] if b["id"] == match["block_id"])
        self.assertEqual(block["text"], "Rehan")

        rendered = self.client.post(
            f"/api/docs/{self.doc['id']}/render",
            json={
                "ops": [{"block_id": block["id"], "text": "Amit", "style": block["style"]}],
                "settings": {"output_format": "pdf"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertEqual(result["kind"], "native")
        self.assertEqual(result["native_report"]["replaced"], 1)
        pdf_id = result["downloads"][0]["id"]
        data = self.client.get(f"/api/download/{pdf_id}").data
        self.assertEqual(data[:5], b"%PDF-")
        out = fitz.open(stream=data, filetype="pdf")
        text = out[0].get_text()
        out.close()
        self.assertNotIn("Rehan", text)
        self.assertIn("Invoice", text)
        self.assertIn("for", text)
        self.assertIn("Amit", text)
        self.assertIn("Khan", text)
        self.assertIn("Account: 4821 7733", text)

    def test_phrase_match_maps_to_chunk_edits(self):
        found = self.client.post(
            f"/api/docs/{self.doc['id']}/find",
            json={"query": "Rehan Khan", "case_sensitive": True, "whole_word": True},
        )
        self.assertEqual(found.status_code, 200)
        matches = found.get_json()["matches"]
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match["block_id"], "pdf-p0-b0-l0-s0-w2")
        self.assertEqual(
            [e["block_id"] for e in match["chunk_edits"]],
            ["pdf-p0-b0-l0-s0-w2", "pdf-p0-b0-l0-s0-w3"],
        )
        self.assertEqual([(e["start"], e["end"]) for e in match["chunk_edits"]], [(0, 5), (0, 4)])

    def test_replace_span_preserves_pdf_text(self):
        found = self.client.post(
            f"/api/docs/{self.doc['id']}/find",
            json={"query": "Rehan", "case_sensitive": True, "whole_word": True},
        )
        self.assertEqual(found.status_code, 200)
        matches = found.get_json()["matches"]
        self.assertEqual(len(matches), 1)
        match = matches[0]
        block = next(b for b in self.doc["blocks"] if b["id"] == match["block_id"])
        new_text = block["text"][: match["start"]] + "Amit" + block["text"][match["end"]:]

        rendered = self.client.post(
            f"/api/docs/{self.doc['id']}/render",
            json={
                "ops": [{"block_id": block["id"], "text": new_text, "style": block["style"]}],
                "settings": {"output_format": "pdf"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertEqual(result["kind"], "native")
        self.assertEqual(result["native_report"]["replaced"], 1)
        pdf_id = result["downloads"][0]["id"]
        data = self.client.get(f"/api/download/{pdf_id}").data
        self.assertEqual(data[:5], b"%PDF-")
        out = fitz.open(stream=data, filetype="pdf")
        text = out[0].get_text()
        out.close()
        self.assertNotIn("Rehan", text)
        self.assertIn("Invoice", text)
        self.assertIn("for", text)
        self.assertIn("Amit", text)
        self.assertIn("Khan", text)
        self.assertIn("Account: 4821 7733", text)

    def test_resized_bbox_moves_replacement_origin(self):
        found = self.client.post(
            f"/api/docs/{self.doc['id']}/find",
            json={"query": "Rehan", "case_sensitive": True, "whole_word": True},
        )
        self.assertEqual(found.status_code, 200)
        match = found.get_json()["matches"][0]
        block = next(b for b in self.doc["blocks"] if b["id"] == match["block_id"])
        x0, y0, x1, y1 = block["bbox"]
        shift = 40.0
        box = [x0 + shift, y0, x1 + shift, y1 + 6]

        src = fitz.open(stream=sample_pdf(), filetype="pdf")
        old_spans = [
            span
            for blk in src[0].get_text("dict")["blocks"] if blk.get("type") == 0
            for line in blk["lines"] for span in line["spans"]
        ]
        old_x0 = next(span for span in old_spans if "Rehan" in span["text"])["bbox"][0]
        src.close()

        rendered = self.client.post(
            f"/api/docs/{self.doc['id']}/render",
            json={
                "ops": [{
                    "block_id": block["id"],
                    "text": "Amit",
                    "style": block["style"],
                    "bbox": box,
                }],
                "settings": {"output_format": "pdf"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertEqual(result["native_report"]["replaced"], 1)
        pdf_id = result["downloads"][0]["id"]
        data = self.client.get(f"/api/download/{pdf_id}").data
        out = fitz.open(stream=data, filetype="pdf")
        spans = [
            span
            for blk in out[0].get_text("dict")["blocks"] if blk.get("type") == 0
            for line in blk["lines"] for span in line["spans"]
        ]
        out.close()
        new_x0 = next(span for span in spans if "Amit" in span["text"])["bbox"][0]
        self.assertGreater(new_x0, old_x0 + 5)

    def test_insert_custom_text_into_pdf(self):
        rendered = self.client.post(
            f"/api/docs/{self.doc['id']}/render",
            json={
                "ops": [{
                    "block_id": "ins-1",
                    "insert": True,
                    "page": 0,
                    "text": "Custom note",
                    "style": {},
                    "bbox": [100, 600, 620, 660],
                }],
                "settings": {"output_format": "pdf"},
            },
        )
        self.assertEqual(rendered.status_code, 200, rendered.get_data(as_text=True))
        result = rendered.get_json()
        self.assertEqual(result["kind"], "native")
        self.assertEqual(result["native_report"]["inserted"], 1)
        self.assertFalse(result["native_report"]["warnings"], result["native_report"]["warnings"])
        pdf_id = result["downloads"][0]["id"]
        data = self.client.get(f"/api/download/{pdf_id}").data
        out = fitz.open(stream=data, filetype="pdf")
        text = out[0].get_text()
        out.close()
        self.assertIn("Custom note", text)
        self.assertIn("Invoice for Rehan Khan", text)
        self.assertIn("Account: 4821 7733", text)

    def test_regex_and_whole_word(self):
        found = self.client.post(
            f"/api/docs/{self.doc['id']}/find",
            json={"query": r"Reh\w+", "regex": True},
        )
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.get_json()["count"], 1)
        none = self.client.post(
            f"/api/docs/{self.doc['id']}/find",
            json={"query": "Reha", "whole_word": True},
        )
        self.assertEqual(none.get_json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
