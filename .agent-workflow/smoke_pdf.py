import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf as fitz
from app import app

doc = fitz.open()
page = doc.new_page(width=595, height=842)
page.insert_text((72, 120), "Invoice for Rehan Khan", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.1))
page.insert_text((72, 170), "Account: 4821 7733", fontsize=14, fontname="helv", color=(0.1, 0.1, 0.1))
pdf_bytes = doc.tobytes()
doc.close()

client = app.test_client()
r = client.post("/api/upload", data={"file": (io.BytesIO(pdf_bytes), "native.pdf"), "settings": json.dumps({"langs": ["eng"], "dpi": 180})}, content_type="multipart/form-data")
print("upload", r.status_code)
data = r.get_json()
if r.status_code != 200:
    print(data); raise SystemExit(1)
docj = data["doc"]
print("kind", docj["kind"], "blocks", len(docj["blocks"]), "pages", docj["pages"])
print([(b["text"], b["bbox"], b["style"]["font"]) for b in docj["blocks"]])
r = client.post(f"/api/docs/{docj['id']}/find", json={"query": "Rehan", "fuzzy": 0.8})
print("find", r.status_code, r.get_json())
matches = r.get_json()["matches"]
if not matches: raise SystemExit("no match")
m = matches[0]
block = next(b for b in docj["blocks"] if b["id"] == m["block_id"])
new_text = block["text"][:m["start"]] + "Amit" + block["text"][m["end"]:]
print("block text", block["text"], "->", new_text)
r = client.post(f"/api/docs/{docj['id']}/render", json={"ops": [{"block_id": m["block_id"], "text": new_text, "style": block["style"]}], "settings": {"output_format": "pdf"}})
print("render", r.status_code)
res = r.get_json()
if r.status_code != 200:
    print(res); raise SystemExit(1)
print(json.dumps({k: v for k, v in res.items() if k != "previews"}, indent=2)[:2000])
dl = res["downloads"][0]
r = client.get(f"/api/download/{dl['id']}")
print("download", r.status_code, len(r.data))
out = fitz.open(stream=r.data, filetype="pdf")
print("output text:", out[0].get_text())
