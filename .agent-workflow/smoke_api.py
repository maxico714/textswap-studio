import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont
from app import app

img = Image.new("RGB", (900, 260), (250, 248, 240))
d = ImageDraw.Draw(img)
font_path = "C:/Windows/Fonts/arial.ttf"
f = ImageFont.truetype(font_path, 40)
d.text((60, 70), "Name: Rehan Khan", font=f, fill=(20, 20, 20))
d.text((60, 140), "Account: 4821 7733", font=f, fill=(20, 20, 20))
buf = io.BytesIO()
img.save(buf, "PNG")
raw = buf.getvalue()

client = app.test_client()
r = client.get("/api/config")
print("config", r.status_code)
r = client.post("/api/upload", data={"file": (io.BytesIO(raw), "test.png"), "settings": json.dumps({"langs": ["eng"], "psm": 6, "dpi": 220, "enhance": True})}, content_type="multipart/form-data")
print("upload", r.status_code)
data = r.get_json()
if r.status_code != 200:
    print(data)
    raise SystemExit(1)
doc = data["doc"]
print("doc", doc["kind"], doc["page_count"], len(doc["blocks"]))
print([b["text"] for b in doc["blocks"][:10]])
if not doc["blocks"]:
    raise SystemExit("no OCR blocks")
block = doc["blocks"][0]
r = client.post(f"/api/docs/{doc['id']}/find", json={"query": "Rehan", "fuzzy": 0.8})
print("find", r.status_code, r.get_json())
matches = r.get_json()["matches"]
if not matches:
    raise SystemExit("no find matches")
m = matches[0]
r = client.post(f"/api/docs/{doc['id']}/render", json={"ops": [{"page": m["page"], "bbox": m["bbox"], "matched_text": m["text"], "text": "Amit", "style": {}}], "settings": {"output_format": "png"}})
print("render", r.status_code)
res = r.get_json()
print(json.dumps({k: v for k, v in res.items() if k != "previews"}, indent=2)[:1500])
print("downloads", res.get("downloads"))
