"""Flask web application for TextSwap Studio."""
from __future__ import annotations

import json
import os
import secrets
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.middleware.proxy_fix import ProxyFix

from textswap import documents, fonts

ROOT = Path(__file__).resolve().parent
ON_RENDER = bool(os.environ.get("RENDER"))
PRODUCTION = os.environ.get("TEXTSWAP_ENV") == "production" or ON_RENDER


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.secret_key = os.environ.get("TEXTSWAP_SECRET") or secrets.token_hex(32)
    app.config.update(
        MAX_CONTENT_LENGTH=documents.MAX_UPLOAD + 8 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=(os.environ.get("TEXTSWAP_HTTPS") == "1") or ON_RENDER,
    )
    if ON_RENDER or os.environ.get("TEXTSWAP_PROXY") == "1":
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": "textswap-studio"})

    def sid() -> str:
        current = session.get("sid")
        sess = documents.read_session(current)
        if current != sess["id"]:
            session["sid"] = sess["id"]
            session.permanent = True
        return sess["id"]

    @app.get("/")
    def index():
        sid()
        return render_template("editor.html")

    @app.get("/api/config")
    def config():
        return jsonify({
            "languages": [{"code": c, "label": label} for c, label in documents.LANG_OPTIONS],
            "available_languages": sorted(documents.available_languages()),
            "max_mb": documents.MAX_UPLOAD // (1024 * 1024),
            "max_files": documents.MAX_FILES,
            "formats": sorted(documents.ALLOWED_FORMATS),
        })

    @app.get("/api/session")
    def session_info():
        return jsonify({"sid": sid()})

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "File select karein."}), 400
        raw = file.read()
        settings = {}
        if request.form.get("settings"):
            try:
                settings = json.loads(request.form["settings"])
            except Exception:
                settings = {}
        doc = documents.create_document(sid(), file.filename, raw, settings)
        return jsonify({"doc": doc})

    @app.get("/api/docs/<did>")
    def get_doc(did: str):
        doc = documents.get_doc(sid(), did)
        return jsonify({"doc": documents.doc_summary(doc)})

    @app.get("/api/docs/<did>/preview")
    def preview(did: str):
        page = request.args.get("page", 0, type=int)
        data = documents.page_preview(sid(), did, page)
        return jsonify({"image": data})

    @app.post("/api/docs/<did>/recognize")
    def recognize(did: str):
        payload = request.get_json(silent=True) or {}
        bid = str(payload.get("block_id") or "")
        return jsonify(documents.recognize_block(sid(), did, bid))

    @app.post("/api/docs/<did>/ocr-area")
    def ocr_area(did: str):
        payload = request.get_json(silent=True) or {}
        return jsonify(documents.ocr_region(sid(), did, payload))

    @app.post("/api/docs/<did>/find")
    def find(did: str):
        payload = request.get_json(silent=True) or {}
        matches = documents.find_text(sid(), did, payload)
        return jsonify({"matches": matches, "count": len(matches)})

    @app.post("/api/docs/<did>/render")
    def render(did: str):
        payload = request.get_json(silent=True) or {}
        result = documents.render(sid(), did, payload)
        return jsonify(result)

    @app.post("/api/docs/<did>/export-text")
    def export_text(did: str):
        return jsonify({"download": documents.export_text(sid(), did)})

    @app.delete("/api/docs/<did>")
    def delete_doc(did: str):
        documents.delete_document(sid(), did)
        return jsonify({"ok": True})

    @app.get("/api/fonts")
    def get_fonts():
        records = fonts.scan_fonts()
        return jsonify({
            "installed": [r.to_dict() for r in records],
            "catalog": fonts.catalog_payload(),
        })

    @app.post("/api/fonts/download")
    def download_font():
        payload = request.get_json(silent=True) or {}
        family = str(payload.get("family") or "")
        weights = payload.get("weights") or [400, 700]
        result = fonts.download_family(family, weights)
        return jsonify(result)

    @app.get("/api/download/<fid>")
    def download(fid: str):
        rec = documents.get_file(sid(), fid)
        if not rec:
            return jsonify({"error": "File expired ya missing hai."}), 404
        return send_file(rec["path"], as_attachment=True, download_name=rec["filename"], mimetype=rec["mime"])

    @app.errorhandler(ValueError)
    def value_error(exc):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({"error": "Upload size limit se badi file."}), 413

    @app.errorhandler(Exception)
    def unhandled(exc):
        app.logger.error("Unhandled: %s\n%s", exc, traceback.format_exc())
        if PRODUCTION:
            return jsonify({"error": "Internal server error."}), 500
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"TextSwap Studio -> http://127.0.0.1:{port}")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False, threaded=True)
