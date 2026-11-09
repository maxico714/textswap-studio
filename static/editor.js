/* TextSwap Studio — vanilla JS editor */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const NS = "http://www.w3.org/2000/svg";
  let suppressClick = false;
  let styleTimer = null;
  let deferredRender = null;

  const state = {
    doc: null,
    docs: [],
    page: 0,
    zoom: 1,
    fitMode: true,
    ops: new Map(),
    boxes: new Map(),
    history: [],
    future: [],
    selected: new Set(),
    originals: {},
    lastRender: null,
    matches: [],
    selectedMatches: new Set(),
    fonts: { installed: [], catalog: [] },
    busy: false,
    tab: "edit",
    renderSeq: 0,
    ocrMode: false,
  };

  /* ---------------------------------------------------------------- utils */
  function toast(message, kind = "") {
    const el = $("#toast");
    el.textContent = message;
    el.className = "toast " + kind;
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 4200);
  }

  function status(text, kind = "") {
    const el = $("#statusBar");
    el.className = "statusbar " + kind;
    el.textContent = text;
  }

  function busy(on, text) {
    state.busy = on;
    $("#canvasWrap").classList.toggle("busy", on);
    if (text) status(text);
  }

  function fmtBytes(n) {
    if (!n && n !== 0) return "";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(1) + " MB";
  }

  async function api(url, { method = "GET", body, form } = {}) {
    const opts = { method, credentials: "same-origin" };
    if (form) opts.body = form;
    else if (body !== undefined) {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    let data = null;
    try { data = await res.json(); } catch (_) { /* ignore */ }
    if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
    return data || {};
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  function selectedBlockId() {
    return state.selected.size === 1 ? Array.from(state.selected)[0] : null;
  }

  function blockById(id) {
    return state.doc ? state.doc.blocks.find((b) => b.id === id) : null;
  }

  function blockBox(block) {
    const box = state.boxes.get(block.id);
    return box ? box.slice() : block.bbox.slice();
  }

  function setBlockBox(id, box) {
    const block = blockById(id);
    if (!block) return;
    const same = block.bbox.every((v, i) => Math.abs(v - box[i]) < 0.5);
    if (same) state.boxes.delete(id);
    else state.boxes.set(id, box.slice());
  }

  function boxesEqual(a, b) {
    return a && b && a.every((v, i) => Math.abs(v - b[i]) < 0.01);
  }

  function pageInfo() {
    if (!state.doc) return { width: 1000, height: 1400 };
    return state.doc.pages[state.page] || state.doc.pages[0] || { width: 1000, height: 1400 };
  }

  function isTyping() {
    const a = document.activeElement;
    return a && (a.tagName === "TEXTAREA" || a.tagName === "INPUT" || a.tagName === "SELECT");
  }

  /* --------------------------------------------------------------- config */
  async function loadConfig() {
    try {
      const cfg = await api("/api/config");
      $("#maxMb").textContent = cfg.max_mb;
      const lang = $("#langSelect");
      lang.innerHTML = "";
      for (const item of cfg.languages) {
        const opt = new Option(item.label, item.code, false, item.code === "eng");
        lang.appendChild(opt);
      }
      const missing = cfg.languages.filter((x) => !cfg.available_languages.includes(x.code));
      if (missing.length) {
        lang.title = "Missing Tesseract packs: " + missing.map((x) => x.code).join(", ");
      }
    } catch (e) {
      toast("Config load failed: " + e.message, "bad");
    }
  }

  async function loadFonts() {
    try {
      state.fonts = await api("/api/fonts");
      populateFontFamilySelect();
      renderFontCatalog();
    } catch (e) {
      toast("Fonts load failed: " + e.message, "bad");
    }
  }

  function populateFontFamilySelect() {
    const sel = $("#fontFamily");
    const current = sel.value;
    const families = new Map();
    for (const rec of state.fonts.installed) families.set(rec.family, "installed");
    for (const entry of state.fonts.catalog) {
      if (!families.has(entry.family)) families.set(entry.family, entry.installed ? "installed" : "catalog");
    }
    sel.innerHTML = "";
    sel.appendChild(new Option("Auto — match source", ""));
    const sorted = Array.from(families.keys()).sort((a, b) => a.localeCompare(b));
    for (const family of sorted) {
      const kind = families.get(family);
      const label = family + (kind === "catalog" ? "  (download)" : "");
      sel.appendChild(new Option(label, family));
    }
    if (current) sel.value = current;
  }

  function renderFontCatalog() {
    const box = $("#fontCatalog");
    const q = ($("#fontSearch").value || "").toLowerCase();
    const installedFamilies = new Set(state.fonts.installed.map((r) => r.family.toLowerCase()));
    box.innerHTML = "";
    const rows = state.fonts.catalog.filter((f) => {
      if (!q) return true;
      return (f.family + " " + (f.aliases || []).join(" ") + " " + (f.google || "")).toLowerCase().includes(q);
    });
    if (!rows.length) {
      box.appendChild(el("div", { class: "empty small", text: "No family found." }));
      return;
    }
    for (const f of rows) {
      const installed = f.installed || installedFamilies.has(f.family.toLowerCase());
      const sub = installed
        ? (f.downloaded ? "downloaded" : "installed") + (f.note ? " · " + f.note : "")
        : (f.google ? "Google Fonts · " + f.google : "not installed");
      const row = el("div", { class: "font-row" }, [
        el("div", {}, [
          el("div", { class: "fam", text: f.family }),
          el("div", { class: "sub", text: sub }),
        ]),
        installed
          ? el("span", { class: "badge", text: "ready" })
          : el("button", {
              class: "btn small primary",
              type: "button",
              text: "Download",
              onclick: () => downloadFamily(f.family),
            }),
      ]);
      box.appendChild(row);
    }
  }

  async function downloadFamily(family) {
    try {
      busy(true, `Downloading ${family}…`);
      const res = await api("/api/fonts/download", { method: "POST", body: { family, weights: [400, 700] } });
      await loadFonts();
      $("#fontFamily").value = family;
      toast(`${family} ready${res.cached ? " (cached)" : ""}.`, "ok");
      const warns = res.warnings || [];
      status(`Font ${family} ready.` + (warns.length ? " " + warns.join(" ") : ""));
    } catch (e) {
      toast("Font download failed: " + e.message, "bad");
    } finally {
      busy(false);
    }
  }

  /* --------------------------------------------------------------- upload */
  async function uploadFiles(files) {
    if (!files || !files.length) return;
    if (!state.doc && state.docs.length === 0) {
      // first upload
    }
    for (const file of Array.from(files)) {
      try {
        const form = new FormData();
        form.append("file", file);
        const langs = Array.from($("#langSelect").selectedOptions).map((o) => o.value);
        form.append("settings", JSON.stringify({
          langs: langs.length ? langs : ["eng"],
          psm: Number($("#psmSelect").value),
          dpi: Number($("#dpiInput").value),
          enhance: $("#enhanceCheck").checked,
          min_conf: Number($("#minConf").value),
          preprocess: { denoise: $("#denoiseCheck").checked, auto_contrast: true },
        }));
        busy(true, `Uploading ${file.name} · OCR running…`);
        const res = await api("/api/upload", { method: "POST", form });
        state.docs.push(res.doc);
        setDoc(res.doc);
        toast(`${res.doc.filename} loaded — ${res.doc.blocks.length} text blocks.`, "ok");
      } catch (e) {
        toast(`${file.name}: ${e.message}`, "bad");
      } finally {
        busy(false);
      }
    }
  }

  function setDoc(doc) {
    state.doc = doc;
    state.page = 0;
    state.zoom = 1;
    state.fitMode = true;
    state.ops.clear();
    state.boxes.clear();
    state.history = [];
    state.future = [];
    state.selected.clear();
    state.originals = {};
    state.lastRender = null;
    state.matches = [];
    state.selectedMatches.clear();
    if (state.ocrMode) setOcrMode(false);
    renderDocList();
    renderPageList();
    updateTop();
    showPagePreview().then(() => drawOverlay());
    renderInspector();
    renderMatches();
    status(`${doc.filename} · ${doc.kind.replace("_", " ")} · ${doc.blocks.length} blocks` +
      (doc.warnings && doc.warnings.length ? " · " + doc.warnings[0] : ""));
  }

  function renderDocList() {
    const box = $("#docList");
    box.innerHTML = "";
    $("#docCount").textContent = state.docs.length;
    if (!state.docs.length) {
      box.appendChild(el("div", { class: "empty small", text: "Upload a file to begin." }));
      return;
    }
    for (const doc of state.docs) {
      const btn = el("button", {
        type: "button",
        class: "doc-item" + (state.doc && state.doc.id === doc.id ? " active" : ""),
        onclick: () => setDoc(doc),
      }, [
        el("span", { class: "fn", text: doc.filename }),
        el("span", { class: "fs", text: `${doc.page_count} page(s) · ${doc.block_count ?? doc.blocks.length} blocks` }),
      ]);
      box.appendChild(btn);
    }
  }

  function renderPageList() {
    const box = $("#pageList");
    box.innerHTML = "";
    $("#pageCount").textContent = state.doc ? state.doc.page_count : 0;
    if (!state.doc) return;
    for (let i = 0; i < state.doc.page_count; i++) {
      const btn = el("button", {
        type: "button",
        class: "page-chip" + (i === state.page ? " active" : ""),
        text: String(i + 1),
        onclick: () => goPage(i),
      });
      box.appendChild(btn);
    }
  }

  function goPage(page) {
    if (!state.doc) return;
    state.page = Math.max(0, Math.min(page, state.doc.page_count - 1));
    renderPageList();
    updateTop();
    showPagePreview().then(() => drawOverlay());
    renderMatches();
  }

  /* --------------------------------------------------------------- canvas */
  async function showPagePreview() {
    if (!state.doc) return;
    const info = pageInfo();
    const key = `${state.doc.id}:${state.page}`;
    let src = null;
    if (state.lastRender && !state.compare) {
      const pv = state.lastRender.previews.find((p) => p.page === state.page);
      if (pv && pv.after) src = pv.after;
    }
    if (!src) {
      if (!state.originals[key]) {
        const res = await api(`/api/docs/${state.doc.id}/preview?page=${state.page}`);
        state.originals[key] = res.image;
      }
      src = state.originals[key];
    }
    const img = $("#pageImg");
    img.onload = () => { layoutCanvas(); drawOverlay(); };
    img.src = src;
    $("#emptyState").hidden = true;
    $("#canvasInner").hidden = false;
    $("#modeBadge").textContent = state.compare ? "Original" : (state.doc.native ? "Native PDF" : "Raster / OCR");
    $("#zoomLabel").textContent = Math.round(state.zoom * 100) + "%";
    $("#pageLabel").textContent = `Page ${state.page + 1} / ${state.doc.page_count}`;
    if (img.complete) { layoutCanvas(); drawOverlay(); }
    return info;
  }

  function layoutCanvas() {
    if (!state.doc) return;
    const info = pageInfo();
    const wrap = $("#canvasWrap");
    const availW = Math.max(120, wrap.clientWidth - 48);
    const availH = Math.max(120, wrap.clientHeight - 48);
    if (state.fitMode) {
      state.zoom = Math.max(0.08, Math.min(availW / info.width, availH / info.height, 1.6));
    }
    const inner = $("#canvasInner");
    inner.style.width = Math.round(info.width * state.zoom) + "px";
    inner.style.height = Math.round(info.height * state.zoom) + "px";
    const svg = $("#overlay");
    svg.setAttribute("viewBox", `0 0 ${info.width} ${info.height}`);
    $("#zoomLabel").textContent = Math.round(state.zoom * 100) + "%";
  }

  function setZoom(z) {
    state.fitMode = false;
    state.zoom = Math.max(0.08, Math.min(4, z));
    layoutCanvas();
    drawOverlay();
  }

  function drawOverlay() {
    if (!state.doc) return;
    const svg = $("#overlay");
    const info = pageInfo();
    svg.setAttribute("viewBox", `0 0 ${info.width} ${info.height}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const blocks = state.doc.blocks.filter((b) => b.page === state.page);
    for (const block of blocks) {
      const [x0, y0, x1, y1] = blockBox(block);
      const key = "block:" + block.id;
      const op = state.ops.get(key);
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", x0);
      rect.setAttribute("y", y0);
      rect.setAttribute("width", Math.max(1, x1 - x0));
      rect.setAttribute("height", Math.max(1, y1 - y0));
      rect.setAttribute("rx", Math.max(1, 2 / Math.max(0.2, state.zoom)));
      rect.setAttribute("class", "block-rect" +
        (state.selected.has(block.id) ? " selected" : "") +
        (op && op.remove ? " removed" : "") +
        (op && !op.remove ? " edited" : ""));
      rect.dataset.id = block.id;
      rect.addEventListener("click", (e) => {
        e.stopPropagation();
        if (suppressClick) { suppressClick = false; return; }
        selectBlock(block.id, e.shiftKey || e.ctrlKey || e.metaKey);
      });
      rect.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        selectBlock(block.id, false);
        $("#blockText").focus();
      });
      rect.addEventListener("pointerdown", (e) => startMove(e, block));
      const title = document.createElementNS(NS, "title");
      title.textContent = block.text.slice(0, 240);
      rect.appendChild(title);
      svg.appendChild(rect);
    }
    for (const m of state.matches.filter((m) => m.page === state.page)) {
      const [x0, y0, x1, y1] = m.bbox;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", x0 - 1);
      rect.setAttribute("y", y0 - 1);
      rect.setAttribute("width", Math.max(2, x1 - x0 + 2));
      rect.setAttribute("height", Math.max(2, y1 - y0 + 2));
      rect.setAttribute("class", "match-rect" + (state.selectedMatches.has(m.id) ? " selected" : ""));
      rect.dataset.id = m.id;
      rect.addEventListener("click", (e) => {
        e.stopPropagation();
        state.selectedMatches.add(m.id);
        switchTab("find");
        renderMatches();
        drawOverlay();
      });
      const title = document.createElementNS(NS, "title");
      title.textContent = m.context || m.text;
      rect.appendChild(title);
      svg.appendChild(rect);
    }
    const single = selectedBlockId();
    const selected = single ? blocks.find((b) => b.id === single) : null;
    if (selected) drawResizeFrame(svg, selected, blockBox(selected));
    svg.onclick = () => {
      if (state.ocrMode) return;
      state.selected.clear(); renderInspector(); drawOverlay(); updateTop();
    };
  }

  function pointerToPage(svg, e) {
    const rect = svg.getBoundingClientRect();
    const info = pageInfo();
    return {
      x: (e.clientX - rect.left) / Math.max(1, rect.width) * info.width,
      y: (e.clientY - rect.top) / Math.max(1, rect.height) * info.height,
    };
  }

  /* ---------------------------------------------------------- move block */
  function startMove(e, block) {
    if (state.busy || state.ocrMode || !state.doc || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const svg = $("#overlay");
    const info = pageInfo();
    const start = pointerToPage(svg, e);
    const startBox = blockBox(block);
    const w = Math.max(1, startBox[2] - startBox[0]);
    const h = Math.max(1, startBox[3] - startBox[1]);
    const threshold = 3 / Math.max(0.2, state.zoom);
    let moved = false;
    let pushed = false;
    let current = startBox.slice();
    suppressClick = false;
    const move = (ev) => {
      const p = pointerToPage(svg, ev);
      const dx = p.x - start.x;
      const dy = p.y - start.y;
      if (!moved && Math.abs(dx) < threshold && Math.abs(dy) < threshold) return;
      if (!moved) {
        moved = true;
        document.body.classList.add("moving");
      }
      if (!pushed) {
        pushHistory();
        pushed = true;
      }
      const x0 = Math.max(0, Math.min(info.width - w, startBox[0] + dx));
      const y0 = Math.max(0, Math.min(info.height - h, startBox[1] + dy));
      const box = [x0, y0, x0 + w, y0 + h];
      if (boxesEqual(box, current)) return;
      current = box;
      setBlockBox(block.id, box);
      updateResizeVisual(block, box);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("moving");
      if (!moved) return;
      suppressClick = true;
      state.selected.clear();
      state.selected.add(block.id);
      const box = blockBox(block);
      const existing = state.ops.get("block:" + block.id);
      const op = existing
        ? { ...existing }
        : { block_id: block.id, text: block.text, style: block.style || {} };
      if (state.boxes.has(block.id)) op.bbox = box.slice();
      else delete op.bbox;
      state.ops.set("block:" + block.id, op);
      renderInspector();
      drawOverlay();
      updateTop();
      renderNow();
      status(`Block moved to ${Math.round(box[0])}, ${Math.round(box[1])}.`, "ok");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
  }

  function moveSelected(dx, dy) {
    const id = selectedBlockId();
    if (!id) return;
    const block = blockById(id);
    if (!block) return;
    const info = pageInfo();
    const box = blockBox(block);
    const w = box[2] - box[0];
    const h = box[3] - box[1];
    const x0 = Math.max(0, Math.min(info.width - w, box[0] + dx));
    const y0 = Math.max(0, Math.min(info.height - h, box[1] + dy));
    if (Math.abs(x0 - box[0]) < 0.01 && Math.abs(y0 - box[1]) < 0.01) return;
    pushHistory();
    const next = [x0, y0, x0 + w, y0 + h];
    setBlockBox(id, next);
    const existing = state.ops.get("block:" + id);
    const op = existing
      ? { ...existing }
      : { block_id: id, text: block.text, style: block.style || {} };
    op.bbox = next.slice();
    state.ops.set("block:" + id, op);
    renderInspector();
    drawOverlay();
    updateTop();
    scheduleRender(180);
  }

  function scheduleRender(delay = 200) {
    clearTimeout(deferredRender);
    deferredRender = setTimeout(() => {
      deferredRender = null;
      renderNow();
    }, delay);
  }

  /* ------------------------------------------------------------- area OCR */
  function setOcrMode(on) {
    state.ocrMode = !!on;
    const btn = $("#btnOcrArea");
    const svg = $("#overlay");
    if (btn) {
      btn.classList.toggle("primary", state.ocrMode);
      btn.textContent = state.ocrMode ? "OCR area… (Esc)" : "OCR area";
    }
    if (svg) svg.classList.toggle("ocr-mode", state.ocrMode);
    if (!state.ocrMode) {
      const marquee = svg && svg.querySelector("rect.ocr-select");
      if (marquee) marquee.remove();
    }
    if (state.ocrMode) {
      status("OCR area: drag a rectangle over the page. Esc to cancel.");
    } else if (state.doc) {
      status(`${state.doc.filename} · ${state.doc.blocks.length} blocks`);
    }
  }

  async function runAreaOcr(box) {
    if (!state.doc) return;
    const langs = Array.from($("#langSelect").selectedOptions).map((o) => o.value);
    try {
      busy(true, "Running OCR on selected area…");
      const res = await api(`/api/docs/${state.doc.id}/ocr-area`, {
        method: "POST",
        body: {
          page: state.page,
          bbox: box,
          langs: langs.length ? langs : ["eng"],
          psm: Number($("#psmSelect").value),
          enhance: $("#enhanceCheck").checked,
          min_conf: Number($("#minConf").value),
          preprocess: { denoise: $("#denoiseCheck").checked, auto_contrast: true },
        },
      });
      if (res.doc) {
        state.doc = mergeCustomBlocks(res.doc);
        state.matches = [];
        state.selectedMatches.clear();
        renderDocList();
        renderMatches();
        updateTop();
      }
      if (res.count) {
        const first = res.blocks && res.blocks.length ? res.blocks[0].id : null;
        state.selected = new Set(first ? [first] : []);
        renderInspector();
        drawOverlay();
        updateTop();
        toast(`${res.count} text block(s) found in the selected area.`, "ok");
        status(`Area OCR: ${res.count} block(s) added on page ${state.page + 1}.`, "ok");
      } else {
        drawOverlay();
        toast("Selected area me koi text nahi mila.", "bad");
        status("Area OCR: no text found in the selection.", "bad");
      }
    } catch (e) {
      toast("Area OCR failed: " + e.message, "bad");
      status("Area OCR failed: " + e.message, "bad");
    } finally {
      busy(false);
    }
  }

  function bindAreaSelection() {
    const svg = $("#overlay");
    svg.addEventListener("pointerdown", (e) => {
      if (!state.ocrMode || state.busy || !state.doc || e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      const info = pageInfo();
      const start = pointerToPage(svg, e);
      const anchor = {
        x: Math.max(0, Math.min(info.width, start.x)),
        y: Math.max(0, Math.min(info.height, start.y)),
      };
      const marquee = document.createElementNS(NS, "rect");
      marquee.setAttribute("class", "ocr-select");
      marquee.setAttribute("x", anchor.x);
      marquee.setAttribute("y", anchor.y);
      marquee.setAttribute("width", 1);
      marquee.setAttribute("height", 1);
      svg.appendChild(marquee);
      let box = null;
      const move = (ev) => {
        const p = pointerToPage(svg, ev);
        const x = Math.max(0, Math.min(info.width, p.x));
        const y = Math.max(0, Math.min(info.height, p.y));
        box = [Math.min(anchor.x, x), Math.min(anchor.y, y), Math.max(anchor.x, x), Math.max(anchor.y, y)];
        marquee.setAttribute("x", box[0]);
        marquee.setAttribute("y", box[1]);
        marquee.setAttribute("width", Math.max(1, box[2] - box[0]));
        marquee.setAttribute("height", Math.max(1, box[3] - box[1]));
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        marquee.remove();
        setOcrMode(false);
        if (box && box[2] - box[0] >= 8 && box[3] - box[1] >= 8) {
          runAreaOcr(box);
        } else {
          toast("Area bahut chhota hai; dobara drag karein.", "bad");
        }
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up, { once: true });
    }, { capture: true });
  }

  function drawResizeFrame(svg, block, box) {
    const [x0, y0, x1, y1] = box;
    const frame = document.createElementNS(NS, "rect");
    frame.setAttribute("x", x0);
    frame.setAttribute("y", y0);
    frame.setAttribute("width", Math.max(1, x1 - x0));
    frame.setAttribute("height", Math.max(1, y1 - y0));
    frame.setAttribute("class", "resize-frame");
    frame.setAttribute("pointer-events", "none");
    svg.appendChild(frame);

    const size = Math.max(3, Math.min(24, 9 / Math.max(0.15, state.zoom)));
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const spots = {
      nw: [x0, y0], n: [cx, y0], ne: [x1, y0], e: [x1, cy],
      se: [x1, y1], s: [cx, y1], sw: [x0, y1], w: [x0, cy],
    };
    for (const [name, [hx, hy]] of Object.entries(spots)) {
      const handle = document.createElementNS(NS, "rect");
      handle.setAttribute("x", hx - size / 2);
      handle.setAttribute("y", hy - size / 2);
      handle.setAttribute("width", size);
      handle.setAttribute("height", size);
      handle.setAttribute("class", "resize-handle handle-" + name);
      handle.dataset.handle = name;
      handle.dataset.block = block.id;
      handle.addEventListener("pointerdown", (e) => startResize(e, block, name));
      handle.addEventListener("click", (e) => e.stopPropagation());
      svg.appendChild(handle);
    }
  }

  function updateResizeVisual(block, box) {
    const svg = $("#overlay");
    const rect = Array.from(svg.querySelectorAll("rect.block-rect")).find((r) => r.dataset.id === block.id);
    if (rect) {
      rect.setAttribute("x", box[0]);
      rect.setAttribute("y", box[1]);
      rect.setAttribute("width", Math.max(1, box[2] - box[0]));
      rect.setAttribute("height", Math.max(1, box[3] - box[1]));
    }
    const frame = svg.querySelector("rect.resize-frame");
    if (frame) {
      frame.setAttribute("x", box[0]);
      frame.setAttribute("y", box[1]);
      frame.setAttribute("width", Math.max(1, box[2] - box[0]));
      frame.setAttribute("height", Math.max(1, box[3] - box[1]));
    }
    const cx = (box[0] + box[2]) / 2;
    const cy = (box[1] + box[3]) / 2;
    const spots = {
      nw: [box[0], box[1]], n: [cx, box[1]], ne: [box[2], box[1]], e: [box[2], cy],
      se: [box[2], box[3]], s: [cx, box[3]], sw: [box[0], box[3]], w: [box[0], cy],
    };
    svg.querySelectorAll("rect.resize-handle").forEach((handle) => {
      const spot = spots[handle.dataset.handle];
      if (!spot) return;
      const size = Number(handle.getAttribute("width")) || 8;
      handle.setAttribute("x", spot[0] - size / 2);
      handle.setAttribute("y", spot[1] - size / 2);
    });
    setBlockMeta(block, box);
  }

  function startResize(e, block, handle) {
    if (state.busy || !state.doc) return;
    e.preventDefault();
    e.stopPropagation();
    const svg = $("#overlay");
    const info = pageInfo();
    const startBox = blockBox(block);
    const start = pointerToPage(svg, e);
    const op = state.ops.get("block:" + block.id);
    const textOp = !!(op && !op.remove && op.text && op.text.trim());
    const startScale = textOp ? Number((op.style && op.style.size_scale) || 1) : 1;
    const nativeSize = textOp && state.doc.native
      ? Number((op.style && op.style.size) || (block.style && block.style.size) || 0)
      : 0;
    const startHeight = Math.max(1, startBox[3] - startBox[1]);
    let pushed = false;
    let current = startBox.slice();
    document.body.classList.add("resizing");
    const move = (ev) => {
      const p = pointerToPage(svg, ev);
      const dx = p.x - start.x;
      const dy = p.y - start.y;
      const min = Math.max(4, 8 / Math.max(0.2, state.zoom));
      let [x0, y0, x1, y1] = startBox;
      if (handle.includes("w")) x0 = Math.max(0, Math.min(startBox[0] + dx, startBox[2] - min));
      if (handle.includes("e")) x1 = Math.min(info.width, Math.max(startBox[2] + dx, startBox[0] + min));
      if (handle.includes("n")) y0 = Math.max(0, Math.min(startBox[1] + dy, startBox[3] - min));
      if (handle.includes("s")) y1 = Math.min(info.height, Math.max(startBox[3] + dy, startBox[1] + min));
      const box = [x0, y0, x1, y1];
      if (boxesEqual(box, current)) return;
      if (!pushed) {
        pushHistory();
        pushed = true;
      }
      current = box;
      setBlockBox(block.id, box);
      if (textOp && (handle.includes("n") || handle.includes("s"))) {
        // Vertical resize scales the replacement text too, so a longer
        // replacement can be made visibly larger/smaller without the slider.
        // Remember the pre-drag size once so "Reset box" can restore it
        // instead of compounding scale on every drag/reset cycle.
        if (!op._size_base) {
          op._size_base = state.doc.native ? { size: nativeSize || null } : { scale: startScale };
        }
        const ratio = (y1 - y0) / startHeight;
        if (state.doc.native) {
          if (nativeSize > 0) op.style = { ...(op.style || {}), size: Math.max(4, Math.min(300, nativeSize * ratio)) };
        } else {
          op.style = { ...(op.style || {}), size_scale: Math.max(0.2, Math.min(5, startScale * ratio)) };
        }
      }
      updateResizeVisual(block, box);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("resizing");
      if (!pushed) return;
      renderInspector();
      drawOverlay();
      updateTop();
      const op = state.ops.get("block:" + block.id);
      if (op) {
        op.bbox = blockBox(block);
        renderNow();
      } else {
        status("Annotation resized. Apply text to use the new region.", "ok");
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
  }

  function selectBlock(id, additive) {
    if (additive) {
      if (state.selected.has(id)) state.selected.delete(id);
      else state.selected.add(id);
    } else {
      state.selected.clear();
      state.selected.add(id);
    }
    switchTab("edit");
    renderInspector();
    drawOverlay();
    updateTop();
  }

  /* ------------------------------------------------------------- inspector */
  function setBlockMeta(block, box) {
    const conf = block.confidence != null ? Math.round(block.confidence * 100) + "%" : "—";
    const size = `${Math.round(box[2] - box[0])} × ${Math.round(box[3] - box[1])} px`;
    const note = state.doc && state.doc.native ? " · PDF geometry" : "";
    $("#blockMeta").textContent =
      `Page ${block.page + 1} · ${conf} confidence · bbox ${box.map((v) => Math.round(v)).join(", ")} · ${size}${note}`;
  }

  function renderInspector() {
    const one = selectedBlockId();
    const count = state.selected.size;
    if (count === 0 || !one) {
      $("#inspectorEmpty").hidden = false;
      $("#inspectorEmpty").textContent = count > 1
        ? `${count} blocks selected. Use the Remove tab for bulk removal.`
        : "Select a text block on the page.";
      $("#blockEditor").hidden = true;
      return;
    }
    const block = blockById(one);
    if (!block) return;
    const op = state.ops.get("block:" + one);
    const style = (op && op.style) || block.style || {};
    $("#inspectorEmpty").hidden = true;
    $("#blockEditor").hidden = false;
    $("#blockSource").textContent = block.synthetic ? "Custom" : (block.source === "pdf" ? "PDF text" : "OCR");
    const box = blockBox(block);
    setBlockMeta(block, box);
    const resetBox = $("#resetBlockBox");
    if (resetBox) resetBox.disabled = !state.boxes.has(block.id);
    $("#blockText").value = op && op.text !== undefined ? op.text : block.text;
    $("#fontFamily").value = style.font_family || style.font || "";
    if ($("#fontFamily").value !== (style.font_family || style.font || "")) {
      // The family may not be in the current list yet; add it.
      const fam = style.font_family || style.font;
      if (fam) $("#fontFamily").appendChild(new Option(fam, fam));
      $("#fontFamily").value = fam || "";
    }
    $("#fontWeight").value = style.weight ? String(style.weight) : "";
    $("#fontItalic").checked = !!style.italic;
    $("#fontSize").value = style.size ? String(style.size) : "";
    $("#autoColor").checked = !style.color;
    if (style.color) $("#textColor").value = style.color;
    $("#textColor").disabled = !style.color;
    $("#alignSelect").value = style.align || "left";
    $("#sizeScale").value = style.size_scale || 1;
    $("#sizeScaleValue").textContent = Number($("#sizeScale").value).toFixed(2) + "×";
    $$("#colorSwatches .swatch").forEach((b) => {
      b.classList.toggle("active", !!style.color &&
        String(style.color).toLowerCase() === String(b.dataset.color).toLowerCase());
    });
    $("#nudgeX").value = style.nudge_x || 0;
    $("#nudgeY").value = style.nudge_y || 0;
    $("#caseMatch").checked = style.case_match !== false;
    $("#weightMatch").checked = style.weight_match !== false;
    $("#stretchMatch").checked = style.stretch_match !== false;
    const native = state.doc.native;
    $("#fontSize").disabled = !native;
    $("#fontSize").value = native && style.size ? style.size : "";
    $("#fontResult").textContent = "";
  }

  function buildStyle() {
    const style = {};
    if ($("#fontFamily").value) style.font_family = $("#fontFamily").value;
    if ($("#fontWeight").value) style.weight = Number($("#fontWeight").value);
    if ($("#fontItalic").checked) style.italic = true;
    if (!$("#autoColor").checked) style.color = $("#textColor").value;
    style.align = $("#alignSelect").value;
    const scale = Number($("#sizeScale").value);
    if (Math.abs(scale - 1) > 1e-6) style.size_scale = scale;
    const nx = Number($("#nudgeX").value || 0);
    const ny = Number($("#nudgeY").value || 0);
    if (nx) style.nudge_x = nx;
    if (ny) style.nudge_y = ny;
    style.case_match = $("#caseMatch").checked;
    style.weight_match = $("#weightMatch").checked;
    style.stretch_match = $("#stretchMatch").checked;
    if (state.doc && state.doc.native && $("#fontSize").value) {
      style.size = Number($("#fontSize").value);
    }
    return style;
  }

  function applyBlock() {
    const id = selectedBlockId();
    if (!id) return;
    const block = blockById(id);
    const text = $("#blockText").value;
    if (block && block.synthetic && !text.trim()) {
      removeBlock();
      return;
    }
    const op = { block_id: id, text, remove: text.trim() === "", style: buildStyle() };
    if (block && block.synthetic) {
      // A custom box has no source block on the server; carry its page and
      // destination geometry so the renderer can draw it from scratch.
      op.insert = true;
      op.page = block.page;
      op.bbox = blockBox(block).slice();
      block.text = text;
    } else {
      const box = state.boxes.get(id);
      if (box) op.bbox = box.slice();
    }
    commitOp("block:" + id, op);
  }

  function nextCustomId() {
    const ids = new Set(state.doc ? state.doc.blocks.map((b) => b.id) : []);
    let n = 1;
    while (ids.has("ins-" + n)) n++;
    return "ins-" + n;
  }

  function mergeCustomBlocks(next) {
    if (!next) return next;
    const custom = (state.doc && state.doc.blocks ? state.doc.blocks : []).filter((b) => b.synthetic);
    if (!custom.length) return next;
    const ids = new Set((next.blocks || []).map((b) => b.id));
    next.blocks = [...(next.blocks || []), ...custom.filter((b) => !ids.has(b.id))];
    return next;
  }

  function addCustomText() {
    if (!state.doc) { toast("Pehle document upload karein."); return; }
    const input = $("#newTextValue");
    const text = (input.value || "").trim() || "New text";
    const info = pageInfo();
    const w = Math.min(info.width * 0.7, Math.max(220, text.length * 16 + 60));
    const h = Math.max(36, Math.min(90, info.height * 0.06));
    const box = [
      +((info.width - w) / 2).toFixed(2),
      +((info.height - h) / 2).toFixed(2),
      +((info.width + w) / 2).toFixed(2),
      +((info.height + h) / 2).toFixed(2),
    ];
    const id = nextCustomId();
    const block = {
      id, page: state.page, text, bbox: box, style: {},
      confidence: null, source: state.doc.native ? "pdf" : "ocr", synthetic: true,
    };
    pushHistory();
    state.doc.blocks.push(block);
    state.ops.set("block:" + id, {
      block_id: id, insert: true, page: state.page, text,
      bbox: box.slice(), style: {},
    });
    state.boxes.delete(id);
    input.value = "";
    state.selected = new Set([id]);
    renderInspector();
    drawOverlay();
    updateTop();
    renderNow();
    status("Custom text box added — drag it anywhere or type in the inspector.", "ok");
  }

  function duplicateBlock() {
    const id = selectedBlockId();
    if (!id) { toast("Pehle koi text box select karein."); return; }
    const src = blockById(id);
    if (!src) return;
    const op = state.ops.get("block:" + id);
    const text = op && op.text !== undefined ? op.text : src.text;
    const style = JSON.parse(JSON.stringify((op && op.style) || src.style || {}));
    const current = blockBox(src);
    const w = current[2] - current[0];
    const h = current[3] - current[1];
    const info = pageInfo();
    const x0 = Math.max(0, Math.min(info.width - w, current[0] + 16));
    const y0 = Math.max(0, Math.min(info.height - h, current[1] + 16));
    const box = [+x0.toFixed(2), +y0.toFixed(2), +(x0 + w).toFixed(2), +(y0 + h).toFixed(2)];
    const newId = nextCustomId();
    const block = { ...src, id: newId, text, bbox: box, style, synthetic: true, confidence: null };
    pushHistory();
    state.doc.blocks.push(block);
    state.ops.set("block:" + newId, {
      block_id: newId, insert: true, page: src.page, text,
      bbox: box.slice(), style: JSON.parse(JSON.stringify(style)),
    });
    state.boxes.delete(newId);
    state.selected = new Set([newId]);
    renderInspector();
    drawOverlay();
    updateTop();
    renderNow();
    status("Text box duplicated.", "ok");
  }

  /* ------------------------------------------------------- quick styling */
  function liveApplySoon() {
    if (!selectedBlockId()) return;
    clearTimeout(styleTimer);
    styleTimer = setTimeout(() => {
      styleTimer = null;
      if (selectedBlockId()) applyBlock();
    }, 280);
  }

  function adjustWeight(delta) {
    const sel = $("#fontWeight");
    let weight = sel.value ? Number(sel.value) : 400;
    weight = Math.max(100, Math.min(900, weight + delta));
    if (!Array.from(sel.options).some((o) => Number(o.value) === weight)) {
      sel.appendChild(new Option(String(weight), String(weight)));
    }
    sel.value = String(weight);
    liveApplySoon();
  }

  function adjustSize(delta) {
    if (state.doc && state.doc.native) {
      const input = $("#fontSize");
      const block = blockById(selectedBlockId() || "");
      const base = Number(input.value) || Number(block && block.style && block.style.size) || 12;
      input.value = Math.max(4, Math.min(300, base + delta)).toFixed(1);
    } else {
      const slider = $("#sizeScale");
      const next = Math.max(0.5, Math.min(2, Number(slider.value || 1) + delta));
      slider.value = next.toFixed(2);
      $("#sizeScaleValue").textContent = next.toFixed(2) + "×";
    }
    liveApplySoon();
  }

  function applyColor(hex) {
    $("#autoColor").checked = false;
    const input = $("#textColor");
    input.disabled = false;
    input.value = hex;
    liveApplySoon();
  }

  function deleteCustomBlock(id, { history = true } = {}) {
    if (!id || !state.doc) return;
    if (history) pushHistory();
    state.doc.blocks = state.doc.blocks.filter((b) => b.id !== id);
    state.ops.delete("block:" + id);
    state.boxes.delete(id);
    state.selected.delete(id);
    renderInspector();
    drawOverlay();
    updateTop();
    renderNow();
  }

  function removeBlock() {
    const id = selectedBlockId();
    if (!id) return;
    const block = blockById(id);
    if (block && block.synthetic) {
      deleteCustomBlock(id);
      status("Text box deleted.", "ok");
      return;
    }
    const op = { block_id: id, remove: true };
    const box = state.boxes.get(id);
    if (box) op.bbox = box.slice();
    commitOp("block:" + id, op);
    status("Text box deleted — Undo to restore.", "ok");
  }

  function revertBlock() {
    const id = selectedBlockId();
    if (!id) return;
    const block = blockById(id);
    if (block && block.synthetic) {
      deleteCustomBlock(id);
      status("Custom text box removed.", "ok");
      return;
    }
    pushHistory();
    state.ops.delete("block:" + id);
    renderNow();
  }

  function resetBlockBox() {
    const id = selectedBlockId();
    if (!id || !state.boxes.has(id)) return;
    pushHistory();
    state.boxes.delete(id);
    const block = blockById(id);
    const op = state.ops.get("block:" + id);
    if (op) {
      // Inserted text stores its only position in `bbox`, so reset returns it
      // to the box it was created in instead of deleting the geometry.
      if (op.insert && block) op.bbox = block.bbox.slice();
      else delete op.bbox;
      const base = op._size_base;
      if (base) {
        // Restore the text size captured before the first vertical drag so
        // reset does not leave an ever-growing scale behind.
        if (state.doc.native) {
          if (base.size) op.style = { ...(op.style || {}), size: base.size };
          else if (op.style) delete op.style.size;
        } else if (op.style) {
          if (base.scale && Math.abs(base.scale - 1) > 1e-6) op.style.size_scale = base.scale;
          else delete op.style.size_scale;
        }
        delete op._size_base;
      }
    }
    renderInspector();
    drawOverlay();
    updateTop();
    if (op) renderNow();
  }

  async function detectFont() {
    const id = selectedBlockId();
    if (!id) return;
    try {
      busy(true, "Recognising font…");
      const res = await api(`/api/docs/${state.doc.id}/recognize`, { method: "POST", body: { block_id: id } });
      const box = $("#fontResult");
      box.innerHTML = "";
      if (!res.candidates || !res.candidates.length) {
        box.textContent = res.note || "No font candidate found.";
        return;
      }
      box.appendChild(el("div", { class: "hint", text: res.note || "" }));
      for (const cand of res.candidates) {
        const row = el("div", { class: "candidate" }, [
          el("div", {}, [
            el("b", { text: cand.font_family || cand.family }),
            el("span", { class: "score", text: ` ${cand.style || ""} · score ${Math.round((cand.score || 0) * 100)}% · ${cand.source}` }),
          ]),
          el("button", {
            class: "btn small ghost",
            type: "button",
            text: "Use",
            onclick: () => {
              const family = cand.family || cand.label;
              if (!Array.from($("#fontFamily").options).some((o) => o.value === family)) {
                $("#fontFamily").appendChild(new Option(family, family));
              }
              $("#fontFamily").value = family;
              toast(`Font set to ${family}. Apply the block to render.`, "ok");
            },
          }),
        ]);
        box.appendChild(row);
      }
      status(res.method === "embedded-pdf-font" ? "Embedded PDF font identified." : "Font candidates ranked from glyph shape.");
    } catch (e) {
      toast("Font detection failed: " + e.message, "bad");
    } finally {
      busy(false);
    }
  }

  /* --------------------------------------------------------------- history */
  function snapshotOps() {
    return {
      ops: Array.from(state.ops.entries()).map(([k, v]) => [k, JSON.parse(JSON.stringify(v))]),
      boxes: Array.from(state.boxes.entries()).map(([k, v]) => [k, v.slice()]),
      // Custom boxes exist only in the editor, so history must carry them too.
      custom: state.doc
        ? state.doc.blocks.filter((b) => b.synthetic).map((b) => JSON.parse(JSON.stringify(b)))
        : [],
    };
  }

  function pushHistory() {
    state.history.push(snapshotOps());
    if (state.history.length > 40) state.history.shift();
    state.future = [];
    updateTop();
  }

  function restoreOps(snapshot) {
    state.ops = new Map(snapshot.ops || []);
    state.boxes = new Map((snapshot.boxes || []).map(([k, v]) => [k, v.slice()]));
    if (state.doc) {
      const originals = state.doc.blocks.filter((b) => !b.synthetic);
      const seen = new Set(originals.map((b) => b.id));
      state.doc.blocks = originals.concat(
        (snapshot.custom || []).filter((b) => !seen.has(b.id)).map((b) => JSON.parse(JSON.stringify(b))),
      );
    }
    state.selected.clear();
    renderInspector();
    updateTop();
    renderNow();
    drawOverlay();
  }

  function undo() {
    if (!state.history.length) return;
    state.future.push(snapshotOps());
    restoreOps(state.history.pop());
    updateTop();
  }

  function redo() {
    if (!state.future.length) return;
    state.history.push(snapshotOps());
    restoreOps(state.future.pop());
    updateTop();
  }

  function commitOp(key, op) {
    pushHistory();
    state.ops.set(key, op);
    updateTop();
    renderInspector();
    drawOverlay();
    renderNow();
  }

  function renderSettings() {
    return {
      output_format: state.doc && state.doc.native ? "pdf" : $("#exportFormat").value,
      quality: 92,
      method: $("#eraseMethod").value,
      watermark_mode: $("#watermarkMode").value,
      watermark_sensitivity: Number($("#watermarkSens").value),
      align: "left",
      case_match: true,
      weight_match: true,
      stretch_match: true,
      pixel_lock: true,
    };
  }

  async function renderNow() {
    if (!state.doc) return null;
    if (!state.ops.size) {
      state.lastRender = null;
      await showPagePreview();
      drawOverlay();
      status("No edits. Showing original.");
      return null;
    }
    const seq = ++state.renderSeq;
    try {
      busy(true, "Rendering edits…");
      const res = await api(`/api/docs/${state.doc.id}/render`, {
        method: "POST",
        body: { ops: Array.from(state.ops.values()), settings: renderSettings() },
      });
      if (seq !== state.renderSeq) return null;
      state.lastRender = res;
      let count = 0;
      if (res.styles && res.styles.length) count = res.styles.length;
      else if (res.native_report) {
        count = (res.native_report.replaced || 0) + (res.native_report.inserted || 0) + (res.native_report.removed || 0);
      }
      const msg = res.changed
        ? `Rendered ${count} edit(s).`
        : "No visible pixel change.";
      status(msg + (res.warnings && res.warnings.length ? " · " + res.warnings[0] : ""), res.changed ? "ok" : "bad");
      await showPagePreview();
      drawOverlay();
      return res;
    } catch (e) {
      status("Render failed: " + e.message, "bad");
      toast("Render failed: " + e.message, "bad");
      return null;
    } finally {
      busy(false);
    }
  }

  /* ------------------------------------------------------------------ find */
  async function findMatches() {
    if (!state.doc) return;
    const query = $("#findQuery").value;
    if (!query) { toast("Find text likhein."); return; }
    try {
      busy(true, "Searching text…");
      const res = await api(`/api/docs/${state.doc.id}/find`, {
        method: "POST",
        body: {
          query,
          case_sensitive: $("#findCase").checked,
          whole_word: $("#findWhole").checked,
          regex: $("#findRegex").checked,
          fuzzy: $("#findFuzzyOn").checked ? Number($("#findFuzzy").value) : 0,
        },
      });
      state.matches = res.matches || [];
      state.selectedMatches = new Set(state.matches.map((m) => m.id));
      renderMatches();
      drawOverlay();
      status(`${state.matches.length} match(es) found.`, state.matches.length ? "ok" : "bad");
    } catch (e) {
      toast("Find failed: " + e.message, "bad");
    } finally {
      busy(false);
    }
  }

  function renderMatches() {
    const box = $("#matchList");
    box.innerHTML = "";
    $("#findSummary").textContent = state.matches.length
      ? `${state.matches.length} match(es) · ${state.selectedMatches.size} selected`
      : "No matches yet.";
    $("#btnReplaceSelected").disabled = state.selectedMatches.size === 0;
    $("#btnReplaceAll").disabled = state.matches.length === 0;
    for (const m of state.matches) {
      const cb = el("input", { type: "checkbox" });
      cb.checked = state.selectedMatches.has(m.id);
      cb.addEventListener("change", () => {
        if (cb.checked) state.selectedMatches.add(m.id);
        else state.selectedMatches.delete(m.id);
        renderMatches();
        drawOverlay();
      });
      const ctx = el("div", { class: "ctx" });
      const raw = m.context || m.text;
      const start = raw.indexOf("[[");
      const end = raw.indexOf("]]");
      if (start >= 0 && end > start) {
        ctx.appendChild(document.createTextNode(raw.slice(0, start)));
        const mark = el("mark", { text: raw.slice(start + 2, end) });
        ctx.appendChild(mark);
        ctx.appendChild(document.createTextNode(raw.slice(end + 2)));
      } else {
        ctx.textContent = raw;
      }
      const item = el("label", { class: "match-item" }, [
        cb,
        ctx,
        el("span", { class: "pg", text: `p${m.page + 1} · ${Math.round((m.score || 0) * 100)}%` }),
      ]);
      box.appendChild(item);
    }
  }

  function nativeBlockGroups(matches, replacement) {
    const groups = new Map();
    const addEdit = (bid, start, end, text) => {
      const block = blockById(bid);
      if (!block) return;
      if (!groups.has(bid)) groups.set(bid, { block, edits: [] });
      groups.get(bid).edits.push({ start, end, replacement: text });
    };
    for (const m of matches) {
      if (Array.isArray(m.chunk_edits) && m.chunk_edits.length) {
        // A phrase can span several word chunks.  Put the replacement in the
        // first affected chunk and remove the matched remainder from the other
        // chunks, leaving their surrounding text intact.
        m.chunk_edits.forEach((ce, i) => {
          addEdit(ce.block_id, ce.start, ce.end, i === 0 ? replacement : "");
        });
        continue;
      }
      const bid = m.block_id;
      if (!bid) continue;
      const block = blockById(bid);
      if (!block) continue;
      let start = Number.isInteger(m.start) ? m.start : block.text.indexOf(m.text);
      let end = Number.isInteger(m.end) ? m.end : start + m.text.length;
      if (start < 0) continue;
      addEdit(bid, start, end, replacement);
    }
    const out = [];
    for (const [bid, group] of groups) {
      let text = group.block.text;
      const edits = group.edits.sort((a, b) => b.start - a.start);
      for (const e of edits) text = text.slice(0, e.start) + e.replacement + text.slice(e.end);
      const op = { block_id: bid, text, style: group.block.style || {} };
      const box = state.boxes.get(bid);
      if (box) op.bbox = box.slice();
      out.push(op);
    }
    return out;
  }

  function replaceMatches(all) {
    if (!state.doc) return;
    const chosen = all ? state.matches : state.matches.filter((m) => state.selectedMatches.has(m.id));
    if (!chosen.length) { toast("Pehle match select karein."); return; }
    const replacement = $("#replaceQuery").value;
    pushHistory();
    if (state.doc.native) {
      const groups = nativeBlockGroups(chosen, replacement);
      for (const op of groups) {
        const existing = state.ops.get("block:" + op.block_id);
        if (existing && existing.style) op.style = { ...existing.style, ...(op.style || {}) };
        state.ops.set("block:" + op.block_id, op);
      }
    } else {
      for (const m of chosen) {
        const key = `match:${m.page}:${Math.round(m.bbox[0])}:${Math.round(m.bbox[1])}:${m.text}`;
        state.ops.set(key, {
          page: m.page,
          bbox: m.bbox,
          matched_text: m.text,
          context: m.context,
          text: replacement,
          remove: replacement === "",
          style: {},
        });
      }
    }
    updateTop();
    renderInspector();
    drawOverlay();
    renderNow();
    toast(`${chosen.length} occurrence(s) ${replacement ? "replaced" : "removed"}.`, "ok");
  }

  /* ----------------------------------------------------------------- fonts */
  function switchTab(tab) {
    state.tab = tab;
    $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab));
    if (tab === "fonts" && !state.fonts.catalog.length) loadFonts();
  }

  /* ---------------------------------------------------------------- export */
  async function exportNow() {
    if (!state.doc) { toast("Pehle document upload karein."); return; }
    try {
      let downloads = [];
      if (state.ops.size) {
        const res = await renderNow();
        downloads = (res && res.downloads) || [];
      }
      const txt = await api(`/api/docs/${state.doc.id}/export-text`, { method: "POST" });
      if (txt.download && !downloads.some((d) => d.filename === txt.download.filename && d.label === txt.download.label)) {
        downloads.push(txt.download);
      }
      if (!downloads.length) { toast("Koi export file nahi bani."); return; }
      const body = $("#modalBody");
      body.innerHTML = "";
      for (const d of downloads) {
        body.appendChild(el("div", { class: "dl-row" }, [
          el("div", {}, [
            el("div", { text: d.label || d.filename }),
            el("div", { class: "size", text: d.filename + " · " + fmtBytes(d.size) }),
          ]),
          el("a", { href: `/api/download/${d.id}`, text: "Download", download: d.filename }),
        ]));
      }
      $("#modalTitle").textContent = "Exports";
      $("#modal").hidden = false;
    } catch (e) {
      toast("Export failed: " + e.message, "bad");
    }
  }

  /* ------------------------------------------------------------------- UI */
  function updateTop() {
    $("#docTitle").textContent = state.doc ? state.doc.filename : "No document loaded";
    $("#btnUndo").disabled = state.history.length === 0;
    $("#btnRedo").disabled = state.future.length === 0;
    $("#btnCompare").textContent = state.compare ? "Showing original" : "Original";
    $("#btnCompare").classList.toggle("primary", state.compare);
    $("#modeBadge").textContent = state.doc
      ? (state.compare ? "Original" : (state.doc.native ? "Native PDF" : "Raster / OCR"))
      : "No document";
    $("#zoomLabel").textContent = Math.round(state.zoom * 100) + "%";
    if (state.doc) {
      $("#pageLabel").textContent = `Page ${state.page + 1} / ${state.doc.page_count}`;
    } else {
      $("#pageLabel").textContent = "Page 0 / 0";
    }
  }

  function bind() {
    $("#dropzone").addEventListener("click", () => $("#fileInput").click());
    $("#dropzone").addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#fileInput").click(); }
    });
    $("#emptyUpload").addEventListener("click", () => $("#fileInput").click());
    $("#fileInput").addEventListener("change", (e) => {
      uploadFiles(e.target.files);
      e.target.value = "";
    });
    const dz = $("#dropzone");
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

    $$(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
    $("#prevPage").addEventListener("click", () => goPage(state.page - 1));
    $("#nextPage").addEventListener("click", () => goPage(state.page + 1));
    $("#zoomIn").addEventListener("click", () => setZoom(state.zoom * 1.18));
    $("#zoomOut").addEventListener("click", () => setZoom(state.zoom / 1.18));
    $("#zoomFit").addEventListener("click", () => { state.fitMode = true; layoutCanvas(); drawOverlay(); });
    $("#btnOcrArea").addEventListener("click", () => {
      if (!state.doc) { toast("Pehle document upload karein."); return; }
      setOcrMode(!state.ocrMode);
    });
    $("#btnCompare").addEventListener("click", async () => {
      state.compare = !state.compare;
      updateTop();
      await showPagePreview();
      drawOverlay();
    });
    $("#btnUndo").addEventListener("click", undo);
    $("#btnRedo").addEventListener("click", redo);
    $("#btnExport").addEventListener("click", exportNow);
    $("#modalClose").addEventListener("click", () => { $("#modal").hidden = true; });
    $("#modal").addEventListener("click", (e) => { if (e.target === $("#modal")) $("#modal").hidden = true; });

    $("#applyBlock").addEventListener("click", applyBlock);
    $("#removeBlock").addEventListener("click", removeBlock);
    $("#revertBlock").addEventListener("click", revertBlock);
    const addTextBtn = $("#addTextBtn");
    if (addTextBtn) addTextBtn.addEventListener("click", addCustomText);
    const newTextValue = $("#newTextValue");
    if (newTextValue) newTextValue.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); addCustomText(); }
    });
    const duplicateBtn = $("#duplicateBlock");
    if (duplicateBtn) duplicateBtn.addEventListener("click", duplicateBlock);
    const resetBoxBtn = $("#resetBlockBox");
    if (resetBoxBtn) resetBoxBtn.addEventListener("click", resetBlockBox);
    $("#weightDown").addEventListener("click", () => adjustWeight(-100));
    $("#weightUp").addEventListener("click", () => adjustWeight(100));
    $("#sizeDown").addEventListener("click", () => adjustSize(state.doc && state.doc.native ? -1 : -0.05));
    $("#sizeUp").addEventListener("click", () => adjustSize(state.doc && state.doc.native ? 1 : 0.05));
    $$("#colorSwatches .swatch").forEach((b) => {
      b.addEventListener("click", () => applyColor(b.dataset.color));
    });
    ["fontFamily", "fontWeight", "fontItalic", "alignSelect", "autoColor", "textColor",
     "nudgeX", "nudgeY", "caseMatch", "weightMatch", "stretchMatch"].forEach((id) => {
      const node = $("#" + id);
      node.addEventListener("change", () => {
        if (id === "autoColor") $("#textColor").disabled = node.checked;
        liveApplySoon();
      });
    });
    $("#sizeScale").addEventListener("change", liveApplySoon);
    $("#fontSize").addEventListener("change", liveApplySoon);
    $("#detectFont").addEventListener("click", detectFont);
    $("#blockText").addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); applyBlock(); }
    });
    $("#autoColor").addEventListener("change", () => { $("#textColor").disabled = $("#autoColor").checked; });
    $("#sizeScale").addEventListener("input", () => { $("#sizeScaleValue").textContent = Number($("#sizeScale").value).toFixed(2) + "×"; });
    $("#findFuzzy").addEventListener("input", () => { $("#findFuzzyValue").textContent = Number($("#findFuzzy").value).toFixed(2); });
    $("#watermarkSens").addEventListener("input", () => { $("#watermarkSensValue").textContent = Number($("#watermarkSens").value).toFixed(2); });
    $("#minConf").addEventListener("input", () => { $("#minConfValue").textContent = $("#minConf").value; });

    $("#btnFind").addEventListener("click", findMatches);
    $("#findQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") findMatches(); });
    $("#btnSelectAllMatches").addEventListener("click", () => {
      state.selectedMatches = new Set(state.matches.map((m) => m.id));
      renderMatches(); drawOverlay();
    });
    $("#btnReplaceSelected").addEventListener("click", () => replaceMatches(false));
    $("#btnReplaceAll").addEventListener("click", () => replaceMatches(true));

    $("#refreshFonts").addEventListener("click", loadFonts);
    $("#fontSearch").addEventListener("input", renderFontCatalog);
    $("#exportFormat").addEventListener("change", () => { if (state.ops.size) renderNow(); });

    $("#removeSelected").addEventListener("click", () => {
      if (!state.selected.size) { toast("Pehle block select karein."); return; }
      pushHistory();
      for (const id of state.selected) {
        const block = blockById(id);
        if (block && block.synthetic) {
          state.doc.blocks = state.doc.blocks.filter((b) => b.id !== id);
          state.ops.delete("block:" + id);
          state.boxes.delete(id);
          continue;
        }
        const op = { block_id: id, remove: true };
        const box = state.boxes.get(id);
        if (box) op.bbox = box.slice();
        state.ops.set("block:" + id, op);
      }
      state.selected.clear();
      renderInspector(); drawOverlay(); updateTop(); renderNow();
    });
    $("#removePage").addEventListener("click", () => {
      if (!state.doc) return;
      pushHistory();
      const pageBlocks = state.doc.blocks.filter((b) => b.page === state.page);
      for (const b of pageBlocks) {
        if (b.synthetic) {
          state.ops.delete("block:" + b.id);
          state.boxes.delete(b.id);
          continue;
        }
        const op = { block_id: b.id, remove: true };
        const box = state.boxes.get(b.id);
        if (box) op.bbox = box.slice();
        state.ops.set("block:" + b.id, op);
      }
      state.doc.blocks = state.doc.blocks.filter((b) => !(b.page === state.page && b.synthetic));
      renderInspector(); drawOverlay(); updateTop(); renderNow();
    });
    $("#clearAll").addEventListener("click", () => {
      pushHistory();
      state.ops.clear();
      state.boxes.clear();
      state.selected.clear();
      if (state.doc) state.doc.blocks = state.doc.blocks.filter((b) => !b.synthetic);
      renderInspector(); drawOverlay(); updateTop(); renderNow();
    });

    document.addEventListener("keydown", (e) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z" && !isTyping()) {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      } else if (mod && e.key.toLowerCase() === "y" && !isTyping()) {
        e.preventDefault(); redo();
      } else if (mod && e.key.toLowerCase() === "f") {
        e.preventDefault(); switchTab("find"); $("#findQuery").focus();
      } else if (!isTyping() && state.selected.size === 1 &&
                 ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
        const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
        moveSelected(dx, dy);
      } else if ((e.key === "Delete" || e.key === "Backspace") && state.selected.size && !isTyping()) {
        e.preventDefault();
        pushHistory();
        for (const id of state.selected) {
          const block = blockById(id);
          if (block && block.synthetic) {
            state.doc.blocks = state.doc.blocks.filter((b) => b.id !== id);
            state.ops.delete("block:" + id);
            state.boxes.delete(id);
            continue;
          }
          const op = { block_id: id, remove: true };
          const box = state.boxes.get(id);
          if (box) op.bbox = box.slice();
          state.ops.set("block:" + id, op);
        }
        state.selected.clear();
        renderInspector(); drawOverlay(); updateTop(); renderNow();
      } else if (e.key === "Escape") {
        if (state.ocrMode) { setOcrMode(false); return; }
        state.selected.clear(); renderInspector(); drawOverlay(); updateTop();
        $("#modal").hidden = true;
      }
    });

    window.addEventListener("resize", () => { if (state.fitMode) { layoutCanvas(); drawOverlay(); } });
    $("#canvasWrap").addEventListener("wheel", (e) => {
      // Ctrl/Cmd + wheel over the page must zoom only the document image; it
      // must never trigger the browser's page/UI zoom.
      if (!state.doc || !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      setZoom(state.zoom * Math.exp(-e.deltaY * 0.0025));
    }, { passive: false });
    bindAreaSelection();
    $("#pageImg").addEventListener("click", () => { state.selected.clear(); renderInspector(); drawOverlay(); updateTop(); });
  }

  async function init() {
    bind();
    await loadConfig();
    await loadFonts();
    try { await api("/api/session"); } catch (_) { /* cookie is set on first page load */ }
    updateTop();
    status("Ready. Upload a document to begin.");
  }

  init();
})();
