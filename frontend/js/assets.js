// ── The asset step ────────────────────────────────────────────────────────
//
// The image overlay's second way in: instead of a file from the user's own
// disk, one of the ready-made pictures shipped in frontend/assets/, painted
// in a colour the user picks. Two views in one dialog — the shelf of assets,
// then the colour for the one chosen — and the answer is a PNG data URL,
// for the reason mask.js gives for its own: everything downstream of an
// overlay already draws "a picture at a position and a scale", and a
// colour baked into the pixels is just another picture to all of it. A tint
// stored on the overlay record would have had to be taught to the canvas,
// the grid thumbnails, the downloads and the carry-over to other frames,
// and the first one somebody forgot would draw the asset in its own red.
//
// No mask step after it. Assets are made cut out on a transparent field;
// squaring one off would put back the rectangle it was drawn without.

import { el } from "./config.js";

const ASSET_DIR = "assets/";

// The preview's side, in CSS pixels — the mask stage's frame, so the two
// dialogs an overlay can come through show their picture at the same size.
const PREVIEW = 320;

// The Drop shadow switch. Measured against the asset's long side, not in
// pixels, so it is the same shadow on a small asset as on a large one and
// grows with the overlay when the overlay is resized — it is part of the
// picture. Soft and only slightly lowered: modelled on the shadow ring.png
// arrived with (a dark glow about 2.5% of its side), which is what the user
// took to be the look, with a nudge downward to make it a DROP shadow.
const SHADOW_BLUR = 0.025;
const SHADOW_DROP = 0.008;
const SHADOW_COLOR = "rgba(0, 0, 0, 0.7)";
// Room left round the asset for the shadow to fall into, so it is not cut
// off at the PNG's edge. Reserved whether the shadow is on or not: the
// switch would otherwise change the picture's size, and the overlay would
// land smaller with a shadow than without one.
const SHADOW_PAD = 0.05;

let _manifest = null;   // the shelf, read once per page and then kept
let _open = null;       // the live session, so a second press cannot stack a second dialog

/**
 * The shelf, from assets/assets.json. Read on first use rather than at
 * page load: most runs never open it, and a page that waited on it would
 * be waiting for nothing. A shelf that will not load is an empty shelf —
 * the dialog says so — rather than an exception out of a button.
 */
async function manifest() {
    if (_manifest) return _manifest;
    try {
        const res = await fetch(`${ASSET_DIR}assets.json`, { cache: "no-cache" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const list = Array.isArray(data && data.assets) ? data.assets : [];
        _manifest = list.filter(a => a && typeof a.file === "string" && a.file)
            .map(a => ({
                file: a.file,
                name: typeof a.name === "string" && a.name ? a.name : a.file.replace(/\.[^.]+$/, ""),
                color: parseHex(a.color) || "#ff0000",
                url: ASSET_DIR + encodeURIComponent(a.file),
            }));
    } catch (e) {
        console.warn("The asset list could not be read:", e);
        return [];   // not cached, so the next press tries again
    }
    return _manifest;
}

/** "#rrggbb" in lower case from "#rgb", "rgb", "#rrggbb" or "rrggbb"; null otherwise. */
function parseHex(value) {
    if (typeof value !== "string") return null;
    let h = value.trim().replace(/^#/, "").toLowerCase();
    if (/^[0-9a-f]{3}$/.test(h)) h = h.replace(/./g, c => c + c);
    return /^[0-9a-f]{6}$/.test(h) ? `#${h}` : null;
}

function rgbOf(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function loadPicture(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`could not load ${url}`));
        img.src = url;
    });
}

/**
 * Splits every pixel of `img` into how much of the asset's paint `base` it
 * holds and the neutral grey under it, so that any colour can be put back in
 * the paint's place by one multiply-add per channel (see recolor).
 *
 * The model is pixel = s·base + q, with q the same on all three channels.
 * Subtracting a grey leaves a pixel's chroma (its brightest channel minus its
 * dimmest) alone, so s is the pixel's chroma over the paint's; q is then what
 * the dimmest channel has beyond the paint's own share of it. A pure paint
 * pixel comes out s = 1, q = 0 and turns into exactly the colour chosen; the
 * grey of a drop shadow comes out s = 0 and is left as it was; the blend
 * along the edge between them is split in proportion and keeps its softness.
 * Alpha is never touched, so the shape and the shadow's fade are the asset's.
 */
function decompose(img, base) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, w, h);
    const px = data.data;

    const [br, bg, bb] = rgbOf(base);
    const bMin = Math.min(br, bg, bb);
    const bChroma = Math.max(br, bg, bb) - bMin;

    const n = w * h;
    const s = new Float32Array(n), q = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
        const r = px[p], g = px[p + 1], b = px[p + 2];
        const lo = Math.min(r, g, b);
        // A grey paint has no chroma to measure against: nothing can be told
        // apart from it, so nothing is recoloured rather than everything.
        const amount = bChroma > 0 ? Math.min(1, (Math.max(r, g, b) - lo) / bChroma) : 0;
        s[i] = amount;
        q[i] = lo - amount * bMin;
    }
    return { w, h, data, s, q, original: new Uint8ClampedArray(px) };
}

/** `parts` repainted in `hex`, onto its own ImageData. */
function recolor(parts, hex) {
    const [tr, tg, tb] = rgbOf(hex);
    const { s, q, original, data } = parts;
    const px = data.data;
    for (let i = 0, p = 0; i < s.length; i++, p += 4) {
        const k = s[i], g = q[i];
        if (k === 0) {
            px[p] = original[p]; px[p + 1] = original[p + 1]; px[p + 2] = original[p + 2];
        } else {
            px[p] = k * tr + g;       // Uint8ClampedArray rounds and clamps
            px[p + 1] = k * tg + g;
            px[p + 2] = k * tb + g;
        }
        px[p + 3] = original[p + 3];
    }
    return data;
}

/**
 * Shows the asset step and resolves with what to place:
 *   - { src, name, color }, src a PNG data URL of the asset in the chosen colour;
 *   - null, when the user cancels — nothing is added to the frame.
 */
export function openAssetStep() {
    if (_open) _open.finish(null);
    return new Promise((resolve) => {
        _open = new AssetSession((result) => {
            _open = null;
            resolve(result);
        });
    });
}

class AssetSession {
    constructor(done) {
        this.done = done;
        this.asset = null;     // the one picked from the shelf
        this.parts = null;     // its pixels split into paint and grey (see decompose)
        this.color = null;
        this.token = 0;        // bumped per pick, so a slow load cannot answer for a later one

        this.root = el("assetStep");
        this.canvas = el("assetCanvas");
        this.picker = el("assetColor");
        this.hex = el("assetHex");
        this.work = document.createElement("canvas");   // the recoloured asset at full size
        this.out = document.createElement("canvas");    // ...flipped and shadowed: what is placed
        this.flipH = this.flipV = this.shadow = false;

        this.bind();
        this.showShelf();
        this.root.classList.remove("hidden");
        // Shares the mask step's flag: the same "nothing behind scrolls" rule.
        document.body.classList.add("mask-open");
    }

    // ── Wiring ────────────────────────────────────────────────────────────
    //
    // Per session and removed by finish(), as in mask.js, so a closed dialog
    // holds nothing and an old session can never answer for a new one.

    bind() {
        el("assetShelfCancelBtn").onclick = () => this.finish(null);
        el("assetCancelBtn").onclick = () => this.finish(null);
        el("assetBackBtn").onclick = () => this.showShelf();
        el("assetApplyBtn").onclick = () => this.apply();
        el("assetFlipHBtn").onclick = () => this.setToggles({ flipH: !this.flipH });
        el("assetFlipVBtn").onclick = () => this.setToggles({ flipV: !this.flipV });
        el("assetShadowBtn").onclick = () => this.setToggles({ shadow: !this.shadow });
        this.root.onpointerdown = (e) => { if (e.target === this.root) this.finish(null); };
        // `input` fires as the picker's own dialog is dragged around, which
        // is what makes the preview follow the pointer rather than the close.
        this.picker.oninput = () => this.setColor(this.picker.value);
        this.hex.oninput = () => {
            const hex = parseHex(this.hex.value);
            if (hex) this.setColor(hex, { fromHex: true });
        };
        // A half-typed value left behind on blur goes back to the colour in
        // use, so the field never shows one thing while the preview shows another.
        this.hex.onblur = () => { this.hex.value = this.color || ""; };

        this.onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); this.finish(null); }
            else if (e.key === "Enter" && e.target.tagName !== "BUTTON" && this.parts) {
                e.preventDefault(); this.apply();
            }
        };
        document.addEventListener("keydown", this.onKey);
    }

    finish(result) {
        for (const id of ["assetShelfCancelBtn", "assetCancelBtn", "assetBackBtn", "assetApplyBtn",
                          "assetFlipHBtn", "assetFlipVBtn", "assetShadowBtn"]) {
            el(id).onclick = null;
        }
        this.root.onpointerdown = null;
        this.picker.oninput = this.hex.oninput = this.hex.onblur = null;
        document.removeEventListener("keydown", this.onKey);
        this.token++;
        this.root.classList.add("hidden");
        document.body.classList.remove("mask-open");
        this.done(result);
    }

    // ── The shelf ─────────────────────────────────────────────────────────

    async showShelf() {
        this.asset = this.parts = null;
        el("assetShelfView").classList.remove("hidden");
        el("assetColorView").classList.add("hidden");
        el("assetStepLabel").textContent = "Assets";

        const grid = el("assetGrid");
        const token = ++this.token;
        const list = await manifest();
        if (token !== this.token) return;
        grid.innerHTML = "";
        el("assetShelfEmpty").classList.toggle("hidden", list.length > 0);
        for (const asset of list) {
            const tile = document.createElement("button");
            tile.type = "button";
            tile.className = "asset-tile";
            tile.title = asset.name;
            const img = document.createElement("img");
            img.src = asset.url;
            img.alt = "";
            const label = document.createElement("span");
            label.textContent = asset.name;
            tile.append(img, label);
            tile.onclick = () => this.choose(asset);
            grid.appendChild(tile);
        }
        const first = grid.querySelector("button");
        (first || el("assetShelfCancelBtn")).focus();
    }

    // ── The colour ────────────────────────────────────────────────────────

    async choose(asset) {
        const token = ++this.token;
        let img;
        try {
            img = await loadPicture(asset.url);
        } catch (e) {
            console.warn(e);
            return;
        }
        if (token !== this.token) return;
        this.asset = asset;
        this.parts = decompose(img, asset.color);
        const { w, h } = this.parts;
        this.work.width = w;
        this.work.height = h;
        const pad = Math.ceil(Math.max(w, h) * SHADOW_PAD);
        this.out.width = w + pad * 2;
        this.out.height = h + pad * 2;

        el("assetShelfView").classList.add("hidden");
        el("assetColorView").classList.remove("hidden");
        el("assetStepLabel").textContent = asset.name;
        this.sizePreview();
        // Every asset opens the way it was drawn — its own colour, the right
        // way round — and with the shadow on, because a mark laid over a
        // busy photo needs one to stand off it far more often than not.
        // Whatever the last asset was given does not carry over.
        this.setToggles({ flipH: false, flipV: false, shadow: true }, { redraw: false });
        this.setColor(asset.color);
        el("assetApplyBtn").focus();
    }

    /** Fits the preview canvas to the placed picture's shape, inside a PREVIEW square. */
    sizePreview() {
        const w = this.out.width, h = this.out.height;
        const k = PREVIEW / Math.max(w, h);
        const cw = Math.max(1, Math.round(w * k)), ch = Math.max(1, Math.round(h * k));
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(cw * dpr);
        this.canvas.height = Math.round(ch * dpr);
        this.canvas.style.width = `${cw}px`;
        this.canvas.style.aspectRatio = `${cw} / ${ch}`;
    }

    setColor(hex, { fromHex = false } = {}) {
        if (!this.parts) return;
        this.color = hex;
        this.picker.value = hex;
        // Not rewritten while it is being typed into: "#f00" is a colour, and
        // turning it into "#ff0000" under the cursor would fight the typing.
        if (!fromHex) this.hex.value = hex;
        this.work.getContext("2d").putImageData(recolor(this.parts, hex), 0, 0);
        this.compose();
    }

    /** Sets any of the three switches, and shows each one's state on its button. */
    setToggles(change, { redraw = true } = {}) {
        Object.assign(this, change);
        for (const [id, on] of [["assetFlipHBtn", this.flipH], ["assetFlipVBtn", this.flipV],
                                ["assetShadowBtn", this.shadow]]) {
            el(id).classList.toggle("btn-active", on);
            el(id).setAttribute("aria-pressed", String(on));
        }
        if (redraw && this.parts) this.compose();
    }

    /**
     * Lays the recoloured asset onto `out` — mirrored as asked, shadowed if
     * asked — and shows the result. The order matters for the shadow: a
     * canvas shadow's offset is not touched by the transform, so it falls
     * downward whichever way round the asset is drawn, which is where a
     * shadow belongs. Flipping the finished picture instead would have put
     * the shadow above an asset flipped upside down.
     */
    compose() {
        const { w, h } = this.parts;
        const W = this.out.width, H = this.out.height, long = Math.max(w, h);
        const ctx = this.out.getContext("2d");
        ctx.clearRect(0, 0, W, H);
        ctx.save();
        if (this.shadow) {
            ctx.shadowColor = SHADOW_COLOR;
            ctx.shadowBlur = long * SHADOW_BLUR;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = long * SHADOW_DROP;
        }
        ctx.translate(W / 2, H / 2);
        ctx.scale(this.flipH ? -1 : 1, this.flipV ? -1 : 1);
        ctx.drawImage(this.work, -w / 2, -h / 2);
        ctx.restore();

        const view = this.canvas.getContext("2d");
        view.clearRect(0, 0, this.canvas.width, this.canvas.height);
        view.imageSmoothingQuality = "high";
        view.drawImage(this.out, 0, 0, this.canvas.width, this.canvas.height);
    }

    apply() {
        if (!this.parts) return;
        // A value still sitting in the hex field counts, so "type, Enter"
        // is the same as "type, wait for the preview, press the button".
        const typed = parseHex(this.hex.value);
        if (typed && typed !== this.color) this.setColor(typed);
        this.finish({
            src: this.out.toDataURL("image/png"),
            name: this.asset.name,
            color: this.color,
        });
    }
}
