// ── Framing an image of the user's own as a frame ─────────────────────────
//
// Between picking an image to add to a frame fetch's grid and its card
// appearing there (see main.addOwnFrame): the user drags and zooms the
// picture behind a window the shape of the VIDEO, and what is inside the
// window becomes the frame.
//
// The video's shape and not the canvas's, because the image is about to
// become one more frame of that video as far as everything downstream is
// concerned. A capture slot points at the uncropped video frame, and the
// backend frames it to the canvas from there (see /add-frame) — the same
// automatic framing, the same room to zoom and reframe on the canvas
// afterwards. An image cropped straight to the canvas would arrive with no
// room left around it, and one sent uncropped would be framed around its
// own middle with nobody having chosen what the frame shows.
//
// The answer is a JPEG at the video's own pixel size, so the image and the
// frames beside it are measured on the same scale: the automatic framing's
// zoom is worked out from the size of the source, and an added frame twice
// the video's resolution would open framed twice as tight as its neighbours.
//
// Same build as the mask step (mask.js), and the same card and controls on
// screen, because it is the same gesture: placing a picture behind a fixed
// window. What differs is that this window is never left with a gap in it —
// a frame is a photograph, and a strip of nothing down one side of it would
// be shipped to the user as part of the picture. To see more of a wide image,
// zoom out on the canvas once it has been added: dragging past the edge there
// fills the gap in from the picture itself.

import { el } from "./config.js";

// The longer side of the window, in CSS pixels, and how far the stage extends
// past it on every side — where the part of the picture that will be thrown
// away stays visible, dimmed, as it does in the mask step.
const FRAME_LONG = 340;
const MARGIN = 36;

// The slider runs from "the image just fills the window" to this many times
// that. There is no "fit" below it: see the header.
const MAX_ZOOM_OVER_FILL = 4;

let _open = null;   // the live session, so a second pick cannot stack a second step

/**
 * A ratio short enough to read as one — "9:16" — or the two sides as they are
 * when they do not reduce to anything a person would recognise.
 */
function ratioLabel(w, h) {
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const g = gcd(w, h);
    return (w / g <= 32 && h / g <= 32) ? `${w / g}:${h / g}` : `${w} × ${h}`;
}

/**
 * Shows the framing step for `img` and resolves with a JPEG Blob of the framed
 * picture at `width` x `height` — the video's size — or with null when the
 * user cancels, in which case nothing is added.
 */
export function openFrameCrop(img, width, height) {
    if (_open) _open.finish(null);
    return new Promise((resolve) => {
        _open = new FrameCropSession(img, width, height, (result) => {
            _open = null;
            resolve(result);
        });
    });
}

class FrameCropSession {
    constructor(img, width, height, done) {
        this.img = img;
        this.done = done;
        this.outW = width;
        this.outH = height;

        // The window, the video's shape with its longer side at FRAME_LONG,
        // and the stage around it.
        const k = FRAME_LONG / Math.max(width, height);
        this.fw = width * k;
        this.fh = height * k;
        this.sw = this.fw + MARGIN * 2;
        this.sh = this.fh + MARGIN * 2;

        // Scale is CSS pixels per image pixel; "fill" is the smallest that
        // covers the window.
        const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
        this.w = w; this.h = h;
        this.fill = Math.max(this.fw / w, this.fh / h);
        this.max = this.fill * MAX_ZOOM_OVER_FILL;
        this.scale = this.fill;
        // Centre of the image relative to the centre of the window, in CSS px.
        this.dx = 0; this.dy = 0;

        this.root = el("frameCropStep");
        this.canvas = el("frameCropCanvas");
        this.slider = el("frameCropZoom");
        this.ctx = this.canvas.getContext("2d");

        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(this.sw * dpr);
        this.canvas.height = Math.round(this.sh * dpr);
        // The width and the shape, and nothing else: on a narrow window the
        // card's max-width shrinks the stage, and the height follows through
        // aspect-ratio rather than a fixed figure that would stretch it.
        this.canvas.style.width = `${this.sw}px`;
        this.canvas.style.aspectRatio = `${this.sw} / ${this.sh}`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        el("frameCropHint").textContent =
            `Drag the picture to place it and zoom to choose how much of it shows. The frame has `
            + `the video's proportions (${ratioLabel(width, height)}), so your image sits in the grid `
            + "like every other frame.";

        this.bind();
        this.syncSlider();
        this.draw();
        this.root.classList.remove("hidden");
        document.body.classList.add("mask-open");
        el("frameCropApplyBtn").focus();
    }

    // ── Wiring ────────────────────────────────────────────────────────────
    //
    // Per session and removed by finish(), as in the mask step, so a closed
    // step holds nothing and an old session can never answer for a new one.

    bind() {
        el("frameCropZoomOutBtn").onclick = () => this.stepZoom(-1);
        el("frameCropZoomInBtn").onclick = () => this.stepZoom(+1);
        el("frameCropCancelBtn").onclick = () => this.finish(null);
        el("frameCropApplyBtn").onclick = () => this.apply();
        this.root.onpointerdown = (e) => { if (e.target === this.root) this.finish(null); };
        this.slider.oninput = () => this.zoomTo(this.sliderToScale(+this.slider.value));

        this.canvas.onpointerdown = (e) => this.dragStart(e);
        this.canvas.onpointermove = (e) => this.dragMove(e);
        this.canvas.onpointerup = this.canvas.onpointercancel = (e) => this.dragEnd(e);
        this.canvas.onwheel = (e) => {
            e.preventDefault();
            // Toward the pointer, so what is under the cursor stays there.
            const r = this.canvas.getBoundingClientRect(), k = this.cssToStage();
            const px = (e.clientX - r.left) * k - this.sw / 2, py = (e.clientY - r.top) * k - this.sh / 2;
            this.zoomTo(this.scale * Math.exp(-e.deltaY * 0.0015), px, py);
        };

        this.onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); this.finish(null); }
            else if (e.key === "Enter" && e.target.tagName !== "BUTTON") { e.preventDefault(); this.apply(); }
        };
        document.addEventListener("keydown", this.onKey);
    }

    finish(result) {
        for (const id of ["frameCropZoomOutBtn", "frameCropZoomInBtn",
                          "frameCropCancelBtn", "frameCropApplyBtn"]) el(id).onclick = null;
        this.root.onpointerdown = null;
        this.slider.oninput = null;
        const c = this.canvas;
        c.onpointerdown = c.onpointermove = c.onpointerup = c.onpointercancel = c.onwheel = null;
        document.removeEventListener("keydown", this.onKey);
        this.root.classList.add("hidden");
        document.body.classList.remove("mask-open");
        this.done(result);
    }

    async apply() {
        if (this.applying) return;   // a double click would send the image twice
        this.applying = true;
        this.finish(await this.render());
    }

    // ── Zoom ──────────────────────────────────────────────────────────────
    //
    // Logarithmic, as in the mask step: zoom is a ratio, and equal steps
    // along the track should be equal ratios.

    scaleToSlider(s) { return Math.log(s / this.fill) / Math.log(this.max / this.fill); }
    sliderToScale(v) { return this.fill * Math.pow(this.max / this.fill, v); }

    stepZoom(dir) {
        this.zoomTo(this.sliderToScale(Math.min(1, Math.max(0, this.scaleToSlider(this.scale) + dir * 0.1))));
    }

    /** Zooms to `s`, keeping the stage point (px, py) — from the centre — still. */
    zoomTo(s, px = 0, py = 0) {
        s = Math.min(this.max, Math.max(this.fill, s));
        const k = s / this.scale;
        this.dx = px - (px - this.dx) * k;
        this.dy = py - (py - this.dy) * k;
        this.scale = s;
        this.clamp();
        this.syncSlider();
        this.draw();
    }

    syncSlider() {
        const v = this.scaleToSlider(this.scale);
        this.slider.value = String(v);
        this.slider.style.setProperty("--count-fill", String(v));
    }

    // ── Pan ───────────────────────────────────────────────────────────────

    /** Stage units per screen pixel — more than 1 where the card has shrunk the stage. */
    cssToStage() {
        return this.sw / (this.canvas.getBoundingClientRect().width || this.sw);
    }

    dragStart(e) {
        this.canvas.setPointerCapture(e.pointerId);
        this.drag = { x: e.clientX, y: e.clientY, dx: this.dx, dy: this.dy, k: this.cssToStage() };
        this.canvas.classList.add("dragging");
    }

    dragMove(e) {
        if (!this.drag) return;
        this.dx = this.drag.dx + (e.clientX - this.drag.x) * this.drag.k;
        this.dy = this.drag.dy + (e.clientY - this.drag.y) * this.drag.k;
        this.clamp();
        this.draw();
    }

    dragEnd(e) {
        if (!this.drag) return;
        this.drag = null;
        this.canvas.classList.remove("dragging");
        if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    }

    /**
     * Keeps the window covered: the picture is never zoomed below fill, so on
     * both axes it is at least as large as the window, and it may not be
     * dragged far enough to open a gap inside it.
     */
    clamp() {
        const hx = Math.max(0, (this.w * this.scale - this.fw) / 2);
        const hy = Math.max(0, (this.h * this.scale - this.fh) / 2);
        this.dx = Math.min(hx, Math.max(-hx, this.dx));
        this.dy = Math.min(hy, Math.max(-hy, this.dy));
    }

    // ── Drawing ───────────────────────────────────────────────────────────

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.sw, this.sh);
        const iw = this.w * this.scale, ih = this.h * this.scale;
        const cx = this.sw / 2 + this.dx, cy = this.sh / 2 + this.dy;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(this.img, cx - iw / 2, cy - ih / 2, iw, ih);

        // Everything outside the window, dimmed, with the window as the hole.
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, this.sw, this.sh);
        ctx.rect(MARGIN, MARGIN, this.fw, this.fh);
        ctx.fillStyle = "rgba(6, 6, 16, 0.72)";
        ctx.fill("evenodd");
        ctx.restore();

        // The edge in the app's yellow, as the mask step draws its own.
        ctx.beginPath();
        ctx.rect(MARGIN, MARGIN, this.fw, this.fh);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#fffa00";
        ctx.stroke();
    }

    /** What is inside the window, as a JPEG at the video's own size. */
    render() {
        const k = this.outW / this.fw;   // output pixels per CSS pixel of the window
        const out = document.createElement("canvas");
        out.width = this.outW;
        out.height = this.outH;
        const ctx = out.getContext("2d");
        ctx.imageSmoothingQuality = "high";
        const iw = this.w * this.scale * k, ih = this.h * this.scale * k;
        const cx = this.outW / 2 + this.dx * k, cy = this.outH / 2 + this.dy * k;
        ctx.drawImage(this.img, cx - iw / 2, cy - ih / 2, iw, ih);
        return new Promise((resolve) => out.toBlob(resolve, "image/jpeg", 0.95));
    }
}
