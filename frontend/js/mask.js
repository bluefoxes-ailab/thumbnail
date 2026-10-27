// ── The mask step ─────────────────────────────────────────────────────────
//
// Between picking an overlay image and seeing it on the canvas: the user can
// cut it to a square or a circle and choose how much of the original goes
// inside, by zooming and dragging the picture behind a fixed frame.
//
// The step's answer is a new PNG, not a description of a crop. Everything
// downstream of an overlay — the canvas, the grid thumbnails, the downloads,
// the carry-over to other frames — already knows how to draw "a picture at a
// position and a scale", and a mask baked into the pixels is just another
// picture to all of it. A crop rectangle and a shape stored on the overlay
// record would have had to be taught to every one of those paths, and the
// first one somebody forgot would draw the full, unmasked image.
//
// PNG and not JPEG because the circle's corners are the whole point: they
// have to come out transparent.
//
// It can also be skipped. Overlays were logos before they were anything else,
// and a logo is usually already cut out on a transparent field — forcing it
// into a square would put back the very rectangle its maker took away.

import { el } from "./config.js";

// The frame the image is cut to, in CSS pixels, and how far the stage
// extends past it on every side. The margin is not decoration: it is where the
// part of the picture that will be thrown away stays visible, dimmed, so the
// user can see what they are dragging into the frame before it arrives.
const FRAME = 320;
const MARGIN = 40;
const STAGE = FRAME + MARGIN * 2;

// The slider runs from "the whole image fits in the frame" to this many times
// "the image just fills it". Below fill there are transparent gaps at two
// edges, which is a real answer — a wide logo in a circle — so zooming out is
// allowed rather than stopped at the point where the frame is first covered.
const MAX_ZOOM_OVER_FILL = 4;

// The exported side, in pixels. Never more than the original pixels the crop
// actually covers — scaling up only makes a bigger blurry file — and never
// less than enough to stay sharp at the overlay's default on-canvas width
// (28% of 1280, see editor.handleLogoFile) on a high-density screen.
const MIN_OUT = 512;
const MAX_OUT = 2048;

// The blur slider's far end, as a radius in CSS pixels of the frame. Measured
// against the frame rather than the image or the output so that what the
// stage shows is what the PNG gets: the export is the frame scaled to `side`,
// and the radius is scaled with it. At 24 of 320 a face is no longer a face,
// which is about as far as a background picture ever needs to go.
const MAX_BLUR = 24;

/**
 * Draws `img` into the rectangle (x, y, iw, ih) of `ctx`, blurred by `blur`
 * units, without the soft transparent rim a plain blur filter leaves.
 *
 * A blur averages each pixel with its neighbours, and past the image's edge
 * the neighbours are nothing — so the edge fades out over the blur's width.
 * At "fill" two of the image's edges sit exactly on the frame's, and that
 * fade would be cut into the exported picture as a dark, half-transparent
 * border. So the edge rows and columns are stretched outward first, giving
 * the blur more of the same colour to average with, and the result is
 * clipped back to the image's own rectangle. Where the image is smaller than
 * the frame the gap stays as crisp and as transparent as it is unblurred.
 *
 * `view` is the side of the square of `ctx` that can be seen, in its own
 * units, and `density` its device pixels per unit — only that square is
 * blurred, however far the image extends past it.
 */
function drawBlurred(ctx, img, x, y, iw, ih, blur, view, density) {
    if (!(blur > 0)) {
        ctx.drawImage(img, x, y, iw, ih);
        return;
    }
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const pad = Math.ceil(blur * 3);   // past three radii a Gaussian adds nothing visible
    const side = view + pad * 2;
    const px = Math.max(1, Math.round(side * density));

    const src = document.createElement("canvas");
    src.width = src.height = px;
    const s = src.getContext("2d");
    s.setTransform(density, 0, 0, density, pad * density, pad * density);
    s.imageSmoothingQuality = "high";
    // Edges, then corners, each stretched far enough to cover the padding
    // whatever side of the view the image happens to end on.
    const far = side * 2;
    s.drawImage(img, 0, 0, 1, h, x - far, y, far, ih);                  // left
    s.drawImage(img, w - 1, 0, 1, h, x + iw, y, far, ih);               // right
    s.drawImage(img, 0, 0, w, 1, x, y - far, iw, far);                  // top
    s.drawImage(img, 0, h - 1, w, 1, x, y + ih, iw, far);               // bottom
    s.drawImage(img, 0, 0, 1, 1, x - far, y - far, far, far);
    s.drawImage(img, w - 1, 0, 1, 1, x + iw, y - far, far, far);
    s.drawImage(img, 0, h - 1, 1, 1, x - far, y + ih, far, far);
    s.drawImage(img, w - 1, h - 1, 1, 1, x + iw, y + ih, far, far);
    s.drawImage(img, x, y, iw, ih);

    // The blur goes on in a second, untransformed pass, so its radius is
    // in device pixels and nothing has to be assumed about whether a
    // context's transform scales its filter.
    const out = document.createElement("canvas");
    out.width = out.height = px;
    const o = out.getContext("2d");
    o.filter = `blur(${blur * density}px)`;
    o.drawImage(src, 0, 0);

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, iw, ih);
    ctx.clip();
    ctx.drawImage(out, -pad, -pad, side, side);
    ctx.restore();
}

let _open = null;   // the live session, so a second upload cannot stack a second step

/**
 * Shows the mask step for `img` and resolves with what to place:
 *   - a PNG data URL of the masked crop, when the user applies it;
 *   - `original` untouched, when they choose to keep the full image;
 *   - null, when they cancel — nothing is added to the frame.
 */
export function openMaskStep(img, original) {
    if (_open) _open.finish(null);
    return new Promise((resolve) => {
        _open = new MaskSession(img, original, (result) => {
            _open = null;
            resolve(result);
        });
    });
}

class MaskSession {
    constructor(img, original, done) {
        this.img = img;
        this.original = original;
        this.done = done;
        this.shape = "square";

        // Scale is CSS pixels per image pixel. "fill" is the smallest that
        // covers the frame, "fit" the largest that shows all of the image.
        const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
        this.w = w; this.h = h;
        this.fill = FRAME / Math.min(w, h);
        this.fit = FRAME / Math.max(w, h);
        this.max = this.fill * MAX_ZOOM_OVER_FILL;
        this.scale = this.fill;   // a square photo opens filled, a wide one centred and covering
        // Centre of the image relative to the centre of the frame, in CSS px.
        this.dx = 0; this.dy = 0;
        this.blur = 0;   // radius in CSS px of the frame, 0..MAX_BLUR

        this.root = el("maskStep");
        this.canvas = el("maskCanvas");
        this.slider = el("maskZoom");
        this.blurSlider = el("maskBlur");
        this.ctx = this.canvas.getContext("2d");

        // Drawn at the screen's own density so the preview is as sharp as
        // the thing it previews; laid out at STAGE CSS pixels either way.
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(STAGE * dpr);
        this.canvas.height = Math.round(STAGE * dpr);
        this.canvas.style.width = this.canvas.style.height = `${STAGE}px`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this.dpr = dpr;

        this.bind();
        this.setShape("square");
        this.syncSlider();
        this.setBlur(0);   // every image starts sharp, whatever the last one was given
        this.root.classList.remove("hidden");
        document.body.classList.add("mask-open");
        el("maskApplyBtn").focus();
    }

    // ── Wiring ────────────────────────────────────────────────────────────
    //
    // Every listener is attached per session and removed by finish(), so a
    // closed step holds nothing and an old session can never answer for a
    // new one. Assigned as properties (onclick and the like) where the
    // element has nothing else listening, because then "remove" is one
    // assignment and cannot miss.

    bind() {
        el("maskSquareBtn").onclick = () => this.setShape("square");
        el("maskRoundBtn").onclick = () => this.setShape("round");
        el("maskZoomOutBtn").onclick = () => this.stepZoom(-1);
        el("maskZoomInBtn").onclick = () => this.stepZoom(+1);
        el("maskBlurUpBtn").onclick = () => this.setBlur(this.blur + MAX_BLUR * 0.1);
        el("maskBlurDownBtn").onclick = () => this.setBlur(this.blur - MAX_BLUR * 0.1);
        el("maskCancelBtn").onclick = () => this.finish(null);
        el("maskSkipBtn").onclick = () => this.finish(this.renderWhole());
        el("maskApplyBtn").onclick = () => this.finish(this.render());
        // A press on the dimmed backdrop is a cancel, the way it is on any
        // dialog; a press inside the card is not, which is what the target
        // check is for.
        this.root.onpointerdown = (e) => { if (e.target === this.root) this.finish(null); };

        this.slider.oninput = () => this.zoomTo(this.sliderToScale(+this.slider.value));
        this.blurSlider.oninput = () => this.setBlur(+this.blurSlider.value * MAX_BLUR);

        this.canvas.onpointerdown = (e) => this.dragStart(e);
        this.canvas.onpointermove = (e) => this.dragMove(e);
        this.canvas.onpointerup = this.canvas.onpointercancel = (e) => this.dragEnd(e);
        this.canvas.onwheel = (e) => {
            e.preventDefault();
            // Zoom toward the pointer, the way every image viewer does, so
            // the thing under the cursor stays under the cursor.
            const r = this.canvas.getBoundingClientRect(), k = this.cssToStage();
            const px = (e.clientX - r.left) * k - STAGE / 2, py = (e.clientY - r.top) * k - STAGE / 2;
            this.zoomTo(this.scale * Math.exp(-e.deltaY * 0.0015), px, py);
        };

        this.onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); this.finish(null); }
            else if (e.key === "Enter" && e.target.tagName !== "BUTTON") {
                // A focused button answers Enter with its own click already;
                // this is for Enter pressed anywhere else in the step.
                e.preventDefault(); this.finish(this.render());
            }
        };
        document.addEventListener("keydown", this.onKey);
    }

    finish(result) {
        for (const id of ["maskSquareBtn", "maskRoundBtn", "maskZoomOutBtn", "maskZoomInBtn",
                          "maskBlurUpBtn", "maskBlurDownBtn",
                          "maskCancelBtn", "maskSkipBtn", "maskApplyBtn"]) el(id).onclick = null;
        this.root.onpointerdown = null;
        this.slider.oninput = this.blurSlider.oninput = null;
        const c = this.canvas;
        c.onpointerdown = c.onpointermove = c.onpointerup = c.onpointercancel = c.onwheel = null;
        document.removeEventListener("keydown", this.onKey);
        this.root.classList.add("hidden");
        document.body.classList.remove("mask-open");
        this.done(result);
    }

    // ── Shape ─────────────────────────────────────────────────────────────

    setShape(shape) {
        this.shape = shape;
        // The app's one way of showing a one-of-two choice: .btn-active held
        // on the chosen button (see .btn in styles.css).
        for (const [id, s] of [["maskSquareBtn", "square"], ["maskRoundBtn", "round"]]) {
            el(id).classList.toggle("btn-active", s === shape);
            el(id).setAttribute("aria-pressed", String(s === shape));
        }
        this.draw();
    }

    // ── Zoom ──────────────────────────────────────────────────────────────
    //
    // The slider is logarithmic: 0 is "fit", 1 is the maximum. Zoom is a
    // ratio, so equal steps along the track should be equal ratios — on a
    // linear track the first half would go by in a blur and the second half
    // would barely move.

    scaleToSlider(s) { return Math.log(s / this.fit) / Math.log(this.max / this.fit); }
    sliderToScale(v) { return this.fit * Math.pow(this.max / this.fit, v); }

    stepZoom(dir) {
        this.zoomTo(this.sliderToScale(Math.min(1, Math.max(0, this.scaleToSlider(this.scale) + dir * 0.1))));
    }

    /** Zooms to `s`, keeping the stage point (px, py) — from the centre — still. */
    zoomTo(s, px = 0, py = 0) {
        s = Math.min(this.max, Math.max(this.fit, s));
        const k = s / this.scale;
        this.dx = px - (px - this.dx) * k;
        this.dy = py - (py - this.dy) * k;
        this.scale = s;
        this.clamp();
        this.syncSlider();
        this.draw();
    }

    syncSlider() {
        // A square image has fit === fill === max / 4; the guard is for the
        // degenerate 1x1 case where the ratio has nothing to divide.
        const v = this.max > this.fit ? this.scaleToSlider(this.scale) : 0;
        this.slider.value = String(v);
        this.slider.style.setProperty("--count-fill", String(v));
    }

    // ── Blur ──────────────────────────────────────────────────────────────
    //
    // Linear, unlike zoom: a radius is an amount, not a ratio, and the
    // difference between 2 and 4 pixels looks about as large as the one
    // between 20 and 22 — which a straight track already gives.

    setBlur(b) {
        this.blur = Math.min(MAX_BLUR, Math.max(0, b));
        const v = this.blur / MAX_BLUR;
        this.blurSlider.value = String(v);
        this.blurSlider.style.setProperty("--count-fill", String(v));
        this.draw();
    }

    // ── Pan ───────────────────────────────────────────────────────────────

    /**
     * Stage units per screen pixel. 1 at full size; more on a window narrow
     * enough that the card's max-width shrinks the stage, where a drag of
     * one screen pixel has to move the picture by more than one stage unit
     * to stay under the finger.
     */
    cssToStage() {
        return STAGE / (this.canvas.getBoundingClientRect().width || STAGE);
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
     * Keeps the picture where it can still be used, one axis at a time.
     * Along an axis where the image is bigger than the frame, it may not be
     * dragged far enough to open a gap inside the frame. Along one where it
     * is smaller (zoomed out past fill), it may not leave the frame — so the
     * gap is always one the user chose by zooming, never one left by a drag.
     */
    clamp() {
        // Both rules come to the same number: half the difference between
        // the image's side and the frame's, whichever of the two is larger.
        const hx = Math.abs(this.w * this.scale - FRAME) / 2;
        const hy = Math.abs(this.h * this.scale - FRAME) / 2;
        this.dx = Math.min(hx, Math.max(-hx, this.dx));
        this.dy = Math.min(hy, Math.max(-hy, this.dy));
    }

    // ── Drawing ───────────────────────────────────────────────────────────

    /** Adds the frame's outline, square or round, to the current path. */
    framePath(ctx, x, y, side) {
        if (this.shape === "round") ctx.arc(x + side / 2, y + side / 2, side / 2, 0, Math.PI * 2);
        else ctx.rect(x, y, side, side);
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, STAGE, STAGE);
        const iw = this.w * this.scale, ih = this.h * this.scale;
        const cx = STAGE / 2 + this.dx, cy = STAGE / 2 + this.dy;
        ctx.imageSmoothingQuality = "high";
        drawBlurred(ctx, this.img, cx - iw / 2, cy - ih / 2, iw, ih, this.blur, STAGE, this.dpr);

        // Everything outside the frame, dimmed: one path holding the whole
        // stage and the frame, filled even-odd so the frame is the hole.
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, STAGE, STAGE);
        this.framePath(ctx, MARGIN, MARGIN, FRAME);
        ctx.fillStyle = "rgba(6, 6, 16, 0.72)";
        ctx.fill("evenodd");
        ctx.restore();

        // The edge itself in the app's yellow — the colour that means "this
        // is the current one" everywhere else on the page, and the one that
        // reads on both a dark photo and a bright one.
        ctx.beginPath();
        this.framePath(ctx, MARGIN, MARGIN, FRAME);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#fffa00";
        ctx.stroke();
    }

    /** The crop as a PNG data URL, at the original's own resolution. */
    render() {
        // How many original pixels the frame covers along its side.
        const covered = FRAME / this.scale;
        const side = Math.round(Math.min(MAX_OUT, Math.max(MIN_OUT, Math.min(covered, Math.max(this.w, this.h)))));
        const k = side / FRAME;   // output pixels per CSS pixel of the frame
        const out = document.createElement("canvas");
        out.width = out.height = side;
        const ctx = out.getContext("2d");
        ctx.imageSmoothingQuality = "high";
        if (this.shape === "round") {
            ctx.beginPath();
            ctx.arc(side / 2, side / 2, side / 2, 0, Math.PI * 2);
            ctx.clip();
        }
        const iw = this.w * this.scale * k, ih = this.h * this.scale * k;
        const cx = side / 2 + this.dx * k, cy = side / 2 + this.dy * k;
        drawBlurred(ctx, this.img, cx - iw / 2, cy - ih / 2, iw, ih, this.blur * k, side, 1);
        return out.toDataURL("image/png");
    }

    /**
     * "No mask": the whole image, uncropped. Untouched — the original file
     * itself — unless it was blurred, because skipping the crop is not
     * asking to lose the blur. A blurred one is redrawn at its own size
     * (capped like the masked export), with the radius carried over from
     * the stage at "fit", where the frame's width is the image's long side.
     */
    renderWhole() {
        if (!(this.blur > 0)) return this.original;
        const k = Math.min(1, MAX_OUT / Math.max(this.w, this.h));
        const w = Math.max(1, Math.round(this.w * k)), h = Math.max(1, Math.round(this.h * k));
        const out = document.createElement("canvas");
        out.width = w; out.height = h;
        const ctx = out.getContext("2d");
        ctx.imageSmoothingQuality = "high";
        drawBlurred(ctx, this.img, 0, 0, w, h, this.blur * Math.max(w, h) / FRAME, Math.max(w, h), 1);
        return out.toDataURL("image/png");
    }
}
