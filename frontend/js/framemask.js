// ── The frame mask step ───────────────────────────────────────────────────
//
// Static Studio's "Create mask": the user draws a window on the frame — a
// rectangle of whatever proportions they drag it to, or a circle — and
// everything outside it is painted one solid colour, with an optional edge
// round the window. What it produces is the record described at
// state.frameMaskFor, and compose.addFrameMask is the one thing that draws it.
//
// A description and not new pixels, which is the opposite of the choice the
// overlay's mask step makes (see mask.js), and for the opposite reason. That
// step cuts a picture that is about to be placed; this one cuts the frame's
// own photograph, which is still being worked on after the mask goes on — it
// is reframed, re-enhanced, replaced by a Variation. A mask baked into the
// photo would be thrown away by the first of those, and one stored as a shape
// is simply drawn again over whatever the photo has become.
//
// The window is drawn on the picture ALONE — no title, no overlays — because
// that is what it cuts: the mask goes down directly on the photograph and
// everything else stands on it (see compose.paintFrame). The miniature beside
// the stage is the whole frame with the mask applied, so the user can see
// how the title reads against the ground before they commit to a colour.

import { CW, CH, el } from "./config.js";
import { current, selectedIndex, channelFor, frameMaskFor, copyFrameMask } from "./state.js";
import { styleFor } from "./channels.js";
import { buildComposite, updatePreview, refreshThumb } from "./compose.js";
import { status } from "./ui.js";

// The stage's longer side, and the miniature's, in CSS pixels. The stage is
// sized to the canvas's own shape, so a portrait Static Studio frame and a
// landscape thumbnail both arrive whole and undistorted.
const STAGE_LONG = 440;
const THUMB_LONG = 170;

// How close to a corner, in CSS pixels of the stage, a press has to land to
// take the corner rather than the whole window. Generous, because the handle
// is small and the window is right next to it: a miss there moves the mask.
const HANDLE_REACH = 14;
const HANDLE_SIZE = 10;

// The smallest window, in canvas pixels. Below this it is a speck the user
// can no longer get hold of to make bigger again.
const MIN_SIDE = 24;

// The edge's width, in canvas pixels. The default is a fraction of the
// canvas's shorter side so that it reads the same on a 540-wide still as on
// a 1280-wide thumbnail.
const OUTLINE_RANGE = { min: 1, max: 40 };
const defaultOutlineWidth = () => Math.max(2, Math.round(Math.min(CW, CH) * 0.012));

const HEX6 = /^#[0-9a-f]{6}$/i;

/**
 * `c` as "#rrggbb", or null when it is not a colour. Through a canvas, which
 * is the browser's own parser, so a pack written in "#FFF" or "white" still
 * gives the colour well something it can show — <input type="color"> accepts
 * nothing but the six-digit form.
 */
let _parser = null;
function toHex(c) {
    if (typeof c !== "string" || !c) return null;
    if (HEX6.test(c)) return c.toLowerCase();
    _parser = _parser || document.createElement("canvas").getContext("2d");
    _parser.fillStyle = "#000001";   // a sentinel: an unparsable colour leaves it unchanged
    _parser.fillStyle = c;
    const out = _parser.fillStyle;
    return (HEX6.test(out) && out !== "#000001") ? out.toLowerCase() : null;
}

/**
 * The colours channel `id` is set in, as "#rrggbb", each once, in order of
 * how much of the brand they carry: the card's fill first (on a Static Studio
 * channel the card IS the brand colour — see the Snapchat packs), then the
 * letters, the picked words, and then whatever the pack offers the user on
 * top. These are the swatches the step offers and where its defaults come
 * from, so a mask opens already in the channel's colours.
 */
function channelPalette(id) {
    const style = styleFor(id);
    const palette = style.colors || {};
    const all = [
        style.panel && style.panel.fill,
        style.color,
        style.highlight && style.highlight.color,
        ...(palette.panel || []), ...(palette.text || []), ...(palette.highlight || []),
    ];
    const out = [];
    for (const c of all) {
        const hex = toHex(c);
        if (hex && !out.includes(hex)) out.push(hex);
    }
    return out.length ? out : ["#000000", "#ffffff"];
}

/**
 * A new mask for a frame on channel `id`: a window in the middle of the frame
 * on a ground of the channel's first colour, with its edge — off until asked
 * for — already set to the next one, so switching it on shows the brand's two
 * colours against each other rather than a colour against itself.
 */
function defaultMask(id) {
    const palette = channelPalette(id);
    const fill = palette[0];
    const outlineColor = palette.find(c => c !== fill) || "#ffffff";
    const w = Math.round(CW * 0.7), h = Math.round(CH * 0.7);
    return {
        shape: "rect",
        x: Math.round((CW - w) / 2), y: Math.round((CH - h) / 2), w, h,
        fill,
        outline: false, outlineColor, outlineWidth: defaultOutlineWidth(),
    };
}

/**
 * The "Create mask" button, made to say what pressing it will do on the frame
 * on screen: create one, or edit the one it has. Called wherever the other
 * buttons under the canvas are synced — a change of frame, Undo and Redo.
 */
export function syncFrameMaskButton() {
    const btn = el("frameMaskBtn");
    if (!btn) return;
    const has = !!frameMaskFor(selectedIndex());
    btn.textContent = has ? "Edit mask" : "Create mask";
    btn.classList.toggle("btn-active", has);
}

// ── Geometry ──────────────────────────────────────────────────────────────
//
// Moving and reshaping a mask's window, in canvas pixels, as the step does
// it. The canvas never reshapes the window — it moves and scales the masked
// picture whole (see Placement below) — but it hit-tests the PLACED window
// with the same corner and inside tests, so a corner is a corner in both.

/** The window's four corners, in the order top-left, top-right, bottom-left, bottom-right. */
export function maskCorners(mask) {
    const { x, y, w, h } = mask;
    return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];
}

/** Which corner (0-3) of `mask` is within `reach` of (x, y), or -1. */
export function maskCornerAt(mask, x, y, reach) {
    return maskCorners(mask).findIndex(([cx, cy]) => Math.abs(cx - x) <= reach && Math.abs(cy - y) <= reach);
}

/** The corner that stays put while corner `k` is dragged. */
export const oppositeCorner = (mask, k) => maskCorners(mask)[3 - k];

/** Whether (x, y) is inside the window's box. */
export function insideMask(mask, x, y) {
    return x >= mask.x && x <= mask.x + mask.w && y >= mask.y && y <= mask.y + mask.h;
}

/**
 * Whether (x, y) is on the photograph showing through the window, as opposed
 * to the ground round it — the shape itself, so the corners of a round
 * window's box count as ground, which is what they look like.
 */
export function inWindow(mask, x, y) {
    if (mask.shape !== "round") return insideMask(mask, x, y);
    const rx = mask.w / 2, ry = mask.h / 2;
    const dx = (x - mask.x - rx) / rx, dy = (y - mask.y - ry) / ry;
    return dx * dx + dy * dy <= 1;
}

/**
 * Moves `mask` so its top-left corner is at (x, y), kept on the canvas: a
 * window dragged half off the frame is a window nobody can get hold of again.
 */
export function moveMaskTo(mask, x, y) {
    mask.x = Math.max(0, Math.min(CW - mask.w, x));
    mask.y = Math.max(0, Math.min(CH - mask.h, y));
}

/**
 * Stretches `mask` between the held corner (ax, ay) and the pointer (x, y).
 *
 * A circle is kept square, and so is a rectangle when `square` is asked for
 * (Shift held), the way every drawing program does it: the side that follows
 * is the longer of the two the pointer asks for, cut back to the room there
 * is on that side of the held corner so the window never leaves the canvas.
 */
export function resizeMask(mask, ax, ay, x, y, square = false) {
    x = Math.max(0, Math.min(CW, x));
    y = Math.max(0, Math.min(CH, y));
    const dirX = x < ax ? -1 : 1, dirY = y < ay ? -1 : 1;
    let w = Math.abs(x - ax), h = Math.abs(y - ay);
    if (mask.shape === "round" || square) {
        const roomX = dirX < 0 ? ax : CW - ax;
        const roomY = dirY < 0 ? ay : CH - ay;
        w = h = Math.min(Math.max(w, h), roomX, roomY);
    }
    w = Math.max(MIN_SIDE, w);
    h = Math.max(MIN_SIDE, h);
    mask.w = w; mask.h = h;
    mask.x = Math.max(0, Math.min(CW - w, dirX < 0 ? ax - w : ax));
    mask.y = Math.max(0, Math.min(CH - h, dirY < 0 ? ay - h : ay));
}

/**
 * Whole pixels, once the hand is off, so the record — and the thumbnail
 * signature built from it — does not carry fractions of a pixel nobody sees.
 */
export function roundMask(mask) {
    for (const key of ["x", "y", "w", "h"]) mask[key] = Math.round(mask[key]);
    const p = mask.place;
    if (p) {
        p.tx = Math.round(p.tx);
        p.ty = Math.round(p.ty);
        p.s = Math.round(p.s * 1000) / 1000;
    }
}

// ── Placement ─────────────────────────────────────────────────────────────
//
// Where the MASKED PICTURE stands on the canvas, as opposed to where the
// window was cut. Two different decisions, made in two different places on
// purpose.
//
// The window is cut in the step, on the photograph alone: `x`, `y`, `w`, `h`
// are measured on the picture, and they say which part of it is kept. Once
// applied that is settled. On the canvas the user is no longer choosing what
// to keep, they are placing what they kept - so a drag there moves the window
// AND the photograph in it together, and a corner scales the two together,
// keeping their proportions. It used to be the other way round: the canvas
// moved and reshaped the window over a photograph that stayed put, which
// re-cut the picture every time the user only meant to move it.
//
// `place` is that placement, as a uniform scale and a shift: a point of the
// picture at (px, py) is drawn at (s*px + tx, s*py + ty). Absent means the
// identity - the masked picture exactly where it was cut - which is what
// every mask made before this existed is, and what a new one starts as.
// Stated as an absolute map rather than as a scale about the window's centre
// so that re-cutting the window in the step leaves the picture where it was
// on the canvas instead of jumping by however far the centre moved.

const PLACE_IDENTITY = Object.freeze({ s: 1, tx: 0, ty: 0 });

// How big the masked picture may be made, as a multiple of the size it was
// cut at. Below the floor its window would be a speck nobody can get hold of
// again (the same reason MIN_SIDE exists); the ceiling is generous because
// blowing a window up past the frame, cropped to a face, is a composition.
const PLACE_SCALE_MAX = 4;

// How much of the window has to stay on the canvas, in canvas pixels, so a
// picture dragged off the edge can still be grabbed and dragged back.
const PLACE_KEEP = 40;

/** The mask's placement, or the identity. Never null. */
export const placeOf = (mask) => (mask && mask.place) || PLACE_IDENTITY;

/** Whether the masked picture stands exactly where it was cut. */
export function placedAsCut(mask) {
    const p = placeOf(mask);
    return p.s === 1 && p.tx === 0 && p.ty === 0;
}

/**
 * The window as it is drawn on the canvas - the cut window carried through
 * the placement. Everything the canvas hit-tests and draws chrome round is
 * this box, never the cut one.
 */
export function placedWindow(mask) {
    const p = placeOf(mask);
    return {
        shape: mask.shape,
        x: p.s * mask.x + p.tx, y: p.s * mask.y + p.ty,
        w: p.s * mask.w, h: p.s * mask.h,
    };
}

/**
 * Shifts the masked picture so its placement's offset is (tx, ty), kept
 * partly on the canvas: PLACE_KEEP pixels of the window always remain.
 */
export function movePlacementTo(mask, tx, ty) {
    const p = { ...placeOf(mask) };
    const w = p.s * mask.w, h = p.s * mask.h;
    const x0 = p.s * mask.x, y0 = p.s * mask.y;   // the window's corner before the shift
    p.tx = Math.max(PLACE_KEEP - w - x0, Math.min(tx, CW - PLACE_KEEP - x0));
    p.ty = Math.max(PLACE_KEEP - h - y0, Math.min(ty, CH - PLACE_KEEP - y0));
    mask.place = p;
}

/**
 * Scales the masked picture uniformly with corner (ax, ay) of its window -
 * in canvas pixels - held still, to whatever size the pointer at (x, y)
 * asks for. Uniform because the picture is a photograph: the pointer's
 * horizontal and vertical travel are read as one size, the larger of the
 * two, and both sides follow it.
 */
export function scalePlacementFrom(mask, ax, ay, x, y) {
    const p = placeOf(mask);
    // The point of the picture under the held corner, which must not move.
    const px = (ax - p.tx) / p.s, py = (ay - p.ty) / p.s;
    const want = Math.max(Math.abs(x - ax) / mask.w, Math.abs(y - ay) / mask.h);
    const floor = MIN_SIDE / Math.min(mask.w, mask.h);
    const s = Math.max(floor, Math.min(want, PLACE_SCALE_MAX));
    mask.place = { s, tx: ax - s * px, ty: ay - s * py };
}

/**
 * The placed mask, painted with a 2D context over the photograph `ctx` has
 * just drawn on `source` (the same canvas, or another of the same size):
 * the ground, the photograph carried through the placement and clipped to
 * the placed window, then the edge.
 *
 * compose.addFrameMask is the one that draws a mask on every real render.
 * This exists only for the live layer the zoom slider previews on (see
 * reframe.drawLiveCrop), which is painted with a bare 2D context many times
 * a second and has no fabric canvas to hand the photograph to; it follows
 * the same geometry, edge outside the window included.
 */
let _copy = null;
export function paintPlacedMask(ctx, mask, source) {
    const p = placeOf(mask);
    const win = placedWindow(mask);
    _copy = _copy || document.createElement("canvas");
    _copy.width = CW;
    _copy.height = CH;
    _copy.getContext("2d").drawImage(source, 0, 0, CW, CH);

    const path = (grow = 0) => {
        const q = new Path2D();
        if (mask.shape === "round") {
            q.ellipse(win.x + win.w / 2, win.y + win.h / 2, win.w / 2 + grow, win.h / 2 + grow, 0, 0, Math.PI * 2);
        } else {
            q.rect(win.x - grow, win.y - grow, win.w + grow * 2, win.h + grow * 2);
        }
        return q;
    };

    ctx.save();
    ctx.fillStyle = mask.fill || "#000000";
    ctx.fillRect(0, 0, CW, CH);
    ctx.clip(path());
    ctx.drawImage(_copy, p.tx, p.ty, CW * p.s, CH * p.s);
    ctx.restore();

    const sw = mask.outline ? Math.max(0, +mask.outlineWidth || 0) * p.s : 0;
    if (sw > 0) {
        ctx.save();
        ctx.lineWidth = sw;
        ctx.strokeStyle = mask.outlineColor || "#ffffff";
        ctx.stroke(path(sw / 2 - 1));
        ctx.restore();
    }
}

let _open = null;   // the live session, so a second press cannot stack a second step

/** Opens the mask step on the selected frame. */
export function openFrameMask() {
    const f = current();
    if (!f) return status("No frame selected");
    if (_open) _open.finish();
    _open = new FrameMaskSession(selectedIndex());
}

class FrameMaskSession {
    constructor(i) {
        this.i = i;
        this.frame = current();
        this.channel = channelFor(i);
        this.palette = channelPalette(this.channel);
        // An existing mask is edited as a copy, so Cancel leaves it exactly
        // as it was; a new one starts from the channel's defaults.
        this.hadMask = !!frameMaskFor(i);
        this.mask = copyFrameMask(frameMaskFor(i)) || defaultMask(this.channel);
        this.base = null;      // the frame's picture alone, once it has been drawn
        this.drag = null;
        this.thumbBusy = false;
        this.thumbDirty = false;
        this.closed = false;

        this.root = el("frameMaskStep");
        this.canvas = el("frameMaskCanvas");
        this.thumb = el("frameMaskThumb");
        this.ctx = this.canvas.getContext("2d");
        this.thumbCtx = this.thumb.getContext("2d");

        // Both drawn at the screen's own density so they are as sharp as the
        // frame they preview; laid out at their CSS size either way.
        const dpr = window.devicePixelRatio || 1;
        this.k = STAGE_LONG / Math.max(CW, CH);   // CSS px per canvas px
        this.sw = CW * this.k; this.sh = CH * this.k;
        this.canvas.width = Math.round(this.sw * dpr);
        this.canvas.height = Math.round(this.sh * dpr);
        this.canvas.style.width = `${this.sw}px`;
        this.canvas.style.height = `${this.sh}px`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const tk = THUMB_LONG / Math.max(CW, CH);
        this.thumb.width = Math.round(CW * tk * dpr);
        this.thumb.height = Math.round(CH * tk * dpr);
        this.thumb.style.width = `${CW * tk}px`;
        this.thumb.style.height = `${CH * tk}px`;

        el("frameMaskLabel").textContent = this.hadMask ? "Edit mask" : "Create mask";
        el("frameMaskRemoveBtn").classList.toggle("hidden", !this.hadMask);
        el("frameMaskApplyBtn").textContent = this.hadMask ? "Update mask" : "Apply mask";
        const slider = el("frameMaskOutlineWidth");
        slider.min = OUTLINE_RANGE.min;
        slider.max = OUTLINE_RANGE.max;

        this.bind();
        this.renderSwatches();
        this.syncControls();
        this.root.classList.remove("hidden");
        document.body.classList.add("mask-open");
        el("frameMaskApplyBtn").focus();

        this.draw();
        this.queueThumb();
        buildComposite(i, { baseOnly: true }).then((cv) => {
            if (this.closed || !cv) return;
            this.base = cv.getElement();
            this.draw();
        });
    }

    // ── Wiring ────────────────────────────────────────────────────────────
    //
    // Every listener is attached per session and removed by finish(), as in
    // the overlay's mask step, so a closed step holds nothing and an old
    // session can never answer for a new one.

    bind() {
        el("frameMaskRectBtn").onclick = () => this.setShape("rect");
        el("frameMaskRoundBtn").onclick = () => this.setShape("round");
        el("frameMaskFill").oninput = (e) => this.set({ fill: e.target.value });
        el("frameMaskOutlineBtn").onclick = () => this.set({ outline: !this.mask.outline });
        // Picking a colour or a width for the edge is asking for the edge,
        // so either one switches it on rather than waiting for a second press.
        el("frameMaskOutlineColor").oninput = (e) => this.set({ outline: true, outlineColor: e.target.value });
        el("frameMaskOutlineWidth").oninput = (e) => this.set({ outline: true, outlineWidth: +e.target.value });
        el("frameMaskCancelBtn").onclick = () => this.finish();
        el("frameMaskRemoveBtn").onclick = () => this.commit(null);
        el("frameMaskApplyBtn").onclick = () => this.commit(this.mask);
        this.root.onpointerdown = (e) => { if (e.target === this.root) this.finish(); };

        const c = this.canvas;
        c.onpointerdown = (e) => this.dragStart(e);
        c.onpointermove = (e) => this.dragMove(e);
        c.onpointerup = c.onpointercancel = (e) => this.dragEnd(e);

        this.onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); this.finish(); }
            else if (e.key === "Enter" && e.target.tagName !== "BUTTON") {
                e.preventDefault(); this.commit(this.mask);
            }
        };
        document.addEventListener("keydown", this.onKey);
    }

    finish() {
        this.closed = true;
        for (const id of ["frameMaskRectBtn", "frameMaskRoundBtn", "frameMaskOutlineBtn",
                          "frameMaskCancelBtn", "frameMaskRemoveBtn", "frameMaskApplyBtn"]) {
            el(id).onclick = null;
        }
        for (const id of ["frameMaskFill", "frameMaskOutlineColor", "frameMaskOutlineWidth"]) {
            el(id).oninput = null;
        }
        this.root.onpointerdown = null;
        const c = this.canvas;
        c.onpointerdown = c.onpointermove = c.onpointerup = c.onpointercancel = null;
        document.removeEventListener("keydown", this.onKey);
        this.root.classList.add("hidden");
        document.body.classList.remove("mask-open");
        if (_open === this) _open = null;
    }

    /**
     * Writes `mask` to the frame (null takes the mask off) and redraws it.
     *
     * To the frame the step was opened on, held rather than looked up: the
     * selection can move under an open dialog (a keyboard shortcut, a frame
     * arriving), and the mask belongs to the picture it was drawn on.
     */
    commit(mask) {
        const f = this.frame;
        const i = this.i;
        this.finish();
        if (!f || f.discarded) return;
        f.frameMask = copyFrameMask(mask);
        // A mask that has just been applied arrives selected, its box and
        // corner handles on, so the canvas shows straight away that the
        // masked picture can be moved and scaled there. The handles come off
        // everything else, as they do when the mask is pressed on. Its
        // placement on the canvas, if it had one, rides along in the copy:
        // re-cutting the window here does not send the picture back to
        // where it started.
        f.frameMaskActive = !!mask;
        if (mask) {
            f.titleActive = false;
            f.activeTextbox = -1;
            f.activeLogo = -1;
            f.activeLayer = -1;
            document.dispatchEvent(new CustomEvent("logo-selection-changed"));
        }
        f.edited = true;
        syncFrameMaskButton();
        // updatePreview is also what Undo watches (see history.noteEdit), so
        // this is one undoable action, like any other edit.
        if (i === selectedIndex()) updatePreview();
        refreshThumb(i);
    }

    // ── State ─────────────────────────────────────────────────────────────

    set(patch) {
        Object.assign(this.mask, patch);
        this.syncControls();
        this.changed();
    }

    /**
     * Switches between a rectangle and a circle. A circle needs a square box,
     * so going round keeps the window's centre and takes its shorter side;
     * going back to a rectangle leaves the square as it is, for the user to
     * pull into whatever proportions they want.
     */
    setShape(shape) {
        const m = this.mask;
        if (shape === "round" && m.w !== m.h) {
            const s = Math.min(m.w, m.h);
            m.x += Math.round((m.w - s) / 2);
            m.y += Math.round((m.h - s) / 2);
            m.w = m.h = s;
        }
        this.set({ shape });
    }

    changed() {
        this.draw();
        this.queueThumb();
    }

    /** Every control in the card, made to agree with the mask. */
    syncControls() {
        const m = this.mask;
        for (const [id, s] of [["frameMaskRectBtn", "rect"], ["frameMaskRoundBtn", "round"]]) {
            el(id).classList.toggle("btn-active", m.shape === s);
            el(id).setAttribute("aria-pressed", String(m.shape === s));
        }
        el("frameMaskFill").value = m.fill;
        const outlineBtn = el("frameMaskOutlineBtn");
        outlineBtn.classList.toggle("btn-active", !!m.outline);
        outlineBtn.setAttribute("aria-pressed", String(!!m.outline));
        outlineBtn.textContent = m.outline ? "Outline on" : "Outline off";
        el("frameMaskOutlineRow").classList.toggle("on", !!m.outline);
        el("frameMaskOutlineColor").value = m.outlineColor;
        el("frameMaskOutlineWidth").value = m.outlineWidth;
        el("frameMaskOutlineWidthValue").textContent = `${m.outlineWidth}px`;
        for (const b of document.querySelectorAll("#frameMaskFillSwatches .btn-swatch")) {
            b.classList.toggle("btn-active", b.dataset.color === m.fill);
        }
        for (const b of document.querySelectorAll("#frameMaskOutlineSwatches .btn-swatch")) {
            b.classList.toggle("btn-active", !!m.outline && b.dataset.color === m.outlineColor);
        }
    }

    /**
     * One swatch per channel colour, in both rows. Built per session because
     * the colours are the channel's, and the frame this step opened on may be
     * on a different channel from the last one.
     */
    renderSwatches() {
        const build = (rowId, onPick) => {
            const row = el(rowId);
            row.innerHTML = "";
            for (const color of this.palette) {
                const b = document.createElement("button");
                b.type = "button";
                b.className = "btn-swatch";
                b.style.background = color;
                b.dataset.color = color;
                b.title = color.toUpperCase();
                b.setAttribute("aria-label", `Colour ${color.toUpperCase()}`);
                b.onclick = () => onPick(color);
                row.appendChild(b);
            }
        };
        build("frameMaskFillSwatches", (fill) => this.set({ fill }));
        build("frameMaskOutlineSwatches", (outlineColor) => this.set({ outline: true, outlineColor }));
    }

    // ── Gesture ───────────────────────────────────────────────────────────
    //
    // Three things a press can mean, decided by where it lands: on a corner
    // handle it reshapes the window from that corner, with the opposite one
    // held still; inside the window it moves the window; anywhere else it
    // draws a new window from that point, which is the quickest way to put
    // one somewhere else entirely.

    /** The pointer, in canvas pixels, clamped to the canvas. */
    point(e) {
        const r = this.canvas.getBoundingClientRect();
        const k = (r.width / this.sw) * this.k;   // screen px per canvas px, whatever CSS did to the stage
        const x = (e.clientX - r.left) / k, y = (e.clientY - r.top) / k;
        return [Math.max(0, Math.min(CW, x)), Math.max(0, Math.min(CH, y)), k];
    }

    dragStart(e) {
        if (e.button !== 0) return;
        const [x, y, k] = this.point(e);
        const m = this.mask;
        const hit = maskCornerAt(m, x, y, HANDLE_REACH / k);
        if (hit !== -1) {
            const [ax, ay] = oppositeCorner(m, hit);
            this.drag = { mode: "resize", ax, ay };
        } else if (insideMask(m, x, y)) {
            this.drag = { mode: "move", sx: x, sy: y, ox: m.x, oy: m.y };
        } else {
            this.drag = { mode: "resize", ax: x, ay: y };
            resizeMask(m, x, y, x, y, e.shiftKey);
        }
        this.canvas.setPointerCapture(e.pointerId);
        this.canvas.classList.add("dragging");
        e.preventDefault();
    }

    dragMove(e) {
        if (!this.drag) {
            this.hover(e);
            return;
        }
        const [x, y] = this.point(e);
        const d = this.drag, m = this.mask;
        if (d.mode === "move") moveMaskTo(m, d.ox + x - d.sx, d.oy + y - d.sy);
        else resizeMask(m, d.ax, d.ay, x, y, e.shiftKey);
        this.changed();
    }

    dragEnd(e) {
        if (!this.drag) return;
        this.drag = null;
        if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
        this.canvas.classList.remove("dragging");
        roundMask(this.mask);
        this.changed();
    }

    /** The cursor, saying which of the three a press here would do. */
    hover(e) {
        const [x, y, k] = this.point(e);
        const m = this.mask;
        const reach = HANDLE_REACH / k;
        const corner = maskCornerAt(m, x, y, reach);
        let cursor = "crosshair";
        if (corner !== -1) cursor = (corner === 0 || corner === 3) ? "nwse-resize" : "nesw-resize";
        else if (insideMask(m, x, y)) cursor = "move";
        this.canvas.style.cursor = cursor;
    }

    // ── Drawing ───────────────────────────────────────────────────────────

    /**
     * The stage: the picture, the ground with the window cut out of it, the
     * edge, and the handles. The ground and the edge are drawn the way
     * compose.addFrameMask draws them — edge outside the window — so the stage
     * shows what the frame will get; the handles are the step's own.
     */
    draw() {
        const ctx = this.ctx, k = this.k, m = this.mask;
        ctx.clearRect(0, 0, this.sw, this.sh);
        if (this.base) ctx.drawImage(this.base, 0, 0, this.sw, this.sh);

        const x = m.x * k, y = m.y * k, w = m.w * k, h = m.h * k;
        const path = (grow = 0) => {
            const p = new Path2D();
            if (m.shape === "round") {
                p.ellipse(x + w / 2, y + h / 2, w / 2 + grow, h / 2 + grow, 0, 0, Math.PI * 2);
            } else {
                p.rect(x - grow, y - grow, w + grow * 2, h + grow * 2);
            }
            return p;
        };

        const ground = new Path2D();
        ground.rect(0, 0, this.sw, this.sh);
        ground.addPath(path());
        ctx.fillStyle = m.fill;
        ctx.fill(ground, "evenodd");

        if (m.outline && m.outlineWidth > 0) {
            const sw = m.outlineWidth * k;
            ctx.lineWidth = sw;
            ctx.strokeStyle = m.outlineColor;
            ctx.stroke(path(sw / 2));
        }

        // The box the handles belong to, dashed, so a circle's corners are
        // visibly corners of something.
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(255, 250, 0, 0.9)";
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        ctx.restore();
        ctx.fillStyle = "#fffa00";
        ctx.strokeStyle = "#060610";
        ctx.lineWidth = 1.5;
        for (const [cx, cy] of maskCorners(m)) {
            const hx = cx * k - HANDLE_SIZE / 2, hy = cy * k - HANDLE_SIZE / 2;
            ctx.fillRect(hx, hy, HANDLE_SIZE, HANDLE_SIZE);
            ctx.strokeRect(hx, hy, HANDLE_SIZE, HANDLE_SIZE);
        }
    }

    /**
     * Redraws the miniature: the whole frame — title, overlays and all — with
     * the mask as it stands in the step. A full composite is a fabric render
     * of the frame, so at most one is in flight; anything that changes while
     * it runs marks it dirty, and one more is drawn when it lands. During a
     * drag that is as fast as the composites can go and never a queue.
     */
    queueThumb() {
        if (this.thumbBusy) { this.thumbDirty = true; return; }
        this.thumbBusy = true;
        this.thumbDirty = false;
        const mask = { ...this.mask };
        buildComposite(this.i, { mask }).then((cv) => {
            if (this.closed) return;
            if (cv) {
                const t = this.thumb;
                this.thumbCtx.clearRect(0, 0, t.width, t.height);
                this.thumbCtx.drawImage(cv.getElement(), 0, 0, t.width, t.height);
            }
        }).finally(() => {
            this.thumbBusy = false;
            if (this.thumbDirty && !this.closed) this.queueThumb();
        });
    }
}
