export const API = "";

// Must match the backend's target_w()/target_h() (reframe_engine.py). Every
// frame and wide-frame the backend sends is in these dimensions, and all text
// layout is relative to them.
//
// 1280×720, the YouTube thumbnail, until a channel says otherwise — the
// Snapchat pack's frames are 540×960 (see `capture` in channels.js). Both
// sides of that decision come from the same pack: the backend is told the size
// in the framing it is sent, and setCanvasSize below tells this side.
//
// `let`, not `const`, and deliberately never copied into a local at module
// level anywhere: an ES module export is a live binding, so every `CW` already
// written across compose.js, reframe.js, text.js and editor.js follows this one
// the moment it changes, with nothing to keep in step by hand.
const THUMBNAIL_CW = 1280, THUMBNAIL_CH = 720;

export let CW = THUMBNAIL_CW, CH = THUMBNAIL_CH;

// Grid thumbnails are drawn well under full size, so a gridful of them stays
// cheap to regenerate. Long side fixed; the short one follows the canvas, so a
// portrait run gets portrait cards rather than letterboxed landscape ones.
export let THUMB_W = 320, THUMB_H = 180;

const THUMB_LONG_SIDE = 320;

/**
 * Points the whole frontend at a canvas of this size.
 *
 * Called once per run, before anything is drawn, from the channel the run was
 * started under. Everything that lays out against the canvas — the fabric
 * canvases, the title metrics, the drag geometry — reads these through the
 * live bindings above and so needs no telling.
 */
export function setCanvasSize(width, height) {
    CW = Math.max(1, Math.round(width));
    CH = Math.max(1, Math.round(height));
    const k = THUMB_LONG_SIDE / Math.max(CW, CH);
    THUMB_W = Math.max(1, Math.round(CW * k));
    THUMB_H = Math.max(1, Math.round(CH * k));
}

/**
 * Back to the thumbnail canvas — what a run under any channel but a capture
 * one produces. Called at the start of every such run rather than only when
 * leaving a capture, so the size is stated by each run instead of inherited
 * from whatever the last one happened to be.
 */
export function resetCanvasSize() {
    setCanvasSize(THUMBNAIL_CW, THUMBNAIL_CH);
}

export const FONT = "'Trade Gothic Next LT Pro Heavy Compressed', Impact, 'Arial Black', sans-serif";

// Every line slot the markup has. How many of them a channel actually SHOWS is
// the channel's (see `layout.lines` in channels.js) — the fields are all in
// index.html at all times and the panel hides the ones this brand does not use,
// for the same reason the freeform block sits beside them: building the fields
// per channel would throw away what had been typed every time the user tried
// another brand.
export const LINE_IDS = ["line1", "line2", "line3", "line4"];

// The one field a channel whose breaks are the user's own is typed into
// instead of those three (see `freeform` in channels.js). Both exist in the
// markup at all times and the panel shows whichever the channel calls for —
// swapping the markup itself would throw away what the user had typed every
// time they tried another brand.
export const BLOCK_ID = "lineBlock";

// Every field a title can be typed into. What main.js binds its listeners to,
// so a field added here is answered the same way the others are.
export const TITLE_FIELD_IDS = [...LINE_IDS, BLOCK_ID];

// The square dragged to resize something on the canvas — an image overlay,
// or a title belonging to a channel that lets its titles be placed by hand.
// One number, because they are one gesture and sizing them apart would only
// make the same handle harder to hit in one of the two places.
export const HANDLE_SIZE = 26;

// YouTube rejects thumbnail files over 2MB, so every export is capped.
export const MAX_EXPORT_BYTES = 2 * 1024 * 1024;

export const DEFAULT_PRESET = "natural";
export const FIDELITY = 0.60;

export const el = (id) => document.getElementById(id);
