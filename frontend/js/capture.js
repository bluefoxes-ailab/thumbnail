/**
 * capture.js — The app as a frame fetcher.
 *
 * One channel does not make thumbnails. It takes a video and hands back a
 * numbered set of stills, edited and ready to use, with no brand furniture
 * and nothing to compose — see `capture` in channels.js and frame_grab.py on
 * the backend. Everything that differs on THIS side of the app is here, in
 * one file, rather than as a condition inside each of the dozen controls that
 * has nothing to do in that mode.
 *
 * The mode is a property of the RUN, not of the page or of the selected
 * channel. That distinction is the whole reason this module holds state at
 * all: the channel picker can be changed at any moment, including while a
 * grid of already-fetched frames is on screen, and the frames in front of the
 * user do not stop being frames because somebody opened the dropdown. So the
 * mode is entered when a run's frames LAND and left when a thumbnail run's do
 * — never when the picker moves.
 *
 * ## What the mode actually changes
 *
 * Four things, and they are deliberately different mechanisms:
 *
 *   the canvas size — config.setCanvasSize, before anything is drawn. 540x960
 *   instead of 1280x720, which the backend was told separately in the same
 *   channel's framing (see channels.framingFor).
 *
 *   which controls exist — a class on <body>, and CSS. Every one of those
 *   buttons is already shown and hidden by its own sync function against its
 *   own channel-shaped question, and teaching all of them a second question
 *   would be a dozen edits that all have to keep agreeing. One rule in the
 *   stylesheet cannot fall out of step with itself.
 *
 *   what the files are called — frameName below, which is the one piece of
 *   this that the user actually receives.
 *
 *   what is written on them — applyCaptions below, for a channel that asks
 *   for it (`capture.captions`). This is the one thing in the list that puts
 *   something BACK: the mode began by taking the title away, on the grounds
 *   that a fetched frame is the deliverable and has nothing over it. A
 *   channel that captions its stills with what was being said over them wants
 *   the title machinery entire — the panel, the drag, the download — and gets
 *   it, because a caption is a title in every respect except where the words
 *   came from.
 */

import { el, setCanvasSize, resetCanvasSize } from "./config.js";
import { captureFor } from "./channels.js";
import { frames } from "./state.js";
import { captionLines } from "./caption.js";
import { prepareTitleFont } from "./text.js";

// The capture block the frames currently on screen were fetched under, or null
// when they are thumbnails. See the module header for why this is not simply
// "what does the picker say".
let _active = null;

/** The capture block this grid was fetched under, or null for a thumbnail run. */
export const captureMode = () => _active;

/** Whether the frames on screen came from a frame fetch. */
export const isCapture = () => _active !== null;

/**
 * Paints the slider's filled portion and its readout to match its value.
 *
 * A range input gives no way to style the part of the track behind the handle,
 * so the fill is a gradient whose stop is this custom property — the same
 * arrangement the zoom bar uses.
 */
export function syncCountValue() {
    const slider = el("frameCount");
    if (!slider) return;
    const min = Number(slider.min), max = Number(slider.max), value = Number(slider.value);
    const fraction = max > min ? (value - min) / (max - min) : 0;
    slider.style.setProperty("--count-fill", fraction);
    el("frameCountValue").textContent = String(value);
}

/**
 * Shows the frame-count slider for a channel that fetches frames, and hides it
 * for one that does not.
 *
 * Bound to the picker above the link box, so it answers as the channel is
 * chosen rather than once the run has started — the count is part of the same
 * question the channel is, and both are spent the moment the video is read.
 *
 * The slider keeps wherever the user last put it, as long as the new channel's
 * range still contains it. Snapping back to the pack's default every time the
 * dropdown moved would throw away a deliberate "I want 120 of them" for
 * nothing.
 */
export function syncCaptureFields(channelId) {
    const capture = captureFor(channelId);
    const row = el("frameCountRow");
    if (!row) return;

    row.classList.toggle("hidden", !capture);
    if (!capture) return;

    const slider = el("frameCount");
    const { min, max, default: preferred } = capture.frames;
    const current = Number(slider.value);
    slider.min = min;
    slider.max = Math.max(min, max);
    slider.value = (current >= min && current <= max) ? current : Math.max(min, preferred);
    syncCountValue();

    el("frameCountHint").textContent =
        `How many frames to cut from the video — drag the handle, or click the track. `
        + `${min} to ${max}, opening at ${preferred}. Blurred and blank frames are `
        + "rejected, so a video without enough usable moments returns fewer.";
}

/**
 * How many frames to ask for, as a whole number no smaller than the channel's
 * floor.
 *
 * Clamped here as well as on the backend (see api.CAPTURE_COUNT_RANGE),
 * because the two clamps answer different people: this one keeps the control
 * from sending nonsense, and that one keeps the backend from trusting
 * anything.
 */
export function requestedCount(channelId) {
    const capture = captureFor(channelId);
    if (!capture) return null;
    const chosen = Math.round(Number(el("frameCount").value));
    const min = capture.frames.min;
    return Number.isFinite(chosen) ? Math.max(min, chosen) : Math.max(min, capture.frames.default);
}

/**
 * Puts the page into (or out of) frame-fetch mode for the frames that have
 * just arrived.
 *
 * Called once per run, after the result lands and before anything is drawn
 * from it: the canvas size decides the shape of every canvas built below it,
 * and re-shaping one after it has been painted is a frame drawn to the wrong
 * aspect ratio.
 */
export function enterMode(capture) {
    _active = capture || null;
    if (_active) setCanvasSize(_active.output.width, _active.output.height);
    else resetCanvasSize();
    document.body.classList.toggle("capture-mode", !!_active);
    // A second class rather than a second question inside the first, for the
    // reason the first one is a class at all: capture mode hides the whole
    // text panel, and a capture channel that writes on its frames needs it
    // back. One rule in the stylesheet says which of the two this run is (see
    // the capture-mode block in styles.css) instead of every control in the
    // panel learning to ask.
    document.body.classList.toggle("captions-mode", !!(_active && _active.captions));
}

/**
 * Fills every frame's title with the sentence that was being said over it.
 *
 * The words arrive with the frames (see `caption` in api.py); what happens
 * here is only the break into lines, and it is separate from makeFrame for
 * one reason: it MEASURES. Every break is chosen by measuring the words in
 * the channel's own face (see caption.js), so it waits for that face the way
 * every other measurement in the app does — against a fallback, each caption
 * would be broken for a font nobody can see, which is exactly the failure
 * fontguard exists to catch. In practice the wait is already over: the face
 * was fetched when the channel was picked, long before the video finished
 * processing.
 *
 * The lines it writes are a starting point and nothing more. The user has a
 * text box with them in it and every break is theirs to move from that moment
 * on; nothing calls this again.
 */
export async function applyCaptions() {
    if (!_active || !_active.captions) return;
    await prepareTitleFont();
    for (const frame of frames()) {
        if (!frame.caption) continue;
        const lines = captionLines(frame.caption);
        // Written into the slots the markup has rather than assigned, so a
        // frame's lines stay the fixed-length array everything else expects
        // (see state.makeFrame and editor.updateLines).
        frame.lines = frame.lines.map((_, i) => lines[i] || "");
    }
}

/**
 * What frame `index` of `total` is called, in the grid and in the file the
 * user ends up with.
 *
 * The grid's numbering, so "frame 4" means the fourth card, which — the grid
 * being in the order the moments occur in the video — is also the fourth
 * moment. That is the whole point of the naming: the user picks a frame out of
 * the grid by eye and finds it in the folder under the number they read.
 *
 * Padded to the width of the largest number in the set. "frame 1" is what was
 * asked for and "frame 01" is what delivers it: without the padding every file
 * manager on every platform sorts frame 10 between frame 1 and frame 2, so the
 * folder would open in an order that contradicts the grid it was named after —
 * which is exactly the correspondence the numbering exists to provide.
 */
export function frameName(index, total) {
    const width = String(Math.max(1, total)).length;
    return `frame ${String(index + 1).padStart(width, "0")}`;
}
