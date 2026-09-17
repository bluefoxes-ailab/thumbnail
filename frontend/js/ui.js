import { el } from "./config.js";
import { current, frames, selectedIndex } from "./state.js";
import { contentProblems } from "./channels.js";

export function status(msg) { el("status").textContent = msg; }

export function showProgress(stage, percent) {
    el("progressSection").classList.remove("hidden");
    el("progressFill").style.width = percent + "%";
    el("progressText").textContent = `${stage} (${percent}%)`;
}

export function hideProgress() {
    el("progressSection").classList.add("hidden");
    el("progressFill").style.width = "0%";
    el("progressText").textContent = "";
}

// frame_ids currently being enhanced.
export const enhancing = new Set();

// Set while a manual gesture owns the canvas: a drag in progress, a fine-zoom
// adjustment, or a committed reframe still in flight. Any of them means the
// frozen drop preview must stay up.
export const pending = { drag: null, zooming: false, reframing: false };

let controlsLocked = false;

export const isLocked = () => controlsLocked;

/**
 * Flip/download/zoom act on whatever the editing pipeline (enhance, reframe,
 * flip compose) is currently producing for the selected frame — locked while
 * one of those is in flight so the user can't flip/download a stale image or
 * reframe-zoom out from under a request already using the current crop.
 *
 * Frame switching and Variation are deliberately NOT gated by this: switching
 * frames must stay free even mid-pipeline, and Variation is meant to
 * interrupt/replace whatever's in flight rather than wait for it.
 */
export function setControlsLocked(locked) {
    controlsLocked = locked;
    for (const id of ["flipImageBtn", "flipTextBtn", "gradientBtn", "downloadBtn",
                      "presetNoneBtn", "presetNaturalBtn", "presetFullBtn"]) {
        el(id).disabled = locked;
    }
    el("zoomInBtn").classList.toggle("locked", locked);
    el("zoomOutBtn").classList.toggle("locked", locked);
    el("zoomSlider").disabled = locked;
    syncBatchButton();
}

export function syncBatchButton() {
    const n = frames().filter(f => f.picked).length;
    const btn = el("batchBtn");
    // The count only reads as yellow while the button is actually active
    // (n > 0) — disabled/zero stays plain text.
    btn.innerHTML = n > 0
        ? `Download selected <span class="count-badge">(${n})</span>`
        : `Download selected (${n})`;
    btn.disabled = n === 0 || controlsLocked;
}

export function showSpinner() {
    el("reframeSpinner").style.display = "flex";
    setControlsLocked(true);
}

/**
 * The spinner tracks the WHOLE editing pipeline, not just the raw re-crop: it
 * only comes down when no enhance pass is in flight for the frame the user is
 * looking at.
 */
export function maybeHideSpinner() {
    const f = current();
    if (f && enhancing.has(f.frame_id)) return;
    el("reframeSpinner").style.display = "none";
    setControlsLocked(false);
}

/**
 * Drops all in-flight bookkeeping, for when a new video replaces the grid.
 *
 * frame_ids restart at 0 for every video, so anything keyed by them from the
 * previous one is not just stale but actively wrong. `enhancing` is the
 * damaging case: an entry left in it makes enhanceFrame return immediately
 * for the NEW video's frame with that id, so that frame silently keeps its
 * plain unrestored crop with no spinner and no error to say why. The gesture
 * flags and the overlays they hold up are cleared for the same reason — a
 * reframe still marked pending keeps hideReframeWait from ever taking the
 * frozen drop preview down, over a frame that no longer exists.
 */
export function resetPending() {
    enhancing.clear();
    pending.drag = null;
    pending.zooming = false;
    pending.reframing = false;
    el("liveCropLayer").style.display = "none";
    el("reframeSpinner").style.display = "none";
    setControlsLocked(false);
}

/**
 * Hides the frozen drop preview left up by a reframe drop — called once a
 * fresh preview render actually reaches the canvas, and on reframe failure
 * paths. No-op while a gesture is still pending: ANY render reaching the
 * canvas (a stale enhance, a title retype) must not tear down the frozen
 * preview, or the old framing flashes until the commit's render lands.
 */
export function hideReframeWait() {
    if (pending.drag || pending.zooming || pending.reframing) return;
    el("liveCropLayer").style.display = "none";
    maybeHideSpinner();
}


/**
 * Turns frame `i`'s card from "still being worked on" into "ready".
 *
 * Only ever called forwards. A card starts pending on a frame fetch (see
 * main.renderGrid), and the one thing that clears it is the frame's finished
 * pixels arriving — which happens exactly once per frame, and never unhappens:
 * a later re-render replaces a picture the user has already been shown, so it
 * is not a return to "not ready yet".
 *
 * A no-op everywhere else. On a thumbnail run no card is ever marked pending,
 * because only the selected frame is restored and the other nineteen would sit
 * grey and unclickable for the whole session.
 */
export function markFrameReady(i) {
    const card = el(`card-${i}`);
    if (card) card.classList.remove("pending");
    // A frame becoming ready is a frame an arrow may now step onto, and on a
    // fetch the whole grid arrives this way - so the two controls are asked
    // again here rather than left describing the set as it was when the run
    // started.
    syncFrameNav();
}

/**
 * The frame an arrow beside the canvas moves to, or -1 when there is none
 * that way.
 *
 * Steps OVER a card the edit has not reached yet, for the same reason the
 * grid makes one inert (see main.renderGrid): a pending card is not something
 * the user can look at, and an arrow that landed on one would answer a click
 * with a grey rectangle. On a thumbnail run nothing is ever pending, so this
 * walks exactly one place and stops.
 *
 * The DOM is asked rather than the frame record, so that this and the grid
 * cannot come to different conclusions about what is clickable - they are
 * reading the same class.
 */
export function nextSelectable(from, step) {
    const total = frames().length;
    for (let i = from + step; i >= 0 && i < total; i += step) {
        const card = el(`card-${i}`);
        if (!card || !card.classList.contains("pending")) return i;
    }
    return -1;
}

/**
 * Shows the two arrows flanking the canvas, and dims whichever has nowhere
 * left to go.
 *
 * Hidden outright while there is a single frame, where they are two controls
 * that could never do anything. Disabled rather than hidden at the two ends
 * of a longer set: which side of the canvas an arrow is on is how the user
 * reads which way it goes, and one that vanishes at the last frame makes the
 * other appear to have moved.
 */
export function syncFrameNav() {
    const prev = el("prevFrameBtn"), next = el("nextFrameBtn");
    if (!prev || !next) return;
    const many = frames().length > 1;
    prev.classList.toggle("hidden", !many);
    next.classList.toggle("hidden", !many);
    prev.disabled = nextSelectable(selectedIndex(), -1) < 0;
    next.disabled = nextSelectable(selectedIndex(), +1) < 0;
}

// ── What failed to load ───────────────────────────────────────────────────────

/**
 * Reports a channel pack that did not load.
 *
 * A pack is data, so the failure mode is silence: a channel with a stray
 * comma in its JSON simply is not in the dropdown, and the app looks like it
 * forgot a brand rather than like it found a broken file. The message names
 * the pack, because with the folder sitting right there that is the whole fix.
 */
export function reportContentProblems() {
    const problems = contentProblems();
    const box = el("contentWarning");
    if (!box || !problems.length) return;
    box.textContent = problems.length === 1
        ? `Channel problem — ${problems[0]}`
        : `${problems.length} channel problems — ` + problems.join("  •  ");
    box.classList.remove("hidden");
}
