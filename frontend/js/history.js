/**
 * Undo and Redo for Static Studio: the last five things the user did to a
 * frame, on the canvas or in the text panel, and the way back to each.
 *
 * WATCHED, NOT REPORTED. The obvious design is for every edit to announce
 * itself — a push() at the end of updateLines, of every drag, of every colour
 * button — and it is the one this deliberately does not take. There are some
 * forty places a frame is changed from, across five modules, and more get
 * added with every feature; a history that depends on each of them
 * remembering to report in is a history with a gap in it the first time one
 * forgets, and nothing would show the gap until a user pressed Undo and the
 * wrong thing came back. Instead the frame is LOOKED AT: after anything that
 * could have changed it, its editable state is compared with what it was
 * last time, and a difference is an action. A control added next year is
 * undoable the day it ships, because it changes the frame and that is all
 * this needs.
 *
 * What "after anything" means is the preview being redrawn (compose.
 * updatePreview calls noteEdit — every edit ends there, including the ones
 * that finish asynchronously, like an overlay that has to be decoded first),
 * backed up by the ordinary input events on the page. Both only SCHEDULE a
 * look, QUIET_MS after the last of them, so that:
 *
 *   - a word typed into the title is one action, not one per letter;
 *   - a drag is one action from where it started to where it was let go —
 *     nothing is looked at while a pointer is held down, however long the
 *     drag takes;
 *   - a slider dragged through twenty values is one action.
 *
 * ONE STACK FOR THE PAGE, not one per frame, because what the user means by
 * "undo" is "the last thing I did", and the last thing may have been on the
 * frame before this one. Each entry remembers which frame it was made on, and
 * undoing it goes to that frame first: changing a picture the user cannot see
 * would look, from where they are sitting, like the button doing nothing.
 *
 * WHAT IS NOT IN IT, and why. Everything here is state the browser holds and
 * can put back for free. The photograph is not: Variation, Upload frame and a
 * reframe each replace it with a new picture from the backend and release the
 * old one (see state.setFrameImage), so there is nothing left to go back to.
 * The cut-out figures are released the same way, which is why a change in
 * WHICH figures a frame has is treated as the frame being re-shot rather than
 * as an edit (see isUserEdit) — their placement, while they stay, is undoable
 * like anything else. And the channel is left out because switching one is a
 * re-branding with its own defaults to apply, not a value to write back.
 */

import { el } from "./config.js";
import { frames, selectedIndex } from "./state.js";
import { textEffects, setTextEffects } from "./channels.js";
import { updatePreview, refreshThumb, refreshAllThumbs } from "./compose.js";
import {
    loadTextPanel, syncFlipButtons, syncGradientButton, syncBackgroundButton,
    syncFigureGlowButton, syncLogoControls, syncTextEffects, recomposeSelected,
} from "./editor.js";
import { syncFrameMaskButton } from "./framemask.js";

/** How many actions can be undone — the five most recent. */
export const HISTORY_LIMIT = 5;

/**
 * How long the page has to be still before a change is written down.
 *
 * Long enough that a word typed at an ordinary pace is one action rather than
 * five; short enough that a user who makes a change and goes straight for
 * Undo finds it there (and if they are quicker than this, undo() looks first
 * anyway — see flush).
 */
const QUIET_MS = 500;

/**
 * The parts of a frame that are the user's edits, and that putting back is
 * enough to put the frame back.
 *
 * The `...Touched` flags ride along with the values they guard: they record
 * that the user decided, and undoing the decision has to undo that too, or a
 * later channel switch would treat a value the user never chose as theirs.
 */
const FIELDS = [
    "lines", "hl", "titleBox", "titleActive", "titleAlign", "glow",
    "textbox", "textboxActive", "frameMask", "frameMaskActive",
    "background", "figureGlow", "textColor", "panelColor", "highlightColor",
    "gradient", "gradientTouched", "flipImage", "flipText", "flipsTouched",
    "logos", "activeLogo", "layers", "activeLayer",
];

/**
 * A copy deep enough that nothing the user does to the frame afterwards can
 * reach into it — except under keys starting with "_", which are held by
 * reference. Those are the decoded pictures and the caches built from them
 * (an overlay's `_img`, its `_alpha`): they are the same picture whatever
 * happens to the record around them, and copying them would be expensive
 * and pointless.
 */
function clone(v) {
    if (Array.isArray(v)) return v.map(clone);
    if (v && typeof v === "object" && Object.getPrototypeOf(v) === Object.prototype) {
        const out = {};
        for (const [k, x] of Object.entries(v)) out[k] = k.startsWith("_") ? x : clone(x);
        return out;
    }
    return v;
}

/**
 * What frame `i` looks like right now, as far as Undo is concerned.
 *
 * The caption shadow and outline are in it although they belong to the page
 * rather than the frame (see channels.textEffects): to the user they are
 * something done to the text on the canvas, and Undo not reaching them would
 * be the exception that has to be explained.
 */
function snapshot(i) {
    const f = frames()[i];
    const snap = {};
    for (const k of FIELDS) snap[k] = clone(f[k]);
    snap.effects = clone(textEffects());
    return snap;
}

/** A string that is equal for two snapshots exactly when they look the same. */
const signature = (snap) =>
    JSON.stringify(snap, (k, v) => (k.startsWith("_") ? undefined : v));

/** Which cut-out figures a snapshot holds — the pictures, not where they stand. */
const figureSet = (snap) => (snap.layers || []).map(l => l.src).join("|");

// Each entry: { i, before, after } — the frame, and the two snapshots either
// side of the one action. Oldest first.
let undoStack = [];
let redoStack = [];

// What the frame being watched looked like the last time it was looked at.
// Null until there is a frame to watch.
let baseline = null;   // { i, frames, snap, sig }

let timer = null;
let pointerHeld = false;
let restoring = false;
let selectFrame = null;   // main.selectFrame, handed over by initHistory

/**
 * Starts looking at frame `i` afresh, without writing anything down: whatever
 * it looks like now is where the next action will be measured from.
 */
function watch(i) {
    const list = frames();
    if (!list[i]) { baseline = null; return; }
    const snap = snapshot(i);
    baseline = { i, frames: list, snap, sig: signature(snap) };
}

/**
 * Whether the difference between `before` and `after` is something the user
 * did.
 *
 * Not when the frame's figures were swapped for others: that is a cutout
 * arriving, a Variation or an upload — the backend answering, not the user
 * editing — and the pictures it replaced have already been released, so an
 * entry built on them could only ever put back figures that no longer exist.
 */
const isUserEdit = (before, after) => figureSet(before) === figureSet(after);

/**
 * Looks at the watched frame and, if it has changed, writes the change down.
 * Then starts watching whichever frame is on screen now.
 *
 * The frame compared is the one that was being WATCHED, not the selected
 * one, which is what lets a title typed on frame 3 still be recorded as frame
 * 3's when the user clicks over to frame 4 before the quiet period is up.
 */
function look() {
    clearTimeout(timer);
    timer = null;
    const list = frames();
    // A new run, or the frames of the old one replaced: none of the entries
    // describe anything that exists any more.
    if (baseline && baseline.frames !== list) reset();
    if (baseline) {
        const f = list[baseline.i];
        if (f && !f.discarded) {
            const now = snapshot(baseline.i);
            const sig = signature(now);
            if (sig !== baseline.sig && isUserEdit(baseline.snap, now)) {
                undoStack.push({ i: baseline.i, before: baseline.snap, after: now });
                if (undoStack.length > HISTORY_LIMIT) undoStack.shift();
                redoStack = [];
            }
        }
    }
    if (list.length) watch(selectedIndex());
    syncHistoryButtons();
}

/**
 * Asks for a look once the page has been still for QUIET_MS. Called from
 * every redraw of the preview and from the page's input events, so it has to
 * cost nothing — it only moves a timer.
 */
export function noteEdit() {
    if (restoring) return;
    clearTimeout(timer);
    timer = pointerHeld ? null : setTimeout(look, QUIET_MS);
    // Lit while a real change is waiting to be written down, so that Undo
    // pressed straight after a first edit is never a dead button. Only worth
    // measuring while the stack is empty — otherwise the button is lit anyway.
    if (!pointerHeld && !undoStack.length) syncHistoryButtons(changePending());
}

/** Whether the watched frame differs from how it was last written down. */
function changePending() {
    if (!baseline || baseline.frames !== frames()) return false;
    const f = frames()[baseline.i];
    return !!f && !f.discarded && signature(snapshot(baseline.i)) !== baseline.sig;
}

/** Writes down anything still waiting, now rather than when the page goes quiet. */
export function flush() {
    if (timer !== null || pointerHeld) look();
}

/**
 * The selection moved. Whatever was done to the frame being left is written
 * down first, as that frame's, and the new frame is watched from the state it
 * arrived in — including whatever the carry-over gave it, which the user did
 * not do and so cannot undo.
 */
export function frameSelected() {
    if (restoring) return;
    look();
}

/**
 * Something changed frames that the user should not be able to undo frame by
 * frame — "Split the text", which rewrites every frame in the grid at once.
 * The caller writes down anything already pending before it (flush), and
 * this watches the frames again from after it.
 */
export function rebaseline() {
    clearTimeout(timer);
    timer = null;
    if (frames().length) watch(selectedIndex()); else baseline = null;
    syncHistoryButtons();
}

/** Forgets everything — the frames it was about are gone. */
export function reset() {
    undoStack = [];
    redoStack = [];
    baseline = null;
    clearTimeout(timer);
    timer = null;
    syncHistoryButtons();
}

/**
 * Puts frame `i` back to `snap` and redraws everything that shows it.
 *
 * The sync calls are the same ones a change of selection makes (see
 * main.selectFrame), because that is the same question — every control on
 * the page has to agree with a frame that has just changed under it.
 */
function apply(i, snap) {
    const f = frames()[i];
    const was = snapshot(i);
    // The figures are put back only if they are still the SAME figures. An
    // entry can outlive them: a cutout that lands after it was written swaps
    // the frame's figures for new ones and releases the old pictures (see
    // isUserEdit), and writing the old list back would stand figures on the
    // frame whose images no longer exist. Then the figures stay as they are
    // and everything else is undone around them.
    const sameFigures = figureSet(was) === figureSet(snap);
    for (const k of FIELDS) {
        if (!sameFigures && (k === "layers" || k === "activeLayer")) continue;
        f[k] = clone(snap[k]);
    }
    f.edited = true;

    const effectsChanged = signature({ e: was.effects }) !== signature({ e: snap.effects });
    if (effectsChanged) setTextEffects(clone(snap.effects));

    loadTextPanel();
    syncFlipButtons();
    syncGradientButton();
    syncBackgroundButton();
    syncFigureGlowButton();
    syncLogoControls();
    syncTextEffects();
    syncFrameMaskButton();

    // The photo's mirror and the backend's gradient are baked into the picture
    // the backend returns, so putting those flags back means asking for the
    // picture again. Everything else is drawn here.
    if (was.flipImage !== snap.flipImage || was.gradient !== snap.gradient) {
        recomposeSelected();
    }
    updatePreview();
    if (effectsChanged) refreshAllThumbs(); else refreshThumb(i);
}

/**
 * Moves one entry from `from` to `to`, putting its frame into the state on
 * `side` of it — "before" for Undo, "after" for Redo.
 */
function step(from, to, side) {
    flush();
    const list = frames();
    while (from.length) {
        const entry = from.pop();
        const f = list[entry.i];
        // A frame thrown out of the run since: there is nothing on screen to
        // put back, so the entry goes with it and the next one is tried.
        if (!f || f.discarded) continue;
        restoring = true;
        try {
            if (entry.i !== selectedIndex() && selectFrame) selectFrame(entry.i);
            apply(entry.i, entry[side]);
        } finally {
            restoring = false;
        }
        to.push(entry);
        if (to.length > HISTORY_LIMIT) to.shift();
        break;
    }
    // Whatever the frame looks like now is the starting point for the next
    // action — the restored state is not itself an edit.
    clearTimeout(timer);
    timer = null;
    if (list.length) watch(selectedIndex());
    syncHistoryButtons();
}

export const undo = () => step(undoStack, redoStack, "before");
export const redo = () => step(redoStack, undoStack, "after");

/**
 * Enables the two buttons for what there is to do. `pending` is a change not
 * yet written down (see noteEdit), which Undo can already act on because it
 * flushes first.
 */
function syncHistoryButtons(pending = false) {
    const u = el("undoBtn"), r = el("redoBtn");
    if (!u || !r) return;
    u.disabled = !(undoStack.length || pending);
    r.disabled = !redoStack.length;
}

/**
 * Whether a key press is meant for a field rather than for the page. Inside
 * a text box Ctrl+Z is the box's own undo, which is the one the user expects
 * while typing, and the page's must not take it from them.
 */
const typingInField = (t) =>
    t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));

/**
 * Binds the buttons, the shortcuts and the events that schedule a look.
 * `select` is main.selectFrame — Undo moves to the frame an entry belongs to
 * the same way a click on its card would, carry-over and all.
 */
export function initHistory({ select, enabled }) {
    selectFrame = select;
    el("undoBtn").addEventListener("click", undo);
    el("redoBtn").addEventListener("click", redo);

    // Held from the first press to the last release, anywhere: a drag that
    // leaves the canvas is still the same drag.
    window.addEventListener("pointerdown", () => { pointerHeld = true; clearTimeout(timer); timer = null; }, true);
    const release = () => {
        if (!pointerHeld) return;
        pointerHeld = false;
        noteEdit();
    };
    window.addEventListener("pointerup", release, true);
    window.addEventListener("pointercancel", release, true);
    // Not every change redraws the preview straight away — a colour well, a
    // checkbox — so the page's own events schedule a look as well.
    for (const type of ["input", "change", "keyup"]) {
        document.addEventListener(type, noteEdit, true);
    }

    // Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y, on the page only (see typingInField),
    // and only where the buttons are offered.
    document.addEventListener("keydown", (e) => {
        if (!(e.ctrlKey || e.metaKey) || e.altKey || typingInField(e.target)) return;
        if (!enabled() || !frames().length) return;
        const k = e.key.toLowerCase();
        if (k === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
        else if ((k === "z" && e.shiftKey) || k === "y") { e.preventDefault(); redo(); }
    });
    syncHistoryButtons();
}
