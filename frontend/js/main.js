
// --- Output Mode Toggle (Frames with text / Frames only / My own images) ---
//
// "text", "framesOnly" or "images". The third is "Frames only" without the
// video: the frames are images the user uploads (see startImageSet), and
// everything after they land is that mode's — no transcript, the user's own
// text split over them, and the whole edit.
let outputMode = "text";

const MODE_BUTTONS = [["btnModeText", "text"], ["btnModeFramesOnly", "framesOnly"],
                      ["btnModeOwnImages", "images"]];

/** Whether the frames carry no words from a video: "Frames only", and the user's own images. */
const framesOnly = () => outputMode !== "text";

// The chosen button is marked with .btn-active and nothing else — the look is
// the stylesheet's, shared with every other one-of-several choice in the app,
// so the colours are written down once rather than here. Outside
// initModeToggle because a run picked back up after a reload sets the mode too.
function syncModeToggle() {
    for (const [id, mode] of MODE_BUTTONS) {
        const btn = document.getElementById(id);
        if (!btn) continue;
        btn.classList.toggle("btn-active", outputMode === mode);
        btn.setAttribute("aria-pressed", String(outputMode === mode));
    }
    // "Max characters per frame" belongs to one of the modes only, and the
    // link box and the video upload to two of them.
    syncCaptionCharsField();
}

function initModeToggle() {
    for (const [id, mode] of MODE_BUTTONS) {
        document.getElementById(id)?.addEventListener("click", () => {
            outputMode = mode;
            syncModeToggle();
        });
    }
    syncModeToggle();
}
import { API, LINE_IDS, TITLE_FIELD_IDS, el } from "./config.js";
import {
    store, frames, current, selectedIndex, setSelectedIndex,
    makeFrame, releaseFrames, carryOverSettings, gridOrder, positionOf, discardFrame, addFrame,
} from "./state.js";
import { getJson, postJson, postFile, postBlobJson, loadImage } from "./net.js";
import { openFrameCrop } from "./framecrop.js";
import { initCanvas, updatePreview, refreshThumb, refreshAllThumbs, forgetThumbSignatures } from "./compose.js";
import {
    status, showProgress, hideProgress, syncBatchButton, resetPending,
    reportContentProblems, syncFrameNav, nextSelectable,
} from "./ui.js";
import * as wide from "./wide.js";
import {
    setupReframeInteraction, syncZoomSliderFor, updateZoomFill, dismissFigureHandles,
} from "./reframe.js";
import {
    enhanceFrame, loadTextPanel, updateLines, setEditPreset, toggleFlip, toggleGradient, toggleGlow,
    varyFrame, syncFlipButtons, syncGradientButton, togglePick, handleLogoFile, handleAssetPick,
    initChannelPicker, selectChannel, syncChannelToFrame, guardChannel, warnChannelRequired,
    handleFrameFile, syncVaryButton, syncLogoControls, ensureCutout, setTitleAlign, showChannel,
    cycleBackground, syncBackgroundButton, cycleFigureGlow, syncFigureGlowButton,
    warmCutouts, applyEditToAll,
} from "./editor.js";
import { hasChannel, activeChannelId, framingFor, captureFor, cleanupFor,
         selectionFor, NO_CHANNEL } from "./channels.js";
import {
    enterMode, isCapture, captureMode, syncCaptureFields, requestedCount, syncCountValue,
    applyCaptions,
} from "./capture.js";
import { downloadFrame, downloadSelected, downloadAll } from "./download.js";
import { titleFontReady, reportMissingTitleFont } from "./fontguard.js";
import {
    showMyText, lockSplit, myText, maxChars, dealOut, captionChars, respreadCaptions,
} from "./mytext.js";
import { initReorder } from "./reorder.js";
import { primeReadyNotice, announceReady } from "./notify.js";

const POLL_MS = 500;
// Consecutive failed polls tolerated before giving up. A tunnel hiccup or a
// momentarily busy server shouldn't abandon a run that is still going fine on
// the backend, so this is generous — 30s at POLL_MS.
const POLL_MAX_FAILURES = 60;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Follows a started run to completion and returns its frames.
 *
 * The pipeline takes minutes, so the backend runs it in the background and
 * every request here is short — see the note on /process-video. Progress and
 * completion arrive through the same poll: the payload carries `status`
 * alongside the stage/percent that drive the bar.
 */
async function waitForResult(runId) {
    let failures = 0;
    for (;;) {
        await sleep(POLL_MS);

        const progress = await getJson("/process-video/progress").catch(() => null);
        if (!progress) {
            if (++failures >= POLL_MAX_FAILURES) throw new Error("Lost contact with the server");
            continue;
        }
        failures = 0;
        // Not while this tab is stopping the run to start over: the bar says
        // so then, and the dying run's stages would talk over it.
        if (!restarting) showProgress(progress.stage, progress.percent);

        // The backend holds one run at a time, so a different id here means
        // another tab of this browser has started a new video in its place —
        // possible in the moment between this run finishing and this poll
        // collecting it. Waiting on would collect THAT run's frames as if they
        // were this one's, under this one's channel; stepping aside lets the
        // run be picked up properly instead (see the storage listener below).
        if (runId && progress.run && progress.run.id !== runId) {
            const replaced = new Error("Another tab started a new video — following it instead.");
            replaced.replaced = true;
            throw replaced;
        }

        // Each of the endings below leaves nothing for a reload to pick
        // up, so the browser stops remembering the run. Losing contact is not
        // one of them: the run may well still be going, and a reload once the
        // connection is back is exactly what should find it.
        if (progress.status === "error") {
            forgetRun(runId);
            throw new Error(progress.detail || "Processing failed");
        }
        // Stopped on purpose, for one of the only two reasons a run is ever
        // stopped: the user started a new video in its place (see
        // startFresh), or the page that owned it was closed (see the leaving
        // section at the end of this file) — the second reached only by a
        // tab that page did not know was still open.
        if (progress.status === "cancelled") {
            forgetRun(runId);
            if (replacedRun() === runId) throw new Error("Stopped — a new video was started in its place.");
            throw new Error("Processing was stopped — the page that started it was closed.");
        }
        // "idle" can only mean the backend forgot the run it had just told us
        // it started — i.e. it restarted. Without this the poll would spin
        // forever on a run that no longer exists, which is the exact hang
        // moving off the long-lived request was meant to eliminate.
        if (progress.status === "idle") {
            forgetRun(runId);
            throw new Error("The run was lost — the server restarted");
        }
        if (progress.status === "done") {
            const result = await getJson("/process-video/result");
            if (!result) throw new Error("Finished, but the frames could not be collected");
            forgetRun(runId);
            // A backend older than this file has no /result and answers the
            // mount's index.html instead, so what arrives is not a frame
            // payload at all. Saying so beats the TypeError the caller would
            // otherwise raise several lines later on data.frames.map.
            if (!Array.isArray(result.frames)) throw new Error("Backend is out of date (no /process-video/result)");
            return result;
        }
    }
}

// The initial grid arrives as one JSON payload with base64 images (a single
// bulk response, where per-request binary transport would buy nothing). They
// become object URLs immediately so that every consumer downstream handles
// exactly one kind of image reference.
function base64ToObjectUrl(b64) {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    return URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
}

// Both source buttons move together, and are off only while a new video is
// being set going (see startFresh) — not while one processes. A click during
// a run is how the user asks to drop it and start another, so it has to
// reach startFresh, which asks before anything is lost.
//
// The image upload's buttons move with them, and are also off while a batch
// of images is being added (see addImages): a second batch started on top of
// the first would interleave the two in the grid.
function setSourceBusy(busy) {
    for (const id of ["processBtn", "uploadVideoBtn", "uploadImagesBtn", "newImageSetBtn"]) {
        el(id).disabled = busy;
    }
}

// The run this browser started and has not collected yet, by the id it was
// sent with. Kept in localStorage because the whole point is to survive the
// page: a reload, a second tab, a browser restarted mid-run.
//
// It is what makes picking a run up safe. A backend holds whatever run it was
// last given, and the page cannot tell from that alone whose it is — a slot
// handed on to someone new after its owner went quiet still holds the old
// owner's frames. Without the id, the newcomer's page would open on them.
//
// Every access is guarded: storage can be switched off or full, and the cost
// of that should be losing the pick-up, not the page.
const RUN_KEY = "thumbnailMaker.runId";

function storedRunId() {
    try { return localStorage.getItem(RUN_KEY); } catch { return null; }
}

function rememberRun(id) {
    try { localStorage.setItem(RUN_KEY, id); } catch { /* see RUN_KEY */ }
}

// Only if it is still the one stored: another tab may have started a newer run
// since, and that one is not this caller's to forget.
function forgetRun(id) {
    try { if (id && localStorage.getItem(RUN_KEY) === id) localStorage.removeItem(RUN_KEY); }
    catch { /* see RUN_KEY */ }
}

// The run the user threw away to start a new video in its place (see
// startFresh), so every tab following it can say that rather than "the page
// was closed". In localStorage because the tab that starts over is not always
// the one following the run.
const REPLACED_KEY = "thumbnailMaker.replacedRun";

function replacedRun() {
    try { return localStorage.getItem(REPLACED_KEY); } catch { return null; }
}

function markReplaced(id) {
    try { if (id) localStorage.setItem(REPLACED_KEY, id); } catch { /* see RUN_KEY */ }
}

// getRandomValues rather than randomUUID: the latter exists only on https and
// localhost, and the app is also reached over plain http by address.
function newRunId() {
    return Array.from(crypto.getRandomValues(new Uint8Array(16)),
                      b => b.toString(16).padStart(2, "0")).join("");
}

/**
 * A run on the backend that this browser started and should finish, or null.
 *
 * The page that starts a run is not always the page that sees it through: a
 * reload, a second tab or a dropped connection puts a fresh page in front of
 * a backend minutes into a video. That page used to learn about the run only
 * as "Your video is being processed" in red when it tried to start another —
 * and then had no way to reach the frames, so the user waited on nothing and
 * ran the video again. Asked on load and before every run instead, so the
 * page simply takes over.
 *
 * A finished run counts only with `finished`, and only while nobody has
 * collected it (see the note on /process-video/progress): otherwise every
 * visit to the page would bring back the last video anyone processed. Page
 * load passes it; a click on Process does not, because there the user has
 * just asked for a NEW video, and frames they abandoned earlier are not an
 * answer to that — a run still going is, since nothing else can start until
 * it ends.
 */
async function runInProgress(finished) {
    // No run of this browser's outstanding, so nothing to ask about — which
    // also keeps a page that is only being opened from sending the backend
    // anything at all.
    const mine = storedRunId();
    if (!mine) return null;

    const progress = await getJson("/process-video/progress").catch(() => null);
    if (!progress) return null;
    const client = progress.run || null;

    // Whatever the backend holds, if it is not this browser's run then this
    // browser's run is gone — the backend keeps one at a time — and there is
    // nothing more to wait for.
    const ours = !!client && client.id === mine;
    const pending = ours && (progress.status === "running"
                             || (progress.status === "done" && progress.collected === false));
    if (!pending) {
        forgetRun(mine);
        return null;
    }
    if (progress.status === "done" && !finished) return null;

    showProgress(progress.stage, progress.percent);
    return { capture: captureFor(client.channel), client, resumed: true };
}

/**
 * Puts the page back the way it was when the run being picked up was started,
 * so its frames land in the tab, channel and mode they were made for. The
 * channel decides the canvas a frame fetch is drawn on and the brand every
 * frame is made under; landing them under whatever this page happens to have
 * selected — nothing, on a page just reloaded — would make them a different
 * run's frames.
 */
function resumeSettings(client) {
    if (!client) return;
    // Only the two video modes are ever picked up: a set of the user's own
    // images is not a run, and a reload ends it (see startImageSet).
    outputMode = client.framesOnly ? "framesOnly" : "text";
    syncModeToggle();
    // The tab before the channel: switching tab rebuilds the dropdowns from
    // that tab's list, and a channel set first would be dropped as not on it.
    setAppMode(client.tab === "static" ? "static" : "thumbnail");
    showChannel(client.channel);
}

/**
 * Starts a run of this page's own and describes it the way runInProgress
 * describes one picked up, so the rest of runPipeline cannot tell them apart.
 */
async function startRun(prepare) {
    const prepared = await prepare();
    const { body, stage, capture } = prepared;
    const client = { ...prepared.client, id: newRunId() };
    showProgress(stage, 0);
    try {
        await postJson("/process-video", { ...body, client });
    } catch (e) {
        if (e.status !== 409) throw e;
        // runPipeline asked this question before preparing anything, but an
        // upload takes long enough for another tab to have started a run in
        // the meantime. The uploaded file is then wasted; the run that won is
        // still the user's, and following it beats an error. Finished runs
        // included: it may have ended between the 409 and this question, and
        // it is still the one to show.
        const run = await runInProgress(true);
        if (run) return run;
        // Not this browser's: the backend is busy with someone else's video.
        // The proxy gives every session a backend of its own, so this takes a
        // slot handed on while its last run was still going — rare, and worth
        // saying plainly rather than "your video", which it is not.
        throw new Error("Another video is being processed on this server — try again when it finishes.");
    }
    // After the backend accepted it, not before: remembered first, a refusal
    // would leave this browser waiting on a run that never existed — and
    // would forget the run it really had going in another tab.
    rememberRun(client.id);
    return { capture, client, resumed: false };
}

/**
 * Runs one video end to end, whatever it came from — or, with no `prepare`,
 * only finishes a run already going (the check made on page load).
 *
 * `prepare` is the only part that differs between a pasted link and an
 * uploaded file: it does whatever getting the video within the backend's
 * reach requires (nothing at all, for a link) and hands back the body to
 * start /process-video with. Everything from that call onward — the poll, the
 * grid, the font check, the first enhance — is shared, because from the
 * backend's side there is no longer any difference to act on.
 *
 * A run of this browser's already going on the backend is followed rather
 * than refused: it is the one the backend will finish first, and a second
 * could not start until it had.
 *
 * `resumeMessage` is what the status line says while following a run this
 * tab did not start — a reload and another tab's new video are both that,
 * and the user is owed different words for each.
 */
async function runPipeline(prepare,
                           resumeMessage = "Picking up the video that was already processing...") {
    // The load-time check, with nothing to check: returns before touching
    // the buttons, so an ordinary page load does not flicker them.
    if (!prepare && !storedRunId()) return;
    pipelineBusy = true;
    status("");
    try {
        const run = (await runInProgress(!prepare)) || (prepare && await startRun(prepare));
        if (!run) return;
        // The box for the user's own text, on a "Frames only" run whose
        // channel writes on its frames: up from the moment the run starts, so
        // the text can be written while the video processes, and locked until
        // every frame has loaded (see mytext.js and the end of this block).
        // Any other run takes it away, since its frames are not ones the text
        // could go on.
        ownTextRun = !!(run.client && run.client.framesOnly && run.capture && run.capture.captions);
        showMyText(ownTextRun);
        if (run.resumed) {
            resumeSettings(run.client);
            status(resumeMessage);
            // Whatever frames this tab is showing belong to a video the
            // backend has already let go of: starting a run wipes the last
            // one's. Left on screen they look fine and edit wrongly — every
            // edit asks the backend for a frame by number, and gets that
            // number from the NEW video, or an error. Hidden until the new
            // frames replace them below.
            el("resultSection").classList.add("hidden");
        }
        const data = await waitForResult(run.client && run.client.id);

        // frame_ids are reused per video (0, 1, 2...), so anything keyed by
        // them from the previous video has to go before the new frames land.
        releaseFrames();
        wide.clearAll();
        forgetThumbSignatures();
        resetPending();

        // Before a single canvas is built: the run's canvas size and the
        // controls that exist under it are both decided here, and a canvas
        // created at the previous run's shape would have to be thrown away
        // (see capture.enterMode). The block is the one captured when the run
        // STARTED, not what the picker says now — the user is free to have
        // moved it while the video was processing.
        enterMode(run.capture);

        store.frames = data.frames.map(f => makeFrame(
            f.frame_id, base64ToObjectUrl(f.frame),
            f.timecode ? { timestamp: f.timestamp, timecode: f.timecode } : null,
            f.caption || ""));
        // The video's own size, which an image the user adds to this grid is
        // framed to (see addOwnFrame). A backend older than /add-frame sends
        // none, and the tile then falls back to the canvas's shape.
        runSourceSize = Array.isArray(data.source_size) ? data.source_size : null;
        // A channel that asked for rewritten captions and did not get them is
        // the one outcome here that looks exactly like success: every frame is
        // captioned, every caption reads, and every one of them is the speech
        // in the video rather than the report the channel is built on. The
        // backend says which happened (see api._capture_frames_sync) and this
        // is the only place the user would ever find out.
        const runFramesOnly = !!(run.client && run.client.framesOnly);
        const spoken = run.capture && run.capture.rewrite && !runFramesOnly && data.rewritten === false;
        // The captions dealt out again under the ceiling the user typed for
        // this run, before anything is broken into lines or drawn. The run's
        // own number, from its record: a page that picked the run up after a
        // reload has an empty field, and the user's figure is still the one
        // owed. A run started before the field existed has none, and keeps
        // its captions as they came.
        const ceiling = !runFramesOnly && run.capture && run.capture.captions
                        && run.client && run.client.captionChars;
        const leftover = ceiling ? respreadOverGrid(ceiling) : 0;
        status(isCapture()
            ? `${data.count} frames, in the order they happen in the video`
              + (ceiling ? `, captions of up to ${ceiling} characters` : "")
              + (leftover ? ` — ${leftover} more ${leftover === 1 ? "piece does" : "pieces do"} `
                          + "not fit. Raise the max characters per frame or fetch more frames" : "")
              + (spoken ? " — captions are the transcript as spoken: the narration "
                        + "model did not load, so they were not rewritten as report" : "")
            : `${data.count} frames — select one and give it a title`);
        // A video's frames, whatever was on screen before them.
        imageSet = false;
        syncSourceFields();
        await presentGrid({ captions: !runFramesOnly, announce: true });
    } catch (e) {
        // Nothing arrived for the text to go on. A run being followed from
        // another tab shows the box again as it starts, if it wants one.
        showMyText(false);
        // Ended because this tab is starting over: startFresh is already
        // saying what is happening, and "stopped" would talk over it.
        if (!restarting) status(e.message);
        if (e.replaced) followPending = true;
    } finally {
        if (!restarting) hideProgress();
        pipelineBusy = false;
    }
    // A run another tab started while this one was busy, now that it is not.
    if (followPending) {
        followPending = false;
        runPipeline(null, "Following the video started in another tab...");
    }
}

/**
 * Puts the frames in store.frames on screen: the grid, the canvas, the panel,
 * the first frame's edit, and the edit pass (or the cutouts) over the rest.
 *
 * The landing of a video's frames, and of the first image of a set of the
 * user's own (see startImageSet) — one path, so the two grids cannot come to
 * differ in anything but where their frames came from. The caller has already
 * entered the run's mode (capture.enterMode) and filled store.frames.
 *
 * `captions` breaks the frames' captions into lines first; `announce` tells a
 * user in another tab that their video is ready, which a set of images they
 * are uploading in this one has no need of.
 */
async function presentGrid({ captions = false, announce = false } = {}) {
    syncPresetHint();
    setSelectedIndex(0);
    el("resultSection").classList.remove("hidden");

    // The title font is self-hosted and may still be downloading at this
    // point — without waiting, the first canvas render/measurement could
    // briefly use the Impact fallback and bake its metrics into the layout.
    // A face that never arrives at all is a broken install rather than a
    // slow one, and is called out rather than silently drawn around.
    const grid = frames();
    if (!(await titleFontReady()).ok) reportMissingTitleFont();

    // ...and only now are the captions of a run that has any broken into
    // lines, because every break is chosen by measuring the words — in
    // the run's OWN channel's face, which is a second thing to wait for
    // and which applyCaptions waits for itself. Before the panel is
    // loaded and before the grid is drawn, so the first thing the user
    // sees already has the words on it.
    //
    // The mode the run was started in, not the toggle's position now, for
    // the reason the capture block is the run's: the backend was told then
    // whether to transcribe.
    if (captions) { await applyCaptions(); }
    // Replaced while the faces loaded — the newer frames draw themselves.
    if (frames() !== grid) return;

    initCanvas(setupReframeInteraction);
    renderGrid();
    // The moment the frames start going onto the grid is the moment a
    // user in another tab wants to hear about (see notify.js).
    if (announce) announceReady(store.frames.length);
    loadTextPanel();
    syncFlipButtons();
    syncGradientButton();
    syncBackgroundButton();
    syncFigureGlowButton();
    syncVaryButton();
    syncLogoControls();
    syncBatchButton();
    updateZoomFill();
    updatePreview();
    enhanceFrame(selectedIndex());
    // A frame fetch edits the whole grid up front, because there the edit
    // IS the deliverable and the user may never open a frame at all (see
    // editor.applyEditToAll). A thumbnail run fills in every card's figures
    // instead, so the grid is twenty comparable options rather than twenty
    // identical backdrops waiting to be clicked (see warmCutouts).
    //
    // Neither is awaited: the first frame is already being worked on above,
    // and this is the rest of them.
    //
    // The end of the edit pass is the end of loading — the last grey card
    // has turned — so it is what unlocks "Split the text" (see runEditPass).
    if (isCapture()) {
        runEditPass();
    } else {
        warmCutouts();
    }
}

// Whether runPipeline is in flight in this tab, and whether another tab
// started a run meanwhile that this one should follow once it is free.
let pipelineBusy = false;
let followPending = false;
// Whether startFresh is between the click and the new run being under way,
// and whether that includes stopping the old one (see startFresh).
let starting = false;
let restarting = false;
// The upload this tab is sending, so starting over can call it off — a file
// half across belongs to the video being dropped.
let uploadAbort = null;
// Whether the frames of the latest run are ones the user's own text can go
// on — a "Frames only" run on a channel that captions (see runPipeline).
let ownTextRun = false;
// The latest frame fetch's video size, [width, height], or null — see
// runPipeline and addOwnFrame.
let runSourceSize = null;
// Whether the frames on screen are a set of the user's own images rather than
// a video's (see startImageSet), and whether a batch of them is still being
// added (see addImages).
let imageSet = false;
let addingImages = false;
// The edit pass over the grid, while one is going, and whether frames joined
// the grid after it had already looked for its last one (see runEditPass).
let editPass = null;
let editPassAgain = false;
// What the last batch of images could not add, waiting to be said once the
// edit pass has finished talking (see runEditPass and addImages).
let skippedImages = "";

/** Whether a set of the user's own images is on screen to add to. */
const imageSetOpen = () => imageSet && frames().length > 0;

/**
 * The edit over every frame in the grid (editor.applyEditToAll), and the end
 * of it unlocking "Split the text" on a run the user's own text goes on.
 *
 * Never two at once. The pass asks the grid's order afresh before every frame,
 * so a frame added while it runs is reached by it without anything being
 * said; the one gap is a frame added in the moment after the pass found
 * nothing left, which `editPassAgain` closes by going round once more.
 *
 * Not while a batch of images is still arriving: the split deals the text
 * over the frames that are there, and until the last image is in, that number
 * is still growing. addImages runs this again once the batch is done.
 */
function runEditPass() {
    if (editPass) {
        editPassAgain = true;
        return;
    }
    editPassAgain = false;
    editPass = applyEditToAll().then((done) => {
        editPass = null;
        // Asked first, whether or not this pass finished: one that a newer
        // grid cut short (done is false) is also how that grid's own request
        // for a pass ended up waiting here.
        if (editPassAgain) return runEditPass();
        if (!done) return;
        if (ownTextRun && !addingImages) lockSplit(false);
        // The images a batch could not add, said after the pass's own last
        // word rather than before it, where that word would replace it.
        if (skippedImages && !addingImages) {
            status(`${el("status").textContent} ${skippedImages}`);
            skippedImages = "";
        }
    }, () => { editPass = null; });
}

/**
 * "Split the text": the user's text, broken at its sentences and clauses into
 * pieces of at most "Max characters per frame", and set one piece per frame
 * from the first frame on (see mytext.dealOut).
 *
 * In the grid's order, which is the user's if they have rearranged it (see
 * reorder.js): the first piece goes on the card in the first place. After a
 * split, a caption belongs to its frame and travels with it when it is
 * dragged; splitting again deals the text out along the new order.
 *
 * Over the frames that actually LANDED, not the slider's count — frames
 * rejected as blurred or blank make the grid shorter than what was asked for
 * — and of those, only the ones still on the grid: a frame the user threw out
 * with its X (see discardFromGrid) is not in the order and gets nothing.
 * The pieces are set where a transcript's sentence would have arrived, so
 * from here on each is a caption like any other: broken into lines in the
 * channel's face (capture.applyCaptions), and the user's to edit per frame.
 *
 * Every frame is cleared first, so splitting again after changing the text
 * replaces what the last split put there instead of leaving its pieces on
 * the frames the new one skips.
 *
 * The order is checked as the pieces are set, not trusted: each piece goes on
 * its frame only once the frame before it in the grid has text of its own.
 * dealOut is built so that always holds, and the check is there because a
 * split that once left frames bare between captioned ones is exactly the
 * mistake the user cannot see from the status line — they find it in the
 * folder, after exporting. If it ever fails, the frames get back the captions
 * they had and the user is told, rather than shown a split with holes in it.
 */
async function splitMyText() {
    const text = myText();
    if (!text.trim()) return status("Add your text first, then split it over the frames");
    const max = maxChars();
    if (max === null) return status("Max characters per frame has to be a whole number of at least 1");
    const grid = frames();
    const order = gridOrder();
    const { pieces, leftover } = dealOut(text, order.length, max);

    const before = grid.map(f => ({ caption: f.caption, lines: f.lines }));
    order.forEach((i) => {
        grid[i].caption = "";
        grid[i].lines = LINE_IDS.map(() => "");   // the fixed slots makeFrame starts with
    });
    for (let position = 0; position < pieces.length; position++) {
        const previous = position > 0 && grid[order[position - 1]];
        if (previous && !previous.caption.trim()) {
            grid.forEach((f, i) => { f.caption = before[i].caption; f.lines = before[i].lines; });
            return status(`The text could not be split in order: frame ${position} was left `
                        + `without text before frame ${position + 1}. Nothing was changed.`);
        }
        grid[order[position]].caption = pieces[position];
    }

    await applyCaptions();
    if (frames() !== grid) return;   // a new run landed while the face loaded
    loadTextPanel();
    refreshAllThumbs();
    updatePreview();
    const placed = pieces.length;
    const where = placed === order.length ? `one on each of the ${order.length} frames`
                : placed === 1 ? "on frame 1" : `on frames 1 to ${placed} of ${order.length}`;
    status(`Your text is split into ${placed} ${placed === 1 ? "piece" : "pieces"} of up to `
         + `${max} characters, ${where}`
         + (leftover ? ` — ${leftover} more ${leftover === 1 ? "piece does" : "pieces do"} not fit. `
                     + "Raise the max characters per frame or "
                     + (imageSet ? "add more images." : "fetch more frames.") : ""));
}

// Every tab of this browser shares one session, so one backend, so one run:
// when any of them starts a video, the frames every other tab is showing stop
// existing on the backend (see runPipeline). The tab that started it says so
// in localStorage (rememberRun), and the browser tells every OTHER tab of the
// change — this is that. They follow the new run, and land on its frames
// instead of being left editing ones the backend has thrown away.
//
// Only a new id counts: the key being removed is a run finishing.
window.addEventListener("storage", (e) => {
    if (e.key !== RUN_KEY || !e.newValue) return;
    if (pipelineBusy) {
        // Busy with a run of its own, or following one: picked up once that
        // ends. If it is the same run, there is nothing left to pick up by
        // then and the follow-up stops at its first check.
        followPending = true;
        return;
    }
    runPipeline(null, "Following the video started in another tab...");
});

/**
 * What the selected channel wants of the frame SELECTION, if anything — sent
 * with the video and read nowhere else.
 *
 * This is the one thing a channel decides that cannot be changed afterwards:
 * which moments of the video become thumbnails at all. By the time the title
 * panel's channel picker exists there are already twenty frames, chosen. Hence
 * the picker above the link box, and hence this being read at the moment a run
 * starts rather than at the moment a frame is drawn.
 */
const runFraming = () => framingFor(activeChannelId());

/**
 * Whether this run should clean burned-in overlays off its frames.
 *
 * Read here rather than per frame because it is a fact about how the frames
 * are MADE: the two passes run once, over the sample, before anything is
 * selected — so like the framing, a channel picked afterwards cannot undo it.
 */
const runCleanup = () => cleanupFor(activeChannelId());

/**
 * How this run should tell two candidate frames apart.
 *
 * Read here for the reason the framing is: it decides which frames the
 * video becomes, once, before any of them exist.
 */
const runSelection = () => selectionFor(activeChannelId());

/**
 * What the selected channel wants of the run ITSELF: a frame fetch, and how
 * many frames of it.
 *
 * Read at the same moment the framing is and for the same reason — this is
 * not a way of drawing a frame that a later choice could change, it is what
 * the run produces. Null for every channel that makes thumbnails, which is
 * what the backend reads as "the pipeline you have always run".
 */
function runCapture() {
    const id = activeChannelId();
    const capture = captureFor(id);
    return capture
        ? {
            block: capture,
            // The count is the user's, from the slider; whether the video
            // is transcribed, and whether that transcript is rewritten as
            // report before it is set, are both the pack's. All three are
            // sent rather than assumed, so the backend never pays for a
            // transcription nothing will draw or a rewrite nothing will read.
            request: {
                count: requestedCount(id),
                captions: framesOnly() ? false : !!capture.captions,
                rewrite: framesOnly() ? false : !!capture.rewrite,
            },
        }
        : null;
}

/**
 * Whether a run started now would caption its frames with what the video
 * says: a "Frames with text" run on a channel that captions. The question
 * runCapture's `captions` answers, asked of the page as it stands.
 */
function captionsRun() {
    const capture = captureFor(activeChannelId());
    return !framesOnly() && !!(capture && capture.captions);
}

/**
 * Shows "Max characters per frame" above the link box exactly when the run it
 * is required for would be the one started — so it answers the channel, the
 * mode and the tab, from wherever each of them is changed. A function
 * declaration, so syncModeToggle can call it from the top of the file.
 */
function syncCaptionCharsField() {
    el("captionCharsRow").classList.toggle("hidden", !captionsRun());
    // Asked from every one of the same places, for the same reason: which
    // source the page offers answers the tab and the mode too.
    syncSourceFields();
}

/** Whether the page is asking for the user's own images rather than a video. */
const imagesSource = () => currentMode === "static" && outputMode === "images";

/**
 * The link box and the video upload, or the image upload, whichever the page
 * is set for — and "Frames to fetch" with the video, since a set of images is
 * however many images there are.
 *
 * Each control is hidden in place rather than the two sources being wrapped
 * in blocks of their own, so the section keeps the spacing it has always had
 * between every one of them.
 */
function syncSourceFields() {
    const images = imagesSource();
    for (const id of ["driveUrl", "processBtn", "sourceOr", "uploadVideoBtn"]) {
        el(id).classList.toggle("hidden", images);
    }
    el("imageSourceRow").classList.toggle("hidden", !images);
    el("uploadImagesHint").classList.toggle("hidden", !images);
    // Adding to a set is only "more" when there is one on screen to add to.
    const open = imageSetOpen();
    el("uploadImagesBtn").textContent = open ? "Add more images" : "Upload images from computer";
    el("newImageSetBtn").classList.toggle("hidden", !(images && open));
    // The slider answers to the channel (see capture.syncCaptureFields), and
    // is put back to that answer whenever the page leaves this source.
    syncCaptureFields(activeChannelId());
    if (images) el("frameCountRow").classList.add("hidden");
    syncPresetHint();
}

/**
 * The edit preset panel's note on which look a frame starts on — the frames
 * on screen, not the toggle: a video's grid stays on "Natural" whichever mode
 * button is pressed after it landed. See IMAGE_SET_PRESET.
 */
function syncPresetHint() {
    el("presetHint").textContent = imageSet
        ? "Applies to this frame. Your own images start on Restore only."
        : "Applies to this frame. Natural is the default look.";
}

/**
 * Refuses to start a "Frames with text" run without a ceiling, and says why.
 * True when it refused. Asked before anything is spent — before an upload's
 * file dialog, before a link is sent — because the captions are dealt out
 * the moment the frames land, and a run cannot be told the number later.
 */
function missingCaptionChars() {
    if (!captionsRun() || captionChars() !== null) return false;
    status("Fill in Max characters per frame with a whole number before processing the video");
    el("captionCharsInput").focus();
    return true;
}

/**
 * The page's own settings for a run: which channel, which tab, which output
 * mode, and the caption ceiling. The backend keeps them only to hand them
 * back, so a page that picks the run up after a reload can be put back the
 * way this one was (see runInProgress and resumeSettings). Read when the run
 * is set going, like everything else a run is made from.
 *
 * The ceiling travels here rather than in the capture request because it is
 * this page's to apply, to the captions as they land (see runPipeline) — and
 * riding in the record the backend hands back is what lets a page that picks
 * the run up after a reload still apply the number the user typed.
 */
const runClient = () => ({
    channel: activeChannelId(), tab: currentMode, framesOnly: framesOnly(),
    captionChars: captionsRun() ? captionChars() : null,
});

/**
 * A "Frames with text" run's captions, dealt out again over the grid under
 * the ceiling the user typed (see mytext.respreadCaptions). In the grid's
 * order — at this point the order the frames arrived in — and over every
 * frame of it, so a frame the new split leaves without text is cleared
 * rather than keeping the caption it came with. Returns how many pieces did
 * not fit.
 */
function respreadOverGrid(max) {
    const grid = frames();
    const order = gridOrder();
    const { pieces, leftover } = respreadCaptions(order.map(i => grid[i].caption), order.length, max);
    order.forEach((i, position) => { grid[i].caption = pieces[position] || ""; });
    return leftover;
}

const STOP_POLL_MS = 500;
// How long starting over waits for the old run to stop. It stops at its next
// checkpoint, and a step already under way — scoring every frame,
// transcribing the voice track — finishes first (see backend/cancel.py).
const STOP_WAIT_MS = 10 * 60 * 1000;

/**
 * Every new video goes through here, from a link or from a file.
 *
 * If one is already processing — in this tab, in another tab of this browser
 * (they share a backend), or still landing on the grid — the user is asked
 * first, because going on throws it away. Saying yes stops it, clears the
 * backend, and starts the new one from nothing; saying no changes nothing.
 * Either way the page is cleared before the new run starts: the backend wipes
 * the last video's frames the moment a run begins, so anything left on screen
 * would only look editable.
 */
async function startFresh(prepare) {
    if (starting) return;          // a second click while the first is still deciding
    starting = true;
    setSourceBusy(true);
    try {
        if (pipelineBusy || await backendRunning()) {
            if (!(await askRestart())) return;
            restarting = true;
            if (!(await stopCurrentRun())) return;   // the status line says why
        }
        clearWork();
        // runPipeline marks itself busy before its first await, so by the time
        // the finally below turns the buttons back on, a click lands on a run.
        return runPipeline(prepare);
    } finally {
        restarting = false;
        starting = false;
        setSourceBusy(false);
    }
}

async function backendRunning() {
    const progress = await getJson("/process-video/progress").catch(() => null);
    return !!progress && progress.status === "running";
}

/**
 * The question before a video in progress is thrown away. Resolves true to
 * go on. Anything but the one button is a no — Escape, the backdrop — since
 * the safe answer is the one that loses nothing.
 */
function askRestart() {
    return new Promise((resolve) => {
        const box = el("restartConfirm");
        const onKey = (e) => {
            if (e.key !== "Escape") return;
            e.stopPropagation();
            done(false);
        };
        const done = (answer) => {
            box.classList.add("hidden");
            el("restartConfirmBtn").onclick = el("restartCancelBtn").onclick = box.onclick = null;
            document.removeEventListener("keydown", onKey, true);
            resolve(answer);
        };
        el("restartConfirmBtn").onclick = () => done(true);
        el("restartCancelBtn").onclick = () => done(false);
        box.onclick = (e) => { if (e.target === box) done(false); };
        document.addEventListener("keydown", onKey, true);
        box.classList.remove("hidden");
        el("restartCancelBtn").focus();
    });
}

/**
 * Stops whatever this browser has processing and leaves the backend empty.
 * True once it is; false, with the status line saying why, if it would not.
 *
 * Marked as replaced before it is stopped, so every tab following it says
 * "a new video was started" instead of "the page was closed" (see
 * waitForResult), and forgotten, so none of them goes looking for it again.
 * Then the backend's own /cleanup, which the user asked for by name: the
 * next run would reset the session anyway, but this also drops the temp
 * files and the channel the last run was framed by.
 */
async function stopCurrentRun() {
    showProgress("Stopping the video that was processing...", 0);
    status("");
    const progress = await getJson("/process-video/progress").catch(() => null);
    const old = (progress && progress.run && progress.run.id) || storedRunId();
    markReplaced(old);
    forgetRun(old);
    if (uploadAbort) uploadAbort.abort();
    // No id: whatever this backend is running is this browser's, since the
    // proxy gives every browser a backend of its own.
    await postJson("/process-video/cancel", { id: null })
        .catch(() => { /* judged below by whether it is still running */ });

    const deadline = Date.now() + STOP_WAIT_MS;
    const giveUp = (why) => {
        hideProgress();
        status(why);
        return false;
    };
    for (;;) {
        const now = await getJson("/process-video/progress").catch(() => null);
        // This tab's own pipeline too: it has to have let go of the old
        // video before the page is cleared for the new one.
        if (now && now.status !== "running" && !pipelineBusy) break;
        if (Date.now() > deadline) return giveUp("The video that was processing would not stop — try again in a moment");
        await sleep(STOP_POLL_MS);
    }
    // 409 while the stopped run is still letting go of the backend: its
    // status changes a moment before its lock is released.
    for (;;) {
        try {
            await postJson("/cleanup", {});
            return true;
        } catch (e) {
            if (e.status !== 409 || Date.now() > deadline) return giveUp(`Could not clear the last video: ${e.message}`);
            await sleep(STOP_POLL_MS);
        }
    }
}

/** Takes the last video off the page: its frames, their titles and text, the grid. */
function clearWork() {
    releaseFrames();
    wide.clearAll();
    forgetThumbSignatures();
    resetPending();
    el("framesGrid").innerHTML = "";
    el("resultSection").classList.add("hidden");
    showMyText(false);
    imageSet = false;
    skippedImages = "";
    syncSourceFields();
}

function processVideo() {
    const url = el("driveUrl").value.trim();
    if (!url) return status("Paste a link, or upload a video file from your computer");
    if (missingCaptionChars()) return;
    primeReadyNotice();
    const capture = runCapture();
    const client = runClient();
    return startFresh(async () => ({
        body: { google_drive_url: url, framing: runFraming(), capture: capture && capture.request,
                cleanup: runCleanup(), selection: runSelection() },
        stage: "Downloading video and extracting frames...",
        capture: capture && capture.block,
        client,
    }));
}

/**
 * The upload is the one stage the backend cannot report progress for — it
 * hasn't received the file yet — so this drives the bar itself until the
 * bytes are across, and the poll takes over from there.
 */
function processUploadedFile(file) {
    // Asked again here, not only before the file dialog opened: the field can
    // be emptied, or the mode switched, while the dialog is up.
    if (missingCaptionChars()) return;
    // Read before the upload starts, not after it finishes: the bytes take
    // their own time to cross, and the channel a run belongs to is the one
    // that was selected when the user set it going.
    const capture = runCapture();
    const client = runClient();
    return startFresh(async () => {
        const label = `Uploading ${file.name}...`;
        showProgress(label, 0);
        const abort = uploadAbort = new AbortController();
        let upload_id;
        try {
            ({ upload_id } = await postFile("/upload-video", file,
                (fraction) => showProgress(label, Math.round(fraction * 100)), abort.signal));
        } finally {
            if (uploadAbort === abort) uploadAbort = null;
        }
        return {
            body: { upload_id, framing: runFraming(), capture: capture && capture.request,
                    cleanup: runCleanup(), selection: runSelection() },
            stage: "Reading uploaded video and extracting frames...",
            capture: capture && capture.block,
            client,
        };
    });
}

// ── A set of the user's own images ────────────────────────────────────────
//
// Static Studio's "Upload my own images": "Frames only" with the video taken
// out. The user uploads images instead — as many as they select, and more
// batches whenever they like — and each becomes a frame of the grid exactly as
// a moment of a video would have: the text box and its split, the ceiling per
// frame, dragging, discarding, every channel's title, the canvas's reframe and
// zoom, the image overlay, the edit presets, and the downloads.
//
// None of that is written again here. The backend opens an empty grid (see
// /start-images) and makes each image an ordinary capture slot through the
// same /add-frame the "Add image" tile uses; this side gives the first image
// the landing a video's frames get (presentGrid) and every later one the card
// the tile gives its image. The one thing a set has fewer of is Variation —
// there is no video either side of an image to look for another moment in.
//
// A set is not a run. Nothing takes minutes and nothing is processed in the
// background, so there is nothing for a reload or another tab to pick up:
// like a video's frames, a set lasts as long as the page that made it.

// The longest side an image is sent at — api.MAX_UPLOADED_FRAME_SIDE, which
// the backend shrinks anything larger to anyway. Shrunk here first, so a batch
// of phone photos is not sent at three times the size that will be kept.
const MAX_IMAGE_SIDE = 4096;

// The edit preset every image of a set starts on: "Restore only", where a
// video's frames start on "Natural" (editor.DEFAULT_PRESET). A video's frames
// are footage, and the grade is part of what turns a moment of it into a
// still worth posting; the user's own images have usually been shot or graded
// the way they want already, and a second grade over them is a change they
// did not ask for. The restoration itself runs at every preset, so nothing is
// given up — and the other two are one click away per frame, as always.
const IMAGE_SET_PRESET = "none";

// Which of the two the image dialog was opened for: more images onto the set
// on screen, or a new set in its place. Set by whatever opened the dialog,
// since a dialog the user cancels says nothing at all.
let imagesIntent = "add";

/** File-name order, numbers compared as numbers, so "2.jpg" comes before "10.jpg". */
const byFileName = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });

/**
 * The image as the JPEG it is sent as, no longer on its longest side than
 * MAX_IMAGE_SIDE, or null when the browser cannot read it as an image at all.
 *
 * Decoded by the browser rather than sent as it is, for two reasons besides
 * the size: what the user sees in the grid is then what their browser shows
 * for the file — a phone photo turned the way its EXIF says, as every browser
 * draws it — and any format the browser opens reaches the backend as one it
 * can read.
 */
async function imageAsJpeg(file) {
    let img;
    const local = URL.createObjectURL(file);
    try {
        img = await loadImage(local);
    } catch (_) {
        return null;
    } finally {
        URL.revokeObjectURL(local);
    }
    const w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return null;
    const k = Math.min(1, MAX_IMAGE_SIDE / Math.max(w, h));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(w * k);
    canvas.height = Math.round(h * k);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
}

/**
 * Refuses to open the image dialog with no channel picked, and says why. True
 * when it refused. Before the dialog rather than after it, for the reason
 * missingCaptionChars is: the user is not sent to choose forty images only to
 * be told afterwards that a field was empty.
 *
 * A Static Studio channel and nothing less, because it is the channel that
 * says what a frame IS here — the canvas it is cut to, and whether it takes
 * text (see `capture` in channels.js). Every channel on the tab is one.
 */
function missingImageChannel() {
    if (captureFor(activeChannelId())) return false;
    status("Pick a channel before uploading your images");
    el("runChannelSelect").focus();
    return true;
}

/**
 * A new set of the user's own images in place of whatever is on screen, made
 * of `files`.
 *
 * As a new video does (see startFresh): a video still processing is stopped
 * only once the user has said yes, and the page is cleared first, because the
 * backend drops the last grid's frames the moment the set begins.
 */
async function startImageSet(files) {
    if (missingImageChannel()) return;
    const capture = captureFor(activeChannelId());
    if (starting || addingImages) return;
    starting = true;
    setSourceBusy(true);
    try {
        if (pipelineBusy || await backendRunning()) {
            if (!(await askRestart())) return;
            restarting = true;
            if (!(await stopCurrentRun())) return;   // the status line says why
            hideProgress();
        }
        clearWork();
        status("");
        try {
            await postJson("/start-images", { framing: runFraming() });
        } catch (e) {
            // A backend older than this file answers with FastAPI's own
            // "Not Found", as for /add-frame.
            const stale = (e.status === 404 && e.message === "Not Found") || e.status === 405;
            return status(stale ? "Backend is out of date (no /start-images)" : e.message);
        }
        // Before anything is drawn, as for a video's frames: the canvas's
        // shape and the controls that exist are the channel's (see
        // capture.enterMode), taken now so that moving the picker while the
        // images upload cannot change what this set is.
        enterMode(capture);
        store.frames = [];
        runSourceSize = null;
        imageSet = true;
        // The text box, on a channel that writes on its frames — up now, so
        // the text can be written while the images upload, and locked until
        // every one of them has loaded (see runEditPass).
        ownTextRun = !!capture.captions;
        showMyText(ownTextRun);
    } finally {
        restarting = false;
        starting = false;
        setSourceBusy(false);
    }
    await addImages(files);
}

/**
 * Adds `files` to the set on screen, one after another, in file-name order,
 * at the end of the grid.
 *
 * One at a time, for the reason the edit pass is: the backend frames them one
 * at a time anyway, and a card appearing as each lands is a grid filling in
 * rather than a bar sitting still. The first image of a new set lands as a
 * video's frames do (presentGrid) and starts the edit pass, which reaches every
 * later image by itself as it joins the grid.
 *
 * An image that cannot be read, or that the backend refuses, is skipped and
 * named at the end; the rest still go in.
 */
async function addImages(files) {
    if (!imageSet || addingImages) return;
    const list = Array.from(files).sort(byFileName);
    if (!list.length) return;
    const grid = frames();
    // The set the batch is for: replaced, by a video another tab started or
    // by a new set, and the batch stops — what is left of it belongs to a grid
    // that is gone.
    const replaced = () => frames() !== grid || !imageSet;

    addingImages = true;
    setSourceBusy(true);
    syncAddTile();
    if (ownTextRun) lockSplit(true);
    const failed = [];
    let added = 0;
    let stale = false;
    try {
        for (const [n, file] of list.entries()) {
            if (replaced()) return;
            showProgress(`Adding image ${n + 1} of ${list.length}...`,
                         Math.round(100 * n / list.length));
            const jpeg = await imageAsJpeg(file);
            if (!jpeg) { failed.push(file.name); continue; }
            let res;
            try {
                res = await postBlobJson("/add-frame", jpeg);
            } catch (e) {
                stale = (e.status === 404 && e.message === "Not Found") || e.status === 405;
                if (stale) break;
                failed.push(file.name);
                continue;
            }
            if (replaced()) return;
            await placeImage(res);
            added++;
        }
    } finally {
        addingImages = false;
        // The buttons whatever happened, unless a new video or set is being
        // set going — that owns them until it is under way (see startFresh).
        if (!starting) setSourceBusy(false);
        syncAddTile();
        // The bar only for a batch that was not replaced: what replaced it
        // may be a video another tab started, whose bar this one would hide.
        if (!replaced()) {
            hideProgress();
            syncSourceFields();
        }
    }
    if (replaced()) return;

    const total = gridOrder().length;
    if (stale) status("Backend is out of date (no /add-frame)");
    else if (!total) {
        // Not one image of a new set went in: nothing for the text to go on.
        showMyText(false);
        status(`None of the images could be added (${failed.join(", ")}). Try other files.`);
    } else {
        status(`${added} ${added === 1 ? "image" : "images"} added — ${total} `
             + `${total === 1 ? "frame" : "frames"} in the set`);
        skippedImages = failed.length ? `Could not be read as images: ${failed.join(", ")}.` : "";
    }
    // Round once more, for the frames that landed after the pass found its
    // last one — and to unlock the split, which waited for the batch.
    if (total) runEditPass();
}

/**
 * One image the backend has made a slot of (see /add-frame), onto the grid:
 * a frame record and a card built as the tile's are (see addOwnFrame), under
 * the channel of the frame it lands beside — or, for the first image of a
 * set, the landing a video's frames get.
 */
async function placeImage(res) {
    const frame = makeFrame(res.frame_id, base64ToObjectUrl(res.frame));
    frame.uploaded = true;
    // Before the edit pass reaches it, which is what sends the preset (see
    // IMAGE_SET_PRESET). Every image of every batch, whatever a neighbour has
    // been switched to since: the user asked for that on that frame.
    frame.editPreset = IMAGE_SET_PRESET;
    const order = gridOrder();
    const neighbour = frames()[order[order.length - 1]];
    if (neighbour) {
        frame.channel = neighbour.channel;
        frame.channelTouched = neighbour.channelTouched;
    }
    const i = addFrame(frame);
    if (!neighbour) {
        status("Your images are being added...");
        return presentGrid();
    }
    const card = frameCard(i, gridOrder().length - 1);
    el("framesGrid").insertBefore(card, el("addFrameTile"));
    refreshThumb(i);
    syncFrameNav();   // the frame on screen may have stopped being the last
}

/** Off while a batch of images is being added, so a second cannot start on top of it. */
function syncAddTile() {
    const tile = el("addFrameTile");
    if (tile) tile.disabled = addingImages;
}

// Laid out in the grid's order (see state.gridOrder), and numbered by place
// in it — the number on a card is the number its file will carry. Each card
// still answers to its frame's own index (`card-${i}`, data-index), which is
// what lets it be moved without anything that addresses it noticing.
function renderGrid() {
    const g = el("framesGrid");
    g.innerHTML = "";
    gridOrder().forEach((i, position) => g.appendChild(frameCard(i, position)));
    // The way to add an image of the user's own to the set, on a frame fetch
    // only — there the grid is the deliverable, and the frame the video did
    // not have is still one the user may want in it (see addOwnFrame). Last,
    // because that is where a new frame lands; the user drags it from there.
    if (isCapture()) g.appendChild(addFrameTile());
    syncFrameNav();
    refreshAllThumbs();
}

/**
 * The card for frame `i`, at `position` in the grid. Built on its own so a
 * frame the user adds (see addOwnFrame) gets exactly the card the run's own
 * frames got, without the whole grid being drawn again for it.
 */
function frameCard(i, position) {
    const f = frames()[i];
    const d = document.createElement("div");
    // Grey and inert until the edit reaches it, on a frame fetch only —
    // there the whole grid is worked through up front and the user's
    // question is "which of these can I open yet" (see ui.markFrameReady
    // and the capture-mode rules in styles.css). On a thumbnail run
    // nothing but the selected frame is ever restored, so a pending card
    // would simply never come back.
    const pending = isCapture() && !f.enhancedUrl;
    d.className = "frame-card"
                + (i === selectedIndex() ? " selected" : "")
                + (pending ? " pending" : "");
    d.id = `card-${i}`;
    d.dataset.index = i;
    // Draggable on a frame fetch only, where the order is the deliverable
    // (see reorder.js). The picture inside is not, or the browser would
    // drag the bare image instead of the card.
    d.draggable = isCapture();
    if (isCapture()) d.title = "Drag to reorder";
    d.innerHTML = `<img id="thumb-${i}" src="${f.plainUrl}" draggable="false">` +
                  `<div class="badge">${position + 1}</div>` +
                  `<div class="pick${f.picked ? " on" : ""}" id="pick-${i}" title="Include in batch download"></div>`;
    // The X that throws the frame out, on a frame fetch only — there the
    // grid is the deliverable and a frame the user does not want would
    // otherwise be edited, captioned and zipped regardless. Live on a grey
    // card too: the earlier a frame is thrown out, the less of the edit
    // pass is spent on it (see discardFromGrid).
    if (isCapture()) {
        const x = document.createElement("button");
        x.type = "button";
        x.className = "discard";
        x.title = "Discard this frame";
        x.setAttribute("aria-label", `Discard frame ${position + 1}`);
        x.draggable = false;
        x.onclick = (e) => { e.stopPropagation(); discardFromGrid(i); };
        d.appendChild(x);
    }
    // A pending card cannot be opened until the edit reaches it. It can
    // still be dragged — rearranging is allowed at any point of the run —
    // so the stylesheet leaves it the pointer, and the refusal is here.
    d.onclick = () => { if (!d.classList.contains("pending")) selectFrame(i); };
    // Ticking a frame for the batch must not also switch the preview to
    // it — the user builds the batch while working on one frame.
    d.querySelector(".pick").onclick = (e) => {
        e.stopPropagation();
        if (d.classList.contains("pending")) return;
        togglePick(i, carryOverForPick);
    };
    return d;
}

/**
 * The "Add image" tile that ends a frame fetch's grid (see addOwnFrame).
 *
 * On a set of the user's own images it takes a batch instead, the way the
 * button above the grid does (see addImages): there is no video for the
 * framing step to shape a picture to, and every image of the set went in
 * without one.
 */
function addFrameTile() {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "frame-add";
    tile.id = "addFrameTile";
    tile.title = imageSet ? "Add more images from your computer to the set"
                          : "Add an image from your computer as a frame";
    tile.draggable = false;
    tile.disabled = addingImages;
    tile.innerHTML = `<span class="frame-add-plus" aria-hidden="true">+</span>`
                   + `<span>${imageSet ? "Add images" : "Add image"}</span>`;
    tile.onclick = () => {
        if (!imageSet) return el("addFrameFileInput").click();
        imagesIntent = "add";
        el("imagesFileInput").click();
    };
    return tile;
}

/**
 * The "Add image" tile: an image of the user's own, framed to the video's
 * proportions in a pop-up (see framecrop.js), becomes one more frame of the
 * run — at the end of the grid, numbered after the rest.
 *
 * From there it is not a special case. The backend makes it an ordinary
 * capture slot (see /add-frame), so the edit, the presets, the reframe and
 * the zoom on the canvas all reach it by its id; and this side gives it a
 * frame record and a card built exactly as the run's own were, so it is
 * dragged, discarded, captioned, given text by "Split the text" and zipped
 * by "Download all" with no more said about it. The two things it does not
 * have are a moment in the video — so no timestamp, and Variation stays off
 * on it (`uploaded`) — and a caption from the transcript: its text starts
 * empty, for the user to type or split onto.
 *
 * The channel is taken from the last frame of the grid, which is the
 * neighbour it lands beside: the run's own brand, whatever the picker above
 * the link box has been moved to since.
 */
async function addOwnFrame(file) {
    const grid = frames();
    if (!isCapture() || !grid.length) return;

    let img;
    const local = URL.createObjectURL(file);
    try {
        img = await loadImage(local);
    } catch (_) {
        return status("That file could not be read as an image");
    } finally {
        URL.revokeObjectURL(local);
    }
    // The video's own size; failing that, the canvas's, which is at least
    // the shape every frame of the run ends up as.
    const output = captureMode() && captureMode().output;
    const [w, h] = runSourceSize || (output ? [output.width, output.height] : [1080, 1920]);
    const blob = await openFrameCrop(img, w, h);
    if (!blob) return;                  // cancelled — nothing added
    if (frames() !== grid) return;      // a new run's frames replaced these meanwhile

    const tile = el("addFrameTile");
    if (tile) tile.disabled = true;
    status("Adding your image to the frames...");
    let res;
    try {
        res = await postBlobJson("/add-frame", blob);
    } catch (e) {
        // A backend older than this file has no /add-frame, and answers with
        // FastAPI's own "Not Found" rather than anything about frames.
        const stale = (e.status === 404 && e.message === "Not Found") || e.status === 405;
        return status(stale ? "Backend is out of date (no /add-frame)" : e.message);
    } finally {
        if (tile) tile.disabled = false;
    }
    if (frames() !== grid) return;

    const frame = makeFrame(res.frame_id, base64ToObjectUrl(res.frame));
    frame.uploaded = true;
    const order = gridOrder();
    const neighbour = grid[order[order.length - 1]];
    if (neighbour) {
        frame.channel = neighbour.channel;
        frame.channelTouched = neighbour.channelTouched;
    }
    const i = addFrame(frame);
    const position = gridOrder().length - 1;
    const card = frameCard(i, position);
    el("framesGrid").insertBefore(card, el("addFrameTile"));
    refreshThumb(i);
    // Opened at once, as the first frame of a run is: the edit starts on it
    // there (selectFrame asks for it), and the card turns from grey when it
    // lands, exactly as the run's own did.
    selectFrame(i);
    card.scrollIntoView({ block: "nearest" });
    status(`Your image was added as frame ${position + 1}`);
}

/**
 * After a drop that changed the grid's order: everything that shows a frame's
 * PLACE follows it. The cards themselves are already where the user put them
 * (reorder.js moved them), so this renumbers them rather than drawing the grid
 * again — a redraw would throw away every thumbnail and repaint forty of them
 * for the sake of a few digits.
 */
function onGridReordered() {
    gridOrder().forEach((i, position) => {
        const card = el(`card-${i}`);
        const badge = card?.querySelector(".badge");
        if (badge) badge.textContent = position + 1;
        card?.querySelector(".discard")?.setAttribute("aria-label", `Discard frame ${position + 1}`);
    });
    syncFrameNav();   // the frame on screen may have moved to an end
    if (!frames().length) return;
    el("previewLabel").textContent = `Preview — Frame ${positionOf(selectedIndex()) + 1}`;
    loadTextPanel();  // its heading carries the frame's number too
}

/**
 * The X on a card: frame `i` leaves the run (see state.discardFrame), and the
 * grid closes up behind it.
 *
 * No confirmation and no undo. The X is small, red and alone in its corner,
 * so it is not pressed by accident, and a question before every discard
 * would make throwing out ten frames twenty clicks. The tick that marks a
 * frame for the batch sits in the opposite corner for the same reason.
 *
 * The edit pass needs telling nothing: it asks the grid's order afresh before
 * every frame (editor.applyEditToAll), so a grey card thrown out is never
 * reached, and one thrown out while it is being edited finishes unseen. The
 * cards that stay keep theirs — the card is removed rather than the grid
 * redrawn, and everything numbered is renumbered as after a drag.
 *
 * The last frame stays: a run with no frames left has nothing to show on the
 * canvas, and starting a new video is the way to be rid of all of them.
 */
function discardFromGrid(i) {
    const order = gridOrder();
    if (!order.includes(i)) return;
    if (order.length === 1) return status("The last frame cannot be discarded");
    const position = order.indexOf(i);
    // Where the selection goes if it is this frame: the next card that is
    // ready, else the previous one, else whatever card took its place —
    // grey or not, as the first frame of a run is selected while it is grey.
    const wasSelected = i === selectedIndex();
    let next = -1;
    if (wasSelected) {
        next = nextSelectable(i, +1);
        if (next < 0) next = nextSelectable(i, -1);
        if (next < 0) next = order[position + 1] ?? order[position - 1];
    }
    discardFrame(i);
    el(`card-${i}`)?.remove();
    syncBatchButton();   // it may have been ticked for the batch
    if (wasSelected) selectFrame(next);
    onGridReordered();
    status(`Frame ${position + 1} discarded — ${gridOrder().length} frames left`);
}

/**
 * One frame forward or back, from the arrows beside the canvas.
 *
 * Routed through selectFrame rather than through setSelectedIndex, so that
 * arriving by arrow and arriving by clicking a card are the same event: the
 * carry-over, the channel switch, the zoom, the panel and the preview all
 * happen either way. An arrow is a second way to reach a frame, not a second
 * kind of selection.
 */
function stepFrame(step) {
    const target = nextSelectable(selectedIndex(), step);
    if (target >= 0) selectFrame(target);
}

// Both entry points into carry-over invalidate the target's cached wide view
// when the inherited preset differs from what it had.
const onPresetCarried = (frame) => wide.invalidate(frame.frame_id);
const carryOverForPick = (i) => carryOverSettings(i, onPresetCarried);

function selectFrame(i) {
    carryOverSettings(i, onPresetCarried);
    setSelectedIndex(i);
    // Before anything is drawn or any panel is filled in: the brand is the
    // frame's, so moving to a frame means moving to its channel (see
    // editor.syncChannelToFrame). Not awaited — it renders again by itself if
    // the face has still to arrive.
    syncChannelToFrame();
    // By the card's own index, not its place among its siblings: on a grid the
    // user has rearranged (see reorder.js) the two are no longer the same.
    document.querySelectorAll(".frame-card").forEach((element) =>
        element.classList.toggle("selected", Number(element.dataset.index) === i));
    syncFrameNav();   // whether either arrow still has somewhere to go
    el("previewLabel").textContent = `Preview — Frame ${positionOf(i) + 1}`;
    syncZoomSliderFor(frames()[i] && frames()[i].frame_id);
    syncFlipButtons();
    syncGradientButton();
    syncBackgroundButton();
    syncFigureGlowButton();
    syncVaryButton();  // Variation is unavailable on a frame the user uploaded
    syncLogoControls();
    loadTextPanel();   // the side panel follows the selection — per-frame title
    updatePreview();
    enhanceFrame(i);
}

/**
 * Scales the "Thumbnail Maker" title's font-size so its rendered width
 * matches the URL textbox's width — text width scales ~linearly with
 * font-size for a fixed string/font/weight, so one measurement at a known
 * baseline size gives the right ratio directly, no iterative search needed.
 * Resets to the baseline before each measurement so repeated calls (resize)
 * scale from a fixed point instead of compounding.
 */
const TITLE_BASE_PX = 32;
let _titleFitTimer = null;

function fitTitleToInputWidth() {
    // The link box, or on "Upload my own images" the row of buttons that
    // stands in its place — the same full width of the same column, and the
    // one of the two that is on screen to be measured.
    const input = imagesSource() ? el("imageSourceRow") : el("driveUrl");
    const h1 = document.querySelector(".app-header h1");
    if (!input || !h1) return;
    const targetWidth = input.getBoundingClientRect().width;
    h1.style.fontSize = TITLE_BASE_PX + "px";
    const baseWidth = h1.getBoundingClientRect().width;
    if (targetWidth <= 0 || baseWidth <= 0) return;
    h1.style.fontSize = (TITLE_BASE_PX * (targetWidth / baseWidth)) + "px";
}

// ── Wiring ────────────────────────────────────────────────────────────────
// Handlers are bound here rather than through inline onclick attributes in
// the markup: with the script loaded as a module, nothing it declares is
// global, so inline attributes would have no way to reach any of it.

el("processBtn").addEventListener("click", processVideo);
el("driveUrl").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !el("processBtn").disabled) processVideo();
});

// Picking a file starts the run there and then, rather than arming a second
// button: choosing a video in the OS file dialog and pressing Open is already
// the deliberate act that "Process Video" is for a link. Same shape as the
// image-overlay picker further down.
// ...with the caption ceiling asked for first, so the user is not sent through
// the file dialog only to be told afterwards that a field was empty.
// The notice is primed on this click rather than on the file being chosen:
// a `change` event is not the user's gesture as far as the permission prompt
// is concerned.
el("uploadVideoBtn").addEventListener("click", () => {
    if (missingCaptionChars()) return;
    primeReadyNotice();
    el("videoFileInput").click();
});
el("videoFileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";   // so re-picking the same file still fires `change`
    if (file) processUploadedFile(file);
});

// "Upload my own images" (see startImageSet). The button above the grid adds
// to the set on screen when there is one and starts one when there is not;
// "Start a new set" always starts one. Both open the same dialog, and say
// first which of the two the files that come back are for.
el("uploadImagesBtn").addEventListener("click", () => {
    if (missingImageChannel()) return;
    imagesIntent = imageSetOpen() ? "add" : "new";
    el("imagesFileInput").click();
});
el("newImageSetBtn").addEventListener("click", () => {
    if (missingImageChannel()) return;
    imagesIntent = "new";
    el("imagesFileInput").click();
});
el("imagesFileInput").addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    e.target.value = "";   // so re-picking the same files still fires `change`
    if (!files.length) return;
    if (imagesIntent === "add" && imageSetOpen()) addImages(files);
    else startImageSet(files);
});

initChannelPicker();
// --- Platform Mode Switching (Optgroup Filter) ---
// --- Platform Mode Switching & Clean Channel Grouping ---
// --- Platform Mode Switching & Clean Channel Grouping ---
const tabThumbnailBtn = document.getElementById("tabThumbnail");
const tabStaticBtn = document.getElementById("tabStatic");
const mainTitle = document.getElementById("mainTitle");

const THUMBNAIL_GROUPS = ["QDS KARMA", "FRANCE TV", "BINGE"];
const STATIC_GROUPS = ["SNAPCHAT"];

const THUMBNAIL_URL_PLACEHOLDER = "Paste Google Drive, Dropbox, or YouTube link here";
const STATIC_URL_PLACEHOLDER = "Paste Instagram, Google Drive, Dropbox, or YouTube link here";

let currentMode = "thumbnail";

function storeAndRenderDropdown(selectEl, mode) {
    if (!selectEl) return;

    // Cache the original complete list on first run
    if (!selectEl._fullChannelStore) {
        selectEl._fullChannelStore = [];
        const optgroups = Array.from(selectEl.querySelectorAll("optgroup"));

        optgroups.forEach(group => {
            const rawLabel = (group.label || "").trim().toUpperCase();
            Array.from(group.children).forEach(opt => {
                selectEl._fullChannelStore.push({
                    value: opt.value,
                    text: opt.textContent.trim(),
                    group: rawLabel
                });
            });
        });
    }

    if (!selectEl._fullChannelStore.length) return;

    const isStatic = mode === "static";
    const selectedVal = selectEl.value;

    // Rebuild the select cleanly from memory
    selectEl.innerHTML = '<option value="">-</option>';

    if (isStatic) {
        // Static Studio: Clean flat list of Snapchat channels (no optgroup headers)
        const snapChannels = selectEl._fullChannelStore.filter(item => 
            STATIC_GROUPS.includes(item.group)
        );

        // Sort alphabetically
        snapChannels.sort((a, b) => a.text.localeCompare(b.text));

        snapChannels.forEach(item => {
            const opt = document.createElement("option");
            opt.value = item.value;
            opt.textContent = item.text;
            selectEl.appendChild(opt);
        });
    } else {
        // Thumbnail Maker: Unified optgroups (QDS KARMA, FRANCE TV, BINGE)
        THUMBNAIL_GROUPS.forEach(groupName => {
            const groupChannels = selectEl._fullChannelStore.filter(item => item.group === groupName);
            if (groupChannels.length > 0) {
                const optgroup = document.createElement("optgroup");
                optgroup.label = groupName;

                groupChannels.sort((a, b) => a.text.localeCompare(b.text));

                groupChannels.forEach(item => {
                    const opt = document.createElement("option");
                    opt.value = item.value;
                    opt.textContent = item.text;
                    optgroup.appendChild(opt);
                });

                selectEl.appendChild(optgroup);
            }
        });
    }

    // Preserve previous selection if it belongs to the current mode
    if (Array.from(selectEl.options).some(o => o.value === selectedVal)) {
        selectEl.value = selectedVal;
    } else {
        selectEl.value = "";
    }
}

function setAppMode(mode) {
    currentMode = mode;
    const isStatic = mode === "static";

    if (mainTitle) {
        mainTitle.textContent = isStatic ? "STATIC STUDIO" : "THUMBNAIL MAKER";
    }

    tabStaticBtn?.classList.toggle("active", isStatic);
    tabThumbnailBtn?.classList.toggle("active", !isStatic);

    // Through the shared .hidden class rather than an inline display, so
    // the row keeps the layout .btn-row gives it whenever it is shown.
    const modeToggle = document.getElementById("outputModeToggle");
    if (modeToggle) modeToggle.classList.toggle("hidden", !isStatic);

    // The backend takes an Instagram link from either tab — it cannot tell
    // which one a run came from — but only Static Studio names it, because
    // reels are where its footage comes from and offering them on the
    // thumbnail tab would only invite links nobody there works with.
    const urlInput = el("driveUrl");
    if (urlInput) urlInput.placeholder = isStatic ? STATIC_URL_PLACEHOLDER : THUMBNAIL_URL_PLACEHOLDER;

    storeAndRenderDropdown(document.getElementById("runChannelSelect"), mode);
    storeAndRenderDropdown(document.getElementById("channelSelect"), mode);

    // The selection follows the list. Rebuilding a dropdown without the
    // selected channel only changes what it SHOWS — it reads "-" while the
    // page is still on, say, a Snapchat channel picked on the other tab. That
    // hidden channel is what the next run is made from: a Thumbnail Maker run
    // would go out as a Snapchat frame fetch, and the frame-count slider,
    // which answers to the selection, stays up on a tab that has no use for
    // it. So a channel the new tab does not list is dropped, through
    // showChannel because nobody picked anything: no frame is re-branded and
    // no usage event is sent. Checked rather than done every time, because
    // this runs again on timers after load (below) and must leave a channel
    // that belongs to the tab alone.
    const listed = Array.from(el("runChannelSelect")?.options || [])
        .some(o => o.value === activeChannelId());
    if (hasChannel() && !listed) showChannel(NO_CHANNEL);
    syncCaptureFields(activeChannelId());
    syncCaptionCharsField();

    if (typeof fitTitleToInputWidth === "function") {
        fitTitleToInputWidth();
    }
}

tabThumbnailBtn?.addEventListener("click", () => setAppMode("thumbnail"));
tabStaticBtn?.addEventListener("click", () => setAppMode("static"));

// Trigger after content loads
const channelSelectEl = document.getElementById("runChannelSelect");
if (channelSelectEl) {
    const observer = new MutationObserver(() => {
        if (channelSelectEl.querySelectorAll("option, optgroup").length > 2) {
            observer.disconnect();
            setAppMode(currentMode);
        }
    });
    observer.observe(channelSelectEl, { childList: true, subtree: true });
}

// The mode the page is in rather than "thumbnail" outright: a run picked up on
// load (see the end of this file) may already have moved the page to the
// static tab, and these must not undo that a moment later.
setTimeout(() => setAppMode(currentMode), 200);
setTimeout(() => setAppMode(currentMode), 600);
// Both pickers are the same selection (see editor.CHANNEL_PICKERS), so both
// answer the same way and each keeps the other in step.
el("channelSelect").addEventListener("change", (e) => selectChannel(e.target.value));
el("runChannelSelect").addEventListener("change", (e) => selectChannel(e.target.value));
// ...and the count box belongs with them, since how many frames to fetch is
// only a question for a channel that fetches frames. Bound to both pickers for
// the reason they are one selection: either one moving is the same answer
// changing, and the box has to follow it from wherever it was changed.
for (const id of ["runChannelSelect", "channelSelect"]) {
    el(id).addEventListener("change", () => {
        syncCaptureFields(activeChannelId());
        syncCaptionCharsField();
    });
}
syncCaptureFields(activeChannelId());
// The slider's fill and its readout follow the handle, wherever the handle was
// moved from — a drag, a click on the track, or an arrow key.
el("frameCount").addEventListener("input", syncCountValue);

// The title fields are readonly until a channel is picked (see
// syncChannelFields), which already refuses every edit — these listeners are
// what turn that silent refusal into an answer. Each route into the field
// gets its own: readonly suppresses `input`/`beforeinput` entirely, so
// keystrokes have to be caught on keydown, while paste and drag-and-drop
// arrive as their own events.
// Every field a title can be typed into — the three line slots and the one
// block that stands in for them (see config.TITLE_FIELD_IDS). One loop,
// because the lock in front of them and the answer it owes the user are the
// channel's business and have nothing to do with which field is on screen.
TITLE_FIELD_IDS.forEach(id => {
    const input = el(id);
    input.addEventListener("input", updateLines);
    input.addEventListener("beforeinput", (e) => { if (!guardChannel()) e.preventDefault(); });
    input.addEventListener("keydown", (e) => {
        if (hasChannel()) return;
        // Shortcuts and navigation keys are left alone: Ctrl+C or an arrow
        // key changes nothing, so answering them with "pick a channel" would
        // be noise. Ctrl+V is covered by the paste handler below.
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const edits = e.key.length === 1 || ["Backspace", "Delete", "Enter"].includes(e.key);
        if (!edits) return;
        e.preventDefault();
        warnChannelRequired();
    });
    input.addEventListener("paste", (e) => { if (!guardChannel()) e.preventDefault(); });
    input.addEventListener("drop", (e) => { if (!guardChannel()) e.preventDefault(); });
    // Clicking into a locked field is the moment the user is about to type,
    // so the reason is up before the first keystroke rather than after it.
    input.addEventListener("focus", () => { if (!hasChannel()) warnChannelRequired(); });
    // ...and it is also the moment the user has finished with whichever figure
    // was wearing the reorder buttons: they have turned from the picture to the
    // words, and three circles sitting on the thumbnail through every keystroke
    // cover the composition they are typing against. See
    // reframe.dismissFigureHandles.
    input.addEventListener("focus", dismissFigureHandles);
});

el("flipImageBtn").addEventListener("click", () => toggleFlip("flipImage"));
el("flipTextBtn").addEventListener("click", () => toggleFlip("flipText"));
el("gradientBtn").addEventListener("click", toggleGradient);
// The other two that only a cutout channel ever shows (see
// editor.syncBackgroundButton and editor.syncFigureGlowButton); bound
// unconditionally for the reason the props button is.
el("bgBtn").addEventListener("click", cycleBackground);
el("figureGlowBtn").addEventListener("click", cycleFigureGlow);
// Only ever visible for a channel that offers a halo (see
// editor.syncTitlePanelMode); bound unconditionally all the same, because a
// listener on a hidden button costs nothing and a button whose handler
// depends on the selected channel would be one more thing to keep in sync.
el("glowBtn").addEventListener("click", toggleGlow);
// Only ever visible for a channel whose block has slack to distribute (see
// editor.alignable); bound unconditionally for the reason the glow button is.
for (const [id, align] of [["alignLeftBtn", "left"], ["alignCenterBtn", "center"], ["alignRightBtn", "right"]]) {
    el(id).addEventListener("click", () => setTitleAlign(align));
}
el("prevFrameBtn").addEventListener("click", () => stepFrame(-1));
el("nextFrameBtn").addEventListener("click", () => stepFrame(+1));
el("varyBtn").addEventListener("click", varyFrame);
el("downloadBtn").addEventListener("click", downloadFrame);
el("batchBtn").addEventListener("click", downloadSelected);
// Only ever visible on a frame fetch (see capture.js), and bound
// unconditionally for the reason the glow button is: a listener on a hidden
// button costs nothing, and one attached and detached with the mode would be
// one more thing to keep in step.
el("downloadAllBtn").addEventListener("click", downloadAll);
// Dragging cards to rearrange a frame fetch (see reorder.js). Bound once, to
// the grid itself, so it keeps working as renderGrid replaces the cards.
initReorder(onGridReordered);
// The "Add image" tile's file dialog (see addOwnFrame). Picking a file goes
// straight to the framing step, which is where the user can still back out.
el("addFrameFileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";   // so re-picking the same file still fires `change`
    if (file) addOwnFrame(file);
});
// Locked until every frame of a "Frames only" run has loaded (see
// runPipeline); a locked button takes no clicks, so nothing here has to ask.
el("splitTextBtn").addEventListener("click", splitMyText);

for (const id of ["presetNoneBtn", "presetNaturalBtn", "presetFullBtn"]) {
    el(id).addEventListener("click", () => setEditPreset(el(id).dataset.preset));
}

// Same shape as the image-overlay picker below: a styled button in front of
// a hidden file input, and picking a file acts immediately.
el("frameUploadBtn").addEventListener("click", () => el("frameFileInput").click());
el("frameFileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";   // so re-picking the same file still fires `change`
    if (file) handleFrameFile(file);
});

el("logoUploadBtn").addEventListener("click", () => el("logoFileInput").click());
el("logoAssetsBtn").addEventListener("click", handleAssetPick);
// Pressing an overlay on the canvas selects it (see reframe.js); the panel's
// list has to follow, and this is where the two sides are joined.
document.addEventListener("logo-selection-changed", () => {
    syncLogoControls();
    // The flip button belongs to whichever cut-out figure is selected on a
    // channel that has them (see editor.syncFlipButtons).
    syncFlipButtons();
});
// ...and a reframe drop replaces the pixels a cutout was made from, which
// reframe.js announces rather than acting on for the same reason: it cannot
// import editor.js without closing a cycle. Both sides are wired here.
document.addEventListener("frame-photo-changed", (e) => ensureCutout(e.detail.index));
el("logoFileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";   // allow re-selecting the same file (e.g. after Remove image)
    if (file) handleLogoFile(file);
});

// The channel packs have already been read by the time this module runs —
// channels.js awaits them at its own top level, which is what keeps every
// titleStyle() call in text.js synchronous. What is left is to say so if any
// of them failed.
reportContentProblems();

// Wait for the web font to finish loading first — measuring against the
// fallback system font would scale to the wrong ratio, then "snap" once
// Raleway swaps in.
titleFontReady().then((font) => {
    if (!font.ok) reportMissingTitleFont();
    fitTitleToInputWidth();
    // Anything already on the canvas was laid out against whatever font
    // was active at the time. text.js drops its measurements here, but a
    // canvas already painted keeps its pixels — so redraw whatever is on
    // screen against the real font instead of leaving a title composed
    // with fallback metrics sitting there.
    if (frames().length) {
        updatePreview();
        refreshAllThumbs();
    }
});
window.addEventListener("resize", () => {
    clearTimeout(_titleFitTimer);
    _titleFitTimer = setTimeout(fitTitleToInputWidth, 150);
});

initModeToggle();

// ── Leaving the page ──────────────────────────────────────────────────────
//
// Closing or reloading the page is the one thing that stops a run and gives
// the backend seat back — nothing else does, however long the page sits
// quiet (see the top of serve.py). So before it goes with something on it,
// it asks; and once it is going, it says goodbye.

// Whether this browser has another tab of the app open. Every tab shares the
// seat and the run (see the storage listener above), so closing one of them
// must not stop the run or give the seat back while another is using both.
//
// Each tab is in the list from the moment it opens until it closes, and
// writes now and then that it is still alive. A tab the browser has frozen in
// the background writes nothing for as long as it is frozen and still counts:
// it is open, and closing its sibling must not take its video down. Only a
// day of silence drops a tab from the list — that is a tab that crashed or
// was killed without the chance to say goodbye, and a seat it holds then
// stays held until the same browser comes back for it.
//
// The id lives in sessionStorage so a tab keeps it across a reload, and
// across the browser discarding the tab and loading it again — otherwise the
// discarded page would sit in the list as a second tab nobody can close.
const TABS_KEY = "thumbnailMaker.tabs";
const TAB_ID_KEY = "thumbnailMaker.tabId";
const TAB_FORGET_MS = 24 * 60 * 60 * 1000;
const TAB_ALIVE_MS = 60 * 1000;

let TAB_ID = (() => {
    try { return sessionStorage.getItem(TAB_ID_KEY) || newRunId(); } catch { return newRunId(); }
})();
function keepTabId() {
    try { sessionStorage.setItem(TAB_ID_KEY, TAB_ID); } catch { /* see RUN_KEY */ }
}
keepTabId();

// A duplicated tab (the browser's "Duplicate" or a middle-click on a link
// back to the page) starts with a COPY of its original's sessionStorage, and
// so with the original's id. The original answers when the copy announces
// itself, and the copy takes an id of its own. A BroadcastChannel because it
// reaches every other tab of this browser and works on plain http.
const PAGE_ID = newRunId();
const tabChannel = "BroadcastChannel" in window ? new BroadcastChannel("thumbnailMaker.tabs") : null;
if (tabChannel) {
    tabChannel.onmessage = ({ data }) => {
        if (!data) return;
        if (data.hello === TAB_ID && data.page !== PAGE_ID) {
            tabChannel.postMessage({ taken: TAB_ID, page: data.page });
        } else if (data.taken === TAB_ID && data.page === PAGE_ID) {
            TAB_ID = newRunId();
            keepTabId();
            markTabAlive();
        }
    };
    tabChannel.postMessage({ hello: TAB_ID, page: PAGE_ID });
}

function readTabs() {
    try { return JSON.parse(localStorage.getItem(TABS_KEY)) || {}; } catch { return {}; }
}

function writeTabs(tabs) {
    try { localStorage.setItem(TABS_KEY, JSON.stringify(tabs)); } catch { /* see RUN_KEY */ }
}

function markTabAlive() {
    const now = Date.now();
    const tabs = readTabs();
    for (const [id, seen] of Object.entries(tabs)) if (now - seen > TAB_FORGET_MS) delete tabs[id];
    tabs[TAB_ID] = now;
    writeTabs(tabs);
}

function otherTabsOpen() {
    const now = Date.now();
    return Object.entries(readTabs()).some(([id, seen]) => id !== TAB_ID && now - seen <= TAB_FORGET_MS);
}

// Nothing here reaches the proxy: it no longer cares how long a page has been
// quiet. This only keeps the list above from forgetting a tab that is open.
markTabAlive();
setInterval(markTabAlive, TAB_ALIVE_MS);
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") markTabAlive();
});

// Set when the page reloads itself, so it does not ask the user to confirm a
// reload they never asked for.
let leavingQuietly = false;

// The question before leaving, when leaving would lose something: a video
// processing, or frames on the screen. Browsers decide the wording and the
// language — a page may only ask for the dialog, never write it — and show it
// only once the user has clicked or typed on the page. Saying no keeps
// everything exactly as it was: no event follows, nothing below runs.
window.addEventListener("beforeunload", (e) => {
    if (leavingQuietly || (!pipelineBusy && !frames().length)) return;
    e.preventDefault();
    e.returnValue = "";   // what older Chrome needs to show the dialog
});

// The page actually going — closed, reloaded, or navigated away from, after a
// yes or with nothing to ask about. If it is the browser's last tab of the
// app, the run it had going is stopped and the seat given back at once (see
// /api/leave in serve.py). A reload then starts from nothing, as the user was
// told it would; the run is forgotten here so the new page does not go looking
// for it.
//
// sendBeacon because nothing else is guaranteed to leave a page that is
// unloading. It can still fail to — a crash, a phone killing the browser —
// and then the seat stays held until this browser comes back to the page:
// there is deliberately no timeout to take it back (see serve.py).
window.addEventListener("pagehide", () => {
    const tabs = readTabs();
    delete tabs[TAB_ID];
    writeTabs(tabs);
    if (otherTabsOpen()) return;
    const runId = storedRunId();
    forgetRun(runId);
    navigator.sendBeacon(`${API}/api/leave`, JSON.stringify({ runId }));
});

// Back to a page the browser kept in memory after leaving it. Its seat was
// given back on the way out, so what it shows no longer exists on any
// backend; starting over is the honest state to come back to.
window.addEventListener("pageshow", (e) => {
    if (!e.persisted) return;
    leavingQuietly = true;
    location.reload();
});

// A run this browser started that the backend is still working on, or has
// finished without anyone collecting it, belongs on this page — the usual way
// to get here with one going is to have reloaded mid-run. See runInProgress.
runPipeline(null);
