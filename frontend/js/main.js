import { API, TITLE_FIELD_IDS, el } from "./config.js";
import {
    store, frames, current, selectedIndex, setSelectedIndex,
    makeFrame, releaseFrames, carryOverSettings,
} from "./state.js";
import { getJson, postJson, postFile } from "./net.js";
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
    varyFrame, syncFlipButtons, syncGradientButton, togglePick, handleLogoFile,
    initChannelPicker, selectChannel, syncChannelToFrame, guardChannel, warnChannelRequired,
    handleFrameFile, syncVaryButton, syncLogoControls, ensureCutout, setTitleAlign,
    cycleBackground, syncBackgroundButton, cycleFigureGlow, syncFigureGlowButton,
    warmCutouts, applyEditToAll,
} from "./editor.js";
import { hasChannel, activeChannelId, framingFor, captureFor, cleanupFor,
         selectionFor } from "./channels.js";
import {
    enterMode, isCapture, syncCaptureFields, requestedCount, syncCountValue,
    applyCaptions,
} from "./capture.js";
import { downloadFrame, downloadSelected, downloadAll } from "./download.js";
import { titleFontReady, reportMissingTitleFont } from "./fontguard.js";

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
async function waitForResult() {
    let failures = 0;
    for (;;) {
        await sleep(POLL_MS);

        const progress = await getJson("/process-video/progress").catch(() => null);
        if (!progress) {
            if (++failures >= POLL_MAX_FAILURES) throw new Error("Lost contact with the server");
            continue;
        }
        failures = 0;
        showProgress(progress.stage, progress.percent);

        if (progress.status === "error") throw new Error(progress.detail || "Processing failed");
        // "idle" can only mean the backend forgot the run it had just told us
        // it started — i.e. it restarted. Without this the poll would spin
        // forever on a run that no longer exists, which is the exact hang
        // moving off the long-lived request was meant to eliminate.
        if (progress.status === "idle") throw new Error("The run was lost — the server restarted");
        if (progress.status === "done") {
            const result = await getJson("/process-video/result");
            if (!result) throw new Error("Finished, but the frames could not be collected");
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

// Both source buttons move together: a run started from either one owns the
// backend until it finishes (/process-video answers 409 to a second), so
// leaving the other live would only offer the user a guaranteed error.
function setSourceBusy(busy) {
    el("processBtn").disabled = busy;
    el("uploadVideoBtn").disabled = busy;
}

/**
 * Runs one video end to end, whatever it came from.
 *
 * `prepare` is the only part that differs between a pasted link and an
 * uploaded file: it does whatever getting the video within the backend's
 * reach requires (nothing at all, for a link) and hands back the body to
 * start /process-video with. Everything from that call onward — the poll, the
 * grid, the font check, the first enhance — is shared, because from the
 * backend's side there is no longer any difference to act on.
 */
async function runPipeline(prepare) {
    setSourceBusy(true);
    status("");
    try {
        const { body, stage, capture } = await prepare();
        showProgress(stage, 0);
        await postJson("/process-video", body);
        const data = await waitForResult();

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
        enterMode(capture);

        store.frames = data.frames.map(f => makeFrame(
            f.frame_id, base64ToObjectUrl(f.frame),
            f.timecode ? { timestamp: f.timestamp, timecode: f.timecode } : null,
            f.caption || ""));
        // A channel that asked for rewritten captions and did not get them is
        // the one outcome here that looks exactly like success: every frame is
        // captioned, every caption reads, and every one of them is the speech
        // in the video rather than the report the channel is built on. The
        // backend says which happened (see api._capture_frames_sync) and this
        // is the only place the user would ever find out.
        const spoken = capture && capture.rewrite && data.rewritten === false;
        status(isCapture()
            ? `${data.count} frames, in the order they happen in the video`
              + (spoken ? " — captions are the transcript as spoken: the narration "
                        + "model did not load, so they were not rewritten as report" : "")
            : `${data.count} frames — select one and give it a title`);
        setSelectedIndex(0);
        el("resultSection").classList.remove("hidden");

        // The title font is self-hosted and may still be downloading at this
        // point — without waiting, the first canvas render/measurement could
        // briefly use the Impact fallback and bake its metrics into the layout.
        // A face that never arrives at all is a broken install rather than a
        // slow one, and is called out rather than silently drawn around.
        if (!(await titleFontReady()).ok) reportMissingTitleFont();

        // ...and only now are the captions of a run that has any broken into
        // lines, because every break is chosen by measuring the words — in
        // the run's OWN channel's face, which is a second thing to wait for
        // and which applyCaptions waits for itself. Before the panel is
        // loaded and before the grid is drawn, so the first thing the user
        // sees already has the words on it.
        await applyCaptions();

        initCanvas(setupReframeInteraction);
        renderGrid();
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
        if (isCapture()) applyEditToAll();
        else warmCutouts();
    } catch (e) {
        status("Error: " + e.message);
    } finally {
        hideProgress();
        setSourceBusy(false);
    }
}

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
                captions: !!capture.captions,
                rewrite: !!capture.rewrite,
            },
        }
        : null;
}

function processVideo() {
    const url = el("driveUrl").value.trim();
    if (!url) return status("Paste a link, or upload a video file from your computer");
    const capture = runCapture();
    return runPipeline(async () => ({
        body: { google_drive_url: url, framing: runFraming(), capture: capture && capture.request,
                cleanup: runCleanup(), selection: runSelection() },
        stage: "Downloading video and extracting frames...",
        capture: capture && capture.block,
    }));
}

/**
 * The upload is the one stage the backend cannot report progress for — it
 * hasn't received the file yet — so this drives the bar itself until the
 * bytes are across, and the poll takes over from there.
 */
function processUploadedFile(file) {
    // Read before the upload starts, not after it finishes: the bytes take
    // their own time to cross, and the channel a run belongs to is the one
    // that was selected when the user set it going.
    const capture = runCapture();
    return runPipeline(async () => {
        const label = `Uploading ${file.name}...`;
        showProgress(label, 0);
        const { upload_id } = await postFile("/upload-video", file,
            (fraction) => showProgress(label, Math.round(fraction * 100)));
        return {
            body: { upload_id, framing: runFraming(), capture: capture && capture.request,
                    cleanup: runCleanup(), selection: runSelection() },
            stage: "Reading uploaded video and extracting frames...",
            capture: capture && capture.block,
        };
    });
}

function renderGrid() {
    const g = el("framesGrid");
    g.innerHTML = "";
    frames().forEach((f, i) => {
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
        d.innerHTML = `<img id="thumb-${i}" src="${f.plainUrl}">` +
                      `<div class="badge">${i + 1}</div>` +
                      `<div class="pick${f.picked ? " on" : ""}" id="pick-${i}" title="Include in batch download"></div>`;
        // The stylesheet already makes a pending card inert; this is the same
        // answer given twice, so a stylesheet that failed to load leaves the
        // grid honest rather than silently clickable.
        d.onclick = () => { if (!d.classList.contains("pending")) selectFrame(i); };
        // Ticking a frame for the batch must not also switch the preview to
        // it — the user builds the batch while working on one frame.
        d.querySelector(".pick").onclick = (e) => {
            e.stopPropagation();
            if (d.classList.contains("pending")) return;
            togglePick(i, carryOverForPick);
        };
        g.appendChild(d);
    });
    syncFrameNav();
    refreshAllThumbs();
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
    document.querySelectorAll(".frame-card").forEach((element, j) => element.classList.toggle("selected", j === i));
    syncFrameNav();   // whether either arrow still has somewhere to go
    el("previewLabel").textContent = `Preview — Frame ${i + 1}`;
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
    const input = el("driveUrl");
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
el("uploadVideoBtn").addEventListener("click", () => el("videoFileInput").click());
el("videoFileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";   // so re-picking the same file still fires `change`
    if (file) processUploadedFile(file);
});

initChannelPicker();
// --- Platform Mode Switching (Optgroup Filter) ---
const tabThumbnailBtn = document.getElementById("tabThumbnail");
const tabStaticBtn = document.getElementById("tabStatic");
const mainTitle = document.getElementById("mainTitle");

// Tab 1: Thumbnail Maker displays QDS KARMA, FRANCE TV, and BINGE
const THUMBNAIL_GROUPS = ["QDS KARMA", "FRANCE TV", "BINGE"];

// Tab 2: Static Studio displays SNAPCHAT only
const STATIC_GROUPS = ["SNAPCHAT"];

function filterDropdownByGroup(selectEl, allowedGroups) {
    if (!selectEl) return;

    // Filter <optgroup> elements
    const optgroups = selectEl.querySelectorAll("optgroup");
    optgroups.forEach(group => {
        const groupLabel = (group.label || "").trim().toUpperCase();
        const shouldShow = allowedGroups.some(g => groupLabel === g.toUpperCase());
        group.style.display = shouldShow ? "" : "none";
        
        // Also toggle child options so keyboard navigation skips hidden groups
        Array.from(group.children).forEach(opt => {
            opt.hidden = !shouldShow;
            opt.disabled = !shouldShow;
        });
    });

    // Reset selection if the currently chosen option belongs to a hidden group
    const currentOpt = selectEl.selectedOptions[0];
    if (currentOpt && (currentOpt.hidden || currentOpt.parentElement?.style.display === "none")) {
        selectEl.value = "";
    }
}

function setAppMode(mode) {
    const isStatic = mode === "static";

    if (mainTitle) {
        mainTitle.textContent = isStatic ? "STATIC STUDIO" : "THUMBNAIL MAKER";
    }

    tabStaticBtn?.classList.toggle("active", isStatic);
    tabThumbnailBtn?.classList.toggle("active", !isStatic);

    const targetGroups = isStatic ? STATIC_GROUPS : THUMBNAIL_GROUPS;

    // Apply filter to both the landing dropdown and the in-editor dropdown
    filterDropdownByGroup(el("runChannelSelect"), targetGroups);
    filterDropdownByGroup(el("channelSelect"), targetGroups);

    fitTitleToInputWidth();
}

tabThumbnailBtn?.addEventListener("click", () => setAppMode("thumbnail"));
tabStaticBtn?.addEventListener("click", () => setAppMode("static"));

// Apply default Thumbnail Maker view on load
setAppMode("thumbnail");
// Both pickers are the same selection (see editor.CHANNEL_PICKERS), so both
// answer the same way and each keeps the other in step.
el("channelSelect").addEventListener("change", (e) => selectChannel(e.target.value));
el("runChannelSelect").addEventListener("change", (e) => selectChannel(e.target.value));
// ...and the count box belongs with them, since how many frames to fetch is
// only a question for a channel that fetches frames. Bound to both pickers for
// the reason they are one selection: either one moving is the same answer
// changing, and the box has to follow it from wherever it was changed.
for (const id of ["runChannelSelect", "channelSelect"]) {
    el(id).addEventListener("change", () => syncCaptureFields(activeChannelId()));
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
