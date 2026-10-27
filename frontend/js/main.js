
// --- Output Mode Toggle (Frames with text vs Frames only) ---
let framesOnlyMode = false;

// The chosen button is marked with .btn-active and nothing else — the look is
// the stylesheet's, shared with every other one-of-two choice in the app, so
// the colours are written down once rather than here. Outside initModeToggle
// because a run picked back up after a reload sets the mode too.
function syncModeToggle() {
    const btnText = document.getElementById("btnModeText");
    const btnOnly = document.getElementById("btnModeFramesOnly");
    if (!btnText || !btnOnly) return;
    btnText.classList.toggle("btn-active", !framesOnlyMode);
    btnOnly.classList.toggle("btn-active", framesOnlyMode);
    btnText.setAttribute("aria-pressed", String(!framesOnlyMode));
    btnOnly.setAttribute("aria-pressed", String(framesOnlyMode));
}

function initModeToggle() {
    const btnText = document.getElementById("btnModeText");
    const btnOnly = document.getElementById("btnModeFramesOnly");
    if (!btnText || !btnOnly) return;
    btnText.addEventListener("click", () => { framesOnlyMode = false; syncModeToggle(); });
    btnOnly.addEventListener("click", () => { framesOnlyMode = true; syncModeToggle(); });
    syncModeToggle();
}
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
    handleFrameFile, syncVaryButton, syncLogoControls, ensureCutout, setTitleAlign, showChannel,
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
        showProgress(progress.stage, progress.percent);

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
        // Stopped because the page that owned it was closed (see the leaving
        // section at the end of this file) — reached only by a tab that page
        // did not know was still open.
        if (progress.status === "cancelled") {
            forgetRun(runId);
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

// Both source buttons move together: a run started from either one owns the
// backend until it finishes (/process-video answers 409 to a second), so
// leaving the other live would only offer the user a second click on a run
// that is already going.
function setSourceBusy(busy) {
    el("processBtn").disabled = busy;
    el("uploadVideoBtn").disabled = busy;
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
    framesOnlyMode = !!client.framesOnly;
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
    setSourceBusy(true);
    status("");
    try {
        const run = (await runInProgress(!prepare)) || (prepare && await startRun(prepare));
        if (!run) return;
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
        // A channel that asked for rewritten captions and did not get them is
        // the one outcome here that looks exactly like success: every frame is
        // captioned, every caption reads, and every one of them is the speech
        // in the video rather than the report the channel is built on. The
        // backend says which happened (see api._capture_frames_sync) and this
        // is the only place the user would ever find out.
        const framesOnly = !!(run.client && run.client.framesOnly);
        const spoken = run.capture && run.capture.rewrite && !framesOnly && data.rewritten === false;
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
        //
        // The mode the run was started in, not the toggle's position now, for
        // the reason the capture block above is the run's: the backend was
        // told then whether to transcribe.
        if (!framesOnly) { await applyCaptions(); }

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
        status(e.message);
        if (e.replaced) followPending = true;
    } finally {
        hideProgress();
        setSourceBusy(false);
        pipelineBusy = false;
    }
    // A run another tab started while this one was busy, now that it is not.
    if (followPending) {
        followPending = false;
        runPipeline(null, "Following the video started in another tab...");
    }
}

// Whether runPipeline is in flight in this tab, and whether another tab
// started a run meanwhile that this one should follow once it is free.
let pipelineBusy = false;
let followPending = false;

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
                captions: framesOnlyMode ? false : !!capture.captions,
                rewrite: framesOnlyMode ? false : !!capture.rewrite,
            },
        }
        : null;
}

/**
 * The page's own settings for a run: which channel, which tab, which output
 * mode. The backend keeps them only to hand them back, so a page that picks
 * the run up after a reload can be put back the way this one was (see
 * runInProgress and resumeSettings). Read when the run is set going, like
 * everything else a run is made from.
 */
const runClient = () => ({
    channel: activeChannelId(), tab: currentMode, framesOnly: framesOnlyMode,
});

function processVideo() {
    const url = el("driveUrl").value.trim();
    if (!url) return status("Paste a link, or upload a video file from your computer");
    const capture = runCapture();
    const client = runClient();
    return runPipeline(async () => ({
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
    // Read before the upload starts, not after it finishes: the bytes take
    // their own time to cross, and the channel a run belongs to is the one
    // that was selected when the user set it going.
    const capture = runCapture();
    const client = runClient();
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
            client,
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

initModeToggle();

// ── Leaving the page ──────────────────────────────────────────────────────
//
// A backend seat is one of a handful, and a user who has gone should give it
// back at once rather than hold it until the proxy gives up on them (see
// SEEN_TIMEOUT_SECONDS in serve.py). So the page says when it is still there
// (a heartbeat) and when it is going (a goodbye), and before it goes with
// something on it, it asks.

// Whether this browser has another tab of the app open. Every tab shares the
// seat and the run (see the storage listener above), so closing one of them
// must not stop the run or give the seat back while another is using both.
// Each tab writes when it was last alive; one that has not written in a while
// crashed or was killed without the chance to say so, and does not count —
// leaving the seat to the proxy's own timeout in the meantime.
const TAB_ID = newRunId();
const TABS_KEY = "thumbnailMaker.tabs";
// Longer than a background tab's slowest heartbeat: browsers hold a hidden
// tab's timers to about once a minute.
const TAB_FRESH_MS = 150 * 1000;
const HEARTBEAT_MS = 60 * 1000;

function readTabs() {
    try { return JSON.parse(localStorage.getItem(TABS_KEY)) || {}; } catch { return {}; }
}

function writeTabs(tabs) {
    try { localStorage.setItem(TABS_KEY, JSON.stringify(tabs)); } catch { /* see RUN_KEY */ }
}

function markTabAlive() {
    const now = Date.now();
    const tabs = readTabs();
    for (const [id, seen] of Object.entries(tabs)) if (now - seen > TAB_FRESH_MS) delete tabs[id];
    tabs[TAB_ID] = now;
    writeTabs(tabs);
}

function otherTabsOpen() {
    const now = Date.now();
    return Object.entries(readTabs()).some(([id, seen]) => id !== TAB_ID && now - seen <= TAB_FRESH_MS);
}

// Whether anyone has touched the page since the last heartbeat. Typing a title
// reaches no backend, so without this the proxy would see a user hard at work
// as an idle one (see IDLE_TIMEOUT_SECONDS in serve.py).
let touchedSinceBeat = true;
for (const type of ["pointerdown", "keydown", "input", "wheel"]) {
    window.addEventListener(type, () => { touchedSinceBeat = true; }, { capture: true, passive: true });
}

function heartbeat() {
    markTabAlive();
    const active = touchedSinceBeat;
    touchedSinceBeat = false;
    fetch(`${API}/api/heartbeat`, {
        method: "POST", body: JSON.stringify({ active }), keepalive: true,
    }).catch(() => { /* the next one will do */ });
}
heartbeat();
setInterval(heartbeat, HEARTBEAT_MS);
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") heartbeat();
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
// which is what the proxy's timeouts are for.
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
