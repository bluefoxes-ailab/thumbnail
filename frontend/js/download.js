import { MAX_EXPORT_BYTES, el } from "./config.js";
import { frames, selectedIndex } from "./state.js";
import { buildComposite } from "./compose.js";
import { ensureCutout, ensureEnhanced } from "./editor.js";
import { isCapture, frameName } from "./capture.js";
import { zipBlob } from "./zip.js";
import { status } from "./ui.js";

/**
 * PNG when it fits under YouTube's 2MB cap, otherwise JPEG stepping quality
 * down until it does (a 1280×720 JPEG lands well under the cap already at the
 * first step — the loop is a guarantee, not the expected path).
 */
async function exportBlob(fabricCv) {
    const toBlob = async (opts) => (await fetch(fabricCv.toDataURL({ multiplier: 1, ...opts }))).blob();
    let blob = await toBlob({ format: "png" });
    if (blob.size <= MAX_EXPORT_BYTES) return { blob, ext: "png" };
    for (const q of [0.92, 0.85, 0.78, 0.7, 0.6]) {
        blob = await toBlob({ format: "jpeg", quality: q });
        if (blob.size <= MAX_EXPORT_BYTES) break;
    }
    return { blob, ext: "jpg" };
}

/**
 * Downloads go through a Blob object-URL, never a data: URL — Chrome silently
 * ignores anchor downloads whose data: href exceeds ~2MB (the click just does
 * nothing, no error anywhere).
 */
function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/**
 * What one frame's file is called, without its extension.
 *
 * A fetched frame is named after its place in the grid and nothing else (see
 * capture.frameName) — that correspondence is the deliverable. A thumbnail
 * keeps the name it has always had.
 */
function baseName(i) {
    return isCapture() ? frameName(i, frames().length) : `thumbnail_frame${i + 1}`;
}

/**
 * One finished frame as a file, with everything it depends on waited for
 * first.
 *
 * Both waits are no-ops most of the time and neither is optional. A frame the
 * user never selected has never been restored, and a frame on a cutout channel
 * may have no subject cut out yet — in both cases what buildComposite would
 * draw is a picture the user has not been shown and would not accept.
 */
async function frameFile(i) {
    await ensureEnhanced(i);
    await ensureCutout(i);
    // Rebuilt from scratch rather than exporting the live preview canvas: that
    // one also carries the logo's dashed selection box and resize handle,
    // which must never end up baked into an exported file.
    const cv = await buildComposite(i);
    const { blob, ext } = await exportBlob(cv);
    return { name: `${baseName(i)}.${ext}`, blob };
}

export async function downloadFrame() {
    if (!frames().length) return status("No frame selected");
    const btn = el("downloadBtn");
    btn.disabled = true;
    status("Preparing download...");
    try {
        const { name, blob } = await frameFile(selectedIndex());
        triggerDownload(blob, name);
        status(`Downloaded ${name} (${(blob.size / 1048576).toFixed(2)}MB)`);
    } catch (err) {
        status("Download failed: " + err.message);
    } finally {
        btn.disabled = false;
    }
}

/**
 * Several frames at once, as one archive or as one file each.
 *
 * Which of the two is the run's, not the user's: a fetched set is a set — the
 * point of asking for forty frames is to receive forty frames, and forty
 * browser downloads is forty dialogs and a folder to sort out afterwards. A
 * handful of thumbnails is the opposite case, each one destined for a
 * different video, so they keep arriving as separate files exactly as before.
 *
 * Every frame is prepared before anything is written either way, because the
 * preparation is the slow part: a frame nobody has selected has never been
 * restored, and that is a full pass per frame. The status line counts them off
 * rather than sitting on one message for what can be minutes.
 */
async function downloadMany(indices, btn, label) {
    btn.disabled = true;
    try {
        const files = [];
        for (let n = 0; n < indices.length; n++) {
            status(`Preparing ${label} ${n + 1} of ${indices.length}...`);
            files.push(await frameFile(indices[n]));
        }

        if (isCapture()) {
            status("Packing the archive...");
            const archive = await zipBlob(files);
            triggerDownload(archive, `frames (${files.length}).zip`);
            status(`Downloaded ${files.length} frames as one zip `
                   + `(${(archive.size / 1048576).toFixed(1)}MB)`);
            return;
        }

        for (const file of files) {
            triggerDownload(file.blob, file.name);
            // Browsers throttle/collapse downloads fired back-to-back; a
            // short gap makes each one land as its own file.
            await new Promise(r => setTimeout(r, 350));
        }
        status(`Downloaded ${files.length} thumbnails`);
    } catch (err) {
        status("Batch download failed: " + err.message);
    } finally {
        btn.disabled = false;
    }
}

export function downloadSelected() {
    const picked = frames().map((_, i) => i).filter(i => frames()[i].picked);
    if (!picked.length) return status("Tick some frames first (checkbox on each thumbnail)");
    return downloadMany(picked, el("batchBtn"), "frame");
}

/**
 * Every frame in the grid, in grid order — which for a fetched set is the
 * order they occur in the video, so the archive unpacks reading the way the
 * grid read.
 *
 * The second of the two clicks the whole mode is built around: process, then
 * this. Nothing in between is required, because the edit has been applied to
 * every frame whether or not the user ever opened one.
 */
export function downloadAll() {
    if (!frames().length) return status("Nothing to download yet");
    return downloadMany(frames().map((_, i) => i), el("downloadAllBtn"), "frame");
}
