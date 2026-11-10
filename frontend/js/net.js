import { API } from "./config.js";

/**
 * POSTs JSON and returns the JPEG the backend replies with as an object URL,
 * plus whatever metadata rode along in the response headers.
 *
 * The editing endpoints return raw image bytes rather than base64 inside
 * JSON. Base64 costs a third more bytes plus an encode on the server and a
 * decode in the browser — per drag, zoom, flip and preset switch. The few
 * scalars that used to travel with the image (whether restoration ran, which
 * backend) come back as headers instead.
 *
 * Returns null on any non-OK response so callers can degrade quietly, which
 * is what every one of them already did.
 */
export async function postImage(path, body) {
    const res = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) return { ok: false, status: res.status, detail: await readDetail(res) };
    const blob = await res.blob();
    const meta = res.headers.get("x-frame-meta");
    return {
        ok: true,
        url: URL.createObjectURL(blob),
        enhanced: res.headers.get("enhanced") === "1",
        backend: res.headers.get("backend") || null,
        // Whatever else the endpoint had to say about what it produced. Only
        // /cutout-frame sends one, and what it carries is which moment of the
        // video this figure came from and how many others there are — see
        // editor.varyFigure, which walks a figure through them.
        meta: meta ? JSON.parse(meta) : null,
    };
}

/**
 * Releases a response whose image turned out not to be wanted after all.
 *
 * postImage hands back an object URL, and the browser keeps that blob alive
 * until it is explicitly revoked — dropping the reference is not enough. Every
 * caller has at least one path where the response arrives but is discarded
 * (the frame was re-cropped or its preset changed while the request was out),
 * and each of those used to strand a full 1280x720 JPEG for the lifetime of
 * the tab.
 */
export function discardImage(res) {
    if (res && res.url) URL.revokeObjectURL(res.url);
}

export async function getImage(path) {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) return null;
    const meta = res.headers.get("x-frame-meta");
    const blob = await res.blob();
    return {
        url: URL.createObjectURL(blob),
        meta: meta ? JSON.parse(meta) : {},
    };
}

export async function postJson(path, body) {
    const res = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        // The status rides along so a caller can tell an answer it knows how
        // to act on (a 409 from /process-video: a run is already going) from
        // a failure, without matching on the wording of the message.
        const error = new Error(await readDetail(res) || "Error");
        error.status = res.status;
        throw error;
    }
    return res.json();
}

/**
 * POSTs a Blob's raw bytes and reads the JSON the backend answers with —
 * postJson's answer, with a binary body. For /add-frame, whose answer is a
 * frame in the shape a run's frames arrive in rather than a bare image, so
 * neither postImageFile nor postFile (which waits for an upload_id) fits it.
 */
export async function postBlobJson(path, blob) {
    const res = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: blob,
    });
    if (!res.ok) {
        const error = new Error(await readDetail(res) || "Error");
        error.status = res.status;
        throw error;
    }
    return res.json();
}

export async function getJson(path) {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) return null;
    return res.json();
}

async function readDetail(res) {
    try {
        const data = await res.json();
        return data.detail || null;
    } catch (_) {
        return null;
    }
}

/** Decodes a URL (object or data) into a ready <img>. */
export function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("image decode failed"));
        img.src = src;
    });
}

/**
 * POSTs a file's raw bytes and reports how much of it has gone out.
 *
 * XMLHttpRequest rather than fetch, which is the whole reason this doesn't
 * just reuse postJson: fetch cannot observe the progress of a request BODY,
 * only of a response. A video takes long enough to send that a bar sitting at
 * zero for two minutes reads as a hung app, and the upload is the one stage
 * whose progress the backend cannot report — it hasn't received it yet.
 *
 * `signal`, an AbortSignal, calls the upload off part-way: the user starting
 * a different video over the one still crossing (see main.startFresh).
 */
export function postFile(path, file, onProgress, signal = null) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        if (signal) {
            if (signal.aborted) return reject(new Error("Upload cancelled"));
            signal.addEventListener("abort", () => xhr.abort(), { once: true });
        }
        // The name travels in the query string, not the body: the body is the
        // file itself, byte for byte (see /upload-video).
        xhr.open("POST", `${API}${path}?name=${encodeURIComponent(file.name)}`);
        xhr.setRequestHeader("Content-Type", "application/octet-stream");
        xhr.upload.addEventListener("progress", (e) => {
            if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total);
        });
        xhr.addEventListener("load", () => {
            let payload = null;
            try { payload = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
            if (xhr.status >= 200 && xhr.status < 300 && payload && payload.upload_id) {
                resolve(payload);
            } else {
                reject(new Error((payload && payload.detail) || `Upload failed (${xhr.status})`));
            }
        });
        xhr.addEventListener("error", () => reject(new Error("Upload failed — is the backend running?")));
        xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));
        xhr.send(file);
    });
}

/**
 * POSTs a file's raw bytes and reads the IMAGE the backend answers with —
 * postImage's shape, with a binary body instead of a JSON one.
 *
 * Kept apart from postFile above because the two differ in what comes back,
 * not in what goes out: that one reports upload progress and reads JSON, for
 * a video whose processing is then followed by polling. This one is an
 * ordinary editing round trip that happens to carry an image up as well as
 * down, so it belongs with postImage and returns exactly what postImage does.
 */
export async function postImageFile(path, file, params = {}) {
    const query = new URLSearchParams(params).toString();
    const res = await fetch(`${API}${path}${query ? "?" + query : ""}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
    });
    if (!res.ok) return { ok: false, status: res.status, detail: await readDetail(res) };
    return { ok: true, url: URL.createObjectURL(await res.blob()) };
}
