/**
 * Makes sure a channel's title face is the real thing and not a fallback.
 *
 * The overlay's whole layout is metric-driven: every size, every highlight
 * box and every line position comes from measuring the string in this face.
 * Impact — first in the fallback stack — is ~17% wider than Trade Gothic at
 * the same size, so when the face fails to load the result isn't "a different
 * font", it's a title composed against numbers that describe a font nobody
 * can see: oversized boxes, text sitting high inside them, lines overflowing
 * the canvas. That is exactly what a packaged build produces when the .otf is
 * missing from the install, and the failure is silent — the canvas draws
 * happily with Impact and nothing anywhere says the font never arrived.
 *
 * So it is checked, once per face, before anything is measured, and a failure
 * is surfaced instead of rendered.
 *
 * Which face that is comes from the selected channel (see channels.js), so
 * every channel's own face is verified in turn as it is picked — a second
 * brand on a second .otf gets the same guarantee as the first, not a check
 * that only ever looked at Trade Gothic.
 */

import { defaultFace } from "./channels.js";

// Used when no channel is selected yet — the face the app ships with, and
// what the page's own measurements (see main.js's title fit) wait on. Read
// through a function rather than imported as a value because the shipped face
// is now itself a content pack, and the pack is what the app should be
// checking once it has loaded (see channels.js).
const DEFAULT_FACE = () => defaultFace();

// Sampled rather than assumed: the check has to survive the face being
// unavailable, and glyph coverage differs enough between faces that a short
// sample can compare equal by accident.
const PROBE = "AVGWy0123 mmmiii";

const specFor = (face) => `${face.weight} 100px "${face.family}"`;

/**
 * True when a face by this name is actually available to the canvas.
 *
 * document.fonts.check() is not enough on its own: it answers for the whole
 * font list it is given, and a browser that can't find the family will still
 * report the request satisfiable via the generic at the end. So the family is
 * measured against a sentinel generic with deliberately unlike metrics — if
 * naming the family in front of monospace changes nothing about the width,
 * nothing by that name is being used and the text is monospace.
 */
function familyIsReal(face) {
    const ctx = document.createElement("canvas").getContext("2d");

    ctx.font = `${face.weight} 100px monospace`;
    const fallbackWidth = ctx.measureText(PROBE).width;

    ctx.font = `${face.weight} 100px "${face.family}", monospace`;
    const familyWidth = ctx.measureText(PROBE).width;

    return Math.abs(familyWidth - fallbackWidth) > 0.5;
}

// Keyed by family: each channel's face is checked once and the answer reused,
// however many times it is asked for afterwards.
const _promises = new Map();

/**
 * Resolves once `face` is loaded, or once it is known not to be.
 *
 * Resolves to { ok, source } — `source` is "webfont" when the @font-face in
 * styles.css served it, "system" when the font is installed on the machine
 * (the installer does this too, as a second line of defence), and null when
 * neither worked.
 *
 * Safe to call as often as convenient: the work happens once per face.
 */
export function titleFontReady(face = DEFAULT_FACE()) {
    const cached = _promises.get(face.family);
    if (cached) return cached;

    const promise = (async () => {
        if (!document.fonts || !document.fonts.load) {
            // No CSS Font Loading API: nothing can be awaited or verified, so
            // assume the CSS did its job rather than blocking the app.
            return { ok: true, source: "webfont" };
        }

        // The family is declared in styles.css but never applied to an
        // element — it exists purely for the canvas — so nothing has asked
        // the browser to fetch it yet. Without this explicit request,
        // document.fonts.ready resolves immediately on a page that has not
        // loaded the font at all, and every measurement taken afterwards is
        // an Impact measurement.
        let faces = [];
        try {
            faces = await document.fonts.load(specFor(face), PROBE);
        } catch {
            faces = [];
        }
        await document.fonts.ready.catch(() => {});

        if (faces.some(f => f.status === "loaded")) return { ok: true, source: "webfont" };

        // The @font-face didn't deliver — most likely the font file 404'd.
        // The installer also installs the bundled face into Windows, so the
        // canvas may still be able to reach it by name.
        //
        // A name is only evidence of the right font while nothing ELSE on the
        // machine answers to it. That holds for the faces this app ships —
        // but not in general: one family name can cover several very
        // differently proportioned cuts, and Windows hands out whichever it
        // has. A face that must come from its own file rather than from
        // anything sharing its name needs this fallback skipped.
        if (familyIsReal(face)) return { ok: true, source: "system" };

        return { ok: false, source: null };
    })();

    _promises.set(face.family, promise);
    return promise;
}

/**
 * Shows a permanent, unmissable banner when a face is missing.
 *
 * Deliberately not a console warning: the symptom (a title that looks wrong)
 * is one a user reads as "the app is broken", and without this they have no
 * way to connect it to a missing file. The file is named because that is the
 * one actionable detail — with several channels on several faces, "the title
 * font" would no longer say which one to go looking for.
 */
export function reportMissingTitleFont(face = DEFAULT_FACE()) {
    const banner = document.getElementById("fontWarning");
    if (!banner) return;
    // Two different failures, and the fix for each is a different sentence.
    // A face with a `file` is one this app ships, and a missing one means a
    // broken install. A face without one is reached by name from the machine
    // itself — a pack naming a face it expects every machine to already have,
    // like Arial — and no reinstall of this app would put it there.
    banner.textContent = face.file
        ? `Title font not loaded — ${face.file} could not be reached, so titles would be laid `
          + "out with a fallback face. Reinstall Thumbnail Maker to restore it."
        : `Title font not found — this channel is set in ${face.family}, which is not installed `
          + "on this machine, so titles would be laid out with a fallback face.";
    banner.classList.remove("hidden");
}
