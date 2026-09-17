import { CW, CH, HANDLE_SIZE } from "./config.js";
import { titleStyle, hasChannel } from "./channels.js";
import { titleFontReady } from "./fontguard.js";

/**
 * The Fabric text properties of whichever channel is selected — the face, its
 * weight and its line height. Read fresh on every call rather than captured
 * once at import: the channel is picked (and can be changed) long after this
 * module loads, and a captured copy would keep setting every title in the
 * first channel's font.
 */
/** "#RRGGBB" + alpha -> the rgba() a canvas wants. Passes anything else through. */
function withOpacity(color, opacity) {
    if (opacity === undefined || opacity === null || opacity >= 1) return color;
    const m = /^#([0-9a-f]{6})$/i.exec(color || "");
    if (!m) return color;
    const n = parseInt(m[1], 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${opacity})`;
}

/**
 * How far a channel's outline reaches OUTSIDE the letterform — 0 when it has
 * none. Half the stroke, since a canvas stroke straddles the path, which is
 * also why the stroke below is twice the configured width.
 */
const outlineOutset = (style = titleStyle()) => (style.outline ? style.outline.width : 0);

/**
 * A stroke around the letterform, `width` of it visible, painted with `paint`.
 *
 * The stroke laid down is twice the width asked for, because a canvas stroke
 * straddles the path: half of it falls inside the letterform and is painted
 * over by the fill, so only the outer half is ever seen.
 *
 * `paintFirst: "stroke"` is what puts it outside the letter instead of
 * through it: the stroke goes down first and the fill covers its inner half.
 * Round joins because at these widths mitred corners throw spikes several
 * times the stroke's own thickness.
 *
 * Shared by the three things that stroke a line — the outline, the sticker
 * under it, and the glow copy that has to match whichever of them is
 * outermost — so all three agree about what a "width" means.
 */
const strokeProps = (width, paint) => ({
    stroke: paint,
    strokeWidth: width * 2,
    paintFirst: "stroke",
    strokeLineJoin: "round",
    strokeLineCap: "round",
});

/**
 * How far into its own box a stroked copy's letters sit.
 *
 * Fabric positions an object by its bounding box, and the stroke is part of
 * that box: the glyphs land half the stroke width down and to the right of
 * `left`/`top`. Measured, at strokeWidth 60, at exactly 30px on both axes.
 *
 * Which means two copies of one line handed the same left/top but different
 * stroke widths do not sit on top of each other — a 25px sticker under a 3px
 * outlined line would be 22px out in each direction. Every extra copy of a
 * row is therefore placed by where its LETTERS will land (see alignedCopy),
 * never by where its box does.
 */
const glyphInset = (props) => (props.strokeWidth || 0) / 2;

/**
 * The properties every text object is built with — the measuring ones in this
 * file as much as the ones actually drawn.
 *
 * The outline lives in here, and that placement is the whole reason the layout
 * survives it. An outline is ink: 25px of it makes a line 50px wider and 50px
 * taller than the letters alone. Nothing measures stroke separately, so if
 * only the DRAWN text carried it, every width fit, every line gap and every
 * block bound would go on describing letters 50px smaller than the ones on
 * screen. One shared property bag means measureVisualBounds rasterises exactly
 * what addTextOverlay will paint, and the numbers cannot drift from the
 * picture.
 *
 * The SECOND outline — the sticker — is deliberately not in here: it is drawn
 * and never measured, for the reason `sticker` in channels.js gives.
 */
const textProps = (style = titleStyle()) => ({
    fontWeight: style.face.weight,
    fontFamily: style.face.stack,
    lineHeight: style.lineHeight,
    ...(style.outline
        ? strokeProps(style.outline.width, withOpacity(style.outline.color, style.outline.opacity))
        : {}),
});

/**
 * The same properties with the outline made INVISIBLE rather than removed —
 * what the glow copies are drawn with.
 *
 * A halo is cast by the letters, not by the black keyline around them, so the
 * obvious move is to drop the stroke properties entirely. That is wrong, and
 * visibly so: fabric shifts a stroked text's glyphs by half the stroke width,
 * so an unstroked copy at the same left/top lands somewhere else. Measured at
 * strokeWidth 12, the copy came out 5px up and 5px left of the real letters,
 * which put its white fill outside the black outline along the top-left of
 * every glyph — reading, entirely reasonably, as an outline that doesn't line
 * up with its letters.
 *
 * Keeping the stroke and painting it in nothing keeps both copies on fabric's
 * one code path, so whatever offset it applies, it applies to both: measured
 * again this way, zero. A fully transparent stroke lays down no pixels, so it
 * contributes nothing to the shadow either — the halo stays the shape of the
 * letters alone, which is the whole point of it.
 */
const glowProps = (style = titleStyle()) => {
    const props = textProps(style);
    return props.stroke ? { ...props, stroke: "rgba(0,0,0,0)" } : props;
};

/**
 * Identifies the face a cached measurement was taken with.
 *
 * Every cache below is keyed by the string and the size, which described the
 * answer completely while there was only ever one font. With a font per
 * channel they no longer do: switching channels would otherwise lay the new
 * brand's titles out against the previous brand's metrics — the same class of
 * failure the fallback-font note below describes, only permanent.
 */
const styleKey = (s = titleStyle()) => (
    `${s.face.family}/${s.face.weight}/${s.lineHeight}/${outlineOutset(s)}`
);

// Both caches below are keyed by exactly what determines their answer, and
// both are hit constantly: the title overlay is rebuilt on every keystroke,
// for the preview, for the drag snapshot, and for each of the twenty grid
// thumbnails. Without them the same handful of strings were re-measured
// hundreds of times per edit.
const _sizeCache = new Map();
const _inkSizeCache = new Map();
const _boundsCache = new Map();
const _wordCache = new Map();
const CACHE_LIMIT = 600;

// A measurement is only meaningful for the font it was taken with, and the
// title face is self-hosted: until it finishes loading, the canvas measures
// the Impact fallback, which is ~17% wider than Trade Gothic at the same size
// (measured: 743.3px vs 634.5px for the same string at 100px).
//
// Caching those numbers is what made them outlive the fallback. Everything
// drawn after the font arrives uses the real face while the highlight box is
// still built from the cached fallback metrics — so the box comes out sized
// and positioned for a font that is no longer on screen: noticeably wider
// than the letters, with the text sitting high inside it, and it stays that
// way for as long as the page lives because nothing ever re-measures.
//
// Nothing is remembered until the font is in, and anything already stored is
// dropped the moment it lands.
//
// The wait goes through fontguard rather than document.fonts.ready directly:
// the family is only ever named on the canvas, never on an element, so
// nothing has asked the browser to fetch it when the page settles and
// fonts.ready resolves on a page whose title face has not loaded at all.
// Gating on that promise is what let fallback metrics get cached as final.
//
// Tracked per face, not once for the app: each channel brings its own, and a
// channel picked later must clear whatever was measured while its font was
// still on the way.
const _readyFamilies = new Set();
const _loadingFamilies = new Map();

function clearMeasurements() {
    _sizeCache.clear();
    _inkSizeCache.clear();
    _boundsCache.clear();
    _wordCache.clear();
}

/**
 * Every face a channel draws its titles in: the block's, and the one its
 * highlight changes voice to where it has one (see `highlight.face` in
 * channels.js).
 *
 * Deduplicated by family, because a pack is free to name the same face twice
 * and neither the loader nor the caches should care that it did.
 */
export function titleFaces(style = titleStyle()) {
    const hl = style.highlight && style.highlight.face;
    if (!hl || hl.family === style.face.family) return [style.face];
    return [style.face, hl];
}

/** Loads and verifies one face, once, and drops anything measured before it landed. */
function prepareFace(face) {
    if (_readyFamilies.has(face.family)) return Promise.resolve({ ok: true, source: "cached" });

    let loading = _loadingFamilies.get(face.family);
    if (!loading) {
        loading = titleFontReady(face).then((res) => {
            // Ready either way: a face that is known to be missing is not
            // going to arrive later, and blocking every measurement forever
            // would take the whole overlay down instead of just its font.
            _readyFamilies.add(face.family);
            clearMeasurements();
            return res;
        });
        _loadingFamilies.set(face.family, loading);
    }
    return loading;
}

/**
 * Loads and verifies every face a channel is set in, and drops any
 * measurement taken before they landed.
 *
 * Resolves to fontguard's { ok, source } for the whole channel, with `face`
 * naming the one that failed — a channel is only as loaded as its worst face,
 * and a caller that put up a banner reading "the title font" would send the
 * user looking for the wrong file on a channel set in two.
 */
export function prepareTitleFont(style = titleStyle()) {
    const faces = titleFaces(style);
    return Promise.all(faces.map(prepareFace)).then((results) => {
        const bad = results.findIndex(r => !r.ok);
        return bad === -1
            ? { ...results[0], face: faces[0] }
            : { ...results[bad], face: faces[bad] };
    });
}

// Per FACE, not per channel: a channel that sets its highlight in a second
// face has two of them in flight, and a measurement taken in one while the
// other was still on the way is as stale as any other fallback measurement.
const fontReady = (style = titleStyle()) => _readyFamilies.has(style.face.family);

if (typeof document !== "undefined") {
    // The bundled face, before any channel has been picked — so the very
    // first render after a channel is selected isn't waiting on a download
    // that could have started at load time.
    prepareTitleFont();
} else {
    _readyFamilies.add(titleStyle().face.family);
}

function remember(cache, key, value, style) {
    if (!fontReady(style)) return value;
    if (cache.size >= CACHE_LIMIT) cache.clear();
    cache.set(key, value);
    return value;
}

function textWidth(str, fontSize, style = titleStyle()) {
    return new fabric.Text(str, { fontSize, ...textProps(style) }).width;
}

/**
 * Width of a string's actual ink, straight from the font's own metrics.
 *
 * measureVisualBounds answers the same question by rasterising and counting
 * pixels, which costs ~30ms at headline sizes — fine for the two or three
 * calls a layout makes, ruinous for a fit that has to try several sizes on
 * every keystroke. actualBoundingBox needs no raster at all (~0.01ms) and
 * agrees with the scan to within a pixel.
 *
 * Only the horizontal extent is taken from here. The vertical offsets still
 * go through Fabric's own draw path for the reason measureVisualBounds
 * documents: Fabric's baseline placement is its own, and a raw metric would
 * disagree with where the glyphs actually land.
 */
const _inkCtx = document.createElement("canvas").getContext("2d");

function inkWidth(str, fontSize, style = titleStyle()) {
    const props = textProps(style);
    _inkCtx.font = `${props.fontWeight} ${fontSize}px ${props.fontFamily}`;
    const m = _inkCtx.measureText(str);
    return m.actualBoundingBoxLeft + m.actualBoundingBoxRight;
}

/**
 * Where each word of `str` begins and ends, in pixels from the line's own INK
 * left edge — what a per-word highlight box is built from.
 *
 * Measured through the font's metrics rather than by rasterising, for the
 * reason inkWidth gives: this is asked for every word of every line on every
 * keystroke, and a raster scan per word would cost more than the whole layout
 * does. The two rulers disagree by a pixel or two at headline sizes, which is
 * a fifth of what the box's own padding adds on each side.
 *
 * Everything is stated relative to the ink edge, and the offsets are taken
 * from the WHOLE string rather than by measuring each word where it stands:
 * the line is drawn as one piece of text and must stay that way — a word
 * measured on its own and drawn on its own would lose whatever the renderer
 * does between it and its neighbours. So the pen advances through the real
 * string, and only the ink bounds come from the word.
 */
function wordSpans(str, fontSize, style = titleStyle()) {
    const key = `${styleKey(style)}|${fontSize}|${str}`;
    const hit = _wordCache.get(key);
    if (hit !== undefined) return hit;

    const props = textProps(style);
    _inkCtx.font = `${props.fontWeight} ${fontSize}px ${props.fontFamily}`;
    // The pen sits this far right of the line's ink edge, so adding it to a
    // pen-relative position gives an ink-relative one.
    const inkOffset = _inkCtx.measureText(str).actualBoundingBoxLeft;

    const spans = [];
    for (const match of str.matchAll(/\S+/g)) {
        const pen = _inkCtx.measureText(str.slice(0, match.index)).width;
        const m = _inkCtx.measureText(match[0]);
        spans.push({
            word: match[0],
            // Where the word begins in the string, and how far the pen has
            // travelled by then. Both are what it takes to draw a piece of
            // the line on its own and have it land where it does in the whole
            // (see the run copies in addTextOverlay).
            index: match.index,
            pen,
            start: inkOffset + pen - m.actualBoundingBoxLeft,
            end: inkOffset + pen + m.actualBoundingBoxRight,
        });
    }
    return remember(_wordCache, key, spans, style);
}

/**
 * A line cut into the pieces a per-word highlight makes of it: alternating
 * runs of picked and unpicked text, together covering the string exactly.
 *
 * `picked(k)` answers for the k-th word, counted the way `wordsOf` counts
 * them, so a run can never cover a different word from the one the button
 * said (see the note on `wordsOf`).
 *
 * Neighbouring picked words are ONE run with the space between them inside
 * it, which is the same rule `pickedSpans` joins boxes by and it is the same
 * reason: a highlighter run over two words is one mark, not two with a seam.
 * The space between a picked word and an unpicked one goes to the unpicked
 * side — a highlight begins and ends at ink, so a run that carried a trailing
 * space would be a run set a space too wide.
 *
 * Whitespace-only pieces are returned as pieces. They have no ink and nothing
 * measures them for bounds, but they carry the advance that puts the piece
 * after them in the right place, so they cannot be dropped.
 */
function treatmentRuns(str, picked) {
    const runs = [];
    let previous = -2;   // never adjacent to word 0
    [...str.matchAll(/\S+/g)].forEach((m, k) => {
        if (!picked(k)) return;
        const end = m.index + m[0].length;
        if (k === previous + 1) runs[runs.length - 1].end = end;
        else runs.push({ start: m.index, end });
        previous = k;
    });

    const pieces = [];
    let at = 0;
    for (const run of runs) {
        if (run.start > at) pieces.push({ text: str.slice(at, run.start), hl: false });
        pieces.push({ text: str.slice(run.start, run.end), hl: true });
        at = run.end;
    }
    if (at < str.length) pieces.push({ text: str.slice(at), hl: false });
    return pieces;
}

/**
 * Those pieces with the pen position each one starts at, and how wide the
 * whole line's ink comes out — a line laid out as runs rather than as a
 * string.
 *
 * The pen accumulates each piece's own advance in its OWN face, which is the
 * whole point: this exists for lines that are set in two faces, where no
 * single measurement of the string describes where anything lands. A piece
 * handed `left = textLeft + pen` therefore draws exactly where it would have
 * drawn had the line been one object, up to the kerning across a join — and
 * every join is a space.
 *
 * Measured through the font's metrics rather than by rasterising, for the
 * reason `wordSpans` gives: this runs for every row of every layout pass on
 * every keystroke. `inkW` is the cheap ruler's answer and is what the size
 * fit uses; the bounds the block is actually stacked and positioned by are
 * rasterised once the size has settled (see `segBounds`).
 */
function penRuns(pieces, fontSize) {
    let pen = 0, left = Infinity, right = -Infinity;
    const segs = pieces.map((piece) => {
        const props = textProps(piece.style);
        _inkCtx.font = `${props.fontWeight} ${fontSize}px ${props.fontFamily}`;
        const m = _inkCtx.measureText(piece.text);
        const seg = { ...piece, props, pen };
        // A piece with no ink in it has no bounds to contribute — measureText
        // reports a zero-width box at the pen, which as a "left edge" would
        // drag the line's ink out to wherever that space happened to fall.
        if (/\S/.test(piece.text)) {
            left = Math.min(left, pen - m.actualBoundingBoxLeft);
            right = Math.max(right, pen + m.actualBoundingBoxRight);
        }
        pen += m.width;
        return seg;
    });
    return { segs, inkW: right > left ? right - left : 0, advance: pen };
}


/**
 * How wide the whitespace a line BEGINS and ENDS with is, in advance pixels.
 *
 * Ink is what a line is normally measured by, and a space has none — which is
 * right for a block of type sitting on a photograph, where two trailing
 * spaces are a typing accident and should change nothing. It is wrong for a
 * line that carries a box behind it: there, a space the user typed is the one
 * way they have of saying "leave room here", and a box that ignored it would
 * make the request unanswerable (see `panel.scope` in channels.js, and the
 * emoji it exists for).
 *
 * So the two are measured apart from the ink and added back only where a box
 * is drawn. Everything else in the layout goes on reading the ink bounds and
 * cannot tell that a line was padded.
 */
function edgeSpace(str, fontSize, style = titleStyle()) {
    const props = textProps(style);
    _inkCtx.font = `${props.fontWeight} ${fontSize}px ${props.fontFamily}`;
    const lead = /^\s+/.exec(str);
    const tail = /\s+$/.exec(str);
    return {
        lead: lead ? _inkCtx.measureText(lead[0]).width : 0,
        tail: tail ? _inkCtx.measureText(tail[0]).width : 0,
    };
}

/**
 * The words of one line, as the panel and the renderer both count them.
 *
 * One function so they cannot disagree. A highlight is stored as "which word
 * of which line" (see state.hlFor), and if the buttons split the line one way
 * and the renderer another, clicking a word would put a box round a different
 * one — a class of bug that is invisible until a title has punctuation in it.
 */
/**
 * How wide `str` is in the selected channel's face, at a fixed reference
 * size — a number that means nothing on its own and everything next to
 * another one from the same call.
 *
 * For code that has to COMPARE candidate strings without laying anything out:
 * caption.js breaks one sentence over up to four lines and needs to know
 * which of several possible breaks come out evenly. Advance width rather than
 * ink, because that is what accumulates along a line, and cheap for the same
 * reason — it is fabric's own measurement of the string, cached, with nothing
 * rasterised.
 */
export const measuredWidth = (str) => textWidth(str, FIT_REFERENCE_SIZE);

export const wordsOf = (line) => (line || "").match(/\S+/g) || [];

/** The key a highlighted word is stored under. */
export const wordKey = (lineIdx, wordIdx) => `${lineIdx}:${wordIdx}`;

// ── Quotation marks on a line of their own ────────────────────────────────
//
// A pull quote is often set with its marks pushed out of the sentence: the
// opening one alone above the first line, the closing one alone below the
// last. Left to the block's own rule those rows are CENTRED like any other
// short line, and a centred quotation mark floating over the middle of a
// title is the one thing it must not look like — it reads as a stray
// character rather than as punctuation belonging to the words.
//
// So a row that is nothing but quotation marks is aligned to the line it
// belongs to instead: an opener to the left edge of the line under it, a
// closer to the right edge of the line over it.

// Written as code points rather than as the characters themselves. These are
// data and not prose — the check that keeps this codebase in English reads a
// row of curly marks in the source as exactly what it is built to stop (see
// installer/check_language.py) — and three of them are the same shape at
// different heights, which no reader can tell apart at a glance anyway.
//
//   2018/2019  single turned and right single quotation mark
//   201A/201B  single low-9 and single high-reversed-9
//   201C/201D  left and right double quotation mark
//   201E/201F  double low-9 and double high-reversed-9
//   00AB/00BB  the French guillemets, and 2039/203A their single form
//   2033/2036  double prime and reversed double prime
const QUOTES_ONLY = /^[\s"'\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F\u00AB\u00BB\u2039\u203A\u2033\u2036]+$/;
// Marks that say for themselves which end they are. A straight " does not,
// which is why position has the last word below.
const OPENING = /[\u201C\u201E\u201F\u00AB\u2039\u2018\u201A\u2036]/;
const CLOSING = /[\u201D\u2019\u00BB\u203A\u2033]/;

const isQuoteRow = (line) => !!line && QUOTES_ONLY.test(line);
const isTextRow = (line) => !!line.trim() && !isQuoteRow(line);

/**
 * Which end of the quotation each line is, or null for the lines that are
 * the quotation.
 *
 * A typographic mark answers for itself. A straight one is the same character
 * at both ends, so it is read from where it sits: a mark with words after it
 * opens, and one with only words before it closes. That is the whole of the
 * ambiguity — nobody sets a quotation mark alone on a line in the middle of a
 * sentence.
 */
function quoteRoles(lines) {
    return lines.map((line, i) => {
        if (!isQuoteRow(line)) return null;
        if (OPENING.test(line) && !CLOSING.test(line)) return "open";
        if (CLOSING.test(line) && !OPENING.test(line)) return "close";
        return lines.slice(i + 1).some(isTextRow) ? "open" : "close";
    });
}

// Reference size the linear fit below measures at. Any size works; larger is
// slightly more precise against rounding.
const FIT_REFERENCE_SIZE = 100;

/**
 * The largest font size at which `str` still renders narrower than `targetW`.
 *
 * Text width is linear in font size for a fixed string, weight and family —
 * the same property the page title's own fit already relies on — so one
 * measurement gives the answer directly. This used to binary-search the
 * range 20..800, constructing and measuring a fresh fabric.Text on each of
 * the ~10 iterations, for every line, on every re-render.
 */
export function fontSizeForWidth(str, targetW, style = titleStyle()) {
    const key = `${styleKey(style)}|${targetW}|${str}`;
    const hit = _sizeCache.get(key);
    if (hit !== undefined) return hit;

    const refWidth = textWidth(str, FIT_REFERENCE_SIZE, style);
    if (!refWidth) return remember(_sizeCache, key, FIT_REFERENCE_SIZE, style);

    let size = Math.max(20, Math.min(800, Math.floor(FIT_REFERENCE_SIZE * targetW / refWidth)));
    // The linear model is exact up to the renderer's own rounding, so at most
    // a step either way is ever needed to land on the true largest fit.
    while (size > 20 && textWidth(str, size, style) >= targetW) size--;
    while (size < 800 && textWidth(str, size + 1, style) < targetW) size++;
    return remember(_sizeCache, key, size, style);
}

/**
 * Visual edges (top/bottom/left/right) of a string's actual glyph pixels,
 * relative to the position Fabric would draw that string at.
 *
 * Rendered through Fabric itself (a real fabric.Text on an offscreen
 * StaticCanvas at top:0), not a bare ctx.fillText — Fabric computes its own
 * baseline offset from a generic font-size fraction rather than the font's
 * real ascent/descent, so a raw canvas measurement and Fabric's actual draw
 * position silently disagree once the font changes (that's what broke
 * centering when the title font was swapped). Measuring through Fabric's own
 * pipeline captures whatever offset Fabric applies empirically, so the
 * highlight box stays centered on the glyphs no matter which font is loaded.
 *
 * The horizontal edges matter for the same reason the vertical ones do: a
 * line's advance width is not its ink. Placing every line at the same `left`
 * lines up the invisible advance boxes, not the letters — the first glyph's
 * side bearing scales with font size, so the block's left edge drifted by up
 * to ~22px between lines and the highlight box's inner padding came out
 * different on every line. Positioning by ink instead makes the edges agree.
 *
 * The scan reads the alpha channel via getImageData and walks in from each
 * side of a row rather than looping every pixel in JS: the old nested loop
 * walked ~175,000 pixels per line per layout attempt, and a full grid refresh
 * ran it sixty-odd times.
 */
// Retina scaling OFF, and this is load-bearing rather than an optimisation.
//
// With it on (Fabric's default whenever devicePixelRatio > 1 — every scaled
// Windows display), setDimensions sizes the backing store to w*DPR × h*DPR
// and scales the context by DPR, so the glyphs are rasterised DPR times
// larger than the numbers handed in. getImageData ignores that transform and
// works in device pixels, so every edge this function scanned came back
// multiplied by DPR while the layout below places text in canvas units — the
// error is a *fraction of the font size*, invisible on small titles and huge
// on large ones. That's what made the highlight box drift: its top and bottom
// come from the cap band, so at 1.25x a 300px line put the box's top edge
// right on the capitals and left the whole (DPR-1)x of cap height as dead
// space under the baseline, with the text riding high in the box.
//
// Measuring at 1:1 keeps the scan in the same coordinate space the text is
// positioned in, at any display scale.
const _measureEl = document.createElement("canvas");
const _measureCanvas = new fabric.StaticCanvas(_measureEl, { enableRetinaScaling: false });

// Slack on both sides of the advance box so a negative side bearing (or a
// glyph overshooting its advance) is measured rather than clipped off. Side
// bearings scale with the font size, so a flat margin that covered a 100px
// line silently cropped a 700px one — and now that a short line gets set as
// large as it takes to fill the width, those sizes are ordinary.
const measureMargin = (fontSize, style) => 20 + Math.ceil(fontSize * 0.1) + outlineOutset(style);

// The same slack, above the draw position. Fabric puts the text box's top at
// `top`, but ink can sit above it: the accent on a capital — Á, Ã, É, ordinary
// in a Portuguese title — rises past the top of the box and used to be sliced
// off by the canvas edge, so every accented line measured as beginning at
// exactly row 0 and got stacked as though the accent weren't there.
const measureSlack = (fontSize, style) => Math.ceil(fontSize * 0.25) + outlineOutset(style);

export function measureVisualBounds(str, fontSize, style = titleStyle()) {
    const key = `${styleKey(style)}|${fontSize}|${str}`;
    const hit = _boundsCache.get(key);
    if (hit !== undefined) return hit;

    const MEASURE_MARGIN = measureMargin(fontSize, style);
    const SLACK = measureSlack(fontSize, style);
    const w = Math.ceil(textWidth(str, fontSize, style)) + MEASURE_MARGIN * 2;
    // Ink reaches ~1.06x the font size below the draw position at the deepest
    // descender; the rest is room for anything unusual. This used to be 1.8x
    // — pure empty canvas, and every pixel of it was read back on each call.
    const h = SLACK + Math.ceil(fontSize * 1.25) + 20 + outlineOutset(style);
    _measureEl.width = w;
    _measureEl.height = h;
    _measureCanvas.setDimensions({ width: w, height: h });
    _measureCanvas.clear();
    _measureCanvas.add(new fabric.Text(str, {
        left: MEASURE_MARGIN, top: SLACK, fontSize, ...textProps(style), selectable: false, evented: false,
    }));
    _measureCanvas.renderAll();

    const data = _measureEl.getContext("2d", { willReadFrequently: true })
        .getImageData(0, 0, w, h).data;

    let top = h, bottom = 0, left = w, right = -1;
    for (let row = 0; row < h; row++) {
        const base = row * w * 4;
        let x = 0;
        while (x < w && data[base + x * 4 + 3] === 0) x++;
        if (x === w) continue;   // no ink on this row
        let xr = w - 1;
        while (data[base + xr * 4 + 3] === 0) xr--;
        if (row < top) top = row;
        bottom = row;
        if (x < left) left = x;
        if (xr > right) right = xr;
    }
    if (right < left) { top = SLACK; bottom = SLACK; left = MEASURE_MARGIN; right = MEASURE_MARGIN; }

    return remember(_boundsCache, key, {
        // Back into coordinates relative to the text's own draw position.
        top: top - SLACK, bottom: bottom - SLACK, visualH: bottom - top + 1,
        left: left - MEASURE_MARGIN, right: right - MEASURE_MARGIN,
        visualW: right - left + 1,
    }, style);
}

/**
 * The largest font size at which `str`'s INK is still narrower than `targetW`.
 *
 * fontSizeForWidth fits the advance width, which is not what the eye reads. A
 * line's side bearings scale with its font size, so two lines fitted to the
 * same advance come out visibly different widths — the shorter string, set
 * larger to reach the target, carries proportionally more air on each end
 * ("SSS" set big has far wider bearings than "IS A TEST" set small). Fitting
 * the ink is what actually makes every line measure the same on screen.
 *
 * Ink is linear in font size for the same reasons the advance is, so one
 * measurement gives the answer directly — same one-shot fit as
 * fontSizeForWidth, just measuring the letters instead of their boxes.
 */
export function fontSizeForInkWidth(str, targetW, style = titleStyle()) {
    const key = `${styleKey(style)}|${targetW}|${str}`;
    const hit = _inkSizeCache.get(key);
    if (hit !== undefined) return hit;

    const refInk = inkWidth(str, FIT_REFERENCE_SIZE, style);
    // A string with no ink at all would send the scale factor to infinity.
    if (!refInk) return remember(_inkSizeCache, key, fontSizeForWidth(str, targetW, style), style);

    const clamp = (sz) => Math.max(20, Math.min(800, sz));
    let size = clamp(FIT_REFERENCE_SIZE * targetW / refInk);
    // Ink is linear in size only up to hinting, so the one-shot guess lands
    // slightly out; scaling by how far off it came back converges in a pass
    // or two.
    //
    // The result is deliberately NOT rounded to a whole number. This used to
    // take the largest INTEGER size that still fit, and at headline sizes one
    // integer step is 4-6px of ink — so each line settled 2-5px short of the
    // target, by a different amount per line, depending on where its own step
    // happened to fall. That is precisely the ragged edge this fit exists to
    // remove: measured on a real title, three lines aimed at 640px came out
    // 638/635/635. Fractional sizes land all three within a pixel, and
    // nothing downstream needs an integer — the box padding rounds itself and
    // every bound is measured, not assumed.
    for (let i = 0; i < 4; i++) {
        const w = inkWidth(str, size, style);
        if (!w) break;
        const next = clamp(size * targetW / w);
        const settled = Math.abs(next - size) < 0.01;
        size = next;
        if (settled) break;
    }

    // Corrections against the measure that actually decides where the glyphs
    // land. inkWidth is ctx.measureText's ink box, and it is what makes the
    // loop above cheap enough to run per keystroke — but it drifts from the
    // RASTERISED bounds by a few pixels at headline sizes, and those bounds
    // are what positions every line and sizes every highlight box. A line
    // fitted only on measureText is therefore flush by one ruler and not by
    // the one on screen: measured on a three-line title, the short line
    // ("BOSSES", set at 264px to reach the width) came out 4px narrower than
    // its neighbours.
    //
    // A loop rather than a single step because of the outline. Letters scale
    // with the font size but an outline does not — it is a fixed 25px however
    // large the type — so the rendered width is (letters x size) + a constant,
    // and scaling by how far off it came back overshoots by however much of
    // the width that constant accounts for. Each pass shrinks the error by
    // roughly that fraction, so two or three land it. Channels with no
    // outline settle on the first pass and pay for one extra measurement to
    // find that out. Affordable either way: _inkSizeCache serves every later
    // layout of the string from here.
    for (let i = 0; i < 4; i++) {
        const vb = measureVisualBounds(str, size, style);
        if (!vb.visualW) break;
        if (Math.abs(vb.visualW - targetW) <= 1) break;
        size = clamp(size * targetW / vb.visualW);
    }

    return remember(_inkSizeCache, key, size, style);
}

// ── Highlight textures ────────────────────────────────────────────────────
//
// A channel can fill its highlighted letters with an image instead of a
// colour (highlight.mode "texture"). Two things have to be true for that to
// work inside a layout that is rebuilt on every keystroke: the image must
// already be decoded when addTextOverlay runs, since that function is
// synchronous and cannot wait; and the scaling work must not be redone per
// render. Hence a load-once cache, and a scaled-once cache on top of it.

const _textureLoads = new Map();    // url -> Promise<{ ok, img }>
const _textureImages = new Map();   // url -> HTMLImageElement, once decoded
const _textureFitted = new Map();   // `url|WxH` -> canvas of the image at that size

/**
 * Loads a channel's highlight texture, resolving to { ok, img }.
 *
 * Called the way prepareTitleFont is — once, when a channel is picked, before
 * anything is drawn with it (see editor.selectChannel). Resolves rather than
 * rejects on failure, because a texture that never arrives should cost the
 * channel its foil and nothing else: the layout, the panel and the type are
 * all unaffected, and the line falls back to highlight.color.
 */
export function prepareTitleTexture(style = titleStyle()) {
    const url = style.highlight && style.highlight.texture;
    if (!url) return Promise.resolve({ ok: true, img: null });

    const cached = _textureLoads.get(url);
    if (cached) return cached;

    const promise = new Promise((resolve) => {
        const img = new Image();
        img.onload = () => { _textureImages.set(url, img); resolve({ ok: true, img }); };
        img.onerror = () => resolve({ ok: false, img: null });
        // Used as it arrives. Encoding spaces and the like used to happen
        // here, and now happens once in channels.assetUrl, which is the one
        // place that turns a path written in a pack into a URL. Doing it in
        // both encoded the percent signs of the first pass — a foil called
        // "best of supermission texture.png" was requested as
        // "best%2520of%2520..." and 404'd, which costs the channel its foil
        // silently, because a texture that never arrives is a highlight drawn
        // in the flat stand-in colour instead.
        img.src = url;
    });
    _textureLoads.set(url, promise);
    return promise;
}

/**
 * The texture drawn at exactly w x h, cropped to cover rather than squashed
 * to fit — the aspect ratio of a foil is part of what makes it read as foil,
 * and a title block is nothing like the shape of the source image.
 *
 * Cached by size: while the user types, the block's box changes only when the
 * line sizes do, so this is a hit on almost every keystroke.
 */
function fittedTexture(url, w, h) {
    const key = `${url}|${w}x${h}`;
    const hit = _textureFitted.get(key);
    if (hit) return hit;

    const img = _textureImages.get(url);
    if (!img) return null;

    const el = document.createElement("canvas");
    el.width = Math.max(1, Math.ceil(w));
    el.height = Math.max(1, Math.ceil(h));
    const scale = Math.max(el.width / img.width, el.height / img.height);
    const dw = img.width * scale, dh = img.height * scale;
    el.getContext("2d").drawImage(img, (el.width - dw) / 2, (el.height - dh) / 2, dw, dh);

    if (_textureFitted.size >= CACHE_LIMIT) _textureFitted.clear();
    _textureFitted.set(key, el);
    return el;
}

// ── Filling the gaps ──────────────────────────────────────────────────────
//
// A sticker is drawn the way it looks best: one stroked copy of each line,
// rasterised by fabric, so every edge of it is fabric's own antialiasing.
// What that leaves behind is holes. Wherever two words land just too far
// apart for their strokes to touch, a sliver of the photograph runs straight
// through the middle of the block — measured on one real title, 5px wide and
// 33px tall, between AT and THE.
//
// This used to be solved by reshaping the silhouette: grow it by a radius and
// shrink it back, which merges anything narrower than that radius. It worked,
// and it cost the edge. Morphology done with offset copies and alpha
// compositing cannot preserve an antialiased boundary — the offsets land on
// whole pixels and the intersections multiply partial alphas away — so the
// outline came back as a staircase. Supersampling the whole slab and
// resolving it down improved that and did not fix it: three samples per pixel
// is not what fabric does, and it showed.
//
// So the shape is left exactly as fabric drew it, and only the holes are
// dealt with. A hole is transparent canvas the slab completely surrounds, so
// it is found the way flood fill finds anything: paint in from every border,
// and whatever the paint could not reach is enclosed. Those pixels, and only
// those, are filled and laid UNDER the strokes — the one edge the patch can
// contribute is one that sits beneath opaque ink, where nothing can see it.

const _maskEl = document.createElement("canvas");
const _maskCanvas = new fabric.StaticCanvas(_maskEl, { enableRetinaScaling: false });

const scratch = (w, h) => {
    const el = document.createElement("canvas");
    el.width = w; el.height = h;
    return el;
};

// Keyed by everything that decides the pixels, and small: an entry is a
// canvas the size of a title block. What it is for is the pair of renders
// every title gets — the preview and that frame's thumbnail — not a page's
// worth of titles.
const _patchCache = new Map();
const PATCH_CACHE_LIMIT = 12;

/**
 * The gaps in a sticker, as one image, with the point to place it at — or
 * null when the strokes met each other everywhere and there is nothing to
 * fill.
 *
 * The mask is drawn through fabric rather than with a bare ctx.strokeText,
 * for the reason measureVisualBounds gives: fabric places a baseline its own
 * way, and a shape drawn by any other route would sit a few pixels off the
 * strokes this is meant to be patching.
 */
function holePatch({ rows, props, box, fill }) {
    // A row may carry its own property bag — a channel whose highlight is set
    // in a second face has rows in two of them (see `highlight.face` in
    // channels.js). The mask has to be drawn in whatever each row is actually
    // drawn in, or the holes it finds are the holes of a block nobody sees;
    // and the family has to be in the key, or the first title cached under a
    // set of positions answers for every later one in another face.
    const propsOf = (row) => row.props || props;
    const key = JSON.stringify([
        rows.map(r => [r.str, Math.round(r.sz * 10), Math.round(r.left), Math.round(r.top),
                       propsOf(r).fontFamily]),
        props.strokeWidth, Math.round(box.w), Math.round(box.h), fill,
    ]);
    if (_patchCache.has(key)) {
        const hit = _patchCache.get(key);
        return hit && { ...hit, left: box.left - hit.margin, top: box.top - hit.margin };
    }

    const margin = 4;
    const w = Math.ceil(box.w) + margin * 2;
    const h = Math.ceil(box.h) + margin * 2;

    _maskEl.width = w; _maskEl.height = h;
    _maskCanvas.setDimensions({ width: w, height: h });
    _maskCanvas.clear();
    for (const row of rows) {
        _maskCanvas.add(new fabric.Text(row.str, {
            left: row.left - box.left + margin,
            top: row.top - box.top + margin,
            fontSize: row.sz, ...propsOf(row),
            stroke: "#000000", fill: "#000000",
            selectable: false, evented: false,
        }));
    }
    _maskCanvas.renderAll();

    const n = w * h;
    const pixels = _maskEl.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
    const clear = new Uint8Array(n);            // 1 where the slab is not
    for (let i = 0; i < n; i++) clear[i] = pixels[i * 4 + 3] <= 128 ? 1 : 0;

    // Flood in from all four borders. An index stack rather than coordinate
    // pairs: this walks a few hundred thousand pixels on every new title, and
    // allocating two-element arrays for each of them is most of the cost.
    const outside = new Uint8Array(n);
    const stack = new Int32Array(n);
    let top = 0;
    const reach = (i) => { if (clear[i] && !outside[i]) { outside[i] = 1; stack[top++] = i; } };
    for (let x = 0; x < w; x++) { reach(x); reach((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { reach(y * w); reach(y * w + w - 1); }
    while (top) {
        const i = stack[--top];
        const x = i % w;
        if (x > 0) reach(i - 1);
        if (x < w - 1) reach(i + 1);
        if (i >= w) reach(i - w);
        if (i < n - w) reach(i + w);
    }

    // Whatever the flood could not reach is a hole.
    const holes = new ImageData(w, h);
    let found = 0;
    for (let i = 0; i < n; i++) {
        if (!clear[i] || outside[i]) continue;
        found++;
        holes.data[i * 4] = 255; holes.data[i * 4 + 1] = 255;
        holes.data[i * 4 + 2] = 255; holes.data[i * 4 + 3] = 255;
    }
    if (!found) {
        if (_patchCache.size >= PATCH_CACHE_LIMIT) _patchCache.clear();
        _patchCache.set(key, null);
        return null;
    }

    const holeEl = scratch(w, h);
    holeEl.getContext("2d").putImageData(holes, 0, 0);

    // Grown by a pixel in every direction before it is used. A hole's border
    // is the strokes' own antialiased edge, where a pixel is part slab and
    // part hole and belongs to neither; without the overlap those pixels stay
    // half transparent and the patch reads as a hairline outline of itself.
    // The growth is hidden under ink by definition — it can only reach into
    // the slab that surrounds the hole.
    const grown = scratch(w, h);
    const gctx = grown.getContext("2d");
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) gctx.drawImage(holeEl, dx, dy);

    const el = scratch(w, h);
    const ctx = el.getContext("2d");
    ctx.fillStyle = patchPaint(ctx, fill, margin, box);
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(grown, 0, 0);

    const patch = { margin, canvas: el };
    if (_patchCache.size >= PATCH_CACHE_LIMIT) _patchCache.clear();
    _patchCache.set(key, patch);
    return { ...patch, left: box.left - margin, top: box.top - margin };
}

/**
 * The sticker's paint, in the coordinates of the patch image.
 *
 * The same ramp the strokes themselves are painted with (see stickerFill),
 * written the other way round: there, a gradient is handed to fabric and
 * anchored per object; here it is an ordinary canvas gradient across the same
 * slab box. Both describe one ramp over one rectangle, so the patch and the
 * strokes around it come out the same colour — which is the whole reason a
 * patch can be dropped in and not be seen.
 */
function patchPaint(ctx, spec, margin, box) {
    if (!spec || typeof spec === "string") return spec || "#000000";
    const [fx, fy] = spec.from || [0, 1];
    const [tx, ty] = spec.to || [1, 0];
    const g = ctx.createLinearGradient(
        margin + fx * box.w, margin + fy * box.h,
        margin + tx * box.w, margin + ty * box.h,
    );
    for (const stop of spec.stops || []) g.addColorStop(stop.at, stop.color);
    return g;
}

/**
 * The channel's style with this frame's chosen colours in place of the
 * channel's own, for a channel that offers a choice (see `colors` in
 * channels.js).
 *
 * A copy, never a mutation: the style handed out by titleStyle() is the
 * channel's and is shared by every frame drawn under it, so writing a colour
 * into it would repaint the other nineteen thumbnails in the grid the moment
 * one of them was recoloured.
 *
 * Null, and an index outside the list, both fall back to the pack's own
 * colour rather than to nothing — the frame nobody has pressed a swatch on,
 * and the frame carried over from a channel with a longer palette. The honest
 * answer for both is what the channel itself said, so neither is a case this
 * function has to tell apart.
 *
 * Nothing here can change a measurement — `styleKey` is face, weight, line
 * height and outline — so recolouring costs nothing off the caches and no
 * layout is taken twice.
 */
function recolored(style, opts) {
    const palette = style.colors;
    if (!palette) return style;
    const at = (list, i) => (i == null ? null : (list || [])[i]);
    const text = at(palette.text, opts.textColor);
    const highlight = at(palette.highlight, opts.highlightColor);
    const panel = at(palette.panel, opts.panelColor);
    if (!text && !highlight && !panel) return style;

    // Whether the picked words are a colour of their OWN, which is a fact
    // about the channel and not about this frame: a pack that states the list
    // is offering the choice, and it goes on offering it on the frame nobody
    // has pressed that row's swatch on. Read from the pack rather than from
    // `highlight` above, or an untouched frame would take the block's colour
    // for one draw and its own the moment a swatch was pressed — the picked
    // words changing colour because the user recoloured the BLOCK.
    const ownHighlight = Array.isArray(palette.highlight) && palette.highlight.length > 0;

    const out = { ...style };
    if (text) {
        out.color = text;
        // The picked lines too — see the note on `colors.text` in
        // channels.js. A channel that offers a palette and no `highlight`
        // list is one whose type is one colour, and its highlight is a
        // different cut of the face rather than a different colour; leaving
        // the highlight behind would mean choosing pink and getting one line
        // still in the pack's white.
        //
        // A channel that DOES offer the list has said the opposite in as many
        // words: the two are separate decisions, and reaching in here would
        // undo the one the user just made in the row above.
        if (out.highlight && !ownHighlight) out.highlight = { ...out.highlight, color: text };
    }
    if (highlight && out.highlight) out.highlight = { ...out.highlight, color: highlight };
    if (panel && out.panel) out.panel = { ...out.panel, fill: panel };
    return out;
}

/**
 * Draws a frame's title, and returns the box its ink came out in — or null
 * when there was nothing to draw. That box is what the on-canvas handles are
 * hung on and what a click is tested against (see compose.titleHitTest); it
 * is the ink alone, so the shadow, the halo and the highlight boxes all
 * spread past it, exactly as they spread past the block in the layout.
 *
 * `opts` carries the three things that are the FRAME's rather than the
 * channel's:
 *
 *   glowOn      whether the optional halo is switched on for this frame,
 *               which is only ever asked of a channel whose glow says
 *               `toggle` (see channels.js).
 *   placement   {dx, dy, scale} — how far the user has dragged the block from
 *               where the guidelines put it, and how much they have resized
 *               it. Null while it is untouched, which is not the same as
 *               {dx: 0, dy: 0, scale: 1} only in that it takes the faster
 *               path: with nothing to transform the objects go straight onto
 *               the canvas, exactly as they did before any of this existed.
 *   interactive whether to draw the dashed box and the resize handle. Never
 *               for a composite — a thumbnail or a download with selection
 *               chrome baked into it is a ruined file.
 *   textColor       which of the colours the channel offers the letters are
 *   highlightColor  set in, which the words picked out of them are set in,
 *   panelColor      and which the slab behind them is filled with — indices
 *                   into `colors.text`, `colors.highlight` and `colors.panel`
 *                   (see channels.js). Ignored, and never read, by a channel
 *                   that offers no palette.
 *
 * `mirror` moves the whole text block to the other side of the canvas: the
 * block is mirrored about the vertical centerline, so the margin it keeps
 * from the edge is the same one it kept from the other edge. What it does not
 * change is the type inside the block, which is centred either way (see
 * inkLeft). Each highlight box is derived from its own line's ink span, which
 * is already mirrored by then, so the boxes follow for free. Glyphs are never mirrored — that would make the
 * title unreadable; it's the layout that flips. The backing gradient follows
 * from the backend (see reframe_engine.apply_dark_gradient's `mirror`).
 */
export function addTextOverlay(canvas, lines, hlIdx, mirror, opts = {}) {
    // A title is set in some channel's house style or it is not set at all:
    // with nothing selected there are no guidelines to lay it out by, so
    // nothing is drawn. The fields that feed this are inert until a channel
    // is picked (see editor.js), so in practice there is nothing here to draw
    // anyway — except right after a channel is cleared back to "-", which is
    // exactly the case this covers.
    if (!hasChannel()) return null;

    // The channel's guidelines, with whatever this frame said about their
    // two open questions. Done once, here, so everything below goes on
    // reading `style.color` and `style.panel.fill` without knowing there was
    // ever a choice to make.
    const style = recolored(titleStyle(), opts);
    const props = textProps(style);

    // Every object this function makes, collected rather than added as it
    // goes. A title the user has moved or resized is drawn as one group and
    // transformed as one thing (see place), which cannot be done to objects
    // already handed to the canvas — and doing it any other way would mean
    // every size, offset and gradient below carrying the placement in it.
    const parts = [];
    const add = (obj) => parts.push(obj);
    const { leftRatio, widthRatio, maxBlockHeightRatio, lineGap } = style.layout;

    const LEFT = CW * leftRatio;
    const TARGET_W = CW * widthRatio;
    const MAX_BLOCK_H = CH * maxBlockHeightRatio;
    const LINE_GAP = lineGap;

    // Highlight padding as a fraction of the line's own font size rather than
    // a flat pixel count, so a box keeps the same proportions whatever size
    // its line came out at — the same amount on all four sides.
    // A channel may carry no shadow at all (see BASE_TITLE_STYLE.shadow), so
    // every shadow is built through this rather than spread straight into a
    // fabric.Shadow — `new fabric.Shadow(null)` is not a no-shadow, it throws.
    const shadowOf = (spec) => (spec ? new fabric.Shadow({ ...spec }) : null);

    // A corner radius as the pair of keys fabric wants, clamped to the box it
    // is rounding. Half the shorter side is a capsule and there is nothing
    // rounder; past that the arcs overlap and the shape folds through itself,
    // so a pack overreaching gets the capsule rather than a rectangle with a
    // bite out of it. Zero — the skeleton's answer everywhere it is not
    // stated — is a square corner, which is what every one of these has always
    // been.
    const rounded = (radius, w, h) => {
        const r = Math.max(0, Math.min(radius || 0, w / 2, h / 2));
        return { rx: r, ry: r };
    };

    // A channel may have no highlight feature at all (see `highlight` in
    // channels.js), and then no line is ever a highlighted one however the
    // buttons were left. Everything the highlight drives — the box, the
    // recolour, the foil, the box's own shadow — hangs off `hl` being true
    // for a row, so answering that question once here is what makes the rest
    // of this function inert for such a channel rather than guarded at every
    // use.
    const highlight = style.highlight;

    // Whether highlighting draws a box or only recolours the line. Anything
    // that isn't the box mode is colour-only, which is also what makes the
    // box geometry below dead weight rather than wrong for such a channel.
    const boxedHighlight = !!highlight && (highlight.mode || "box") === "box";

    // Whether a highlight covers a line or a single word (see `scope` in
    // channels.js). Independent of the mode: a picked word takes a box in
    // "box" and the highlight's colour in every other mode, exactly as a
    // picked LINE does.
    const wordScope = !!highlight && highlight.scope === "word";

    // Line-scope: the whole of line `i` is picked out, or none of it. Answered
    // up here rather than beside the other row predicates because the fit
    // below now needs it — a picked line may be set in a face of its own, and
    // which face a line is measured in cannot wait until after it has been
    // measured.
    const isHighlighted = (i) => !!highlight && !wordScope && hlIdx.includes(i);

    /**
     * The style a row is measured and drawn with — the channel's, or the
     * channel's with the highlight's own face in place of the block's (see
     * `highlight.face` in channels.js).
     *
     * A face is the only thing that differs, and it is spread over the whole
     * style rather than passed alongside it because the metrics are not the
     * only thing that follows from it: `styleKey` keys every measurement
     * cache on the face, so two rows in two faces have to be two styles or
     * they share one cache entry and the second row is laid out against the
     * first row's letters.
     *
     * Line scope only — under word scope a row is not one face at all, and
     * `runRows` below is what handles it.
     */
    const hlFace = highlight && highlight.face;
    const hlStyle = hlFace ? { ...style, face: hlFace } : style;
    const styleFor = (i) => (hlFace && isHighlighted(i) ? hlStyle : style);

    /**
     * Whether a row has to be built out of RUNS rather than set as one string.
     *
     * One string is how a line is normally drawn, and that is not an
     * implementation detail — the space between two words, and the kerning
     * pair across it, come from the font because the font is what set them.
     * A per-word BOX keeps it: the boxes go under the whole line and the
     * picked words are drawn again on top of them, so nothing is ever split
     * (see `runCopy`). Two things defeat that trick, and they are the two
     * cases here:
     *
     *   a per-word face    Two faces in one line means two sets of advances,
     *                      and there is no single string that has both.
     *   a per-word colour  A recolour is not something a second copy can add.
     *                      Drawing the picked words again in another colour
     *                      leaves the copy underneath showing as a fringe
     *                      round every letter — the same compounding that
     *                      stops the boxed version painting a glyph twice
     *                      (see `runCopy`), except that there both copies are
     *                      the same white and here they are two colours.
     *
     * Such a row is cut at its treatment boundaries into runs, each measured
     * and drawn on its own, each placed where the runs before it left the pen.
     * The spacing inside a run is still the font's; the only place arithmetic
     * decides anything is the join between two runs, and every join is a
     * space.
     *
     * Faces are compared by family, the way `titleFaces` deduplicates them: a
     * pack is free to name the block's own face as its highlight face, and a
     * boxed row cut into runs that are all in one face would pay the whole
     * cost of this for a line it could have drawn whole.
     */
    const runRows = wordScope && (!boxedHighlight
        || (!!hlFace && hlFace.family !== style.face.family));

    // The halo, unless it is one the channel merely OFFERS and this frame has
    // not asked for (see `glow.toggle` in channels.js). Resolved once, here,
    // so everything downstream goes on asking the one question "is there a
    // glow" rather than each site having to know about the button as well.
    const glow = (style.glow && (!style.glow.toggle || opts.glowOn)) ? style.glow : null;
    const sticker = style.sticker;

    const HL_PAD_RATIO = highlight ? highlight.padRatio : 0;
    // ...but never so much that the box would run off the left edge, which a
    // very short line (set enormous to reach TARGET_W) otherwise could.
    const padFor = (sz) => Math.min(Math.round(sz * HL_PAD_RATIO), LEFT);

    const cased = (l) => (style.uppercase ? l.toUpperCase() : l);

    /**
     * The slab, and whether there is one slab or one per line.
     *
     * "block" — one rectangle around the whole title, which is what a panel
     *           has always been. Its width is the block's, so every line
     *           shares it and a short line sits in a wide card.
     * "line"  — one rectangle per line, each the width of that line. The
     *           block stops being a card with type on it and becomes a
     *           stack of tags, which is what burned-in social captions
     *           actually look like: the shape of the graphic follows the
     *           shape of the sentence, so a two-word line reads as a short
     *           line rather than as a long line with air on both sides.
     *
     * Resolved here, above the layout, because it is not only a drawing
     * decision. Per line, the slab is part of what gets STACKED — the gap
     * between two lines is the gap between two boxes, and the margin the
     * block stands on is the bottom of the last box — so the rows have to be
     * measured knowing about it.
     */
    const panel = style.panel;
    const linePanel = !!panel && panel.scope === "line";
    /**
     * Whether those per-line boxes are one surface or several.
     *
     * Several — the default — is a stack of separate tags with the channel's
     * `lineGap` of photograph showing between them.
     *
     * Joined is one background, shaped per line. The boxes still take their
     * own widths, so the silhouette still steps in and out with the sentence,
     * but they overlap into a single continuous shape with one outline and
     * one shadow round the whole of it.
     *
     * It changes what `lineGap` measures, and it has to. Separate, the gap is
     * between the BOXES, because the boxes are what is visible between two
     * lines. Joined there is no gap between the boxes — that is the whole
     * point — so the thing being spaced is the type, and `lineGap` becomes
     * the gap between the lines themselves. Which is why a joined panel wants
     * a much smaller number than a separated one: it is spacing letters, not
     * slabs.
     */
    const joinedPanel = linePanel && panel.joined === true;
    const panelPad = panel ? panel.padding : 0;
    const panelBorder = panel ? panel.border.width : 0;

    // A fact about the words, not about the size they come out at, so it is
    // answered once for the block rather than on every layout pass.
    const roles = quoteRoles(lines.map(cased));

    /**
     * The size the lines are set at — ONE size for the whole block, taken
     * from the line that needs the smallest one to fit `targetW`. The longest
     * line reaches the target width; the shorter ones simply come out
     * narrower, at that same size.
     *
     * This used to fit each line to `targetW` INDIVIDUALLY, setting a short
     * line as large as it took to fill the width. That gave flush left and
     * right edges, but it made a line's size a function of how many
     * characters it happens to contain: "WOMAN" against "IN OUR HOUSE?" came
     * out ~2.5x larger, and since a highlight box is the cap band plus
     * padding, its box came out ~2.5x taller — two boxes in one title at
     * wildly different heights, over nothing but a difference in character
     * count.
     *
     * The boxes are what has to stay constant, so the letters are what
     * adapts: at one shared size every cap band is identical, so every box is
     * exactly as tall as every other, whatever gets typed. Box WIDTH still
     * follows each line's own ink (see where the box is built) — a box
     * stretched to the widest line leaves a slab of empty red past the end of
     * a short one.
     *
     * Alignment is unaffected: lines are still positioned by their INK, by
     * whichever edge their channel aligns on (see inkLeft), and each box is
     * still derived from the cap band, so the capitals sit centred in it.
     */
    const uniformSize = style.layout.uniformSize !== false;

    /**
     * A size the BRAND states, which every line is then set at — or null,
     * which is every channel that leaves the size to the fit above.
     *
     * See `fontSize` in channels.js for why a caption wants one and a headline
     * does not. Here it is simply the answer to the question the rest of this
     * section exists to work out, so it short-circuits the fit, and it also
     * turns off the height budget's pull-in below: a size the renderer is free
     * to shrink is not a fixed size.
     */
    const fixedSize = style.layout.fontSize || null;

    // A line with nothing in it is a line break the user typed and meant (see
    // state.linesFor), and it is never measured: an empty string has no ink,
    // so the fit that answers "how big must this be to fill the width" would
    // answer with the largest size there is. It is given the block's own size
    // and takes the height of a line of capitals, which is what makes it read
    // as the blank line it is.
    const blankRow = (i) => !cased(lines[i]).trim();

    const sizesFor = (targetW) => {
        if (fixedSize) return lines.map(() => fixedSize);
        const perLine = lines.map((l, i) => (
            blankRow(i) ? null : sizeOf(i, targetW)));
        const measured = perLine.filter(sz => sz !== null);
        const shared = measured.length ? Math.min(...measured) : FIT_REFERENCE_SIZE;
        if (!uniformSize) return perLine.map(sz => (sz === null ? shared : sz));
        return perLine.map(() => shared);
    };

    /**
     * Line `i` as the pieces it is DRAWN in — one per treatment, in the face
     * that treatment is set in.
     *
     * One piece covering the whole line for every channel that does not pick
     * words out, and that is not a special case being tolerated: it is the
     * ordinary line, expressed in the same shape, so everything downstream
     * stacks, measures and draws rows the one way. A single piece at pen 0 in
     * the row's own face IS the line drawn as one string, which is what it has
     * always been.
     */
    const piecesFor = (i) => {
        const str = cased(lines[i]);
        if (!runRows) return [{ text: str, style: styleFor(i), hl: isHighlighted(i) }];
        return treatmentRuns(str, (k) => hlIdx.includes(wordKey(i, k)))
            .map(piece => ({ ...piece, style: piece.hl ? hlStyle : style }));
    };

    /**
     * The size line `i` needs to reach `targetW` — the ordinary ink fit, or,
     * for a row built out of runs, the same fit over those runs.
     *
     * Such a row cannot go through fontSizeForInkWidth at all: that measures
     * one string in one face, and this row may be several of each. So the runs
     * are laid out at a reference size and scaled to the target, which is
     * exact for the same reason the ordinary fit's first pass is — ink is
     * linear in font size, and every face in the row scales by the same
     * factor.
     *
     * The ruler is measureText's ink box rather than the rasterised bounds the
     * block is finally positioned by, so such a row lands within the pixel or
     * two the two rulers disagree by (see fontSizeForInkWidth, which spends a
     * second loop closing exactly that gap). Worth knowing and not worth
     * closing: the channels that pick words out this way state their size
     * outright (`fontSize` above) and never reach here.
     */
    const sizeOf = (i, targetW) => {
        const str = cased(lines[i]);
        if (!runRows) return fontSizeForInkWidth(str, targetW, styleFor(i));
        const ink = penRuns(piecesFor(i), FIT_REFERENCE_SIZE).inkW;
        if (!ink) return fontSizeForInkWidth(str, targetW, style);
        return Math.max(20, Math.min(800, FIT_REFERENCE_SIZE * targetW / ink));
    };

    let sizes = sizesFor(TARGET_W);

    /**
     * Word-scope: the ink spans of the words of line `i` that the user picked
     * — as RUNS, with neighbours joined into one.
     *
     * Two words picked side by side are one phrase and get one box, the space
     * between them filled in rather than left as a gap between two boxes.
     * That is what the eye expects of a highlighter, and the version that
     * drew a box per word was visibly not it: at these sizes the gap came out
     * a few pixels wide, and each box cast its own shadow into the one beside
     * it, so a picked-out phrase read as a bar with seams cut through it.
     *
     * Joined on the words being NEIGHBOURS, not on the boxes colliding. The
     * boxes only collide when the space between two words happens to be
     * narrower than the padding on either side of it, which is a fact about
     * the font and the size and has nothing to do with what the user meant —
     * so the same two words joined at one size and came apart at another. Two
     * words next to each other in the line are one run whatever the space
     * between them measures.
     *
     * Words on either side of a LINE BREAK are not neighbours: each line is
     * its own set of runs, because a box cannot span two lines.
     */
    const pickedSpans = (i, sz) => {
        if (!wordScope) return [];
        const str = cased(lines[i]);
        const spans = wordSpans(str, sz, style);
        const runs = [];
        let previous = -2;   // never adjacent to word 0
        spans.forEach((span, k) => {
            if (!hlIdx.includes(wordKey(i, k))) return;
            const last = runs[runs.length - 1];
            // Extending to this word's end takes in the space before it, which
            // is exactly the gap being filled — in the box, and in the text
            // the run covers.
            if (k === previous + 1) {
                last.end = span.end;
                last.text = str.slice(last.index, span.index + span.word.length);
            } else {
                runs.push({
                    start: span.start, end: span.end,
                    index: span.index, pen: span.pen, text: span.word,
                });
            }
            previous = k;
        });
        return runs;
    };

    /**
     * Top and bottom of the CAP BAND — flat cap height and baseline — at a
     * given size, read off an "H" through the same measuring path as any
     * other string.
     *
     * The highlight box is padded from this, not from the line's own ink.
     * Uppercase text reads as the mass between cap height and baseline; a J
     * or Q tail is not part of it. Padding the raw ink meant a line carrying
     * descenders had the tail's empty space padded as though it were text, so
     * the capitals sat visibly above the box's centre (worse the more
     * descenders the line had), and two highlighted lines at the same size
     * ended up with different box heights. Off the cap band, every box at a
     * size is the same height and the letters sit centred in it.
     */
    const capBand = (sz, rowStyle = style) => measureVisualBounds("H", sz, rowStyle);

    /**
     * The ink bounds of a row, from the bounds of the pieces it is drawn in.
     *
     * A row of one piece is one string in one face, and this hands back
     * exactly what measuring that string hands back — the same call, the same
     * cache entry. That identity is deliberate: every channel here is a row of
     * one piece, and none of them may be laid out one pixel differently for
     * having gone through a function written for the one that isn't.
     *
     * A row of several is the union of theirs, each shifted by the pen it
     * starts at. The vertical edges need no shifting: fabric derives a text
     * object's baseline from its font SIZE and line height and not from the
     * face's own metrics, so two faces at one size sit on one baseline —
     * which is also what lets the pieces be handed the same `top`.
     */
    const segBounds = (segs, sz) => {
        const inked = segs.filter(seg => /\S/.test(seg.text));
        if (inked.length === 1 && !inked[0].pen) {
            return measureVisualBounds(inked[0].text, sz, inked[0].style);
        }
        const parts = inked.map(seg => ({ pen: seg.pen, vb: measureVisualBounds(seg.text, sz, seg.style) }));
        const top = Math.min(...parts.map(part => part.vb.top));
        const bottom = Math.max(...parts.map(part => part.vb.bottom));
        const left = Math.min(...parts.map(part => part.pen + part.vb.left));
        const right = Math.max(...parts.map(part => part.pen + part.vb.right));
        return { top, bottom, visualH: bottom - top + 1, left, right, visualW: right - left + 1 };
    };

    /**
     * Stacks lines top-to-bottom with a uniform LINE_GAP between the *visible*
     * edges of consecutive lines: normally the glyphs' own pixel bounds, but
     * the red highlight box's border wherever a line (or its neighbor) is
     * highlighted. That's what lets toggling highlight redistribute the block
     * instead of leaving a lopsided gap — the box border, not the invisible
     * font bounding box, is what the eye measures the spacing from.
     */
    function computeLayout(szArr) {
        const rows = lines.map((line, i) => {
            const str = cased(line);
            const quote = roles[i];
            const sz = szArr[i];
            const blank = blankRow(i);
            // The face this row is in, and the property bag every measurement
            // and every copy of it goes through. Carried on the row rather
            // than looked up again where it is drawn: the fit above already
            // decided which face this line's size describes, and a draw that
            // asked the question a second time is a draw that could answer it
            // differently.
            const rowStyle = styleFor(i);
            const rowProps = textProps(rowStyle);
            // The pieces this row is drawn in, each at the pen its
            // predecessors advanced to. One piece for every row that is drawn
            // as one string, which is all of them outside a channel that picks
            // words out (see piecesFor).
            const run = blank ? { segs: [], advance: 0 } : penRuns(piecesFor(i), sz);
            const segs = run.segs;
            // A blank row occupies a line of capitals and no width: it is a
            // gap, and a gap that measured as narrow as its own (nonexistent)
            // ink would not be one.
            const band0 = capBand(sz, rowStyle);
            const vb = blank
                ? { top: band0.top, bottom: band0.bottom, visualH: band0.visualH, left: 0, right: 0, visualW: 0 }
                : segBounds(segs, sz);
            // What a box behind this row has to cover: its ink, widened by
            // whatever whitespace the user typed at either end (see
            // edgeSpace). Identical to the ink bounds for every row that was
            // not padded, and for every channel that draws no box per line —
            // which is what keeps this out of the way of everything else that
            // reads a row's width.
            const space = (linePanel && !blank) ? edgeSpace(str, sz, rowStyle) : { lead: 0, tail: 0 };
            const span = (space.lead || space.tail)
                ? { left: vb.left - space.lead, right: vb.right + space.tail,
                    visualW: vb.visualW + space.lead + space.tail }
                : vb;
            const words = (blank || runRows) ? [] : pickedSpans(i, sz);
            const hl = wordScope
                ? (runRows ? segs.some(seg => seg.hl) : words.length > 0)
                : isHighlighted(i);
            // Whether any word in the row was left unpicked. A row where some
            // words are boxed and some are not needs the line drawn twice —
            // see the note where it is drawn.
            // Only ever asked of a row drawn as one string: a row of runs
            // needs no second copy of the line, so nothing here would read
            // the answer and measuring for it would be work thrown away.
            const clearWords = wordScope && !runRows && !blank
                && wordSpans(str, sz, style).some((_, k) => !hlIdx.includes(wordKey(i, k)));
            // A colour-only highlight occupies exactly the space the same
            // line would unhighlighted, so it must not enter the stacking at
            // all: spacing lines by a box that is never drawn would leave the
            // block holding gaps around nothing.
            const boxed = hl && boxedHighlight;
            // Off the face the BOX's own letters are in, not the row's: the
            // band is what the box is padded from, and on a row of runs the
            // letters inside it are the highlight's.
            const band = boxed ? capBand(sz, runRows ? hlStyle : rowStyle) : null;
            const pad = padFor(sz);
            const boxTop = boxed ? band.top - pad : 0;
            const boxBottom = boxed ? band.bottom + pad : 0;
            return {
                i, quote, style: rowStyle, props: rowProps, segs, span,
                str, sz, blank, vb, hl, words, clearWords, boxed, pad, boxTop, boxBottom,
                // Descenders can now hang past the box, so the edges the next
                // line is spaced from are whichever reaches further — the
                // stacking must never let a tail collide with the line below.
                topOffset: boxed ? Math.min(boxTop, vb.top) : vb.top,
                bottomOffset: boxed ? Math.max(boxBottom, vb.bottom) : vb.bottom,
            };
        });

        // One band for every box in the block, not one per line.
        //
        // A box sized to its own row's ink would be a different height on
        // every line — taller under an accented capital, taller again under a
        // descender — and a stack of tags at four different heights reads as
        // four mistakes. So the band is the union of what every row needs,
        // and every box takes it: the boxes differ in WIDTH, which is the
        // sentence showing through, and in nothing else.
        //
        // Measured from the cap band as well as the ink, so a block whose
        // lines happen to carry no tall letter at all still gets boxes of the
        // height its capitals ask for rather than boxes hugging an x-height.
        if (linePanel) {
            const inked = rows.filter(row => !row.blank);
            if (inked.length) {
                const bandTop = Math.min(...inked.map(row =>
                    Math.min(row.vb.top, capBand(row.sz, row.style).top)));
                const bandBottom = Math.max(...inked.map(row =>
                    Math.max(row.vb.bottom, capBand(row.sz, row.style).bottom)));
                for (const row of rows) {
                    if (row.blank) continue;
                    row.panelTop = bandTop - panelPad - panelBorder;
                    row.panelBottom = bandBottom + panelPad + panelBorder;
                }

                if (joinedPanel) {
                    // Every row is stacked by the BAND, so `lineGap` is the
                    // gap between the lines and the boxes — which are the band
                    // plus the padding on both sides — overlap each other by
                    // twice that padding less the gap. That overlap is what
                    // welds them into one shape, and it is why nothing here
                    // has to compute an outline: two rectangles that overlap
                    // are one rectangle as far as the canvas is concerned.
                    //
                    // The two ENDS are the exception, and only the ends: the
                    // block's own top and bottom edges are the background's,
                    // not the type's, or the margin the block stands on would
                    // be measured to the letters and the slab would hang past
                    // it (see bottomRatio in channels.js).
                    for (const row of rows) {
                        if (row.blank) continue;
                        row.topOffset = bandTop;
                        row.bottomOffset = bandBottom;
                    }
                    inked[0].topOffset = inked[0].panelTop;
                    inked[inked.length - 1].bottomOffset = inked[inked.length - 1].panelBottom;
                } else {
                    // Separate boxes: the slab is what the eye measures the
                    // spacing from, so every one of them is what its row is
                    // stacked by — exactly as a highlight box is (see
                    // topOffset above).
                    for (const row of rows) {
                        if (row.blank) continue;
                        row.topOffset = Math.min(row.topOffset, row.panelTop);
                        row.bottomOffset = Math.max(row.bottomOffset, row.panelBottom);
                    }
                }
            }
        }

        rows[0].T = 0;
        for (let i = 1; i < rows.length; i++) {
            rows[i].T = rows[i - 1].T + rows[i - 1].bottomOffset + LINE_GAP - rows[i].topOffset;
        }

        const topEdge0 = rows[0].T + rows[0].topOffset;
        const bottomEdgeLast = rows[rows.length - 1].T + rows[rows.length - 1].bottomOffset;
        return { rows, topEdge0, totalVisualH: bottomEdgeLast - topEdge0 };
    }

    // Vertical chrome the panel adds around the block. Taken out of the
    // height budget BEFORE the block is fitted, so what gets held inside
    // MAX_BLOCK_H is the panel — the thing actually on screen — rather than
    // the text alone, which would let the slab grow past the limit by exactly
    // the amount of its own border and padding.
    // A per-line slab is already inside the rows' own offsets (see
    // computeLayout), so taking it out again here would charge the block for
    // it twice. Only the one-rectangle kind sits outside the block.
    const panelChrome = (panel && !linePanel) ? (panelPad + panelBorder) : 0;
    const BLOCK_BUDGET = MAX_BLOCK_H - panelChrome * 2;

    let layout = computeLayout(sizes);
    // A pack that states its size is not asking for a size the renderer may
    // reconsider (see `fontSize` in channels.js): the block grows past the
    // budget instead, and the user drags or resizes it from there. Everything
    // else is pulled back in.
    if (!fixedSize && layout.totalVisualH > BLOCK_BUDGET) {
        // Too tall to fit: pull the whole block in by re-fitting it to a
        // narrower width. Re-fitted rather than having the sizes scaled and
        // rounded — rounding independently let the lines drift a few pixels
        // apart from each other, which is the one thing this layout must not
        // do.
        //
        // Iterated for the same reason the width fit is: an outline adds a
        // fixed amount to a line's height whatever size the type is, so
        // narrowing the block by a given fraction does not shorten it by that
        // fraction, and one pass lands short. Measured on an outlined channel,
        // a single pass left a three-line block 4% over its budget. Runs at
        // all only for blocks that overflow, and each pass is served from the
        // measurement caches on every later render of the same title.
        let width = TARGET_W;
        for (let i = 0; i < 3 && layout.totalVisualH > BLOCK_BUDGET; i++) {
            width *= BLOCK_BUDGET / layout.totalVisualH;
            sizes = sizesFor(width);
            layout = computeLayout(sizes);
        }
    }

    // The block's own width: the widest line that actually came out of the
    // layout. Usually every line was fitted to the same target and they all
    // match, but they need not — a channel that sets one size for the whole
    // block leaves its short lines short, and a line of two characters can be
    // stopped by the size cap before it ever reaches the target width.
    const blockWidth = Math.max(...layout.rows.map(r => r.span.visualW));

    // Where the line's ink should start, then backed off by the first glyph's
    // side bearing to get the position Fabric has to be handed.
    //
    // Which side of the CANVAS the block sits on is the brand's decision — the
    // margin above, mirrored or not. This is the question INSIDE the block:
    // where a line narrower than its neighbours takes its slack.
    //
    // Centred by default, and that default is not laziness. Flush left, all of
    // the slack goes to one end and a short line reads as a line that has
    // slipped rather than as a shorter line — the ragged edge is what the eye
    // takes for the shape of the block, so a title with one straight edge and
    // one ragged one looks misaligned even when nothing is. Split evenly,
    // neither edge claims to be the straight one.
    //
    // A channel says otherwise when it has a reason to. This one sets its
    // titles hard against the right-hand margin because the left of its canvas
    // belongs to the subject: the block's right edge is the frame's edge, which
    // is a real straight edge on screen rather than one the eye has to infer,
    // and a short line hanging off it centred would read as detached from the
    // margin the rest of the block is set to.
    // The frame's own answer outranks the channel's, for the reason its
    // gradient and its halo do: the channel states what its titles are set
    // like, and one particular title — a last line of two words, a block that
    // has to lean away from something in the picture — may want another. See
    // editor.setTitleAlign.
    const align = opts.align || style.layout.align || "center";

    /**
     * Where the BLOCK sits across the column, before its lines are placed
     * inside it.
     *
     * Flush to the margin, which is where it has always been, except for a
     * centred block set at a stated size — that one is centred in the column
     * as well as inside itself.
     *
     * The exception exists because the two cases are not the same shape of
     * answer. A fitted block fills the column by construction: its widest
     * line is fitted to TARGET_W, so there is nothing left over to centre and
     * flush-to-the-margin and centred-in-the-column are the same place. A
     * block at a stated size fills nothing — a short caption at 40px may be
     * half the column wide — and left at the margin it would sit visibly off
     * to one side of a frame whose whole look is centred (see `fontSize` in
     * channels.js, and the packs that state one).
     */
    const columnLeft = (fixedSize && align === "center")
        ? LEFT + Math.max(0, TARGET_W - blockWidth) / 2
        : (mirror ? CW - LEFT - blockWidth : LEFT);

    const aligned = (span) => {
        const slack = blockWidth - span.visualW;
        if (align === "right") return columnLeft + slack;
        if (align === "left") return columnLeft;
        return columnLeft + slack / 2;
    };

    /**
     * The nearest row above or below `row` that is words rather than
     * punctuation — the line a lone quotation mark belongs to.
     *
     * Blank rows are stepped over: a break the user typed between the mark and
     * its sentence does not make it a different sentence.
     */
    const nearestText = (row, step) => {
        for (let k = row.i + step; k >= 0 && k < layout.rows.length; k += step) {
            const other = layout.rows[k];
            if (!other.blank && !other.quote) return other;
        }
        return null;
    };

    /**
     * Where a row's span starts — `aligned`, with one exception.
     *
     * The SPAN and not the ink, and on every channel but one they are the
     * same thing: a row's span is its ink widened by whatever whitespace the
     * user typed at either end, and nothing widens it unless a box is being
     * drawn behind the line (see `space` in computeLayout). Positioning by
     * the span is what makes a padded line's BOX centre rather than its
     * letters — the room the user asked for stays where they put it instead
     * of being centred away.
     *
     * A row that is nothing but a quotation mark is that exception, and it is
     * the exception BECAUSE of the alignment: whatever the block's own rule
     * is, a mark has no width to speak of and lands nowhere near the sentence
     * it opens. It is set against its own line instead — an opener flush with
     * the left edge of the line below, a closer flush with the right edge of
     * the line above — so it sits at the corner of the block where a quotation
     * mark belongs.
     *
     * With no line to belong to (a title that is only punctuation) it falls
     * back to the block's own alignment like everything else.
     */
    const inkLeft = (row) => {
        if (row.quote === "open") {
            const line = nearestText(row, +1);
            return line ? aligned(line.span) : aligned(row.span);
        }
        if (row.quote === "close") {
            const line = nearestText(row, -1);
            // Its RIGHT edge against theirs, which is what "justified right"
            // means for a mark narrower than the line it is closing.
            return line
                ? aligned(line.span) + line.span.visualW - row.span.visualW
                : aligned(row.span);
        }
        return aligned(row.span);
    };
    const textLeft = (row) => inkLeft(row) - row.span.left;

    // Where the whole visual block (glyphs + highlight padding) sits down the
    // canvas. A panel needs no term of its own here: it is the block plus the
    // same padding on every side, so whatever places the block places it too.
    //
    // Centred, stood on a margin off the bottom edge, or hung from one off the
    // top edge — see `vAlign` in channels.js. All three are expressed as where
    // the block's TOP goes, because that is what the rows below are stacked
    // from, and that is why the three arms differ: centring splits what is
    // left over, the bottom form has to subtract the block's own height to
    // keep the margin under it constant as lines come and go, and the top form
    // subtracts nothing at all. A block hung from the top edge grows downward
    // into the picture, so its first line lands on the margin and stays there
    // whatever the sentence turns out to be — which is the same constancy the
    // bottom form buys with the subtraction, for free.
    const blockTop0 = style.layout.vAlign === "bottom"
        ? CH * (1 - style.layout.bottomRatio) - layout.totalVisualH
        : style.layout.vAlign === "top"
            ? CH * style.layout.topRatio
            : (CH - layout.totalVisualH) / 2;
    const offsetY = blockTop0 - layout.topEdge0;

    // The block's own extent, measured from what was laid out rather than
    // assumed from TARGET_W — a block whose lines all came out short (one
    // word, or a run pulled in by the height budget) must not be given a slab,
    // or a sheet of texture, sized for text that isn't there.
    // Named for the block explicitly: the per-row loop below has a `top` of
    // its own, and a plain `top` out here would be shadowed inside it —
    // silently, and only for the code that reads it from within the loop.
    const blockLeft = Math.min(...layout.rows.map(r => inkLeft(r)));
    const blockRight = Math.max(...layout.rows.map(r => inkLeft(r) + r.span.visualW));
    const blockTop = layout.topEdge0 + offsetY;
    const blockW = blockRight - blockLeft, blockH = layout.totalVisualH;

    // The highlight texture, fitted to the block ONCE. Mapping it per line
    // would give each highlighted line its own copy of the whole image; one
    // map across the block makes them read as one sheet with the lines cut
    // out of it, which is what a foil actually looks like.
    const textureSheet = (highlight && highlight.mode === "texture" && highlight.texture)
        ? fittedTexture(highlight.texture, blockW, blockH)
        : null;

    // The ramp a "gradient" highlight is filled with, or null. Read once here
    // rather than per line for the same reason the texture sheet is fitted
    // once: what every highlighted line shares is ONE ramp over ONE box, and
    // a per-line lookup is the shape of code that eventually grows a per-line
    // ramp by accident.
    const highlightRamp = (highlight && highlight.mode === "gradient" && highlight.gradient)
        ? highlight.gradient
        : null;

    /**
     * What a highlighted line is painted with. Fabric anchors an object's
     * paint at that object's own top-left, so shifting it by the gap between
     * that corner and the block's puts every line on the one shared map.
     *
     * That shift is the whole reason both surfaces are built here and not in
     * the pack: a texture and a gradient are stated against the BLOCK, which
     * is the shape the eye reads, and each line then has to be told where it
     * sits inside it. Mapped per line instead, every highlighted line would
     * carry its own copy of the whole ramp and the block would come apart
     * into stripes — the same failure `stickerFill` exists to avoid, and the
     * reason these two do the same arithmetic.
     *
     * Falls back to the flat colour whenever there is neither — a texture
     * still loading, or one that failed to, or a pack that named "gradient"
     * and stated no ramp. The line stays highlighted and the layout is
     * untouched either way; only the surface is missing.
     */
    const highlightFill = (objLeft, objTop) => {
        if (textureSheet) {
            return new fabric.Pattern({
                source: textureSheet, repeat: "repeat",
                offsetX: blockLeft - objLeft, offsetY: blockTop - objTop,
            });
        }
        if (highlightRamp) return blockGradient(highlightRamp, objLeft, objTop);
        return highlight.color;
    };

    /**
     * A pack's ramp as a fabric gradient in one line's own coordinates.
     *
     * `from`/`to` are fractions of the block's box, so the ramp describes the
     * block on screen and not the letters buried in it; `px`/`py` then move
     * that description into the coordinates of the object being painted.
     *
     * A radial ramp's radius is taken from the block's LONGER side rather
     * than from its diagonal or its height. The longer side is the one the
     * eye measures the block by, and it is the one number that does not lurch
     * when a title gains or loses a line — a radius stated against the height
     * would tighten every ramp in the channel the moment a four-line title
     * came in three lines long.
     */
    function blockGradient(spec, objLeft, objTop) {
        const px = (f) => blockLeft + f * blockW - objLeft;
        const py = (f) => blockTop + f * blockH - objTop;
        const [fx, fy] = spec.from || [0, 1];
        const stops = (spec.stops || []).map(stop => ({ offset: stop.at, color: stop.color }));
        if (spec.type === "radial") {
            const r = (spec.radius !== undefined ? spec.radius : 1) * Math.max(blockW, blockH);
            return new fabric.Gradient({
                type: "radial", gradientUnits: "pixels",
                // Both circles share a centre and the inner one has no size:
                // this is a ramp spreading out from a point, not light through
                // a ring, and a non-zero r1 would leave a flat disc of the
                // first colour with the ramp starting outside it.
                coords: { x1: px(fx), y1: py(fy), r1: 0, x2: px(fx), y2: py(fy), r2: r },
                colorStops: stops,
            });
        }
        const [tx, ty] = spec.to || [1, 0];
        return new fabric.Gradient({
            type: "linear", gradientUnits: "pixels",
            coords: { x1: px(fx), y1: py(fy), x2: px(tx), y2: py(ty) },
            colorStops: stops,
        });
    }

    // How far inside its own box the real line's letters sit.
    const inkInset = glyphInset(props);

    /**
     * A copy of a row placed so that ITS letters land on the real line's.
     *
     * The extra copies a channel can ask for — the sticker under the type,
     * the glow behind it — are stroked at their own widths, and a fabric
     * object's letters sit half its stroke width inside its box (glyphInset).
     * Handing every copy the same left/top would therefore scatter them by
     * however much their strokes differ. Positioning by the letters instead
     * is the one thing that keeps 25px of slab centred on a 3px-outlined
     * line.
     *
     * `pen` is how far along its row the piece being copied starts — zero for
     * a row drawn as one string, which is every row outside a channel that
     * picks words out (see `runRows`).
     */
    const alignedCopy = (row, copyProps, pen = 0) => {
        const shift = inkInset - glyphInset(copyProps);
        return { left: textLeft(row) + pen + shift, top: row.T + offsetY + shift };
    };

    // Where a row's letters actually land on the canvas — the point every
    // aligned copy of that row shares, whatever it is stroked at, and the
    // origin fabric anchors a gradient at (measured: the paint starts at the
    // object's left plus half its stroke, exactly where the glyphs do).
    const inkOrigin = (row, pen = 0) => ({
        x: textLeft(row) + pen + inkInset,
        y: row.T + offsetY + inkInset,
    });

    // The slab the sticker's strokes add up to: the block, grown by the
    // sticker's width on every side. It is what the paint is mapped across,
    // so the ramp describes the shape on screen rather than the letters
    // buried inside it.
    const stickerPropsOf = (rowProps) => (
        sticker ? { ...rowProps, ...strokeProps(sticker.width, null) } : null);
    const stickerProps = stickerPropsOf(props);
    const slabBox = sticker ? {
        left: blockLeft - sticker.width, top: blockTop - sticker.width,
        w: blockW + sticker.width * 2, h: blockH + sticker.width * 2,
    } : null;

    /**
     * What a row's slab is painted with, for a row whose letters land at
     * `origin`.
     *
     * A flat colour is taken as written. A gradient is mapped across the
     * WHOLE slab — `from` and `to` are corners of it, as fractions of its box,
     * so [0, 1] to [1, 0] runs from its bottom-left corner to its top-right —
     * and then moved into the row's own coordinates, because fabric anchors a
     * gradient at the object it is painting rather than on the canvas. That is
     * the same move highlightFill makes with a pattern, and for the same
     * reason: per row, each line would carry its own copy of the ramp and the
     * merged shape would come apart into bands.
     */
    const stickerFill = (origin) => {
        const spec = sticker.fill;
        if (!spec || typeof spec === "string") return spec || style.color;
        const [fx, fy] = spec.from || [0, 1];
        const [tx, ty] = spec.to || [1, 0];
        const px = (f) => slabBox.left + f * slabBox.w - origin.x;
        const py = (f) => slabBox.top + f * slabBox.h - origin.y;
        return new fabric.Gradient({
            type: "linear",
            gradientUnits: "pixels",
            coords: { x1: px(fx), y1: py(fy), x2: px(tx), y2: py(ty) },
            colorStops: (spec.stops || []).map(stop => ({ offset: stop.at, color: stop.color })),
        });
    };

    // Anything the strokes will enclose but not cover (see holePatch).
    const patch = sticker ? holePatch({
        rows: layout.rows.filter(row => !row.blank).flatMap(row => row.segs.map((seg) => {
            const slabProps = stickerPropsOf(seg.props);
            return {
                str: seg.text, sz: row.sz, props: slabProps,
                ...alignedCopy(row, slabProps, seg.pen),
            };
        })),
        props: stickerProps,
        box: slabBox,
        fill: sticker.fill || style.color,
    }) : null;

    /**
     * One slab — the block's, or a line's — as its border rectangle and its
     * fill on top.
     *
     * Both kinds are built here so that a per-line box and a per-block one
     * cannot drift apart in how they handle their border, their radius or
     * their shadow. What differs between them is only which rectangle is
     * handed in.
     */
    // Border first and larger, fill inset on top of it — see the note on
    // `panel` in channels.js for why this isn't one stroked rectangle.
    //
    // The inner rectangle's radius is the outer one's less the border, which
    // is what keeps the two corners concentric. A border of even thickness
    // drawn round a rounded corner IS a corner of a smaller radius; giving
    // both rectangles the same number would leave the keyline pinched at the
    // corners and full width along the sides.
    //
    // A pack that states NO border still gets both rectangles, and the outer
    // one is filled with the slab's own colour rather than with the border's.
    //
    // That is not a formality. At width 0 the two are the same rectangle, so
    // the top one lands exactly on whatever is under it — and along a rounded
    // edge, where both are antialiased, "exactly on" means the two colours
    // blend. A white border stated at zero width behind a pink slab drew a
    // pale rim round the whole shape, one pixel wide, following every step of
    // it: an outline nobody asked for, sitting slightly off the fill it was
    // tracing. Two rectangles of the SAME colour cannot do that.
    //
    // Dropping the outer rectangle instead would have been the obvious fix
    // and it is the wrong one: the shadow hangs off it, and the whole reason
    // it is a separate pass is that every shadow has to be laid down before
    // any fill (see the boxes below). Moving the shadow onto the fills puts
    // one box's shadow back on top of the box before it.
    const slabUnder = (x, y, w, h, bw) => new fabric.Rect({
        left: x - bw, top: y - bw, width: w + bw * 2, height: h + bw * 2,
        ...rounded(panel.radius, w + bw * 2, h + bw * 2),
        fill: bw > 0 ? panel.border.color : panel.fill,
        shadow: shadowOf(panel.shadow),
        selectable: false, evented: false,
    });
    const slabFill = (x, y, w, h, bw) => new fabric.Rect({
        left: x, top: y, width: w, height: h,
        ...rounded((panel.radius || 0) - bw, w, h),
        fill: panel.fill, selectable: false, evented: false,
    });

    // One box per line, each the width of its own line, stacked with the rows
    // they belong to. Drawn in one pass before any type, for the reason the
    // sticker's slabs are: a box is wider than its line and a box drawn row by
    // row would be laid over the letters of the row above it.
    //
    // Every box takes the same vertical band and its own horizontal span (see
    // computeLayout), so the stack differs in width and in nothing else.
    if (linePanel) {
        // Never so much padding that a box would hang off the canvas edge,
        // which is the same guard the block slab and the highlight box carry.
        const pad = Math.min(panelPad, Math.max(0, LEFT - panelBorder));
        const boxes = layout.rows.filter(row => !row.blank).map(row => [
            inkLeft(row) - pad, row.T + offsetY + row.panelTop + panelBorder,
            row.span.visualW + pad * 2,
            row.panelBottom - row.panelTop - panelBorder * 2,
            panelBorder,
        ]);

        // Every shadow, and only then every fill. Drawn box by box instead,
        // each one's shadow would be laid over the fill of the box before it —
        // which for separated boxes is invisible (nothing overlaps) and for a
        // joined background is the whole failure: a dark seam across the
        // shape at every line, in exactly the places the boxes were overlapped
        // to hide. One pass each puts every shadow under every fill, so what
        // survives is the shadow outside the union of them and nothing else.
        for (const box of boxes) add(slabUnder(...box));
        for (const box of boxes) add(slabFill(...box));
    }

    if (panel && !linePanel) {
        // ...but never so much padding that the slab would hang off the
        // canvas edge, which is the same guard the highlight box carries and
        // for the same reason: LEFT is all the room there is beside the block.
        const pad = Math.min(panel.padding, Math.max(0, LEFT - panel.border.width));
        const bw = panel.border.width;

        const fillX = blockLeft - pad, fillY = blockTop - pad;
        const fillW = blockW + pad * 2, fillH = blockH + pad * 2;

        add(slabUnder(fillX, fillY, fillW, fillH, bw));
        add(slabFill(fillX, fillY, fillW, fillH, bw));
    }

    // Every halo first, then every line — one pass each, not both per row.
    // A blur this wide reaches well past its own line: drawn row by row, the
    // second line's halo would land on top of the first line's letters and
    // veil them. The glow belongs under ALL the type, so all of it goes down
    // before any of the type does.
    //
    // Stacked rather than drawn once at a higher alpha: each pass compounds
    // into what is already there, so the falloff stays soft while the core
    // gains density — one pass at triple the opacity would instead read as a
    // hard bright rim.
    //
    // Each copy carries the glow BOTH as its own fill and as a shadow of the
    // same colour: the fill is the duplicate letterform, the shadow is that
    // letterform blurred. The solid part ends up covered by what is drawn
    // over it — strictly larger, whether that is an outlined line or a slab
    // of the same width — so what survives on screen is the blur alone.
    //
    // What throws the halo is whatever the block's OUTERMOST ink is. With a
    // sticker that is the slab, not the letters: a halo the shape of letters
    // sitting well inside an opaque slab escapes it only as a rim that
    // follows nothing you can see. So the copy is stroked in the glow's own
    // colour at the sticker's width, and the shadow is told to take the
    // stroke into account — a fabric shadow is cast by the fill alone
    // otherwise, which here would be the letters all over again.
    if (glow) {
        for (const row of layout.rows) {
            if (row.blank) continue;
            for (const seg of row.segs) {
                const glowColor = glow.color || (seg.hl ? highlight.color : style.color);
                const copyProps = sticker
                    ? { ...seg.props, ...strokeProps(sticker.width, glowColor) }
                    : glowProps(seg.style);
                for (let i = 0; i < (glow.layers || 1); i++) {
                    add(new fabric.Text(seg.text, {
                        ...alignedCopy(row, copyProps, seg.pen), fontSize: row.sz,
                        ...copyProps,
                        fill: glowColor, opacity: glow.opacity,
                        shadow: new fabric.Shadow({
                            color: glowColor, blur: glow.blur, offsetX: 0, offsetY: 0,
                            affectStroke: !!sticker,
                        }),
                        selectable: false, evented: false,
                    }));
                }
            }
        }
    }

    // The patch before the strokes, so that where the two meet it is the
    // stroke's antialiased edge that is laid over solid paint, and not a hard
    // edge laid over the photograph.
    if (patch) {
        add(new fabric.Image(patch.canvas, {
            left: patch.left, top: patch.top,
            selectable: false, evented: false, objectCaching: false,
        }));
    }

    // Then every slab, and all of them before any of the type. A sticker is
    // wide enough to reach its neighbours in both directions — that overlap is
    // what merges the separate strokes into one shape — so a slab drawn row by
    // row would be laid over the letters of the row above it.
    //
    // Stroked AND filled with the same paint. The stroke alone would leave the
    // inside of a wide letter as a hole straight through the slab; the real
    // letters go over the filled letterform in a moment anyway.
    if (sticker) {
        for (const row of layout.rows) {
            if (row.blank) continue;
            for (const seg of row.segs) {
                const paint = stickerFill(inkOrigin(row, seg.pen));
                const slabProps = stickerPropsOf(seg.props);
                add(new fabric.Text(seg.text, {
                    ...alignedCopy(row, slabProps, seg.pen), fontSize: row.sz,
                    ...slabProps, stroke: paint, fill: paint,
                    selectable: false, evented: false,
                }));
            }
        }
    }

    for (const row of layout.rows) {
        const { str, sz, vb, hl, boxed, pad } = row;
        if (row.blank) continue;
        const top = row.T + offsetY;

        // Where a piece of the row starts on the canvas. The pen is zero for a
        // row drawn as one string, so this is `textLeft` for every channel
        // that is not built out of runs (see `runRows`).
        const penLeft = (seg) => textLeft(row) + seg.pen;

        /**
         * The line, in its place, carrying whatever shadow this copy of it is
         * meant to carry.
         *
         * A highlighted line takes the highlight's colour whether or not a box
         * went under it — in a colour-only channel that recolouring IS the
         * highlight. Not so under a word-scope highlight: there the highlight
         * belongs to some of the words and not to the line, and recolouring
         * the line would repaint the words nobody picked.
         */
        const lineCopy = (shadow) => new fabric.Text(str, {
            left: textLeft(row), top, fontSize: sz, ...row.props,
            fill: (hl && !wordScope) ? highlightFill(textLeft(row), top) : style.color,
            shadow, selectable: false, evented: false,
        });

        /**
         * One piece of a row that is drawn as runs — its own text, in its own
         * face, at its own pen, painted by its own treatment.
         */
        const segCopy = (seg, shadow) => new fabric.Text(seg.text, {
            left: penLeft(seg), top, fontSize: sz, ...seg.props,
            fill: seg.hl ? highlightFill(penLeft(seg), top) : style.color,
            shadow, selectable: false, evented: false,
        });

        /**
         * One run of picked words, drawn on its own so that it can be drawn
         * over its box.
         *
         * Placed by the pen rather than by the ink: fabric starts a string at
         * `left`, so a piece of a line put at the point the pen had reached
         * by then lands exactly where it does in the whole. The measure is
         * the one the box beside it was built from (see wordSpans), which is
         * what keeps the two agreeing to the pixel — and a run always begins
         * and ends at a space, so no pair of letters is ever split apart.
         *
         * Painted with the highlight's own surface rather than the block's. On
         * every channel that has used this they are the same white, but they
         * are two keys because they answer two questions, and a run set in the
         * block's colour over the highlight's fill is the one place that would
         * show.
         */
        const runCopy = (run) => new fabric.Text(run.text, {
            left: textLeft(row) + run.pen, top, fontSize: sz, ...row.props,
            fill: highlightFill(textLeft(row) + run.pen, top), shadow: null,
            selectable: false, evented: false,
        });

        /**
         * The boxes behind a highlighted row: one round the whole of it under
         * line scope, one per run of picked words under word scope.
         *
         * Two rectangles each. Fabric casts a shadow from the fill, so a box
         * that carried both would throw its shadow and then paint over it —
         * the shadow rectangle is the same shape filled with nothing, and the
         * solid one goes on top of it.
         *
         * Both are derived from the run's own ink rather than from the widest
         * line in the block: a box sized to the block leaves a slab of empty
         * colour past the end of anything shorter.
         */
        const addBoxes = (runs) => {
            const boxH = row.boxBottom - row.boxTop;
            const by = top + row.boxTop;
            for (const run of runs) {
                const boxLeft = run.left - pad;
                const boxW = run.width + pad * 2;
                const corners = rounded(highlight.radius, boxW, boxH);
                add(new fabric.Rect({
                    left: boxLeft, top: by, width: boxW, height: boxH, ...corners,
                    fill: "transparent",
                    shadow: shadowOf(highlight.shadow),
                    selectable: false, evented: false,
                }));
                add(new fabric.Rect({
                    left: boxLeft, top: by, width: boxW, height: boxH, ...corners,
                    fill: highlight.fill, selectable: false, evented: false,
                }));
            }
        };

        /**
         * A row built out of runs draws each of its pieces exactly once, in
         * its own face and its own paint.
         *
         * There is no whole-line copy laid under the boxes here, and on a
         * two-face row there could not be: the pieces are in different faces,
         * so no single string covers them. Which is also why none is needed —
         * the trick that copy exists for (a picked word losing its shadow
         * while its neighbours keep theirs, without splitting the line) is
         * answered directly once every piece is its own object.
         */
        if (runRows) {
            if (boxed) {
                addBoxes(row.segs.filter(seg => seg.hl).map((seg) => {
                    const ink = measureVisualBounds(seg.text, sz, seg.style);
                    return { left: penLeft(seg) + ink.left, width: ink.visualW };
                }));
            }
            for (const seg of row.segs) {
                // A piece that is only the space between two runs has nothing
                // to draw, and a shadow under nothing is still a shadow.
                if (!/\S/.test(seg.text)) continue;
                // Whatever sits on a box takes no shadow of its own: the box
                // is what lifts it off the photo now, and a shadowed letter
                // inside a shadowed box reads as a printing fault. Everything
                // else keeps the shadow its own treatment carries.
                add(segCopy(seg, (boxed && seg.hl)
                    ? null
                    : shadowOf(seg.hl ? highlight.shadow : style.shadow)));
            }
            continue;
        }

        // A word picked out loses its own drop shadow. The box behind it is
        // what lifts it off the photo now, and a shadow on the letters inside
        // a shadowed box is one lift too many — it reads as a printing fault
        // rather than as depth. The words BESIDE it keep theirs: nothing has
        // changed for them, they are still white type on a photograph.
        //
        // Which is two answers for one line, and the line has to stay one
        // text object — split into words, the space between them would come
        // from somewhere other than the font. So the line is drawn whole and
        // shadowed UNDER the boxes, where every picked word and its shadow
        // end up covered by that word's own box, and only the picked RUNS are
        // drawn again over the top, unshadowed.
        //
        // Only the runs, and not the whole line a second time: two copies of
        // the same white glyph in the same place do not cancel out, they
        // compound in the antialiased rim — measured at up to 49/255 on the
        // edge pixels, which is a letter that has quietly got harder edges
        // than the same letter one line below it. Nothing here paints a glyph
        // twice.
        if (boxed && row.clearWords) add(lineCopy(shadowOf(style.shadow)));

        // Where the boxes go across the line: line scope is one box round all
        // of it, word scope one per run of picked words, each measured from
        // that run's own ink (see pickedSpans).
        if (boxed) {
            addBoxes(wordScope
                ? row.words.map(w => ({ left: inkLeft(row) + w.start, width: w.end - w.start }))
                : [{ left: inkLeft(row), width: vb.visualW }]);
        }

        // Wherever a box was drawn it is the box that casts, so the letters
        // on it get no shadow of their own — one rule, whether the box covers
        // a line or a word. A word-scope row has already had everything the
        // boxes do not cover drawn under them, so all that is left is what
        // goes over: the runs.
        if (boxed && wordScope) row.words.forEach(run => add(runCopy(run)));
        else add(lineCopy(boxed ? null : shadowOf(hl ? highlight.shadow : style.shadow)));
    }

    // Nothing has reached the canvas until here: the objects go on as they
    // are, or as one group carrying the user's drag and resize.
    const bounds = place(canvas, parts,
        { left: blockLeft, top: blockTop, w: blockW, h: blockH }, opts.placement);
    if (opts.interactive && style.movable) addTitleHandles(canvas, bounds);
    return bounds;
}

/**
 * Puts the finished objects on the canvas, transformed by whatever the user
 * has done to the block, and answers with the box its ink ended up in.
 *
 * The transform is applied to the objects as a GROUP rather than being folded
 * into the numbers that produced them, and that is the whole design. A resize
 * that re-fitted the type to a wider target would re-flow the block — lines
 * would take different sizes, a short line would stop being short, the height
 * budget would pull the whole thing back in — so dragging the handle would
 * change the title's composition and not merely its size. Scaling what was
 * already laid out is what makes a resize a resize.
 *
 * Anchored at the block's own top-left corner, so the numbers stored on the
 * frame mean what they say: `dx`/`dy` is how far that corner has moved, and
 * `scale` is what the block's width was multiplied by, whatever the objects
 * inside it happen to extend to.
 *
 * With nothing to do it does nothing — the objects are added one by one,
 * which is the path every channel that cannot be moved takes, unchanged.
 */
function place(canvas, parts, box, placement) {
    const dx = (placement && placement.dx) || 0;
    const dy = (placement && placement.dy) || 0;
    const scale = (placement && placement.scale) || 1;
    const bounds = {
        left: box.left + dx, top: box.top + dy,
        width: box.w * scale, height: box.h * scale,
        baseW: box.w, baseH: box.h, scale,
    };

    if (!dx && !dy && scale === 1) {
        for (const part of parts) canvas.add(part);
        return bounds;
    }

    // objectCaching off, and not as an optimisation. A cached group is
    // rasterised into a canvas sized from its objects' bounding boxes, and a
    // shadow is drawn outside those — the halo especially, which reaches 55px
    // past letters it is deliberately not measured into. Cached, that light
    // comes out clipped to a rectangle around the type.
    const group = new fabric.Group(parts, {
        selectable: false, evented: false, objectCaching: false,
    });
    // Fabric has just computed the group's own bounding box, which is not the
    // ink box: it includes every stroke, and a sticker channel's is 25px
    // wider on every side. The scaling is about the INK's corner, so the
    // group's corner has to be carried through the same transform to keep the
    // two in the same place.
    const gx = group.left, gy = group.top;
    group.set({
        scaleX: scale, scaleY: scale,
        left: box.left + (gx - box.left) * scale + dx,
        top: box.top + (gy - box.top) * scale + dy,
    });
    group.setCoords();
    canvas.add(group);
    return bounds;
}

// How far outside the ink the dashed box sits. Zero would draw it through the
// letters' own shadow and along the edge of every highlight box, which reads
// as part of the design rather than as a selection.
export const TITLE_CHROME_PAD = 8;

/**
 * The dashed box and the resize handle, drawn OUTSIDE the group.
 *
 * Outside because they are not part of the title: inside, they would be
 * scaled with it, so the handle would grow as the user dragged it and the
 * dashed line would thicken. They are chrome, and chrome is the same size
 * whatever it is wrapped around.
 *
 * Same shapes and the same blue as the image overlay's (see
 * compose.addLogoOverlay) — two things that are dragged and resized the same
 * way should not look like two different mechanisms.
 */
export function addTitleHandles(canvas, bounds) {
    const pad = TITLE_CHROME_PAD;
    const left = bounds.left - pad, top = bounds.top - pad;
    const w = bounds.width + pad * 2, h = bounds.height + pad * 2;
    canvas.add(new fabric.Rect({
        left, top, width: w, height: h, fill: "transparent",
        stroke: "#1b2fea", strokeWidth: 2, strokeDashArray: [6, 4],
        selectable: false, evented: false,
    }));
    canvas.add(new fabric.Rect({
        left: left + w - HANDLE_SIZE / 2, top: top + h - HANDLE_SIZE / 2,
        width: HANDLE_SIZE, height: HANDLE_SIZE,
        fill: "#1b2fea", stroke: "#fff", strokeWidth: 2, rx: 3, ry: 3,
        selectable: false, evented: false,
    }));
}

/**
 * Hit-tests a canvas-space point against a title's box and its resize handle
 * — the same regions addTitleHandles draws, so what the user aims at is what
 * they hit. The counterpart of compose.logoHitTest, and deliberately the same
 * shape as it.
 */
export function titleHitTest(bounds, cx, cy) {
    if (!bounds) return null;
    const pad = TITLE_CHROME_PAD;
    const left = bounds.left - pad, top = bounds.top - pad;
    const w = bounds.width + pad * 2, h = bounds.height + pad * 2;
    const hs = HANDLE_SIZE;
    if (cx >= left + w - hs && cx <= left + w + hs / 2 && cy >= top + h - hs && cy <= top + h + hs / 2) return "resize";
    if (cx >= left && cx <= left + w && cy >= top && cy <= top + h) return "move";
    return null;
}
