// ── My text, on a "Frames only" run ───────────────────────────────────────
//
// "Frames with text" captions every still with what was being said over it;
// "Frames only" hands the stills back bare. This is what a "Frames only" run
// offers on top: the USER'S words on them — a script, a set of facts, a story
// already written — broken into pieces and dealt out over the grid in order.
//
// The workflow, in the order the user meets it:
//
//   1. They pick "Frames only" and process the video. Nothing is transcribed
//      or rewritten, because nothing the video says will be used.
//   2. A text box appears under the progress bar the moment the run starts,
//      so the text can be written or pasted while the video processes.
//   3. Under it, "Split the text" — locked, with a bubble saying why, until
//      every frame has finished loading (see main.runPipeline, which unlocks
//      it when editor.applyEditToAll reaches the end of the grid). The split
//      is computed against the frames that arrived, and until the last one is
//      in, that number is not settled.
//   4. Clicking it sets the pieces as the frames' captions. Clicking again
//      after editing the text deals it out afresh, replacing what is there.
//
// The text never leaves the page. A run picked up after a reload shows an
// empty box, because what was typed lived in the page that was closed.
//
// ## Where the text is broken
//
// By rule, not by model, and the rules are the ones the automatic captions
// already live under (see transcriber.split_caption), for the reason they were
// arrived at there: a piece of text alone on a still has to read as a
// sentence, because the viewer sees one still at a time and cannot repair a
// fragment from the one before it.
//
//   1. A line break the user typed is always a break. They put it there.
//   2. A sentence end — . ! ? … — is a break, unless what follows starts in
//      lower case or the mark closes an abbreviation ("Dr.", "e.g."). Case is
//      the cheap, language-free signal for "something new starts here".
//   3. A sentence still longer than CAPTION_CHARS is divided at a clause mark
//      — , ; : — – — and only where both halves keep at least MIN_WORDS. A
//      half that opens with a capital is preferred (a name, a new clause with
//      its own subject), then the stronger marks over the comma, then the
//      most even division. A comma followed by lower case is a cut only in a
//      sentence too long for two stills (see divide). Recursively, until
//      every piece fits or has nowhere left to come apart.
//
// A sentence with no such place stays whole. It is too long for one still
// rather than two pieces that are not sentences, and the renderer wraps it
// (see caption.captionLines), so nothing is lost. No word is ever changed,
// dropped or cut in half: these are the user's words and only the breaks are
// this file's.
//
// ## Where the pieces go
//
// Spread over the grid evenly, in order (see spreadOver): ten pieces over
// forty stills land on every fourth one, starting at the first. The grid is
// the count the user chose on the slider, and the backend may hand back fewer
// than asked for when frames are rejected as blurred or blank — so the spread
// is computed against the frames that actually LANDED, never against the
// slider.
//
// More pieces than stills is answered by joining pieces back together, the
// shortest neighbouring pair first and never across a line break the user
// typed while any other pair is left: the text is all set, on fewer and longer
// captions, rather than its ending silently going missing.

import { el } from "./config.js";

// The length one still's caption is written to. A mirror of
// transcriber.CAPTION_CHARS, which says at length why eighty: four lines of
// the caption type at twenty characters each, taken in at a glance on a phone.
const CAPTION_CHARS = 80;

// ...and the length past which a sentence would take more than two stills
// even divided as evenly as it can be, from transcriber.CAPTION_CHARS_MAX.
// See divide for the one cut that is allowed only past it.
const CAPTION_CHARS_MAX = CAPTION_CHARS * 2;

// The fewest words a piece of a divided sentence may keep, from
// transcriber.MIN_WORDS: below it a half is a scrap, not a line.
const MIN_WORDS = 4;

// A sentence end: the mark, any closing quote or bracket after it, and the
// space that proves it is not the point inside "9.5".
const SENTENCE_END = /[.!?…]+["'”’)\]]*(?=\s)/g;

// A clause mark, under the same condition. The dashes are marks only when
// they stand between spaces or close a word — the hyphen in "well-known" is
// not one of them.
const CLAUSE_MARK = /[,;:—–]["'”’)\]]*(?=\s)/g;

// What a piece may not END on once it stands alone: a comma says "this goes
// on", which is true of the sentence and false of the still.
const DANGLING = /[,;:—–]+$/;

// Words a full stop belongs to rather than ends a sentence after. Short and
// deliberately so — a missed abbreviation costs one break the user can see in
// the preview and fix with a line break of their own, while a list that tried
// to be complete would start eating real sentence ends.
const ABBREVIATIONS = new Set([
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc",
    "e.g", "i.e", "no", "vol", "fig", "approx", "inc", "ltd", "co",
]);

const wordCount = (s) => (s.match(/\S+/g) || []).length;
const startsLower = (s) => {
    const first = s.trim()[0] || "";
    return first !== first.toUpperCase() && first === first.toLowerCase();
};

/** The text as sentences: hard breaks at line ends, soft ones at sentence marks. */
function sentencesOf(paragraph) {
    const sentences = [];
    let start = 0;
    for (const mark of paragraph.matchAll(SENTENCE_END)) {
        const end = mark.index + mark[0].length;
        const head = paragraph.slice(start, end).trim();
        const tail = paragraph.slice(end).trim();
        if (!tail) break;
        // "Dr. Smith", "e.g. this": the word before the mark owns the point.
        const lastWord = paragraph.slice(start, mark.index).trim().split(" ").pop().toLowerCase();
        const isAbbreviation = mark[0][0] === "." &&
            (ABBREVIATIONS.has(lastWord) || /^[a-z]$/i.test(lastWord));
        if (isAbbreviation || startsLower(tail)) continue;
        sentences.push(head);
        start = end;
    }
    const rest = paragraph.slice(start).trim();
    if (rest) sentences.push(rest);
    return sentences;
}

/**
 * One sentence divided into pieces that each fit a still, or left whole when
 * it has nowhere to come apart. See the module header for the order of
 * preference between cuts.
 *
 * A comma followed by lower case is the weakest cut there is, and it is only
 * taken from a sentence too long for even two stills (CAPTION_CHARS_MAX). What
 * follows such a comma is as often an aside — "born in Tennessee, the fourth
 * of twelve children" — as a clause, and an aside alone on a still is a
 * caption of nothing. Below that length the sentence stays whole: long, but a
 * sentence.
 *
 * The pieces keep the mark they were cut at, so that pieces joined back
 * together by spreadOver read exactly as the user wrote them; it is stripped
 * only from what is finally set (see piecesOf).
 */
function divide(sentence) {
    if (sentence.length <= CAPTION_CHARS) return [sentence];

    let best = null;
    for (const mark of sentence.matchAll(CLAUSE_MARK)) {
        const cut = mark.index + mark[0].length;
        const head = sentence.slice(0, cut).trim();
        const tail = sentence.slice(cut).trim();
        if (wordCount(head) < MIN_WORDS || wordCount(tail) < MIN_WORDS) continue;
        const comma = mark[0][0] === ",";
        const lower = startsLower(tail);
        if (comma && lower && sentence.length <= CAPTION_CHARS_MAX) continue;
        // Lower is better, compared in this order: a half with its own
        // subject, then a strong mark over a comma, then evenness.
        const rank = [lower ? 1 : 0, comma ? 1 : 0, Math.abs(head.length - tail.length)];
        if (!best || compare(rank, best.rank) < 0) best = { rank, head, tail };
    }
    if (!best) return [sentence];
    return [...divide(best.head), ...divide(best.tail)];
}

function compare(a, b) {
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
    return 0;
}

/**
 * The user's text as the pieces it will be set in, each with the line of the
 * text (`para`) it came from — the joining in spreadOver needs to know which
 * neighbours the user separated with a line break of their own.
 *
 * `raw` is the piece exactly as written, mark and all; `shown` is what a
 * still carries, with a DANGLING mark taken off its end.
 */
function piecesOf(text) {
    const pieces = [];
    (text || "").split(/\r?\n/).forEach((line, para) => {
        const paragraph = line.replace(/\s+/g, " ").trim();
        if (!paragraph) return;
        for (const sentence of sentencesOf(paragraph)) {
            for (const raw of divide(sentence)) pieces.push({ raw, para });
        }
    });
    return pieces;
}

const shown = (piece) => piece.raw.replace(DANGLING, "").trim();

/** The text's pieces as plain strings — what the pop-up previews. */
export function splitText(text) {
    return piecesOf(text).map(shown);
}

/**
 * The caption each of `count` frames gets, in grid order: the pieces spread
 * evenly over the grid, "" on the frames between them. See the module header.
 */
export function spreadOver(text, count) {
    const captions = new Array(Math.max(0, count)).fill("");
    const pieces = piecesOf(text);
    if (!count || !pieces.length) return captions;

    // Too many pieces for the grid: join the cheapest neighbouring pair until
    // they fit. A pair the user separated with a line break costs more than
    // any pair they did not, so those are joined last.
    while (pieces.length > count) {
        let at = 0, cheapest = Infinity;
        for (let i = 0; i < pieces.length - 1; i++) {
            const cost = pieces[i].raw.length + pieces[i + 1].raw.length
                       + (pieces[i].para !== pieces[i + 1].para ? 1e6 : 0);
            if (cost < cheapest) { cheapest = cost; at = i; }
        }
        pieces.splice(at, 2, { raw: `${pieces[at].raw} ${pieces[at + 1].raw}`, para: pieces[at].para });
    }

    pieces.forEach((piece, k) => {
        captions[Math.floor(k * count / pieces.length)] = shown(piece);
    });
    return captions;
}

// ── The box and its button ────────────────────────────────────────────────
//
// Two states, kept in one place so the button, the class the bubble hangs
// off, and the box's visibility can never disagree. What the click DOES is
// main.splitMyText's: it touches the frames, the captions and every view of
// them, which is main's business and none of this file's.

/**
 * Shows the box for a run whose frames the user's text can go on, or hides
 * it for any other run. Either way the button starts locked: a run that has
 * just started has no frames loaded yet. What was typed stays in the box, so
 * processing the video again does not mean writing it again.
 */
export function showMyText(on) {
    el("myTextSection").classList.toggle("hidden", !on);
    lockSplit(true);
}

/** Locks or unlocks "Split the text" — see the module header for when. */
export function lockSplit(locked) {
    el("splitTextBtn").disabled = locked;
    el("splitWrap").classList.toggle("locked", locked);
}

/** The text in the box, as typed. */
export const myText = () => el("myTextInput").value;
