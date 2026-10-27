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
//      Beside it, "Max characters per frame" — a whole number, 80 unless the
//      user changes it — the most text any one frame may be given.
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
//   3. A sentence still longer than the user's maximum is divided at a
//      clause mark — , ; : — – — and only where both halves keep at least
//      MIN_WORDS. A half that opens with a capital is preferred (a name, a
//      new clause with its own subject), then the stronger marks over the
//      comma, then the most even division. Recursively, until every piece
//      fits or has nowhere left to come apart.
//   4. A piece that still does not fit is divided between words, into as few
//      pieces as the maximum allows and as even as they can be made (see
//      wrapWords).
//
// Rule 4 is the one that makes fragments, and it is there because the
// maximum is a ceiling the user set, not a target: it used to be that a
// sentence with no clause mark stayed whole and ran long, and a user who
// typed 80 and found 140 characters on a frame would rightly read the field
// as broken. The clause marks are tried first so a word cut is only ever the
// last resort. No word is ever changed, dropped or cut in half — a single
// word longer than the maximum stays whole, since there is no place in it to
// break that would still be the user's word.
//
// ## Where the pieces go
//
// One per frame, in order, from the first frame of the grid with no frame
// skipped (see dealOut): ten pieces over forty stills land on frames 1 to 10,
// and 11 to 40 stay bare. It used to spread them evenly — every fourth still —
// and that read as frames the text had missed, which is what a gap between
// captioned stills looks like to anyone who did not write the spreading. The
// grid is the frames that actually LANDED, never the slider's count: the
// backend hands back fewer than asked for when frames are rejected as blurred
// or blank.
//
// More pieces than stills is answered by joining pieces back together, the
// shortest neighbouring pair first and never across a line break the user
// typed while any other pair is left — but never past the maximum either. A
// text that will not fit on the grid even so is placed as far as it goes and
// the rest is reported as left over (see main.splitMyText), rather than the
// ceiling quietly giving way.

import { el } from "./config.js";

// What "Max characters per frame" starts at, and what it falls back to if
// the field is emptied. A mirror of transcriber.CAPTION_CHARS, which says at
// length why eighty: four lines of the caption type at twenty characters
// each, taken in at a glance on a phone. The markup's value="80" is this.
export const CAPTION_CHARS = 80;

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
 * One sentence divided into pieces of at most `max` characters: at its
 * clause marks where it has any, between words where it has not. See the
 * module header for the order of preference between cuts.
 *
 * A comma followed by lower case is the weakest clause cut there is — what
 * follows it is as often an aside ("born in Tennessee, the fourth of twelve
 * children") as a clause — and it is ranked last. It used to be refused
 * outright below twice the length, leaving the sentence whole; with the
 * maximum a ceiling, the alternative is now a cut between two arbitrary
 * words, and the comma is still the better of the two.
 *
 * The pieces keep the mark they were cut at, so that pieces joined back
 * together by dealOut read exactly as the user wrote them; it is stripped
 * only from what is finally set (see shown).
 */
function divide(sentence, max) {
    if (sentence.length <= max) return [sentence];

    let best = null;
    for (const mark of sentence.matchAll(CLAUSE_MARK)) {
        const cut = mark.index + mark[0].length;
        const head = sentence.slice(0, cut).trim();
        const tail = sentence.slice(cut).trim();
        if (wordCount(head) < MIN_WORDS || wordCount(tail) < MIN_WORDS) continue;
        const comma = mark[0][0] === ",";
        const lower = startsLower(tail);
        // Lower is better, compared in this order: a half with its own
        // subject, then a strong mark over a comma, then evenness.
        const rank = [lower ? 1 : 0, comma ? 1 : 0, Math.abs(head.length - tail.length)];
        if (!best || compare(rank, best.rank) < 0) best = { rank, head, tail };
    }
    if (!best) return wrapWords(sentence, max);
    return [...divide(best.head, max), ...divide(best.tail, max)];
}

/**
 * A piece with no clause mark left to cut at, divided between words into the
 * fewest pieces of at most `max` characters — and, among those, the most even
 * ones. Packing greedily to `max` alone would leave "...a long sentence
 * that" on one frame and "ends." on the next; packing to the smallest limit
 * that still needs no more pieces spreads the words out instead.
 *
 * A word longer than `max` gets a piece to itself and keeps all its letters.
 */
function wrapWords(piece, max) {
    const words = piece.split(" ");
    const pack = (limit) => {
        const out = [];
        let line = "";
        for (const word of words) {
            if (line && line.length + 1 + word.length > limit) { out.push(line); line = word; }
            else line = line ? `${line} ${word}` : word;
        }
        if (line) out.push(line);
        return out;
    };
    const fewest = pack(max).length;
    if (fewest <= 1) return [piece];
    for (let limit = Math.ceil(piece.length / fewest); limit < max; limit++) {
        const even = pack(limit);
        if (even.length <= fewest) return even;
    }
    return pack(max);
}

function compare(a, b) {
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
    return 0;
}

/**
 * The user's text as the pieces it will be set in, each with the line of the
 * text (`para`) it came from — the joining in dealOut needs to know which
 * neighbours the user separated with a line break of their own.
 *
 * `raw` is the piece exactly as written, mark and all; `shown` is what a
 * still carries, with a DANGLING mark taken off its end. A piece that would
 * show as nothing — a stray dash on a line of its own — is not a piece: set
 * on a frame it would be a blank between two captioned ones.
 */
function piecesOf(text, max) {
    const pieces = [];
    (text || "").split(/\r?\n/).forEach((line, para) => {
        const paragraph = line.replace(/\s+/g, " ").trim();
        if (!paragraph) return;
        for (const sentence of sentencesOf(paragraph)) {
            for (const raw of divide(sentence, max)) {
                if (shown({ raw })) pieces.push({ raw, para });
            }
        }
    });
    return pieces;
}

const shown = (piece) => piece.raw.replace(DANGLING, "").trim();

/** The text's pieces as plain strings — what the pop-up previews. */
export function splitText(text, max = CAPTION_CHARS) {
    return piecesOf(text, max).map(shown);
}

/**
 * The user's text dealt out over `count` frames, one piece per frame from the
 * first, none over `max` characters. See the module header.
 *
 * Returns `pieces`, in the order they go on the grid — never more of them
 * than there are frames — and `leftover`, how many would not fit on the grid
 * even joined as far as the maximum allows. What sets them on the frames is
 * main.splitMyText, which checks the order again as it goes.
 */
export function dealOut(text, count, max = CAPTION_CHARS) {
    const pieces = piecesOf(text, max);
    if (count <= 0) return { pieces: [], leftover: pieces.length };

    // Too many pieces for the grid: join the cheapest neighbouring pair whose
    // join still fits under the maximum, until they fit the grid or no such
    // pair is left. A pair the user separated with a line break costs more
    // than any pair they did not, so those are joined last.
    while (pieces.length > count) {
        let at = -1, cheapest = Infinity;
        for (let i = 0; i < pieces.length - 1; i++) {
            const joined = shown({ raw: `${pieces[i].raw} ${pieces[i + 1].raw}` });
            if (joined.length > max) continue;
            const cost = joined.length + (pieces[i].para !== pieces[i + 1].para ? 1e6 : 0);
            if (cost < cheapest) { cheapest = cost; at = i; }
        }
        if (at < 0) break;
        pieces.splice(at, 2, { raw: `${pieces[at].raw} ${pieces[at + 1].raw}`, para: pieces[at].para });
    }

    return {
        pieces: pieces.slice(0, count).map(shown),
        leftover: Math.max(0, pieces.length - count),
    };
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

/**
 * "Max characters per frame" as a whole number of at least 1, or null when
 * the field holds nothing usable — main.splitMyText says so rather than
 * splitting against a number the user did not type.
 */
export function maxChars() {
    const value = el("maxCharsInput").value.trim();
    if (!/^\d+$/.test(value)) return null;
    const n = parseInt(value, 10);
    return n >= 1 ? n : null;
}

// The field takes whole numbers only. type="number" alone does not see to
// that: every browser lets "e", "-", "+" and a decimal point through, since
// they are all part of some number, and a pasted "80 chars" lands as typed in
// some of them. Those keys are refused as they are pressed, and anything that
// still gets in — a paste, a drop — is cut down to its digits.
el("maxCharsInput").addEventListener("keydown", (e) => {
    if (!e.ctrlKey && !e.metaKey && e.key.length === 1 && !/\d/.test(e.key)) e.preventDefault();
});
el("maxCharsInput").addEventListener("input", (e) => {
    const digits = e.target.value.replace(/\D/g, "");
    if (digits !== e.target.value) e.target.value = digits;
});
// An emptied field goes back to the default when the user leaves it, so the
// number on screen is always the one the next split will use.
el("maxCharsInput").addEventListener("blur", (e) => {
    if (maxChars() === null) e.target.value = String(CAPTION_CHARS);
});
