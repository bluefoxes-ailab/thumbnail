/**
 * caption.js — one sentence, broken into the lines it will be set on.
 *
 * Two channels' worth of behaviour live here, and which one runs is decided by
 * whether the channel states a `fontSize`. A channel that does not is fitted,
 * and the long note below on choosing the number of lines by height is about
 * that case. A channel that does wants the fewest lines the sentence fits on,
 * which is ordinary wrapping — see the top of captionLines.
 *
 * The backend hands each captured frame the sentence that was being said over
 * it (see backend/transcriber.py). That is a single string, and the title
 * renderer does not wrap: it sets the lines it is given, one per row, each
 * fitted to the block's width. So something has to decide where the sentence
 * breaks, and this is it.
 *
 * The breaks it makes are only a STARTING point. The Snapchat pack's titles
 * are freeform (see `freeform` in channels.js), which means the line breaks
 * belong to the user — they get the caption pre-filled in the text box and
 * every break in it is theirs to move. Nothing here runs again after that.
 *
 * ## Why the number of lines is chosen by height
 *
 * This channel sets each line to the full width of the block at its own size
 * (`uniformSize: false`), so the two edges of the block are straight and a
 * line's size is decided by how much text is on it: four words across 540px
 * are large, eleven words across the same 540px are small.
 *
 * That inverts the usual wrapping question. Fitting as many words as possible
 * onto each line — what every text wrapper does — produces the FEWEST lines
 * and therefore the SMALLEST type, which is the opposite of what a caption
 * wants. What it wants is the most lines the frame can hold, because that is
 * the largest the words can be set.
 *
 * So the number of lines is whatever fits the height the channel allows
 * (`maxBlockHeightRatio`), and the words are then distributed evenly over
 * them. One useful consequence: every caption ends up filling roughly the
 * same band at the bottom of the frame, whether it is five words or eighteen.
 * A grid of forty stills reads as one set rather than as forty different
 * decisions about type size.
 *
 * The height is estimated here rather than asked of the renderer, and the
 * estimate is made out of the renderer's own measurements — the same
 * per-line fit (fontSizeForInkWidth) and the same ink bounds
 * (measureVisualBounds) that addTextOverlay lays the block out with. It can
 * still be a little off, and it is allowed to be: the renderer pulls a block
 * that overflows back in by itself, so the cost of an optimistic answer here
 * is a caption set slightly smaller than it could have been, not one hanging
 * off the frame.
 */

import { CW, CH } from "./config.js";
import { titleStyle } from "./channels.js";
import { measuredWidth, measureVisualBounds, fontSizeForInkWidth } from "./text.js";

// The size every relative measurement below is taken at. Sizes are linear in
// the string, so one measurement at a reference size scales to any other —
// the same property fontSizeForInkWidth is built on.
const REF_SIZE = 100;


/**
 * The sentence split over `count` lines with as little difference between
 * their lengths as possible.
 *
 * Even lengths are not a matter of taste here: every line is stretched to the
 * same width, so a line half as long as its neighbour is set half again as
 * large, and a block whose rows are three different sizes reads as three
 * different thoughts. Balanced lines come out at nearly one size, which is
 * what makes the block look like one sentence.
 *
 * Exact, not greedy. A greedy fill packs the early lines and leaves the last
 * one holding two words — the single worst case for a channel that stretches
 * it. This is the standard minimum-raggedness dynamic program: `best[k][i]`
 * is the smallest total squared line width achievable by setting the first
 * `i` words on `k` lines. Squared, because the sum of the widths is very
 * nearly fixed however the words are divided, and minimising the sum of
 * squares under a fixed sum is the same thing as minimising the spread.
 *
 * The strings are measured with their spaces in them (`widthOf` joins before
 * measuring) so the answer accounts for what is actually set, not for a sum
 * of word widths that no line ever has.
 */
function balancedSplit(words, count, widthOf) {
    if (count <= 1) return [words.join(" ")];
    if (words.length <= count) return [...words];

    const n = words.length;
    // Line width for words [i, j), squared. Precomputed because the loop below
    // asks for the same spans repeatedly.
    const cost = [];
    for (let i = 0; i < n; i++) {
        cost[i] = [];
        for (let j = i + 1; j <= n; j++) {
            const w = widthOf(words.slice(i, j).join(" "));
            cost[i][j] = w * w;
        }
    }

    // best[k][i] and the break that produced it. Infinity is "no way to set
    // the first i words on k lines" — which is every k above i, since a line
    // with nothing on it is not a line.
    const best = [new Array(n + 1).fill(Infinity)];
    const cut = [new Array(n + 1).fill(0)];
    best[0][0] = 0;
    for (let k = 1; k <= count; k++) {
        best[k] = new Array(n + 1).fill(Infinity);
        cut[k] = new Array(n + 1).fill(0);
        for (let i = k; i <= n; i++) {
            for (let j = k - 1; j < i; j++) {
                if (best[k - 1][j] === Infinity) continue;
                const total = best[k - 1][j] + cost[j][i];
                if (total < best[k][i]) {
                    best[k][i] = total;
                    cut[k][i] = j;
                }
            }
        }
    }

    const lines = [];
    let end = n;
    for (let k = count; k >= 1; k--) {
        const start = cut[k][end];
        lines.unshift(words.slice(start, end).join(" "));
        end = start;
    }
    return lines;
}


/**
 * How tall the block these lines make will be, in canvas pixels.
 *
 * Built the way addTextOverlay builds it: each line is fitted to the block's
 * width at its own size, and its height is its own ink at that size. The gaps
 * between them are the channel's `lineGap`, which is measured between the
 * lines' visible edges — the same number this adds up.
 *
 * Every measurement is taken once at REF_SIZE and scaled, which is both
 * cheaper and exactly what the renderer's own fit does, so the two cannot
 * disagree about a line by more than rounding.
 */
function blockHeight(lines, cased, targetW, lineGap) {
    let height = 0;
    for (const line of lines) {
        const str = cased(line);
        const size = fontSizeForInkWidth(str, targetW);
        height += measureVisualBounds(str, REF_SIZE).visualH * size / REF_SIZE;
    }
    return height + lineGap * Math.max(0, lines.length - 1);
}


/**
 * A sentence as the lines a frame's title will be set on — at most as many as
 * the channel has slots for, and as many of those as the frame can hold.
 *
 * Answers an empty array for an empty caption, which is what a frame cut from
 * silence gets: an empty text box rather than a line of nothing.
 *
 * Called once per frame as a capture run's frames land, and never again — see
 * the module header on whose the breaks are after that.
 */
export function captionLines(text) {
    const words = (text || "").trim().split(/\s+/).filter(Boolean);
    if (!words.length) return [];

    const style = titleStyle();
    const layout = style.layout;
    const cased = (l) => (style.uppercase ? l.toUpperCase() : l);
    const targetW = CW * layout.widthRatio;
    const budget = CH * layout.maxBlockHeightRatio;

    // A channel that states its size is not playing the game above at all
    // (see `fontSize` in channels.js): more lines no longer means larger type,
    // so there is nothing to be won by using them up. What is left is the
    // ordinary question a wrapper asks — the FEWEST lines the sentence fits
    // on at the size it is being set at.
    //
    // Upward and taking the first fit, which is the mirror image of the search
    // below and lands on the same kind of answer from the other end. Not
    // capped by `layout.lines`: that cap exists because more lines was the
    // knob that made type bigger, and stopping at four here would not make a
    // long sentence smaller, it would run it off both sides of the frame. A
    // caption that needs six lines at 20px is six lines at 20px, and the block
    // grows from its margin INTO the picture rather than off the edge of it —
    // upward from the foot of the frame, downward from the head of it,
    // whichever end the channel hung it from (see `vAlign` in channels.js).
    if (layout.fontSize) {
        const scale = layout.fontSize / REF_SIZE;
        // Two ceilings, and a line has to be under both. Width is what the
        // frame can hold; `maxChars` is what a viewer reads at a glance (see
        // channels.js). A channel that states only one of them is held to
        // that one — an absent `maxChars` is not a limit of zero.
        const room = layout.maxChars || Infinity;
        const fits = (lines) => lines.every(l =>
            measuredWidth(cased(l)) * scale <= targetW && cased(l).length <= room);
        for (let count = 1; count < words.length; count++) {
            const candidate = balancedSplit(words, count, measuredWidth);
            if (fits(candidate)) return candidate;
        }
        // A word too long to fit the block on its own — a URL, a hashtag —
        // has nowhere left to break, and one word per line is the shortest
        // any of them can be set. The renderer sets it as it is; nothing here
        // may drop a word, or break one, to make a sentence fit: a caption
        // that is one character over its character limit is a caption, and a
        // caption with a word cut in half is not.
        return balancedSplit(words, words.length, measuredWidth);
    }

    // Never more lines than the channel keeps slots for, and never more than
    // there are words — a line has to have something on it.
    const most = Math.min(layout.lines, words.length);

    // Downward, taking the first that fits: more lines is larger type (see the
    // module header), so the first fit found this way is the largest the block
    // can be set at.
    for (let count = most; count >= 2; count--) {
        const candidate = balancedSplit(words, count, measuredWidth);
        if (blockHeight(candidate, cased, targetW, layout.lineGap) <= budget) return candidate;
    }
    // Nothing fitted, so the shortest block these words can make. It may still
    // be over the budget — a caption of two very long words has nowhere else
    // to go — and the renderer pulls it in, which is the one place that can do
    // it without changing where the lines break.
    return balancedSplit(words, 1, measuredWidth);
}
