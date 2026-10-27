// ── Rearranging a frame fetch's grid ──────────────────────────────────────
//
// A frame fetch hands back its stills in the order they happen in the video,
// and that order is the deliverable: the files are numbered after it (see
// capture.frameName), and the user's own text is split along it. It is also
// only the video's order, not necessarily the one the user wants to publish,
// so the cards can be dragged into another.
//
// At any point of the run, including while the edit pass is still turning the
// grey cards one by one. That is safe because nothing here moves a frame
// record: the arrangement is a list laid over the records (state.gridOrder),
// and every piece of work in flight keeps addressing its frame by the index
// it started with. A pending card can be dragged and dropped on like any
// other — it is only its CLICK that waits for the edit (see main.renderGrid).
//
// Native drag and drop, with the card itself moving through the grid while it
// is held: the gap it leaves and the gap it would fill are the whole of the
// feedback, and the user sees the arrangement they will get before they let
// go. The card under the pointer trades places with the one being held —
// rather than "before or after, by which half of it the pointer is over",
// which in a grid of equal cards makes a card dither back and forth across
// the boundary it was just moved over.
//
// Let go anywhere outside the grid, or press Escape, and the grid goes back
// to how it was when the drag began. Only a drop inside the grid is a choice.
//
// Thumbnail runs are left alone. Their grid is a set of alternatives to pick
// from, and an order among alternatives means nothing.

import { el } from "./config.js";
import { gridOrder, setGridOrder } from "./state.js";
import { isCapture } from "./capture.js";

let _held = null;      // the card being dragged
let _before = null;    // the grid's card order when the drag began, to undo to
let _dropped = false;  // whether it was let go inside the grid

const cardsOf = (grid) => [...grid.querySelectorAll(".frame-card")];
const indexOf = (card) => Number(card.dataset.index);

/**
 * Makes the frames grid rearrangeable. `onReordered` is called once per drop
 * that changed the order, after state.gridOrder holds the new one.
 */
export function initReorder(onReordered) {
    const grid = el("framesGrid");

    grid.addEventListener("dragstart", (e) => {
        const card = e.target.closest && e.target.closest(".frame-card");
        if (!card || !isCapture()) return;
        _held = card;
        _before = cardsOf(grid);
        _dropped = false;
        e.dataTransfer.effectAllowed = "move";
        // Firefox starts no drag without some data on it.
        e.dataTransfer.setData("text/plain", card.dataset.index);
        // Dimmed on the next frame, after the browser has taken its picture
        // of the card for the pointer to carry — dimmed first, the picture
        // would be of a ghost.
        requestAnimationFrame(() => card.classList.add("dragging"));
    });

    grid.addEventListener("dragover", (e) => {
        if (!_held) return;
        e.preventDefault();   // what makes the grid a place a drop can happen
        e.dataTransfer.dropEffect = "move";
        const over = e.target.closest(".frame-card");
        if (!over || over === _held) return;
        // Into the hovered card's place: after it when the held card comes
        // from earlier in the grid, before it when from later.
        const cards = cardsOf(grid);
        const forward = cards.indexOf(_held) < cards.indexOf(over);
        grid.insertBefore(_held, forward ? over.nextSibling : over);
    });

    grid.addEventListener("drop", (e) => {
        if (!_held) return;
        e.preventDefault();
        _dropped = true;
    });

    grid.addEventListener("dragend", () => {
        if (!_held) return;
        _held.classList.remove("dragging");
        if (!_dropped) {
            // Undone in full, from the arrangement the drag began with:
            // putting back only the held card would be wrong whenever the
            // ones it passed over had shifted under it.
            for (const card of _before) grid.appendChild(card);
        } else {
            const order = cardsOf(grid).map(indexOf);
            const changed = order.some((i, p) => i !== gridOrder()[p]);
            setGridOrder(order);
            if (changed) onReordered();
        }
        _held = _before = null;
    });
}
