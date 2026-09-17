import { CW, CH, THUMB_W, THUMB_H, HANDLE_SIZE, el } from "./config.js";
import {
    frames, linesFor, hlFor, logosFor, activeLogoFor, framePresentation, activeUrl,
    selectedIndex, channelFor, titleBoxFor, titleActiveFor, glowFor,
    layersFor, activeLayerFor, titleAlignFor, backgroundFor, figureGlowFor,
    textColorFor, panelColorFor, highlightColorFor,
} from "./state.js";
import { usingChannel, decorFor } from "./channels.js";
import { addTextOverlay } from "./text.js";
import { loadImage } from "./net.js";
import { hideReframeWait } from "./ui.js";

export let fabricCanvas = null;

export function initCanvas(onReady) {
    // A canvas already built for a DIFFERENT size is resized rather than
    // reused as it is: the canvas outlives a run, and two runs in one page
    // session can be two channels with two canvas shapes (see
    // config.setCanvasSize). Without this the second run's frames would be
    // drawn into the first run's aspect ratio.
    if (fabricCanvas) {
        if (fabricCanvas.getWidth() !== CW || fabricCanvas.getHeight() !== CH) {
            fabricCanvas.setDimensions({ width: CW, height: CH });
        }
        sizeOverlayCanvases();
        return fabricCanvas;
    }
    // StaticCanvas, not fabric.Canvas: nothing here uses Fabric's object
    // interaction (every object is selectable:false, and drag-to-reframe is
    // our own listener), and the interactive Canvas stacks a transparent
    // "upper-canvas" event layer over the content — which is what the
    // browser's right-click → "Save image as" used to hit, saving an empty
    // PNG. StaticCanvas renders straight into the visible canvas element, so
    // right-click saving works on it too.
    fabricCanvas = new fabric.StaticCanvas("thumbCanvas", { width: CW, height: CH });
    sizeOverlayCanvases();
    if (onReady) onReady();
    return fabricCanvas;
}

/**
 * Sizes the bare canvases stacked over the preview to match it.
 *
 * They are not fabric's, so nothing resizes them for us, and the markup can
 * only carry one size — 1280x720, which is the thumbnail canvas and was the
 * only one there had ever been. On a 540x960 run the live-crop layer was
 * therefore a 16:9 buffer under a 9:16 canvas: the drag preview was drawn into
 * it at 540x960, lost every row past 720, and the stylesheet then displayed
 * the whole buffer at the canvas's width — a small landscape panel in the top
 * left corner, with the bottom of the frame missing.
 *
 * Called from initCanvas because that is the one place that knows the canvas
 * geometry has been decided, on a first build and on a later change alike.
 */
function sizeOverlayCanvases() {
    const live = el("liveCropLayer");
    if (live && (live.width !== CW || live.height !== CH)) {
        live.width = CW;
        live.height = CH;
    }
}

// ── Image overlay ─────────────────────────────────────────────────────────
// Sizes/positions are stored in the same CW×CH canvas space every other
// layout number uses — the displayed canvas is only ever CSS-scaled, so
// canvas-space units stay proportionally correct at any window size.

/** Decodes a logo's dataURL once and caches it on the logo record itself. */
export function loadLogoImage(logo) {
    if (logo._img) return Promise.resolve(logo._img);
    return loadImage(logo.src).then(img => { logo._img = img; return img; });
}

/**
 * Adds the logo image at its stored position/scale. `interactive` also draws
 * a dashed bounding box and a resize handle — only wanted on the live preview
 * canvas, never on a composite that becomes a thumbnail or a download.
 *
 * `shadow` is for the one overlay that casts one: the cutout subject (see
 * `cutout.shadow` in channels.js). A fabric shadow follows the image's ALPHA,
 * not its bounding box, so on a picture with the background taken away it is
 * thrown by the person's silhouette — which is the only reason a shadow behind
 * an image overlay is worth anything at all. A logo with a solid rectangular
 * edge would get a rectangle, so nothing asks for one.
 */
export function addLogoOverlay(canvas, logo, img, interactive, shadow = null) {
    const w = logo.natW * logo.scale, h = logo.natH * logo.scale;
    const left = logo.x - w / 2, top = logo.y - h / 2;
    canvas.add(new fabric.Image(img, {
        left, top, width: logo.natW, height: logo.natH,
        scaleX: logo.scale, scaleY: logo.scale,
        // Mirrored HERE rather than by the backend, for the one overlay that
        // is ever mirrored: a flipped figure is the same cutout drawn the
        // other way round, and asking for it over the network would mean a
        // second restoration to produce pixels the canvas already holds. It
        // also makes the flip a property of one figure, which is what a
        // thumbnail carrying three of them needs it to be.
        flipX: !!logo.flip,
        shadow: shadow ? new fabric.Shadow({ ...shadow }) : null,
        selectable: false, evented: false,
    }));
    if (!interactive) return;
    canvas.add(new fabric.Rect({
        left, top, width: w, height: h, fill: "transparent",
        stroke: "#1b2fea", strokeWidth: 2, strokeDashArray: [6, 4],
        selectable: false, evented: false,
    }));
    const hs = HANDLE_SIZE;
    canvas.add(new fabric.Rect({
        left: left + w - hs / 2, top: top + h - hs / 2, width: hs, height: hs,
        fill: "#1b2fea", stroke: "#fff", strokeWidth: 2, rx: 3, ry: 3,
        selectable: false, evented: false,
    }));
}

/**
 * Hit-tests a canvas-space point against a logo's bounding box and resize
 * handle — shared by the drag-start and hover-cursor handlers so both agree
 * on exactly the same regions.
 */
export function logoHitTest(logo, cx, cy) {
    const w = logo.natW * logo.scale, h = logo.natH * logo.scale;
    const left = logo.x - w / 2, top = logo.y - h / 2;
    const hs = HANDLE_SIZE;
    if (cx >= left + w - hs && cx <= left + w + hs / 2 && cy >= top + h - hs && cy <= top + h + hs / 2) return "resize";
    if (cx >= left && cx <= left + w && cy >= top && cy <= top + h) return "move";
    return null;
}

export function clientToCanvasPoint(e, rect, displayScale) {
    return [(e.clientX - rect.left) / displayScale, (e.clientY - rect.top) / displayScale];
}

// ── The cutout subject ────────────────────────────────────────────────────
// Drawn with the same call an image overlay is, and moved by the same drag
// (see state.layersFor). Two things about it are deliberately NOT like an
// overlay, and both come from the same fact: it is a person on a transparent
// field, not a picture in a rectangle.
//
// It wears no selection chrome. A dashed box round a cutout is a box round
// mostly nothing — it draws a rectangle where the design has a person, and it
// invites the user to believe the rectangle is the thing they are arranging.
// The cursor is the affordance instead.
//
// And it is hit-tested on its ALPHA, not on its box. Where the channel places
// a subject there is a great deal of transparent canvas inside their bounding
// box — above a shoulder, between an arm and a body — and every pixel of it is
// backdrop the user can see straight through. Grabbing the subject by pressing
// what is visibly the background is the exact thing that made the placement
// feel like a fixed frame rather than a suggestion.

/**
 * How much of the subject has to stay on the canvas.
 *
 * The ONLY limit on where they may be dragged. `fitWidthRatio` decides where
 * they arrive; it does not decide where they may go, because a channel's
 * placement is a starting composition and the user is the one composing. All
 * this guarantees is that a drag can always be undone by another drag —
 * dragged fully off the edge, there would be nothing left on screen to grab.
 */
const CUTOUT_KEEP_ON_CANVAS = 60;

/** The width a figure is fitted into when it first arrives, in canvas pixels. */
const cutoutFitWidth = (spec) => CW * (spec.fitWidthRatio !== undefined ? spec.fitWidthRatio : 0.5);

/**
 * ...and where that room SITS across the canvas, as the x of its centre.
 *
 * Two questions rather than one, because they have separate answers and a
 * channel can want either without the other. `fitWidthRatio` is how big the
 * subject arrives: it is the width budget the fit is computed against, so a
 * half-width room is a subject fitted to half the frame whatever else is true.
 * This is where that room is put, and it changes nothing about the size.
 *
 * Absent, the room hugs the LEFT edge, which is what every channel that had
 * one before this key existed was drawn with: a subject in the left half with
 * the title in the right. Stating it moves the room without resizing it — a
 * channel whose subject stands in the middle of a card, on its own, with the
 * type above them rather than beside them, says `anchorX: 0.5` and keeps the
 * size its `fitWidthRatio` already gave it.
 *
 * The same 0-to-1 fraction of the canvas that `framing.anchorX` is, and it
 * means the same thing on the same axis — one vocabulary for "where across
 * the frame", rather than a second one that happens to agree.
 */
const cutoutRoomCentre = (spec) => spec.anchorX !== undefined
    ? CW * spec.anchorX
    : cutoutFitWidth(spec) / 2;

/**
 * How far the row of arriving figures may reach, measured centre to centre, as
 * a share of the room the channel gave them.
 *
 * A ceiling on the spacing rather than a spacing of its own — with a gap
 * stated as a fraction of the figures' own width (`copyOffsetGap`) a wide
 * figure asks for a wide gap, and three head-and-shoulders cutouts, each most
 * of the fitted width by themselves, ask between them for a row wider than the
 * canvas. The far one lands under the title and the near one is half off the
 * opposite edge.
 *
 * Just under the full width, so the ordinary case — full-length figures, which
 * are a third as wide and never come near this — is untouched, and the case
 * that does hit it comes back to three figures overlapping heavily inside
 * their own half, which is what a trio of close-ups has to be.
 */
const CUTOUT_SPREAD_MAX = 0.9;

/**
 * The scale the channel's placement puts a figure of this size at — the
 * reference every later size is expressed against, including the zoom
 * control's whole range.
 *
 * Fitted to `fitRatio` of the room rather than filling it, so that the
 * placement reads as a subject standing in their half with air around them
 * rather than as one wedged into it.
 */
export function cutoutFitScale(spec, natW, natH) {
    const margin = spec.fitRatio !== undefined ? spec.fitRatio : 0.92;
    return Math.min(cutoutFitWidth(spec) / Math.max(1, natW), CH / Math.max(1, natH)) * margin;
}

/**
 * Where `pictures` figures go when they first arrive — bottom of the stack
 * first, so the LAST entry is the one nearest the viewer.
 *
 * One figure is the plain case: fitted, standing on the bottom edge, centred
 * in the width the channel gave it — and that width sits wherever the channel
 * put it, which is against the left edge unless the pack said otherwise (see
 * cutoutRoomCentre).
 *
 * More than one is a look rather than an accident (see `copiesBySlot`), and
 * each of them is a different moment of the same shot. They arrive SPREAD, and
 * they have to: the whole point of putting the subject on the thumbnail three
 * times is that a viewer sees three of them, and three figures landing within
 * a few pixels of each other do not read as three of anybody. They read as one
 * figure with a fringe of itself showing down one side — and the user, looking
 * at what appears to be a single subject, has no reason to suspect there are
 * two more behind it waiting to be dragged out.
 *
 * Spread by a fraction of how wide they are DRAWN (`copyOffsetGap`) rather
 * than by a count of pixels, because the figures are not a fixed size: a
 * full-length shot arrives a third of the width a head-and-shoulders does, and
 * one number of pixels cannot be the right gap for both. Measured between
 * neighbours and accumulated, so a wide figure pushes the next one further
 * along than a narrow one does and every gap in the row looks the same size.
 *
 * The row is then centred on the room the channel gave them, wherever that
 * room is, so a slot showing three figures sits where a slot showing one does
 * instead of marching off to the right of it.
 *
 * All of them stand on the same ground line, because that ground line is the
 * bottom of the frame. And all of it is only where they ARRIVE — every figure
 * is draggable, resizable, flippable and reorderable on its own from there.
 */
export function fitCutout(spec, pictures) {
    const bottom = CH * (spec.bottomRatio !== undefined ? spec.bottomRatio : 1);
    const step = spec.copyScale !== undefined ? spec.copyScale : 1;
    const gap = spec.copyOffsetGap;
    const shift = spec.copyOffsetPx !== undefined ? spec.copyOffsetPx : 30;

    // Front figure first here, so `behind` counts back from the viewer exactly
    // as copyScale describes; the array is reversed at the end so that the
    // bottom of the stack comes out first, which is the order paintFrame draws.
    const drawn = pictures.map((picture, behind) => {
        const scale = cutoutFitScale(spec, picture.natW, picture.natH) * Math.pow(step, behind);
        return { picture, scale, width: picture.natW * scale };
    });

    // Each figure's own offset from the one in front of it. With a gap stated
    // as a fraction, that is a fraction of the two widths' mean — the pair's
    // own size, so neither a wide neighbour nor a narrow one decides it alone.
    const offsets = [0];
    for (let behind = 1; behind < drawn.length; behind++) {
        const apart = gap !== undefined
            ? gap * (drawn[behind].width + drawn[behind - 1].width) / 2
            : shift;
        offsets.push(offsets[behind - 1] + apart);
    }

    // Pulled in together if the row reaches further than it may (see
    // CUTOUT_SPREAD_MAX). Scaled rather than clipped, so the gaps stay equal to
    // each other — a row where only the last figure was pulled back in would
    // have one gap visibly tighter than the others.
    const reach = offsets[offsets.length - 1];
    const room = cutoutFitWidth(spec) * CUTOUT_SPREAD_MAX;
    if (reach > room) {
        const squeeze = room / reach;
        for (let behind = 0; behind < offsets.length; behind++) offsets[behind] *= squeeze;
    }
    const middle = offsets[offsets.length - 1] / 2;

    const layers = [];
    for (let behind = drawn.length - 1; behind >= 0; behind--) {
        const { picture, scale } = drawn[behind];
        layers.push(clampCutout(spec, {
            ...picture, scale,
            x: cutoutRoomCentre(spec) + offsets[behind] - middle,
            y: bottom - (picture.natH * scale) / 2,
            flip: false,
        }));
    }
    return layers;
}

/**
 * The one rule a placed figure cannot escape: enough of it stays on the canvas
 * to be grabbed again.
 *
 * This used to hold them inside the half of the canvas their channel gave
 * them, on every drag and at every zoom. That was a misreading of what the
 * channel's placement is for. Where a figure STARTS is the design; where it
 * ends up is the user's composition, and a subject who cannot be pulled across
 * the middle — to bleed off the far edge, to sit behind the title, to be
 * cropped the way the user wants them cropped — is a layout the tool is
 * imposing rather than offering.
 *
 * Mutates and returns the layer it was given, as the drag handlers expect.
 */
export function clampCutout(spec, c) {
    const fit = cutoutFitScale(spec, c.natW, c.natH);
    const floor = spec.minZoom !== undefined ? spec.minZoom : 0.35;
    const ceiling = spec.maxZoom !== undefined ? spec.maxZoom : 3;
    c.scale = Math.max(fit * floor, Math.min(c.scale, fit * ceiling));

    const w = c.natW * c.scale, h = c.natH * c.scale;
    // Positions are centres, so every edge is expressed as a centre here.
    const keep = CUTOUT_KEEP_ON_CANVAS;
    c.x = Math.max(keep - w / 2, Math.min(c.x, CW - keep + w / 2));
    c.y = Math.max(keep - h / 2, Math.min(c.y, CH - keep + h / 2));
    return c;
}

// Alpha at or above which a pixel of a figure counts as the subject rather
// than as the backdrop showing through. Low: a hair strand or a motion-blurred
// hand is faint and is still the person, and the cost of being generous is a
// press landing on the subject a pixel or two before they visibly begin.
const HIT_ALPHA_MIN = 20;

// Width of the alpha map the hit test reads. A megapixel of ImageData per
// figure would be a waste of memory for a question whose answer only has to be
// right to within a finger's width; at this size the map is a few kilobytes
// and each of its cells covers a few pixels of a subject at full size.
const HIT_MAP_W = 160;

/**
 * A small alpha map of a figure, built once and kept on the layer beside the
 * decoded image it came from.
 *
 * Cached there rather than in a module map because it describes THAT picture,
 * and a re-cut figure is a whole new record (see editor.cutOut) — so the map
 * cannot outlive the image it was measured from.
 */
function alphaMap(layer) {
    if (layer._alpha) return layer._alpha;
    const img = layer._img;
    if (!img || !img.width || !img.height) return null;
    const w = Math.max(1, Math.min(HIT_MAP_W, img.width));
    const h = Math.max(1, Math.round(img.height * w / img.width));
    const element = document.createElement("canvas");
    element.width = w; element.height = h;
    const ctx = element.getContext("2d");
    ctx.drawImage(img, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    const map = new Uint8Array(w * h);
    for (let i = 0; i < map.length; i++) map[i] = data[i * 4 + 3];
    layer._alpha = { w, h, map };
    return layer._alpha;
}

/**
 * Whether a canvas-space point is ON this figure — "move" if it is, null if it
 * is the backdrop showing through its bounding box, or beside it.
 *
 * Answers "move" and never "resize", because a figure has no resize handle:
 * the zoom control is what changes its size, over a range far wider than a
 * corner drag could reach.
 *
 * Sampled over a one-cell neighbourhood rather than at the single cell under
 * the cursor. At this map's resolution a thin arm can fall between samples,
 * and a limb that is visibly there but cannot be grabbed is worse than a few
 * pixels of slop around one that can.
 */
export function cutoutHitTest(layer, cx, cy) {
    const w = layer.natW * layer.scale, h = layer.natH * layer.scale;
    const left = layer.x - w / 2, top = layer.y - h / 2;
    if (cx < left || cx > left + w || cy < top || cy > top + h) return null;

    const a = alphaMap(layer);
    // Not decoded yet, or unreadable: the box is the best answer there is, and
    // it is the answer this had before the map existed.
    if (!a) return "move";

    // The map is of the picture as it came; a mirrored figure is the same
    // picture drawn the other way round, so the point is mirrored to ask it.
    const acrossFrac = layer.flip ? 1 - (cx - left) / w : (cx - left) / w;
    const mx = Math.max(0, Math.min(a.w - 1, Math.floor(acrossFrac * a.w)));
    const my = Math.max(0, Math.min(a.h - 1, Math.floor((cy - top) / h * a.h)));
    for (let dy = -1; dy <= 1; dy++) {
        const y = my + dy;
        if (y < 0 || y >= a.h) continue;
        for (let dx = -1; dx <= 1; dx++) {
            const x = mx + dx;
            if (x < 0 || x >= a.w) continue;
            if (a.map[y * a.w + x] >= HIT_ALPHA_MIN) return "move";
        }
    }
    return null;
}

// ── Reordering and removing the figures ───────────────────────────────────
// The one piece of selection chrome a figure wears, and it appears only when
// there is more than one of them and one has been pressed. Three buttons in a
// column: up and down push that figure in front of or behind the one next to
// it, and the cross takes it off the thumbnail.
//
// There is no "below the background" to reach: the backdrop is not in the
// stack at all, it is what the stack is drawn ON. The ends of the array are
// therefore the ends of the movement, and the arrow that would go past one is
// drawn dim.
//
// The cross is red rather than blue, and it is last in the column. Two
// reversible buttons and one that is not should not look like three of a kind,
// and the one that is not should not sit between them where a mis-aimed click
// lands on it.

export const LAYER_ARROW_R = 17;
const LAYER_ARROW_GAP = 10;
// Kept clear of the canvas edge so a button never has half of itself outside.
const LAYER_ARROW_MARGIN = 6;

/**
 * Where this figure's three buttons sit: stacked at the top of the figure they
 * belong to, and pulled inside the canvas when that top is off it — which it
 * very often is, since a subject standing on the bottom edge is routinely
 * taller than the frame.
 */
export function layerArrows(layer) {
    const h = layer.natH * layer.scale;
    const r = LAYER_ARROW_R, m = LAYER_ARROW_MARGIN;
    const pitch = r * 2 + LAYER_ARROW_GAP;
    const span = r * 6 + LAYER_ARROW_GAP * 2;   // all three, top edge to bottom edge
    const x = Math.max(r + m, Math.min(layer.x, CW - r - m));
    const wanted = layer.y - h / 2 + r + m;
    const y = Math.max(r + m, Math.min(wanted, CH - span + r - m));
    return { up: [x, y], down: [x, y + pitch], remove: [x, y + pitch * 2] };
}

/** "up", "down", "remove" or null — the same three circles addLayerArrows draws. */
export function layerArrowHitTest(layer, cx, cy) {
    const spots = layerArrows(layer);
    const near = ([x, y]) => (cx - x) ** 2 + (cy - y) ** 2 <= LAYER_ARROW_R ** 2;
    for (const name of ["up", "down", "remove"]) {
        if (near(spots[name])) return name;
    }
    return null;
}

/**
 * Draws them. `atTop`/`atBottom` dim the arrow that has nowhere to go, rather
 * than hiding it — a button that comes and goes as the stack is reordered
 * moves the other one under the cursor between clicks.
 */
function addLayerArrows(canvas, layer, atTop, atBottom) {
    const spots = layerArrows(layer);
    const r = LAYER_ARROW_R;
    const buttons = [
        [spots.up, "up", atTop, "#1b2fea"],
        [spots.down, "down", atBottom, "#1b2fea"],
        [spots.remove, "remove", false, "#d21f3c"],
    ];
    for (const [point, dir, spent, fill] of buttons) {
        canvas.add(new fabric.Circle({
            left: point[0] - r, top: point[1] - r, radius: r,
            fill, stroke: "#fff", strokeWidth: 2,
            opacity: spent ? 0.35 : 1,
            selectable: false, evented: false,
        }));
        if (dir === "remove") {
            // Two crossed bars rather than a glyph: nothing here loads a font,
            // and a letter X in whatever face the canvas falls back to is not
            // the same shape twice.
            for (const angle of [45, -45]) {
                canvas.add(new fabric.Rect({
                    left: point[0], top: point[1], width: 17, height: 3.5,
                    originX: "center", originY: "center", angle,
                    fill: "#fff", rx: 1.5, ry: 1.5,
                    selectable: false, evented: false,
                }));
            }
            continue;
        }
        // Drawn as explicit points rather than a rotated triangle: fabric
        // rotates about an origin, and two shapes that must line up in one
        // column are easier to trust when neither of them is rotated.
        const points = dir === "up"
            ? [{ x: 0, y: 13 }, { x: 8, y: 1 }, { x: 16, y: 13 }]
            : [{ x: 0, y: 1 }, { x: 16, y: 1 }, { x: 8, y: 13 }];
        canvas.add(new fabric.Polygon(points, {
            left: point[0] - 8, top: point[1] - 7,
            fill: "#fff", opacity: spent ? 0.35 : 1,
            selectable: false, evented: false,
        }));
    }
}

/** Decodes a figure's blob URL once and caches it on the layer, as a logo's is. */
export const loadCutoutImage = loadLogoImage;

// Every picture a CHANNEL brings with it — its backdrop, its brand mark —
// decoded once and kept, keyed by the URL it was served from.
//
// Cached at module level rather than per frame because there is exactly one of
// each per channel and twenty frames share them: decoding the backdrop again
// for every grid thumbnail would be twenty decodes of one file on every
// keystroke that changes a title.
const _decorImages = new Map();

function decorImage(url) {
    if (!url) return Promise.resolve(null);
    if (_decorImages.has(url)) return _decorImages.get(url);
    // The PROMISE is cached, not the image, so twenty thumbnails composed in
    // the same tick share one decode instead of starting twenty.
    const p = loadImage(url).catch(() => null);
    _decorImages.set(url, p);
    return p;
}

/**
 * Draws a channel's backdrop to COVER the canvas: scaled up until it fills
 * both dimensions and centred, with whatever overhangs cropped off.
 *
 * Never stretched to fit. A backdrop is a photograph somebody chose, and one
 * squashed to 16:9 because it was 16:10 is not the picture they chose — where
 * a few pixels off each end is not something anyone can see.
 */
function addBackground(canvas, img) {
    const scale = Math.max(CW / img.width, CH / img.height);
    canvas.add(new fabric.Image(img, {
        left: (CW - img.width * scale) / 2, top: (CH - img.height * scale) / 2,
        width: img.width, height: img.height, scaleX: scale, scaleY: scale,
        selectable: false, evented: false,
    }));
}

// ── The photo card ────────────────────────────────────────────────────────
//
// A channel that states `photo` does not draw the photograph edge to edge (see
// `photo` in channels.js). It draws it as a card: scaled down, centred,
// rounded at the corners, tinted, on a painted ground.
//
// Three things go down and they are one idea, which is why they are computed
// in one place and drawn in one call. The margin only exists because the card
// was shrunk, the backdrop only exists to fill that margin, and the tint only
// exists to tie the picture to the ground it is now lying on. A pack that
// stated any of them without the others would be stating half a composition.

/**
 * Where the card lands and what is painted on and under it, or null for the
 * channels — every other one — whose photograph is the canvas.
 *
 * Geometry in canvas pixels and rounded to whole ones. The card is a rectangle
 * with a hard edge on a ground of a different colour, and a left edge at
 * 80.9999 is that edge drawn across two columns of pixels: on a 13px radius it
 * is the difference between a corner and a smudge.
 *
 * Exported because the drag preview draws the same composition from a 2D
 * context rather than through fabric (see reframe.drawLiveCrop), and the two
 * have to agree about where the picture is to the pixel — a drag that shows
 * the frame full-bleed and drops it into a card is a drag nobody can aim.
 */
export function photoPlate(spec) {
    if (!spec) return null;
    // A fraction of the canvas, and both ends of the range are refused rather
    // than clamped to something arbitrary: 0 is a card with no picture in it
    // and anything above 1 is a card larger than the frame it is centred in,
    // which is the full-bleed photograph with its edges cut off — a channel
    // wanting that states no `photo` at all.
    const fill = Math.min(1, Math.max(0.01, spec.fillRatio !== undefined ? spec.fillRatio : 1));
    const width = Math.round(CW * fill), height = Math.round(CH * fill);
    return {
        left: Math.round((CW - width) / 2), top: Math.round((CH - height) / 2),
        width, height,
        // Clamped to half the card, as the panel's radius is: past that a
        // rounded rectangle folds through itself, and a capsule is the honest
        // end of the range.
        radius: Math.min(Math.max(0, spec.radius || 0), Math.min(width, height) / 2),
        // What the card is MADE of, when it is not made of the photograph.
        // Null is the answer every card gave until now and still means "the
        // still goes here"; a colour means the still does not, and every
        // caller reads it as that one question rather than as a colour it
        // might paint behind something.
        fill: spec.fill || null,
        backdrop: spec.backdrop || null,
        tint: spec.tint || null,
    };
}

/**
 * Whether the frame's own photograph has to be decoded to draw this card.
 *
 * Asked because decoding one is the most expensive thing on the paint path and
 * a card can now be drawn without it: a card with a `fill` is a colour, and a
 * gradient backdrop is a colour too, so a channel that states both wants no
 * photograph on screen at all and should not pay for one per keystroke.
 *
 * The frame backdrop is the case that keeps it honest — the picture is gone
 * from the card and has moved underneath it, which needs the decode as much as
 * ever.
 */
export const plateNeedsPhoto = (plate) =>
    !!plate && (!plate.fill || isFrameBackdrop(plate.backdrop));

/** Whether a backdrop spec is the blurred still rather than a gradient. */
const isFrameBackdrop = (spec) => !!spec && spec.source === "frame";

/**
 * A {from, to, stops} gradient spec as coordinates inside a box.
 *
 * `from` and `to` are corners of that box as fractions of it, which is the
 * form every gradient in a pack is written in (see `sticker.fill` in text.js
 * and `photo` in channels.js): the pack describes the direction the ramp runs
 * in and nothing about the size of the thing it is running across, so the same
 * four numbers hold for the canvas and for a card two thirds of it.
 */
function rampCoords(spec, box) {
    const [fx, fy] = spec.from || [0, 0];
    const [tx, ty] = spec.to || [0, 1];
    return { x1: fx * box.width, y1: fy * box.height, x2: tx * box.width, y2: ty * box.height };
}

const rampStops = (spec) => (spec.stops || []).map(stop => ({ offset: stop.at, color: stop.color }));

/** The same ramp as a fabric paint, anchored in the object's own box. */
function rampFill(spec, box) {
    return new fabric.Gradient({
        type: "linear", gradientUnits: "pixels",
        coords: rampCoords(spec, box), colorStops: rampStops(spec),
    });
}

/** ...and as a 2D context gradient, offset to where the box sits on the canvas. */
function rampGradient(ctx, spec, box) {
    const c = rampCoords(spec, box);
    const left = box.left || 0, top = box.top || 0;
    const paint = ctx.createLinearGradient(left + c.x1, top + c.y1, left + c.x2, top + c.y2);
    for (const stop of (spec.stops || [])) paint.addColorStop(stop.at, stop.color);
    return paint;
}

/**
 * Traces the card's rounded rectangle on a 2D context.
 *
 * Written out rather than left to ctx.roundRect, which is recent enough that a
 * browser without it would throw in the middle of a drag — and a drag that
 * dies halfway leaves the live layer on screen showing the last frame it
 * managed to draw, which reads as the app having frozen rather than as a
 * missing method.
 */
function platePath(ctx, plate) {
    const { left: x, top: y, width: w, height: h, radius: r } = plate;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

/**
 * Paints the ground, clips to the card and puts the context in CANVAS
 * coordinates scaled into it — so a caller that knows how to draw a frame full
 * size draws it into the card by doing nothing differently.
 *
 * Paired with closePlate, which is what lays the tint on and hands the context
 * back. Two calls rather than a callback because the drawing between them is
 * the drag preview's own overpan arithmetic, and moving that inside a closure
 * would move the one piece of this that has nothing to do with the card.
 */
export function openPlate(ctx, plate) {
    if (plate.backdrop) {
        // A FRAME backdrop is painted flat here rather than blurred, and that
        // is a limit worth stating out loud. The still this ground is made of
        // is the one the drag is re-cropping, and the wide view the drag holds
        // is a different bitmap in a different geometry — so drawing it
        // properly would mean blurring the wide view through the crop window
        // on every pointer move. It is unreachable work today: the only kind
        // of channel that has ever wanted a frame backdrop composes from a
        // cutout, and a crop drag is refused outright on those (see the press
        // handler in reframe.js). The flat fill is here so that a channel
        // which one day wants both gets a dark ground rather than a
        // transparent one, and so that this comment is what it finds.
        ctx.fillStyle = isFrameBackdrop(plate.backdrop)
            ? "#181818"
            : rampGradient(ctx, plate.backdrop, { left: 0, top: 0, width: CW, height: CH });
        ctx.fillRect(0, 0, CW, CH);
    }
    ctx.save();
    platePath(ctx, plate);
    ctx.clip();
    // The card's own dark ground, under the picture and inside the clip. It is
    // what shows through where a drag has panned past the edge of the source
    // frame, and it has to be the card's colour rather than the backdrop's or
    // the overpan strip would look like a hole cut in the picture.
    //
    // On a card with a `fill` that colour IS the card, so it stands in: the
    // still is not drawn on such a card at all, which means everything the
    // caller is about to paint lands on this and nothing else.
    ctx.fillStyle = plate.fill || "#181818";
    ctx.fillRect(plate.left, plate.top, plate.width, plate.height);
    ctx.translate(plate.left, plate.top);
    ctx.scale(plate.width / CW, plate.height / CH);
}

/** Ends what openPlate began: drops the clip and the transform, then tints. */
export function closePlate(ctx, plate) {
    ctx.restore();
    if (!plate.tint) return;
    ctx.save();
    platePath(ctx, plate);
    ctx.clip();
    ctx.fillStyle = rampGradient(ctx, plate.tint, plate);
    ctx.fillRect(plate.left, plate.top, plate.width, plate.height);
    ctx.restore();
}

// How far past the canvas a blurred backdrop is drawn before it is cut back to
// size, as a multiple of the blur radius.
//
// A canvas blur is a convolution, and outside the bitmap it convolves with
// transparent black — so a still blurred at exactly canvas size comes back
// with its own edges faded into nothing, four soft grey borders that read as a
// vignette nobody asked for. The fix is to blur a LARGER picture and keep the
// middle of it, and this is how much larger.
//
// Twice the radius, because the browser's blur(r) is a Gaussian of standard
// deviation r/2 and a Gaussian is spent by three deviations — 1.5r of real
// reach, with the rest of the margin covering the cheaper approximations some
// engines use at large radii. Cheap at the sizes involved: at 30px of blur on
// a 540-wide frame it is a bitmap 22% wider than the one that gets kept.
const BACKDROP_BLEED = 2;

/**
 * The frame's own still, cropped in and blurred, at canvas size — what a
 * channel gets instead of a gradient under its card (see `photo.backdrop` in
 * channels.js).
 *
 * Two moves and the order matters. The crop comes first: `cropRatio` of the
 * still, from its centre, scaled back out to cover the canvas. That is what
 * makes the result read as a soft version of the picture rather than as a grey
 * field — blur spreads whatever detail is in the source across its own radius,
 * so a tighter crop is fewer and larger shapes surviving it. Blurring first
 * and enlarging afterwards would spread the detail and THEN magnify the
 * mush.
 *
 * The scale is the one the crop asks for, floored by whatever it takes to
 * cover the bled-out box. A still smaller than the canvas, or a cropRatio near
 * 1, would otherwise leave the bleed margin uncovered — which is the faded
 * edge this whole arrangement exists to avoid, arriving by the other road.
 *
 * Cached on the decoded image, the way a figure's alpha map is cached on its
 * layer: it describes THAT bitmap and cannot outlive it. That makes it free
 * for the preview, which redraws the same decoded still on every keystroke,
 * and no help at all to a composite, which decodes twenty different ones —
 * exactly the shape of the photo cache it rides on (see framePhoto).
 */
function frameBackdrop(img, spec) {
    if (!img || !img.width || !img.height) return null;
    const crop = Math.min(1, Math.max(0.01, spec.cropRatio !== undefined ? spec.cropRatio : 1));
    const blur = Math.max(0, spec.blur || 0);
    const key = `${crop}|${blur}`;
    if (img._backdrop && img._backdrop.key === key) return img._backdrop.element;

    const bleed = Math.ceil(blur * BACKDROP_BLEED);
    const bw = CW + bleed * 2, bh = CH + bleed * 2;
    const scale = Math.max(
        CW / (img.width * crop), CH / (img.height * crop),
        bw / img.width, bh / img.height,
    );
    const dw = img.width * scale, dh = img.height * scale;

    const bled = document.createElement("canvas");
    bled.width = bw; bled.height = bh;
    const bctx = bled.getContext("2d");
    if (blur > 0) bctx.filter = `blur(${blur}px)`;
    bctx.drawImage(img, (bw - dw) / 2, (bh - dh) / 2, dw, dh);

    const element = document.createElement("canvas");
    element.width = CW; element.height = CH;
    element.getContext("2d").drawImage(bled, -bleed, -bleed);
    img._backdrop = { key, element };
    return element;
}

/**
 * The card's layers as fabric objects: the ground, what the card is made of,
 * and the tint.
 *
 * The photograph is stretched to the card exactly as it was stretched to the
 * canvas before there was a card — the backend delivers it at the canvas's own
 * dimensions (see capture.output), so with a card that is a uniform scale of
 * the canvas this is a resize and never a distortion.
 *
 * ...unless the pack gave the card a `fill`, in which case the card is that
 * colour and the still is not drawn in it at all. The card is still the card:
 * same rectangle, same rounding, same everything standing on it — only the
 * thing it is made of has changed, so this is one branch and not a second
 * composition.
 *
 * A photo that failed to decode still gets the ground and the tint. That is
 * the channel drawn with one layer missing rather than a blank canvas, which
 * is the same answer a missing backdrop gets in paintFrame.
 */
function addPhotoPlate(canvas, plate, img) {
    const ground = isFrameBackdrop(plate.backdrop)
        ? frameBackdrop(img, plate.backdrop)
        : null;
    if (ground) {
        // An element rather than an Image: fabric takes either, and what came
        // back is a canvas already drawn at exactly canvas size.
        canvas.add(new fabric.Image(ground, {
            left: 0, top: 0, selectable: false, evented: false,
        }));
    } else if (plate.backdrop && !isFrameBackdrop(plate.backdrop)) {
        canvas.add(new fabric.Rect({
            left: 0, top: 0, width: CW, height: CH,
            fill: rampFill(plate.backdrop, { width: CW, height: CH }),
            selectable: false, evented: false,
        }));
    }
    if (plate.fill) {
        canvas.add(new fabric.Rect({
            left: plate.left, top: plate.top, width: plate.width, height: plate.height,
            rx: plate.radius, ry: plate.radius, fill: plate.fill,
            selectable: false, evented: false,
        }));
    } else if (img) {
        canvas.add(new fabric.Image(img, {
            left: plate.left, top: plate.top,
            width: img.width, height: img.height,
            scaleX: plate.width / img.width, scaleY: plate.height / img.height,
            // absolutePositioned, so the clip is a rectangle ON THE CANVAS
            // rather than one carried around by the image it clips. The two
            // are the same rectangle here, but only the absolute one stays
            // where the pack put it if the image is ever given a transform of
            // its own.
            clipPath: new fabric.Rect({
                left: plate.left, top: plate.top, width: plate.width, height: plate.height,
                rx: plate.radius, ry: plate.radius, absolutePositioned: true,
            }),
            selectable: false, evented: false,
        }));
    }
    if (plate.tint) {
        canvas.add(new fabric.Rect({
            left: plate.left, top: plate.top, width: plate.width, height: plate.height,
            rx: plate.radius, ry: plate.radius,
            fill: rampFill(plate.tint, plate),
            selectable: false, evented: false,
        }));
    }
}

/**
 * Draws a channel's scrim: a gradient laid over the backdrop AND over the
 * figures standing on it.
 *
 * Over the figures, which is the whole point and the one thing that separates
 * it from the backdrop having a dark foot painted into it. A gradient under
 * the subject leaves them lit from the ankles up and floating; over them, the
 * same shade falls across their legs, and the eye reads the pair as one
 * photograph taken in one light rather than as a person pasted onto a plate.
 *
 * `heightRatio` is how far up the canvas it reaches, and the ramp runs from
 * the bottom edge upward — dark where it starts, gone by the top of its own
 * span. Stated bottom-first because that is the direction the shade is thrown
 * in: it is the floor the figures stand on going dark, not a vignette.
 *
 * Straight up, and only straight up, which is what lets it end on a fully
 * transparent stop and disappear. A scrim is a rectangle: a ramp aimed across
 * it rather than up it would be crossed by the band's own top edge instead of
 * running out along it, so the shade would stop on a visible horizontal line.
 * A channel wanting a ramp at an angle wants `photo.backdrop` or `photo.tint`,
 * which are painted on shapes that have somewhere to end.
 *
 * Drawn under the brand marks and the title, which are the two things that
 * have to stay legible whatever is behind them — a scrim over the logo is a
 * dimmed logo, and over the type it is the shadow the type already carries,
 * twice.
 */
function addScrim(canvas, spec) {
    const height = CH * (spec.heightRatio !== undefined ? spec.heightRatio : 0.5);
    const top = CH - height;
    const paint = new fabric.Gradient({
        type: "linear",
        gradientUnits: "pixels",
        // Bottom to top: y1 is the canvas floor, y2 the top of the span.
        coords: { x1: 0, y1: height, x2: 0, y2: 0 },
        colorStops: (spec.stops || []).map(stop => ({ offset: stop.at, color: stop.color })),
    });
    canvas.add(new fabric.Rect({
        left: 0, top, width: CW, height,
        fill: paint, selectable: false, evented: false,
    }));
}

/**
 * Draws a channel's brand mark at its stated inset from a canvas corner.
 *
 * At the file's own pixel size unless the pack says otherwise, and at an exact
 * offset: a lock-up is a measurement, so "10px from each edge" is drawn as
 * 10px and not as a ratio that lands somewhere near it.
 *
 * `right` is measured from the other edge and wins over `left`, which is how a
 * mark belonging to the right-hand corner stays in it. Computed here, from the
 * width the mark is actually drawn at, rather than written into the pack as a
 * `left`: a pack cannot know that number without knowing the file's pixels,
 * and a `left` worked out from them by hand stops being true the first time
 * the artwork is re-exported a few pixels wider.
 */
function addStamp(canvas, spec, img) {
    const scale = spec.width ? spec.width / img.width : 1;
    const left = spec.right !== undefined
        ? CW - img.width * scale - spec.right
        : (spec.left !== undefined ? spec.left : 10);
    canvas.add(new fabric.Image(img, {
        left, top: spec.top !== undefined ? spec.top : 10,
        width: img.width, height: img.height, scaleX: scale, scaleY: scale,
        // Follows the mark's own alpha, as the cutout's shadow does — a badge
        // cut to a torn edge throws that edge, not the rectangle it arrived
        // in.
        shadow: spec.shadow ? new fabric.Shadow({ ...spec.shadow }) : null,
        selectable: false, evented: false,
    }));
}

// -- The contour around a figure -------------------------------------------
//
// A cut-out person has a hard edge by construction, and on a busy stage that
// edge is the one thing still saying they were not photographed there. The
// contour answers it by making the edge deliberate: a band of light lying
// INSIDE the silhouette and drawn OVER the person.
//
// Two properties define it, and everything here exists to hold them.
//
// It never adds a pixel OUTSIDE the cutout. The silhouette's outline is the
// outline, exactly as the backend cut it, lit or not. Anything that grows the
// shape — a glow behind it, a stroke around it, even a blur allowed to bleed
// past the edge — puts a coloured margin between the person and the stage
// where the photograph had none, and the eye reads that margin as the join.
//
// And it FADES INWARD. Depth is measured perpendicular to the outline, going
// into the body: the band is at full strength at depth zero — the pixel
// touching the cut edge — and falls away monotonically from there, reaching
// nothing at `edgeRatio`. The brightest pixel is always the one on the edge.
// That is what makes it light catching an edge rather than a stripe of paint:
// a band of even alpha has a second edge, the inner one, and a second edge is
// the thing that reads as drawn.
//
// The falloff is made by subtraction. Erode the silhouette, soften what is
// left, and take it away from the silhouette: `destination-out` multiplies the
// destination's alpha by one minus the source's, so where the softened core is
// solid — deep inside the person — nothing survives, and where it has faded to
// nothing — at the outline — all of the silhouette does. In between it is the
// blur's own curve, which is exactly the smooth ramp wanted.
//
// -- Why the silhouette is hardened first ----------------------------------
//
// A segmentation mask is not a stencil. What comes back has a soft alpha
// fringe — measured at about nine pixels on real footage — where the model is
// unsure whether it is looking at hair or at curtain. The band is that alpha
// times the ramp, so on a slope the ramp is spent inside the fringe and the
// band washes out. Hardening puts the ramp's zero on the mask's own half-way
// contour, which is a boundary and not a slope.
//
// Everything is a fraction of the cutout's own size and the sprite is built at
// the cutout's own resolution, once — so the band holds its proportion as the
// figure is resized, and the zoom control never triggers a rebuild.

const _glowSprites = new Map();
// One entry per cutout per colour. No zoom term: the sprite does not depend on
// the drawn size.
const GLOW_CACHE_LIMIT = 64;

// How the band's reach is split between pulling the core in and softening it.
//
//   erode  0.55 of the reach — where the core's own edge sits.
//   soften 0.25 of the reach — the blur's standard deviation, so that edge is
//          2.2 deviations from the outline. The core has fallen to about 1%
//          by the time it reaches the outline, so the band is at full strength
//          exactly there, and it is gone by about one reach in.
//
// Soften more (or erode less) and the core lifts over the outline, taking the
// strength out of the band at the one place it is meant to be strongest —
// measured, this pair starts it at 0.87 against the outline where 0.5/0.35
// starts it at 0.76 and then holds on further in, which is the wrong shape
// twice over. Soften less and the core's own edge survives the blur as a
// visible inner line, which is the second edge this design exists not to have.
//
// The property to preserve if these are ever touched: the band must be
// monotonic from the outline inward, and its strongest point must be the
// outline itself.
const GLOW_ERODE = 0.55;
const GLOW_SOFTEN = 0.25;

// Rounds of the contrast curve that hardens the mask's fringe. Each round is
// a := (2a - a^2)^2, which drives everything above about 0.38 to one and
// everything below it to zero; convergence is quadratic, so five rounds turn
// even a wide slope into a boundary.
const GLOW_HARDEN_ROUNDS = 5;
// ...and the softness handed back afterwards, in pixels of the sprite, so the
// band's outer edge is antialiased rather than stepped. It scales with the
// figure along with everything else.
const GLOW_EDGE_FEATHER = 0.8;
// Self-draws used to turn the cutout's footprint into a stencil. Each one
// squares the transparency, so eight take an alpha of a hundredth to one and
// leave a true zero alone.
const GLOW_SUPPORT_ROUNDS = 8;
// Where in the band's own distribution the ordinary outline sits, and so what
// the band is normalised against — see inwardBand. Measured on a real cutout:
// the first fully-opaque ring came out at the 92nd percentile of the non-zero
// band, because erosion eats more off a convex bump than off a flat run and
// the true maximum belongs to whichever corner of the figure is sharpest.
const GLOW_NORMALISE_AT = 0.92;

/** One ring of eight offsets at radius `r`, in the order they are drawn. */
function ring(r) {
    const points = [];
    for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        points.push([Math.cos(a) * r, Math.sin(a) * r]);
    }
    return points;
}

/**
 * The band: light lying inside `sil`, reaching `reach` pixels in from its
 * outline, painted `color` and bent by `falloff`.
 *
 * `falloff` above 1 makes the fade aggressive — the outline keeps very nearly
 * all its strength while everything behind it is pushed down hard, so the
 * light drops away from the edge instead of holding and then trailing. It is a
 * curve on the finished band rather than another turn of the erode and blur,
 * because those two set WHERE the fade happens and this sets HOW STEEP it is:
 * a wider blur that starts the fade at the outline also takes the strength out
 * of it there.
 *
 * The band is NORMALISED before the curve is applied, so that an ordinary
 * pixel on the outline comes out at 1 and the object's own opacity is
 * therefore the opacity AT THE OUTLINE. Without it the two knobs are tangled:
 * the subtraction leaves a peak under 1, raising the falloff drives it down
 * with everything else, and "70% at a falloff of 5" silently rendered at 53%.
 * Normalised, `falloff` changes the shape and nothing else.
 */
function inwardBand(sil, scratch, W, H, reach, falloff, color) {
    // The core: the hardened silhouette pulled inward, which is the part the
    // band is taken OUT of. The ring is intersected rather than unioned, which
    // is the difference between eroding and dilating — and dilating is the one
    // thing this must never do.
    const core = scratch();
    const cctx = core.getContext("2d");
    cctx.drawImage(sil, 0, 0);
    cctx.globalCompositeOperation = "destination-in";
    for (const [dx, dy] of ring(Math.max(0.5, reach * GLOW_ERODE))) cctx.drawImage(sil, dx, dy);

    // ...softened, so what gets subtracted has a gradient in it rather than an
    // edge. This is where the ramp comes from: the blur's own curve, read
    // backwards.
    const soft = scratch();
    const sctx = soft.getContext("2d");
    sctx.filter = `blur(${Math.max(0.3, reach * GLOW_SOFTEN)}px)`;
    sctx.drawImage(core, 0, 0);

    const band = scratch();
    const bctx = band.getContext("2d");
    bctx.drawImage(sil, 0, 0);
    bctx.globalCompositeOperation = "destination-out";
    bctx.drawImage(soft, 0, 0);
    bctx.globalCompositeOperation = "source-in";
    bctx.fillStyle = color;
    bctx.fillRect(0, 0, W, H);

    // Normalise, then bend. One pass over the alpha, which is affordable
    // because this sprite is a function of the cutout and the colour and is
    // built once per subject.
    //
    // Normalised against a high PERCENTILE rather than the maximum, and the
    // difference is the whole point of doing it at all. The band's value at a
    // given depth is not constant along the outline: it is the silhouette
    // minus an eroded copy of itself, and erosion takes more off a convex bump
    // than off a flat run, so the band runs brighter around a shoulder than
    // down a sleeve. Its true maximum is therefore some lucky pixel on the
    // sharpest corner of the figure — measured on a real cutout, the ordinary
    // outline sits at 0.76 of it. Normalised by the maximum, `edgeAlpha` would
    // be the opacity of that one corner and the edge the user actually sees
    // would render at three quarters of the number they wrote.
    //
    // The percentile is where the ordinary outline lands, measured: the first
    // fully-opaque ring of a real cutout came out at the 92nd of the non-zero
    // band. Normalising there makes `edgeAlpha` the opacity of the EDGE, which
    // is the thing anyone setting it is looking at.
    const pixels = bctx.getImageData(0, 0, W, H);
    const data = pixels.data;
    const histogram = new Uint32Array(256);
    let lit = 0;
    for (let i = 3; i < data.length; i += 4) {
        const a = data[i];
        if (a > 1) { histogram[a]++; lit++; }
    }
    if (lit > 0) {
        let seen = 0, level = 255;
        const want = lit * GLOW_NORMALISE_AT;
        for (let v = 2; v < 256; v++) {
            seen += histogram[v];
            if (seen >= want) { level = v; break; }
        }
        const curve = new Uint8Array(256);
        for (let v = 0; v < 256; v++) {
            curve[v] = Math.round(255 * Math.pow(Math.min(1, v / level), falloff));
        }
        for (let i = 3; i < data.length; i += 4) data[i] = curve[data[i]];
        bctx.putImageData(pixels, 0, 0);
    }
    return band;
}

/**
 * The contour sprite for one figure in one colour.
 *
 * At the cutout's own resolution and a function of nothing else, so this is
 * built once per subject per colour and the zoom control never touches it.
 * Exactly the cutout's size, with no margin: nothing here is allowed outside
 * the silhouette, so there is nothing to leave room for — which also makes the
 * sprite line up with the figure by simply sharing its box.
 */
function glowSprite(img, spec, color) {
    const side = Math.max(img.width, img.height);
    const num = (key, fallback) => (spec[key] !== undefined ? spec[key] : fallback);
    const reach = Math.max(1, num("edgeRatio", 0.03) * side);
    const falloff = num("edgeFalloff", 5);
    const key = `${img.src}|${color}|${reach}|${falloff}`;
    const hit = _glowSprites.get(key);
    if (hit) return hit;

    const W = Math.max(1, img.width), H = Math.max(1, img.height);
    const scratch = () => {
        const c = document.createElement("canvas");
        c.width = W; c.height = H;
        return c;
    };

    // The mask, hardened. Two buffers, swapped, rather than a fresh pair per
    // round.
    let front = scratch();
    front.getContext("2d").drawImage(img, 0, 0);
    let back = scratch();
    for (let i = 0; i < GLOW_HARDEN_ROUNDS; i++) {
        const bc = back.getContext("2d");
        // Stacked on itself: 2a - a^2, which lifts the middle of the slope.
        bc.globalCompositeOperation = "copy";
        bc.drawImage(front, 0, 0);
        bc.globalCompositeOperation = "source-over";
        bc.drawImage(front, 0, 0);
        // ...then squared in place, which drops the bottom of it. Drawn from
        // itself: a canvas is snapshotted before it is composited onto, so
        // this is a square and not a feedback loop.
        bc.globalCompositeOperation = "destination-in";
        bc.drawImage(back, 0, 0);
        const swap = front; front = back; back = swap;
    }

    // Everywhere the cutout has ANY alpha at all, made solid: its own
    // footprint as a stencil, which is what the rule is enforced with below.
    const support = scratch();
    const supctx = support.getContext("2d");
    supctx.drawImage(img, 0, 0);
    for (let i = 0; i < GLOW_SUPPORT_ROUNDS; i++) supctx.drawImage(support, 0, 0);

    // A little softness given back so the outer edge is antialiased instead of
    // stepped, then clipped to that footprint — because the softening spreads
    // outward as readily as inward. The hardened boundary can sit a pixel
    // inside the mask's own edge, and without this the feather steps over it:
    // measured before it was added, 81 pixels landed outside the cutout.
    // Stencilled rather than multiplied by the raw alpha, which would
    // attenuate the whole outer edge and undo the hardening.
    const sil = scratch();
    const silctx = sil.getContext("2d");
    silctx.filter = `blur(${GLOW_EDGE_FEATHER}px)`;
    silctx.drawImage(front, 0, 0);
    silctx.filter = "none";
    silctx.globalCompositeOperation = "destination-in";
    silctx.drawImage(support, 0, 0);

    const sprite = { band: inwardBand(sil, scratch, W, H, reach, falloff, color) };
    if (_glowSprites.size >= GLOW_CACHE_LIMIT) _glowSprites.clear();
    _glowSprites.set(key, sprite);
    return sprite;
}

/**
 * Draws one figure's contour, over that figure, in its place and at its size.
 *
 * The sprite is the cutout's own size and holds nothing outside the
 * silhouette, so it takes the figure's box and scale unchanged — which is the
 * whole of what makes the light scale with the subject.
 *
 * No shadow: the figure keeps its own, thrown from underneath, and a second
 * shadow on a band sitting on top of the person would be a shadow cast by
 * their outline onto their chest.
 *
 * `edgeAlpha` is the opacity AT THE OUTLINE — the band is normalised so an
 * ordinary edge pixel is 1 before its falloff is applied (see inwardBand), so
 * this number is not quietly scaled by the shape of the fade.
 */
function addFigureGlow(canvas, layer, img, spec, color) {
    const sprite = glowSprite(img, spec, color);
    const w = layer.natW * layer.scale, h = layer.natH * layer.scale;
    canvas.add(new fabric.Image(sprite.band, {
        left: layer.x - w / 2, top: layer.y - h / 2,
        width: sprite.band.width, height: sprite.band.height,
        scaleX: w / sprite.band.width, scaleY: h / sprite.band.height,
        opacity: spec.edgeAlpha !== undefined ? spec.edgeAlpha : 0.7,
        // Screen by default: it can only lighten, and it lightens least where
        // the pixel underneath is already bright — so the light bites on a
        // dark shoulder, fades across a lit cheek, and never replaces what it
        // crosses with a flat tone. It also means it can never darken
        // anything, so there is no way to leave a dirty seam.
        globalCompositeOperation: spec.blend || "screen",
        // Mirrored with the figure, for the reason the figure is: a flipped
        // subject's light is their flipped silhouette, and one left facing the
        // other way shows on any figure that is not symmetrical.
        flipX: !!layer.flip,
        selectable: false, evented: false,
    }));
}

/**
 * The contour colour frame `i`'s figures wear, or null for none.
 *
 * The frame holds an index and the channel holds the list, so the step past
 * the last colour is what "off" means — see `glow` in channels.js. An index
 * carried over from a channel offering more colours than this one lands on
 * none rather than out of bounds, which is the same wrap the button walks.
 */
function figureGlowColor(decor, i) {
    const spec = decor.cutout && decor.cutout.glow;
    const colors = spec && spec.colors;
    if (!colors || !colors.length) return null;
    return colors[figureGlowFor(i)] || null;
}

// ── Title ─────────────────────────────────────────────────────────────────

/**
 * The parts of a title that belong to the FRAME rather than to its channel —
 * where the user dragged it, whether they switched its optional halo on, what
 * colours they set it in, and whether it is wearing the on-canvas handles.
 *
 * Gathered in one place because all of them go to the same call and every
 * caller wants the same answer for a given frame; `interactive` is the one
 * exception and is passed separately, since the same frame is drawn with
 * handles on the preview and without them in every composite.
 */
const titleOpts = (i, interactive) => ({
    glowOn: glowFor(i),
    placement: titleBoxFor(i),
    align: titleAlignFor(i),
    textColor: textColorFor(i),
    highlightColor: highlightColorFor(i),
    panelColor: panelColorFor(i),
    interactive: interactive && titleActiveFor(i),
});

// Where the previewed frame's title ink actually landed, as of the last
// render — what a click on the canvas is tested against (see reframe.js).
//
// Published from the render rather than recomputed on demand, because the box
// is the OUTPUT of a layout that measures fonts, fits every line and may pull
// the whole block in to fit the canvas. Asking that question a second time,
// from a mousedown handler, would be both expensive and a second answer that
// could differ from the one on screen.
let _titleBounds = null;

export const titleBounds = () => _titleBounds;

// ── Preview ───────────────────────────────────────────────────────────────

// Bumped on every updatePreview call and captured by paintFrame — it awaits
// image decodes, and switching frames quickly enough can start a newer render
// before an older one's decode finishes. Without this guard the older call's
// continuation would still run afterward and layer a PREVIOUS frame's
// text/logo on top of the canvas the newer call had already drawn. Whichever
// call's id no longer matches when it resumes just bails.
let _previewRenderId = 0;

// Caches the last decoded preview image by its source URL so that retyping a
// title (which re-renders the same frame's unchanged photo on every
// keystroke) doesn't force a fresh decode each time — without this the
// canvas.clear() below has to wait for a real image load, leaving a blank
// frame on screen for a beat and making the preview flicker.
let _lastPreviewUrl = null, _lastPreviewImg = null;

export function updatePreview() {
    if (!fabricCanvas || !frames().length) return;
    const i = selectedIndex();
    if (!activeUrl(i)) return;
    const renderId = ++_previewRenderId;
    paintFrame(fabricCanvas, i, { interactive: true, renderId })
        .then(() => { if (renderId === _previewRenderId) hideReframeWait(); })
        .catch(() => {});
}

/**
 * The photo a frame is drawn on, decoded.
 *
 * `cached` is for the PREVIEW only, and the cache it uses holds exactly one
 * image: retyping a title redraws the same unchanged photo on every keystroke,
 * and without it the canvas.clear() below waits on a real image load each time,
 * leaving the frame blank for a beat and making the preview flicker.
 *
 * A composite deliberately does not touch it. Rebuilding the grid walks twenty
 * different frames, and a one-entry cache walked in that order never hits —
 * it only evicts the previewed frame's photo, so the next keystroke pays for a
 * decode the cache exists to prevent.
 */
function framePhoto(url, cached) {
    if (!cached) return loadImage(url);
    if (_lastPreviewUrl === url && _lastPreviewImg) return Promise.resolve(_lastPreviewImg);
    return loadImage(url).then(img => {
        _lastPreviewUrl = url;
        _lastPreviewImg = img;
        return img;
    });
}

/**
 * Paints frame `i` onto `cv`, in full, in order — the ONE description of what
 * a thumbnail is made of.
 *
 * The preview and every composite go through here rather than through two
 * parallel copies of the same sequence, which is what they used to be: the
 * layers came out in the same order in both places by nobody's guarantee, and
 * a layer added to one would be a layer missing from every download until
 * somebody noticed.
 *
 * Bottom to top:
 *
 *   1. the backdrop, or the photograph. A channel that composes from a cutout
 *      states a backdrop of its own, and the photo is then only where the
 *      subject was cut FROM — but only once that cutout exists. Until then the
 *      photo is what is drawn, because a grid of twenty identical empty stages
 *      is not a grid anyone can pick a frame out of.
 *   2. the subject, with the shadow its channel gives it.
 *   3. the channel brand mark — above the subject, so a tall person cannot
 *      stand in front of the logo, and below everything the user controls.
 *   4. the title.
 *   5. the user own image overlays, newest on top. Last, because they are the
 *      thing most recently placed by hand, and burying them under the
 *      channel furniture would make them impossible to work with.
 *
 * `renderId` belongs to the preview: this awaits image decodes, and switching
 * frames fast enough starts a newer paint before an older one resumes. A paint
 * whose id no longer matches stops where it is rather than laying a previous
 * frame layers over the current one.
 */
async function paintFrame(cv, i, { interactive = false, renderId } = {}) {
    const live = () => renderId === undefined || renderId === _previewRenderId;

    const channel = channelFor(i);
    const decor = decorFor(channel);
    const layers = layersFor(i);

    // Asked first and on its own, because the answer decides whether the
    // photograph has to be decoded at all: with a backdrop under the subject
    // the frame's own photo is never drawn, and decoding a 1280x720 JPEG per
    // thumbnail per keystroke to throw it away is the most expensive thing on
    // this path. Free after the first frame — decorImage caches the promise.
    //
    // Asked for whatever the frame is carrying, which is the difference
    // between a grid of twenty photographs and a grid of twenty of THIS
    // channel's thumbnails: the backdrop and the mark are the channel's, they
    // are the same on every slot, and waiting for each slot's figures before
    // showing them made every card look like it belonged to some other brand
    // until it was clicked on.
    // Which of the channel's stages, and not just whether it has one: the set
    // is the channel's and the choice is the frame's (see state.backgroundFor).
    // Wrapped rather than clamped, so a frame carried over from a channel with
    // more backdrops than this one lands on a real stage instead of on nothing.
    const stages = decor.background ? decor.background.textures : null;
    const background = (stages && stages.length)
        ? await decorImage(stages[backgroundFor(i) % stages.length])
        : null;
    if (!live()) return null;

    // Asked here for the same reason the backdrop was: whether the frame's own
    // still has to be decoded at all. A card the channel paints a colour is a
    // card the photograph never reaches, and on a channel like that the only
    // thing that still wants the decode is a backdrop made OF the still — see
    // plateNeedsPhoto, which is the whole of that question.
    const plate = photoPlate(decor.photo);
    const wantsPhoto = !background && (!plate || plateNeedsPhoto(plate));

    // The rest together rather than one after another: sequential awaits are
    // round trips through the event loop before a single pixel is drawn, on a
    // path that runs per keystroke.
    const [photo, stamps, figures] = await Promise.all([
        // A backdrop that failed to load leaves the photo as the only picture
        // there is, which is a channel drawn wrong rather than a blank canvas.
        wantsPhoto ? framePhoto(activeUrl(i), renderId !== undefined) : null,
        Promise.all((decor.stamp || []).map(mark => decorImage(mark.texture))),
        Promise.all(layers.map(l => loadCutoutImage(l).catch(() => null))),
    ]);
    if (!live()) return null;

    cv.clear();
    // Three ways the bottom of a frame can be drawn, in the order of how much
    // of the photograph is left: a backdrop replaces it, a card shrinks it onto
    // a ground of the channel's own, and every other channel IS it.
    if (background) addBackground(cv, background);
    else if (plate) addPhotoPlate(cv, plate, photo);
    else if (photo) cv.add(new fabric.Image(photo, {
        left: 0, top: 0, width: CW, height: CH, selectable: false, evented: false,
    }));

    // Every figure, bottom first, so array order is stacking order. Never with
    // handles: see the note at the top of the cutout section.
    //
    // A figure wearing a contour is drawn as two objects — the person, then
    // the band over them — and the pair goes down together rather than every
    // figure first and every band after. Stacking order is the user's: they
    // can put one figure in front of another, and a sheet of bands over the
    // whole group would lay the BACK figure's edge light across the front
    // one's chest.
    //
    // The figure keeps its own drop shadow throughout. Nothing here competes
    // with it: the bands sit on top of the person and inside their outline,
    // and the shadow is thrown from underneath.
    //
    // The caption goes down FIRST on a channel that asked for it (see
    // `cutout.overTitle` in channels.js), which is the one place in this
    // sequence that is a pack's decision rather than this file's. Written as a
    // closure called from one of two points rather than as a second copy of
    // the call, because everything about what gets drawn — the channel it is
    // set in, the hit-testable bounds it hands back — is the same either way
    // and only the moment differs.
    const lines = linesFor(i);
    const drawTitle = () => {
        const bounds = lines.length
            ? usingChannel(channel, () => addTextOverlay(
                cv, lines, hlFor(i), framePresentation(i).flip_text, titleOpts(i, interactive)))
            : null;
        if (interactive) _titleBounds = bounds;
    };
    const titleUnderFigures = !!(decor.cutout && decor.cutout.overTitle);
    if (titleUnderFigures) drawTitle();

    const glowSpec = decor.cutout && decor.cutout.glow;
    const glowColor = figureGlowColor(decor, i);
    const figureShadow = decor.cutout && decor.cutout.shadow;
    layers.forEach((layer, k) => {
        if (!figures[k]) return;
        addLogoOverlay(cv, layer, figures[k], false, figureShadow);
        if (glowSpec && glowColor) addFigureGlow(cv, layer, figures[k], glowSpec, glowColor);
    });
    // Over the backdrop and the figures both, and under everything that has to
    // stay readable — see addScrim.
    //
    // Switched by the Gradient button, which is the same per-frame answer that
    // turns the backend's dark gradient on and off for every other channel.
    // One button for one idea: on a channel that draws the photograph it is
    // baked into what the backend returns, and on one that composes from a
    // cutout it is this. A channel with a scrim is also the only kind where
    // that button does anything at all — with no photograph on screen there
    // was nothing for the backend's version to darken.
    if (decor.scrim && framePresentation(i).gradient) addScrim(cv, decor.scrim);

    // In the order the pack lists them, which is the order a designer wrote
    // them in and the only one that means anything when two marks overlap.
    (decor.stamp || []).forEach((mark, k) => {
        if (stamps[k]) addStamp(cv, mark, stamps[k]);
    });

    // In this frame own channel, whatever the last frame was drawn in. What
    // comes back is where the title ink landed, which is what the canvas
    // gestures hit-test against — null whenever nothing was drawn, so a click
    // cannot land on the box of a title that is not there.
    //
    // ...unless the channel already had it drawn under its figures above, in
    // which case this is where it would have gone and the bounds it left are
    // still the bounds: what moved is the layer the ink is on, not where the
    // ink is.
    if (!titleUnderFigures) drawTitle();

    // In array order, so the stack on screen matches the stack in the record.
    // Only the active one wears the selection chrome, and never on a composite:
    // a thumbnail or a download with a dashed selection box baked into it is a
    // ruined file.
    const active = activeLogoFor(i);
    for (const logo of logosFor(i)) {
        const img = await loadLogoImage(logo).catch(() => null);
        if (!live()) return null;
        if (img) addLogoOverlay(cv, logo, img, interactive && logo === active);
    }
    // The reorder arrows go on last of everything, on the preview only and
    // only when there is more than one copy to reorder — so they are never
    // buried under a layer stacked over the one they belong to, and never
    // baked into a thumbnail or a download.
    const selectedLayer = interactive && layers.length > 1 ? activeLayerFor(i) : null;
    if (selectedLayer) {
        const at = layers.indexOf(selectedLayer);
        addLayerArrows(cv, selectedLayer, at === layers.length - 1, at === 0);
    }

    cv.renderAll();
    return cv;
}

// ── Composites (grid thumbnails and downloads) ────────────────────────────

/**
 * Composes frame `i` (not necessarily the previewed one) at CW×CH onto a
 * canvas of its own.
 *
 * A single shared canvas here used to leak state between frames: an enhance
 * pass finishing in the background and the user editing a DIFFERENT frame's
 * text/logo can both call this concurrently, and with one shared canvas the
 * later clear() could wipe out the earlier call's in-progress composite —
 * observed as one frame's thumbnail briefly showing another frame's title.
 */
export async function buildComposite(i) {
    if (!frames()[i]) return null;
    const element = document.createElement("canvas");
    element.width = CW; element.height = CH;
    const cv = new fabric.StaticCanvas(element, { width: CW, height: CH });
    return paintFrame(cv, i, { interactive: false });
}

// Everything a composite depends on. Regenerating a thumbnail is by far the
// most expensive thing typing a title triggers (a full 1280×720 fabric render
// plus a JPEG encode, times twenty frames), and until this signature existed
// every keystroke redrew all twenty even though nineteen were unchanged.
function thumbSignature(i) {
    const f = frames()[i];
    if (!f) return "";
    const p = framePresentation(i);
    return [
        // The channel decides the font, the colors and the layout a title is
        // drawn with, so a composite made under one channel is not the same
        // composite as under another even when every other input matches.
        // This frame's channel, not the page's: with a brand per frame, the
        // page's would rebuild all twenty thumbnails every time the user tried
        // a channel on one of them, and miss the one that actually changed.
        channelFor(i),
        activeUrl(i), linesFor(i).join(" "), hlFor(i).join(","),
        p.flip_text ? 1 : 0,
        // The halo and the placement change the pixels as surely as the words
        // do. Whether the title is SELECTED is left out for the same reason
        // the active overlay is: selection chrome is never composited.
        glowFor(i) ? 1 : 0,
        titleBoxFor(i) ? `${titleBoxFor(i).dx}:${titleBoxFor(i).dy}:${titleBoxFor(i).scale}` : "",
        // Every overlay, in order. Which one is ACTIVE is deliberately absent:
        // selection chrome is never baked into a composite, so a change of
        // selection cannot change one.
        logosFor(i).map(l => `${l.src.length}:${l.x}:${l.y}:${l.scale}`).join(";"),
        // Which way the title is set, when the user has overruled the channel.
        titleAlignFor(i) || "",
        // ...and which of the channel's colours it is set in, which its
        // picked words are set in, and which its slab is filled with. All
        // three are the FRAME's (see state.textColorFor) and all three are
        // read by every render of it (see titleOpts), so a composite made
        // under one pairing is not the composite made under another.
        //
        // They were missing from this list, and the shape of that bug is
        // worth keeping in mind for the next thing added to titleOpts: the
        // preview changed on the click and the card in the grid did not,
        // because refreshThumb was called, found the signature unchanged and
        // returned. Nothing looked broken — the card simply kept the colours
        // it had until the user happened to edit something else about that
        // frame, and then caught up all at once. Anything paintFrame reads
        // belongs here.
        textColorFor(i) === null ? "" : textColorFor(i),
        highlightColorFor(i) === null ? "" : highlightColorFor(i),
        panelColorFor(i) === null ? "" : panelColorFor(i),
        // Every figure, in stacking order: which picture it is, where it is,
        // and which way round. A figure's src is a blob URL that changes
        // whenever the backend produces a new one, which is exactly when the
        // picture changed. WHICH figure is selected is left out, for the
        // reason the active overlay is: the arrows are chrome and chrome is
        // never composited.
        layersFor(i).map(l => `${l.src}:${l.x}:${l.y}:${l.scale}:${l.flip ? 1 : 0}`).join(";"),
        // Whether the scrim is drawn. It used to be safe to leave out: the
        // gradient was baked into the photograph by the backend, so it
        // arrived through `activeUrl` above. Drawn on this canvas instead, it
        // changes the composite without changing any of the inputs already
        // listed, and a card would keep the picture it had before the button
        // was pressed.
        framePresentation(i).gradient ? 1 : 0,
        // Which of the channel's stages this frame stands on, and which rim
        // its figures wear. Both change every pixel of the composite and
        // neither is visible in anything above, so without them the card in
        // the grid keeps the picture it had before the button was pressed —
        // and keeps it until something else about the frame happens to change.
        backgroundFor(i), figureGlowFor(i),
    ].join("|");
}

const _thumbSignatures = new Map();
let _thumbEl = null, _thumbCtx = null;
let _thumbTimer = null;

export async function refreshThumb(i, force = false) {
    const img = el(`thumb-${i}`);
    if (!img || !frames()[i]) return;

    const signature = thumbSignature(i);
    if (!force && _thumbSignatures.get(i) === signature) return;
    _thumbSignatures.set(i, signature);

    // With no channel selected no title is drawn even when lines are stored
    // (see addTextOverlay), so there is nothing to composite either — and a
    // channel with a backdrop, a brand mark or a cutout always has something,
    // whatever the title says.
    // ...and a channel that draws the photograph as a card always has
    // something too: the card, the ground under it and the tint over it are
    // the composite even on a frame with no title, no overlay and nothing cut
    // out. Left off this list the grid would show twenty raw frames while the
    // preview showed the brand.
    const decor = decorFor(channelFor(i));
    const bare = (!linesFor(i).length || !channelFor(i))
        && !logosFor(i).length && !layersFor(i).length && !decor.stamp
        && !decor.background && !decor.photo;
    if (bare) {   // nothing to composite — the raw frame is the thumbnail
        img.src = activeUrl(i);
        return;
    }
    const cv = await buildComposite(i);
    if (!cv) return;
    if (!_thumbCtx) {
        _thumbEl = document.createElement("canvas");
        _thumbCtx = _thumbEl.getContext("2d");
    }
    // Sized on every call, not only on the first: the thumbnail's shape
    // follows the canvas's, which is the run's (see config.setCanvasSize).
    // Assigning width also clears the element, which is what the explicit
    // clear used to be for.
    _thumbEl.width = THUMB_W; _thumbEl.height = THUMB_H;
    _thumbCtx.drawImage(cv.getElement(), 0, 0, THUMB_W, THUMB_H);
    img.src = _thumbEl.toDataURL("image/jpeg", 0.82);
}

// Which indices have a live composite running, and whether another was asked
// for while it ran.
const _liveThumbs = new Map();

/**
 * The thumbnail for frame `i`, kept up to date DURING a gesture rather than
 * after it.
 *
 * refreshThumb is a full canvas render and a JPEG encode. Called straight from
 * a mousemove handler it would queue one of those per mouse event, and the
 * grid would still be working through the backlog seconds after the drag
 * finished — every card in it a frame of the gesture, arriving in order, long
 * after the gesture was over.
 *
 * So this runs one at a time and remembers only that another was wanted. The
 * card follows the drag at whatever rate the machine can actually composite,
 * and because the flag is a boolean rather than a queue, the render that
 * eventually lands is always built from the CURRENT state — the last position
 * of the drag, not the twentieth-from-last.
 *
 * Deliberately not debounced, unlike refreshAllThumbs. A debounce would hold
 * the card still until the user stopped moving, which is exactly the thing
 * this exists to stop.
 */
export function refreshThumbLive(i) {
    if (_liveThumbs.has(i)) { _liveThumbs.set(i, true); return; }
    _liveThumbs.set(i, false);
    (async () => {
        try {
            do {
                _liveThumbs.set(i, false);
                await refreshThumb(i);
            } while (_liveThumbs.get(i));
        } finally {
            _liveThumbs.delete(i);
        }
    })();
}

/** Debounced: typing a title would otherwise recompose thumbnails on each keystroke. */
export function refreshAllThumbs() {
    clearTimeout(_thumbTimer);
    _thumbTimer = setTimeout(async () => {
        for (let i = 0; i < frames().length; i++) await refreshThumb(i);
    }, 250);
}

export function forgetThumbSignatures() {
    _thumbSignatures.clear();
    _lastPreviewUrl = null;
    _lastPreviewImg = null;
}
