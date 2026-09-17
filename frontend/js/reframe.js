import { CW, CH, el } from "./config.js";
import {
    frames, current, selectedIndex, linesFor, hlFor, logosFor, activeLogoFor,
    framePresentation, backendPresentation, setFrameImage, titleBoxFor, titleActiveFor,
    glowFor, layersFor, activeLayerFor, titleAlignFor, channelFor,
    textColorFor, panelColorFor, highlightColorFor,
} from "./state.js";
import { postImage, discardImage } from "./net.js";
import {
    updatePreview, refreshThumb, refreshThumbLive, logoHitTest, clientToCanvasPoint,
    addLogoOverlay, loadLogoImage, titleBounds,
    cutoutHitTest, layerArrowHitTest, clampCutout, cutoutFitScale,
    photoPlate, openPlate, closePlate,
} from "./compose.js";
import { addTextOverlay, titleHitTest } from "./text.js";
import { titleStyle, decorFor } from "./channels.js";
import { pending, showSpinner, hideReframeWait } from "./ui.js";
import { fetchWide, loadWideImage, peek, viewStamp } from "./wide.js";

let _logoDrag = null;
// The gesture moving or resizing the cutout subject. Its own, for the reason
// the title's is its own: it acts on a different thing and ends differently,
// while behaving identically in between.
let _cutoutDrag = null;
// The gesture moving or resizing the title, for a channel whose titles are
// placed by hand (`movable`). Held apart from the logo's because the two act
// on different things and end differently, not because they behave
// differently — everything about the drag itself is deliberately the same.
let _titleDrag = null;
let _zoomCommitTimer = null;

// How much of the block has to stay on the canvas. A title dragged entirely
// off the edge is not gone — it is still in the frame's record and still
// composited — but there is nothing left on screen to grab it by, so it
// cannot be brought back.
const TITLE_KEEP_ON_CANVAS = 40;

// A title can be pulled down to a quarter of the size its channel sets it at,
// and up to four times it. Bounds rather than freedom: past them the block is
// either too small to read or larger than the canvas, and both are states the
// user has to drag their way back out of.
const TITLE_MIN_SCALE = 0.25, TITLE_MAX_SCALE = 4;

/**
 * Selects the title on the canvas, taking the handles off whatever had them.
 *
 * One thing wears them at a time, exactly as one image overlay does: two
 * dashed boxes on one canvas and a drag has no unambiguous target. The event
 * is what the side panel's overlay list listens to (see main.js) — the same
 * announcement pressing an overlay makes, for the same reason.
 */
function selectTitle(f) {
    if (f.titleActive && f.activeLogo === -1 && f.activeLayer === -1) return;
    f.titleActive = true;
    f.activeLogo = -1;
    f.activeLayer = -1;
    document.dispatchEvent(new CustomEvent("logo-selection-changed"));
}

/**
 * Takes the reorder buttons off whichever figure is wearing them.
 *
 * What pressing the TITLE does on a channel that composes from cutouts. The
 * arrows and the cross belong to a figure, they are drawn over the picture,
 * and there was no way to put them away again short of pressing another
 * figure — so the title, which is the other thing on the canvas and is drawn
 * above the figures anyway, is where a press means "I am done with that one".
 * Pressing any figure brings them back.
 */
function deselectLayers(f) {
    if (f.activeLayer === -1) return false;
    f.activeLayer = -1;
    document.dispatchEvent(new CustomEvent("logo-selection-changed"));
    return true;
}

/**
 * Puts the figure buttons away from outside the canvas.
 *
 * The arrows and the cross are drawn ON the picture, so every way of putting
 * them down used to be a press on the picture too — press the title, press
 * another figure. That leaves the one moment they are most in the way with no
 * answer at all: the user turns from the canvas to the title fields and types,
 * and three circles stay sitting on the thumbnail through every keystroke,
 * over the very composition they are trying to read.
 *
 * So focusing a title field is also a way of saying "I am done with that
 * figure" — see main.js, where the fields are wired. Repaints only when
 * something was actually selected, since this fires on every focus.
 */
export function dismissFigureHandles() {
    const f = current();
    if (!f || !deselectLayers(f)) return;
    updatePreview();
    refreshThumb(selectedIndex());
}

/**
 * The topmost layer of the cutout under a canvas-space point, as an index, or
 * -1.
 *
 * Topmost FIRST, because the layers are drawn in array order and the one the
 * user can see under the cursor is the last one that covers it. Tested on each
 * layer's own alpha, so a press that lands on the backdrop showing through the
 * front copy still reaches the copy behind it — which, with two or three of
 * the same person overlapping, is most of the canvas.
 */
function layerUnder(i, cx, cy) {
    const layers = layersFor(i);
    for (let k = layers.length - 1; k >= 0; k--) {
        if (cutoutHitTest(layers[k], cx, cy)) return k;
    }
    return -1;
}

/** The cutout rules of the frame on screen, or null when its channel has none. */
const cutoutSpec = (i) => decorFor(channelFor(i)).cutout;

/**
 * Pushes one copy up or down the stack, and keeps the selection on the copy
 * that moved rather than on the position it left.
 *
 * `direction` is +1 toward the viewer and -1 away from them. There is no
 * "below the background" to fall off: the backdrop is what the stack is drawn
 * on rather than a member of it, so the ends of the array are the ends of the
 * movement.
 */
function moveLayer(f, at, direction) {
    const to = at + direction;
    if (at < 0 || to < 0 || to >= f.layers.length) return;
    const [layer] = f.layers.splice(at, 1);
    f.layers.splice(to, 0, layer);
    f.activeLayer = to;
    f.edited = true;
}

/**
 * Takes one figure off the thumbnail.
 *
 * Never the last one: a cutout channel with nobody on the backdrop is not a
 * thumbnail, and there would be nothing left on the canvas to press to get a
 * figure back. The button is only ever drawn where there is more than one, so
 * this guard is the belt to that brace.
 *
 * The frame's own record of what it was cut under goes with it, which is what
 * stops ensureCutout quietly putting the figure back: that count no longer
 * matches the channel's, and the frame is now an arrangement the user made
 * rather than the one the channel proposed.
 */
function removeLayer(f, at) {
    if (f.layers.length <= 1 || at < 0 || at >= f.layers.length) return;
    const [layer] = f.layers.splice(at, 1);
    if (layer.src) URL.revokeObjectURL(layer.src);
    f.activeLayer = Math.min(at, f.layers.length - 1);
    f.cutoutKey = null;
    f.figuresRemoved = true;
    f.edited = true;
}

// Identifies the mouse-down currently holding the button. Arming a reframe
// drag needs the frame's pre-crop view, which is a fetch (and, on a frame
// whose base hasn't been built yet, a slow one) — so a plain click can be
// completely over, mouseup and all, before the awaits in the mousedown
// handler resume. Resuming anyway put the drag ghost layers up over the
// canvas with no gesture left to take them down: the user saw a second,
// half-resolution image appear over the preview a moment after clicking it
// and stay there, and because pending.drag was left set, every later render
// was blocked from clearing it (see ui.hideReframeWait) and mouse movement
// alone panned the frame with no button held. Bumped on each mousedown and
// zeroed on mouseup, so the setup can tell whether its gesture still exists.
let _gestureSeq = 0;
let _activeGesture = 0;

export function updateZoomFill() {
    const slider = el("zoomSlider");
    const min = parseFloat(slider.min), max = parseFloat(slider.max), val = parseFloat(slider.value);
    const frac = (Number.isFinite(min) && Number.isFinite(max) && max > min && Number.isFinite(val))
        ? (val - min) / (max - min) : 0.5;
    slider.style.setProperty("--zoom-fill", frac);
}

/**
 * What the zoom control means on the frame in front of the user, expressed as
 * the slider's own bounds and value.
 *
 * Two answers, because there are two things a channel can mean by "zoom".
 *
 * Ordinarily it resizes the CROP WINDOW over the photograph: zooming in shows
 * less of the source, which is a re-render on the backend and is bounded by
 * how much upscale the frame's automatic framing left unspent.
 *
 * On a channel that composes from a cutout there is no photograph on screen to
 * zoom — the background is the channel's own backdrop, and the only thing the
 * user can mean is the person standing on it. So the control resizes the
 * SUBJECT, the backdrop does not move, and nothing is sent anywhere.
 *
 * Expressed as a multiple of where the CHANNEL put them (see
 * compose.cutoutFitScale), so 1 is always the placement they arrived at and
 * the two ends of the slider are "much smaller than intended" and "much
 * larger". The range is deliberately wide and is bounded by nothing on the
 * canvas: a subject blown up past the frame, cropped to a face, is a
 * composition somebody may want, and the placement is a suggestion.
 */
function cutoutZoomOf(i) {
    const spec = cutoutSpec(i);
    if (!spec) return null;
    // The selected figure, or the front one when nothing is selected. Not "all
    // of them": on a frame carrying three figures at three deliberate depths,
    // one control resizing all three would flatten the arrangement the moment
    // it was touched.
    const layers = layersFor(i);
    const layer = activeLayerFor(i) || layers[layers.length - 1];
    if (!layer) return null;
    return { spec, layer, value: layer.scale / cutoutFitScale(spec, layer.natW, layer.natH) };
}

// The crop-zoom range to show before the frame's own bounds have arrived —
// what the markup used to carry alone. Stated here as well, because the
// slider now has a SECOND meaning with entirely different bounds (see
// cutoutZoomOf), and leaving the markup's numbers as the only fallback meant
// a frame moved off a cutout channel kept the subject's 0.35-to-1 range and
// silently refused to zoom the photograph past its automatic framing.
const CROP_ZOOM_MIN = 0.5, CROP_ZOOM_MAX = 4;

/** Reflects a newly selected frame's zoom level and its own slider bounds. */
export function syncZoomSliderFor(frameId) {
    const slider = el("zoomSlider");
    const subject = cutoutZoomOf(selectedIndex());
    if (subject) {
        slider.min = subject.spec.minZoom !== undefined ? subject.spec.minZoom : 0.35;
        slider.max = subject.spec.maxZoom !== undefined ? subject.spec.maxZoom : 3;
        slider.value = subject.value;
        updateZoomFill();
        return;
    }
    const w = peek(frameId);
    slider.min = (w && w.zoom_min) || CROP_ZOOM_MIN;
    slider.max = (w && w.zoom_max) || CROP_ZOOM_MAX;
    slider.value = (w && w.zoom) || 1;
    updateZoomFill();
}

export function setupReframeInteraction() {
    // StaticCanvas renders straight into #thumbCanvas (no upper-canvas event
    // layer), so drag listeners attach to it directly.
    const dragCanvas = el("thumbCanvas");
    const wideLayer = el("wideLayer");
    const liveCropLayer = el("liveCropLayer");
    const liveCtx = liveCropLayer.getContext("2d");
    if (!dragCanvas) return;

    // Offscreen fabric canvas holding everything that sits ON the photo — the
    // title and the image overlay — but not the photo itself. Reused instead
    // of a bare canvas 2D re-draw so the drag preview renders them through the
    // exact same font-fit/shadow/highlight/handle logic as the real preview,
    // with nothing duplicated.
    let overlayCanvas = null, overlayEl = null;

    /**
     * Everything drawn here is what the live-crop layer composites over the
     * moving photo. The photo pans underneath; these stay put, because they're
     * positioned in canvas space and reframing doesn't move them.
     *
     * The image overlay belongs here for the same reason the title does: the
     * live-crop layer covers the whole preview canvas for the duration of the
     * gesture, so anything left behind on that canvas is simply not on screen.
     * Leaving the logo out made it vanish the moment a reframe drag started
     * and reappear on drop.
     */
    function snapshotOverlays() {
        if (!overlayCanvas) {
            overlayEl = document.createElement("canvas");
            overlayEl.width = CW;
            overlayEl.height = CH;
            overlayCanvas = new fabric.StaticCanvas(overlayEl, { width: CW, height: CH });
        } else if (overlayEl.width !== CW || overlayEl.height !== CH) {
            // Built once and kept, so a run at a different canvas size would
            // otherwise go on compositing the title and the overlays into the
            // previous run's shape — see compose.sizeOverlayCanvases for the
            // same fault on the layer this one is drawn into.
            overlayEl.width = CW;
            overlayEl.height = CH;
            overlayCanvas.setDimensions({ width: CW, height: CH });
        }
        overlayCanvas.clear();
        const i = selectedIndex();
        // The cutout is deliberately absent. This snapshot sits over the
        // PHOTOGRAPH while it pans, and on a cutout channel that photograph is
        // not the thumbnail — it is the window the subject will be cut out of,
        // which is the whole of what this gesture chooses. Compositing the
        // previous window's subject over it would put two of the same person
        // on screen, one of whom is about to stop existing, and hide part of
        // the frame the user is trying to see.
        if (linesFor(i).length) {
            // Every one of the frame's own answers about its title, because
            // this snapshot sits over the preview for the length of a drag
            // and any of them left out is a title that visibly changes the
            // moment the mouse goes down. (compose.titleOpts is the same
            // list; it is not shared because that one also decides the
            // handles from a second argument, and here they are simply the
            // preview's.)
            addTextOverlay(overlayCanvas, linesFor(i), hlFor(i), framePresentation(i).flip_text, {
                glowOn: glowFor(i),
                placement: titleBoxFor(i),
                align: titleAlignFor(i),
                textColor: textColorFor(i),
                highlightColor: highlightColorFor(i),
                panelColor: panelColorFor(i),
                // Handles included, matching the preview underneath — see the
                // note on the overlay's below.
                interactive: titleActiveFor(i),
            });
        }
        // Selection chrome included, matching the preview underneath — the
        // handles disappearing for the length of a drag would read as the
        // overlay having been deselected.
        const active = activeLogoFor(i);
        for (const logo of logosFor(i)) {
            if (logo._img) addLogoOverlay(overlayCanvas, logo, logo._img, logo === active);
        }
        overlayCanvas.renderAll();
    }

    /**
     * The crop window lives in the zoom-1 wide-view space the backend serves;
     * the fine-zoom multiplier shrinks (zoom in) or grows (zoom out) how much
     * of that space the TARGET-sized output window covers. Pannable bounds
     * scale with the window: the overpan margin stays a fixed fraction OF THE
     * WINDOW, so the autofill strip can never grow past the same share of the
     * output regardless of zoom.
     */
    function panLimits(wide, zoom) {
        const winW = wide.target_w / zoom, winH = wide.target_h / zoom;
        const ov = wide.overpan || {
            min_x: 0, max_x: wide.scaled_w - wide.target_w,
            min_y: 0, max_y: wide.scaled_h - wide.target_h,
        };
        const overX = -ov.min_x / zoom, overY = -ov.min_y / zoom;
        return {
            min_x: -overX, max_x: wide.scaled_w - winW + overX,
            min_y: -overY, max_y: wide.scaled_h - winH + overY,
        };
    }

    function clampCrop(wide, x, y, zoom) {
        const lim = panLimits(wide, zoom);
        return [
            Math.max(lim.min_x, Math.min(x, lim.max_x)),
            Math.max(lim.min_y, Math.min(y, lim.max_y)),
        ];
    }

    function positionWideLayer(wide, cropX, cropY, displayScale) {
        const zoom = wide.zoom || 1;
        const f = displayScale * zoom;
        const mirror = framePresentation(selectedIndex()).flip_image;
        // The element is stretched to the view's FULL-resolution size even
        // though the served bitmap is smaller (see preview_scale) — the CSS
        // box is what has to line up with the canvas, not the pixel count.
        wideLayer.style.width  = (wide.scaled_w * f) + "px";
        wideLayer.style.height = (wide.scaled_h * f) + "px";
        // scaleX(-1) mirrors the element about its own center while `left`
        // still positions the untransformed box, so the offset has to be
        // measured from the far edge: the window's left edge sits at
        // (scaled_w - cropX - windowW) once the image is mirrored.
        wideLayer.style.transform = mirror ? "scaleX(-1)" : "";
        wideLayer.style.left = (mirror
            ? -(wide.scaled_w - cropX - wide.target_w / zoom) * f
            : -cropX * f) + "px";
        wideLayer.style.top  = (-cropY * f) + "px";
    }

    /**
     * Draws the exact window that would be produced by dropping at (cropX,
     * cropY) at the current zoom — this is what makes the image *inside* the
     * frame move live with the drag, not just the desaturated surroundings.
     * The window may extend past the source frame's edges (overpan); the
     * uncovered strip is drawn as a flat dark placeholder here and gets
     * autofilled by inpainting on the backend when the user drops. The title
     * and image overlay are composited on top from the cached snapshot so
     * they stay visible throughout the drag.
     *
     * Crop coordinates are in the view's FULL-resolution space while the
     * bitmap is served at `preview_scale` of it, so every source rectangle
     * has to be mapped through that factor.
     *
     * On a channel that draws the photograph as a card (see `photo` in
     * channels.js) the whole of that arithmetic is unchanged and lands inside
     * the card instead, because openPlate leaves the context in canvas
     * coordinates scaled into it. The drag has to show the composition it will
     * drop into: a preview that goes full-bleed the moment the mouse goes down
     * and snaps back to a card on release is a crop chosen against a picture
     * the user is not going to get.
     */
    function drawLiveCrop(wideImg, cropX, cropY, wide) {
        const zoom = wide.zoom || 1;
        const ps = wide.preview_scale || 1;
        const winW = wide.target_w / zoom, winH = wide.target_h / zoom;
        const plate = photoPlate(decorFor(channelFor(selectedIndex())).photo);
        if (plate) {
            openPlate(liveCtx, plate);
        } else {
            liveCtx.fillStyle = "#181818";
            liveCtx.fillRect(0, 0, wide.target_w, wide.target_h);
        }

        const sx = Math.max(cropX, 0), sy = Math.max(cropY, 0);
        const sw = Math.min(cropX + winW, wide.scaled_w) - sx;
        const sh = Math.min(cropY + winH, wide.scaled_h) - sy;
        if (sw > 0 && sh > 0) {
            // Mirror only the photo — the text overlay composited below is
            // drawn unmirrored (it flips by layout, not by glyph).
            liveCtx.save();
            if (framePresentation(selectedIndex()).flip_image) {
                liveCtx.translate(wide.target_w, 0);
                liveCtx.scale(-1, 1);
            }
            liveCtx.drawImage(
                wideImg,
                sx * ps, sy * ps, sw * ps, sh * ps,
                (sx - cropX) * zoom, (sy - cropY) * zoom, sw * zoom, sh * zoom,
            );
            liveCtx.restore();
        }
        // The tint goes on before the snapshot and not after: it is a
        // treatment of the photograph, and the title and the overlays are
        // things standing on top of it. Painting it last would drag the
        // caption's colour with the picture's.
        if (plate) closePlate(liveCtx, plate);
        if (overlayEl) liveCtx.drawImage(overlayEl, 0, 0, wide.target_w, wide.target_h);
    }

    /**
     * Posts the frame's current crop/zoom and hands the result to the
     * preview, spinner flow included — shared by the drag drop and the
     * fine-zoom commit.
     */
    async function commitReframe(wide, frameId) {
        showSpinner();
        pending.reframing = true;
        let handedOff = false;
        try {
            const idx = frames().findIndex(fr => fr.frame_id === frameId);
            const res = await postImage("/reframe-frame", {
                frame_id: frameId,
                crop_x: Math.round(wide.crop_x),
                crop_y: Math.round(wide.crop_y),
                zoom: wide.zoom || 1,
                ...backendPresentation(idx),
            });
            if (res.ok && idx === -1) discardImage(res);   // the grid moved on; nothing will show this
            if (res.ok && idx !== -1) {
                const f = frames()[idx];
                // /reframe-frame crops the frame's already-restored base, so
                // this IS the final enhanced result at whatever preset is
                // locked in. No follow-up /enhance-frame call is needed —
                // that would just re-request the same cached pipeline output
                // over the network on every single drag/zoom commit.
                setFrameImage(f, "plainUrl", res.url);
                setFrameImage(f, "enhancedUrl", null);   // stale — held the previous crop's image
                // New canvas generation: any enhance still in flight belongs
                // to the previous crop, and the version bump makes it discard
                // its result and redo.
                f.version = (f.version || 0) + 1;
                pending.reframing = false;   // before updatePreview, so its render may tear the overlay down
                if (selectedIndex() === idx) {
                    updatePreview();
                    handedOff = true;
                }
                refreshThumb(idx);
                // The subject was cut out of the window this drag has just
                // replaced, so a channel that composes from one needs another.
                // Announced rather than called: ensureCutout is editor.js's,
                // and reaching for it from here would close an import cycle
                // (editor already imports this module). main.js owns both and
                // does the wiring, exactly as it does for the selection event.
                document.dispatchEvent(new CustomEvent("frame-photo-changed", { detail: { index: idx } }));
            }
        } catch (_) { /* network error — fall through to the cleanup below */ }
        pending.reframing = false;
        if (!handedOff) hideReframeWait();   // failure/edge paths: don't leave the overlay stuck
    }

    dragCanvas.addEventListener("mousedown", async (e) => {
        if (e.button !== 0) return;
        const f = current();
        if (!f) return;
        const gesture = (_activeGesture = ++_gestureSeq);

        // Logo hit-test runs first, synchronously — a click on the logo starts
        // a move/resize gesture instead of falling through to the background
        // reframe-pan drag (which would kick off an unnecessary fetch).
        if (f.logos.length) {
            const rect0 = dragCanvas.getBoundingClientRect();
            const displayScale0 = rect0.width / CW;
            const [cx, cy] = clientToCanvasPoint(e, rect0, displayScale0);
            // Topmost first: the overlays are drawn in array order, so the one
            // the user sees under the cursor is the LAST that covers it, and
            // testing bottom-up would hand the press to a picture buried
            // behind the one they aimed at.
            for (let i = f.logos.length - 1; i >= 0; i--) {
                const logo = f.logos[i];
                const hit = logoHitTest(logo, cx, cy);
                if (!hit) continue;
                // Pressing an overlay is also how it becomes the active one,
                // so the handles follow the thing being dragged.
                if (f.activeLogo !== i || f.titleActive) {
                    f.activeLogo = i;
                    f.titleActive = false;
                    // Announced rather than called: the side panel's list of
                    // overlays is editor.js's, and reaching for it directly
                    // from here would close an import cycle (editor already
                    // imports this module). main.js owns both and does the
                    // wiring, the way it does for every other control.
                    document.dispatchEvent(new CustomEvent("logo-selection-changed"));
                }
                dragCanvas.classList.remove("hover-move", "hover-resize");
                _logoDrag = {
                    mode: hit, logo, displayScale: displayScale0,
                    startClientX: e.clientX, startClientY: e.clientY,
                    startX: logo.x, startY: logo.y, startScale: logo.scale,
                    natW: logo.natW, natH: logo.natH,
                };
                dragCanvas.classList.add(hit === "resize" ? "hover-resize" : "hover-move");
                e.preventDefault();
                return;
            }
        }

        // Then the subject, for a channel that composes from a cutout. After
        // the user's own overlays and the title for the same reason they come
        // before the pan: what is on top is what the click is aiming at, and
        // the subject sits under both of them.
        //
        // The test is on the subject's own pixels rather than on their box
        // (see compose.cutoutHitTest), which is what lets a press on the
        // backdrop showing through between an arm and a body fall through to
        // whatever is behind it — another copy, or nothing at all.
        const layers = layersFor(selectedIndex());
        if (layers.length) {
            const rect0 = dragCanvas.getBoundingClientRect();
            const displayScale0 = rect0.width / CW;
            const [cx, cy] = clientToCanvasPoint(e, rect0, displayScale0);

            // The title first, because it is drawn over the figures and a
            // press on it is aimed at it. On a channel whose titles are placed
            // by hand that press starts a drag further down; here it means
            // "put the buttons away" (see deselectLayers), and it is consumed
            // so that a figure lying under the words does not take it back.
            if (!titleStyle().movable && titleHitTest(titleBounds(), cx, cy)) {
                if (deselectLayers(f)) {
                    updatePreview();
                    refreshThumb(selectedIndex());
                }
                e.preventDefault();
                return;
            }

            // The reorder arrows before the figure they belong to: they are
            // drawn over everything and a press on one is aimed at the button,
            // not at whichever figure happens to be underneath it.
            const selected = layers.length > 1 ? activeLayerFor(selectedIndex()) : null;
            const button = selected && layerArrowHitTest(selected, cx, cy);
            if (button === "remove") {
                removeLayer(f, layers.indexOf(selected));
                updatePreview();
                refreshThumb(selectedIndex());
                e.preventDefault();
                return;
            }
            if (button) {
                moveLayer(f, layers.indexOf(selected), button === "up" ? 1 : -1);
                updatePreview();
                refreshThumb(selectedIndex());
                e.preventDefault();
                return;
            }

            const k = layerUnder(selectedIndex(), cx, cy);
            if (k !== -1) {
                const layer = layers[k];
                // Pressing a figure is also how it becomes the selected one,
                // so the arrows — and the zoom, and the flip — follow the
                // thing being dragged.
                if (f.activeLayer !== k) {
                    f.activeLayer = k;
                    // Announced as well as drawn: "Flip image" acts on the
                    // selected figure and has to report ITS state, and the
                    // panel is editor.js's to update — see the note on the
                    // overlay list for why this module announces rather than
                    // reaches across.
                    document.dispatchEvent(new CustomEvent("logo-selection-changed"));
                    updatePreview();
                }
                dragCanvas.classList.remove("hover-move", "hover-resize");
                _cutoutDrag = {
                    layer, spec: cutoutSpec(selectedIndex()),
                    displayScale: displayScale0,
                    startClientX: e.clientX, startClientY: e.clientY,
                    startX: layer.x, startY: layer.y,
                };
                dragCanvas.classList.add("hover-move");
                e.preventDefault();
                return;
            }
        }

        // Then the title, for a channel that lets its titles be placed by
        // hand. After the overlays because they are drawn over it and the
        // topmost thing under the cursor is the one being aimed at; before
        // the pan below for the reason the overlays are — a press that lands
        // on something has no business fetching a wide view to drag.
        if (titleStyle().movable) {
            const rect0 = dragCanvas.getBoundingClientRect();
            const displayScale0 = rect0.width / CW;
            const [cx, cy] = clientToCanvasPoint(e, rect0, displayScale0);
            const bounds = titleBounds();
            const hit = titleHitTest(bounds, cx, cy);
            if (hit) {
                selectTitle(f);
                const box = f.titleBox || { dx: 0, dy: 0, scale: 1 };
                dragCanvas.classList.remove("hover-move", "hover-resize");
                _titleDrag = {
                    mode: hit, displayScale: displayScale0,
                    startClientX: e.clientX, startClientY: e.clientY,
                    startDx: box.dx, startDy: box.dy, startScale: box.scale,
                    // Where the block sits with nothing applied to it, and how
                    // wide it is at that size — both taken from the render the
                    // user is looking at, so the drag moves what they pressed.
                    homeLeft: bounds.left - box.dx, homeTop: bounds.top - box.dy,
                    baseW: bounds.baseW, baseH: bounds.baseH,
                };
                dragCanvas.classList.add(hit === "resize" ? "hover-resize" : "hover-move");
                // Pressing the title is also how it becomes the selected
                // thing, so the handles have to appear on the press rather
                // than on the first movement.
                updatePreview();
                e.preventDefault();
                return;
            }
        }

        // Nothing else was hit, so this is a press on the background — which
        // on a channel that composes from a cutout is the channel's own
        // backdrop, and there is no photograph behind it to pan. The frame's
        // crop no longer decides anything there either: the subject is cut out
        // of the WHOLE photo (see api._cutout_sync), so dragging the window
        // around would spend a re-render and a re-cut to produce the same
        // picture. Nothing happens, which is the honest answer.
        if (cutoutSpec(selectedIndex())) return;

        const wide = await fetchWide(f.frame_id);
        if (!wide) return;
        const wideImg = await loadWideImage(wide);
        if (!wideImg) return;
        // Decoded before the snapshot, which is synchronous. Normally a no-op
        // (the upload already cached it), but a logo carried over from another
        // frame can reach here undecoded, and that must not be the difference
        // between the overlay surviving the drag and not.
        await Promise.all(f.logos.map(l => loadLogoImage(l).catch(() => {})));
        if (_activeGesture !== gesture) return;   // button already released — this was a click, not a drag

        const rect = dragCanvas.getBoundingClientRect();
        pending.drag = {
            frameId: f.frame_id, wide, wideImg,
            stamp: viewStamp(f.frame_id),
            displayScale: rect.width / CW,
            startClientX: e.clientX, startClientY: e.clientY,
            cropX: wide.crop_x, cropY: wide.crop_y,
        };

        wideLayer.src = wide.url;
        positionWideLayer(wide, wide.crop_x, wide.crop_y, pending.drag.displayScale);
        wideLayer.style.display = "block";
        snapshotOverlays();
        drawLiveCrop(wideImg, wide.crop_x, wide.crop_y, wide);
        liveCropLayer.style.display = "block";
        dragCanvas.classList.remove("hover-move", "hover-resize");
        dragCanvas.classList.add("dragging");
        e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
        if (_titleDrag) {
            const s = _titleDrag;
            const f = current();
            if (!f) { _titleDrag = null; return; }
            const dx = (e.clientX - s.startClientX) / s.displayScale;
            const dy = (e.clientY - s.startClientY) / s.displayScale;
            const box = { ...(f.titleBox || { dx: 0, dy: 0, scale: 1 }) };
            if (s.mode === "move") {
                // Clamped by where the BLOCK would land rather than by the
                // offset itself: the same offset means something different for
                // a title the user has already scaled up, and what has to stay
                // reachable is the thing on screen.
                const left = s.homeLeft + s.startDx + dx;
                const top = s.homeTop + s.startDy + dy;
                const w = s.baseW * box.scale, h = s.baseH * box.scale;
                const keep = TITLE_KEEP_ON_CANVAS;
                box.dx = Math.max(keep - w, Math.min(left, CW - keep)) - s.homeLeft;
                box.dy = Math.max(keep - h, Math.min(top, CH - keep)) - s.homeTop;
            } else {
                // Corner drag, uniform: the block keeps its proportions and
                // its layout, and only its size changes (see text.place).
                const width = Math.max(1, s.baseW * s.startScale + dx);
                box.scale = Math.max(TITLE_MIN_SCALE, Math.min(width / s.baseW, TITLE_MAX_SCALE));
            }
            f.titleBox = box;
            f.edited = true;
            updatePreview();
            // ...and the card in the grid with it. The preview and the
            // thumbnail are the same frame drawn twice, and a grid that only
            // catches up when the button is released spends every drag
            // showing the user something that is no longer true.
            refreshThumbLive(selectedIndex());
            return;
        }

        if (_cutoutDrag) {
            const s = _cutoutDrag;
            const f = current();
            // The record is held rather than looked up, exactly as the
            // overlay's is: a cutout replaced mid-drag (an enhance pass
            // landing) is a different picture, and the gesture must not
            // silently continue on it.
            // The layer itself is held, not its index: a re-cut landing
            // mid-drag replaces every figure on the frame, and the gesture
            // must not silently continue on a different picture.
            if (!f || !layersFor(selectedIndex()).includes(s.layer)) {
                _cutoutDrag = null;
                return;
            }
            const dx = (e.clientX - s.startClientX) / s.displayScale;
            const dy = (e.clientY - s.startClientY) / s.displayScale;
            s.layer.x = s.startX + dx;
            s.layer.y = s.startY + dy;
            // Only so that a subject dragged off the edge can be dragged back
            // — the drag is otherwise free of the channel's placement, which
            // is where they started and not where they must stay. See
            // compose.clampCutout.
            clampCutout(s.spec, s.layer);
            f.edited = true;
            updatePreview();
            refreshThumbLive(selectedIndex());
            return;
        }

        if (_logoDrag) {
            const s = _logoDrag;
            const f = current();
            // The overlay itself is held, not its index: deleting another one
            // mid-drag would shift every index after it and the gesture would
            // silently continue on a different picture.
            if (!f || !f.logos.includes(s.logo)) { _logoDrag = null; return; }
            const dx = (e.clientX - s.startClientX) / s.displayScale;
            const dy = (e.clientY - s.startClientY) / s.displayScale;
            if (s.mode === "move") {
                s.logo.x = Math.max(0, Math.min(CW, s.startX + dx));
                s.logo.y = Math.max(0, Math.min(CH, s.startY + dy));
            } else {
                // Corner-drag resize, uniform scale (no distortion) — width
                // follows horizontal mouse travel, height follows via the
                // image's own aspect ratio.
                const newW = Math.max(20, s.natW * s.startScale + dx);
                s.logo.scale = Math.max(0.02, newW / s.natW);
            }
            updatePreview();
            refreshThumbLive(selectedIndex());
            return;
        }

        const s = pending.drag;
        if (!s) {
            // No active gesture — just hover feedback over the logo, so the
            // user can tell where to click before they do.
            const f = current();
            let hit = null;
            if (f && (f.logos.length || layersFor(selectedIndex()).length || titleStyle().movable)) {
                const rect = dragCanvas.getBoundingClientRect();
                const [cx, cy] = clientToCanvasPoint(e, rect, rect.width / CW);
                for (let i = f.logos.length - 1; i >= 0 && !hit; i--) {
                    hit = logoHitTest(f.logos[i], cx, cy);
                }
                // Same order as the press: whatever the click would act on is
                // what the cursor promises it will.
                if (!hit && layerUnder(selectedIndex(), cx, cy) !== -1) hit = "move";
                if (!hit && titleStyle().movable) hit = titleHitTest(titleBounds(), cx, cy);
            }
            dragCanvas.classList.toggle("hover-move", hit === "move");
            dragCanvas.classList.toggle("hover-resize", hit === "resize");
            return;
        }

        // Screen pixels -> zoom-1 wide-view units: through the display scale
        // AND the current zoom (zoomed in, the same mouse travel covers less
        // of the source).
        //
        // On a mirrored photo the horizontal delta is negated. Panning moves
        // the crop window through SOURCE space, but a mirrored window maps
        // source x to screen as (target_w - (x - cropX)*zoom) — so the same
        // window movement pushes the picture the opposite way on screen.
        // Without this the image ran away from the cursor, which is what made
        // dragging a flipped frame feel inverted.
        const zoom = s.wide.zoom || 1;
        const xDir = framePresentation(selectedIndex()).flip_image ? -1 : 1;
        const dx = xDir * (e.clientX - s.startClientX) / (s.displayScale * zoom);
        const dy = (e.clientY - s.startClientY) / (s.displayScale * zoom);

        [s.cropX, s.cropY] = clampCrop(s.wide, s.wide.crop_x - dx, s.wide.crop_y - dy, zoom);

        positionWideLayer(s.wide, s.cropX, s.cropY, s.displayScale);
        drawLiveCrop(s.wideImg, s.cropX, s.cropY, s.wide);
    });

    window.addEventListener("mouseup", async () => {
        // First, and unconditionally: whatever gesture the button was holding
        // is over, including one whose setup hasn't finished awaiting yet.
        _activeGesture = 0;
        if (_cutoutDrag) {
            _cutoutDrag = null;
            refreshThumb(selectedIndex());
            return;
        }
        if (_titleDrag) {
            _titleDrag = null;
            refreshThumb(selectedIndex());
            return;
        }
        if (_logoDrag) {
            _logoDrag = null;
            refreshThumb(selectedIndex());
            return;
        }
        const s = pending.drag;
        if (!s) return;
        pending.drag = null;
        dragCanvas.classList.remove("dragging");
        wideLayer.style.display = "none";

        // A Variation landing mid-drag put a different photo in this slot: the
        // ghost that was just panned belongs to the one it replaced, and the
        // window it ended on was measured against that photo's geometry.
        // Dropping it would commit a crop for a frame that no longer exists.
        if (viewStamp(s.frameId) !== s.stamp) {
            liveCropLayer.style.display = "none";
            return;
        }

        if (Math.round(s.cropX) === Math.round(s.wide.crop_x) &&
            Math.round(s.cropY) === Math.round(s.wide.crop_y)) {
            liveCropLayer.style.display = "none";   // no real movement — nothing to do
            return;
        }

        s.wide.crop_x = s.cropX;
        s.wide.crop_y = s.cropY;

        // Keep the live-crop layer up, frozen at the dropped position, with a
        // spinner over it while the backend re-renders — hiding it right away
        // would flash the old crop on the canvas underneath and then "blink"
        // to the new one. The frozen layer comes down when the fresh render
        // lands on the canvas; the spinner stays through the whole pipeline.
        await commitReframe(s.wide, s.frameId);
    });

    // ── Fine zoom ─────────────────────────────────────────────────────────
    const zoomSlider = el("zoomSlider");

    async function applyZoom(rawZoom) {
        const f = current();
        if (!f) return;

        // On a cutout channel the control belongs to the subject and to
        // nothing else — see cutoutZoomOf. Handled before the wide view is
        // fetched, not after: there is no crop being changed here, so there is
        // no reason to pull a pre-crop render of a photograph nobody is
        // looking at over the network.
        const subject = cutoutZoomOf(selectedIndex());
        if (subject) {
            const { spec, layer } = subject;
            const fit = cutoutFitScale(spec, layer.natW, layer.natH);
            // About the FEET, not the middle: the channel stands the subject on
            // the ground, and one that grew from its centre would sink half of
            // whatever it gained through the floor.
            const bottom = layer.y + (layer.natH * layer.scale) / 2;
            layer.scale = rawZoom * fit;
            layer.y = bottom - (layer.natH * layer.scale) / 2;
            clampCutout(spec, layer);   // where the slider's own ends come from
            f.edited = true;
            zoomSlider.value = layer.scale / fit;
            updateZoomFill();
            updatePreview();
            refreshThumb(selectedIndex());
            return;
        }

        // Nothing else was hit, so this is a press on the background — which
        // on a channel that composes from a cutout is the channel's own
        // backdrop, and there is no photograph behind it to pan. The frame's
        // crop no longer decides anything there either: the subject is cut out
        // of the WHOLE photo (see api._cutout_sync), so dragging the window
        // around would spend a re-render and a re-cut to produce the same
        // picture. Nothing happens, which is the honest answer.
        if (cutoutSpec(selectedIndex())) return;

        const wide = await fetchWide(f.frame_id);
        if (!wide) return;
        const wideImg = await loadWideImage(wide);
        if (!wideImg) return;
        await Promise.all(f.logos.map(l => loadLogoImage(l).catch(() => {})));   // see the drag path

        if (wide.zoom_min) { zoomSlider.min = wide.zoom_min; zoomSlider.max = wide.zoom_max; }
        const zoom = Math.max(parseFloat(zoomSlider.min), Math.min(rawZoom, parseFloat(zoomSlider.max)));
        const oldZoom = wide.zoom || 1;

        // Anchor on the FACE, holding its fractional position in the window —
        // so zooming resizes around the subject and leaves the composition
        // intact. Anchoring on the window's own center instead pushed the
        // face out of frame as the window shrank, because the automatic
        // framing deliberately places it off-center.
        const oldW = wide.target_w / oldZoom, oldH = wide.target_h / oldZoom;
        const newW = wide.target_w / zoom,    newH = wide.target_h / zoom;
        const ax = (wide.face_x !== undefined) ? wide.face_x : wide.crop_x + oldW / 2;
        const ay = (wide.face_y !== undefined) ? wide.face_y : wide.crop_y + oldH / 2;
        const fracX = (ax - wide.crop_x) / oldW;
        const fracY = (ay - wide.crop_y) / oldH;

        wide.zoom = zoom;
        [wide.crop_x, wide.crop_y] = clampCrop(wide, ax - fracX * newW, ay - fracY * newH, zoom);
        zoomSlider.value = zoom;
        updateZoomFill();

        // Live preview through the same layers the drag uses; the commit
        // (debounced past the last adjustment) reuses the drop flow.
        pending.zooming = true;
        wideLayer.src = wide.url;
        positionWideLayer(wide, wide.crop_x, wide.crop_y, dragCanvas.getBoundingClientRect().width / CW);
        wideLayer.style.display = "block";
        snapshotOverlays();
        drawLiveCrop(wideImg, wide.crop_x, wide.crop_y, wide);
        liveCropLayer.style.display = "block";

        clearTimeout(_zoomCommitTimer);
        _zoomCommitTimer = setTimeout(() => {
            pending.zooming = false;
            wideLayer.style.display = "none";
            commitReframe(wide, f.frame_id);
        }, 500);
    }

    zoomSlider.addEventListener("input", () => applyZoom(parseFloat(zoomSlider.value)));
    el("zoomInBtn").onclick  = () => applyZoom((parseFloat(zoomSlider.value) || 1) + 0.1);
    el("zoomOutBtn").onclick = () => applyZoom((parseFloat(zoomSlider.value) || 1) - 0.1);

    // The zoom slider is a horizontal <input> rotated -90deg to read as
    // vertical — its pre-rotation "length" is expressed as `width`, but the
    // space it actually has to fill is the wrap's rendered HEIGHT. There's no
    // CSS-only way to size an element off its own containing block's
    // cross-axis, so this measures it directly and keeps it in sync.
    const wrap = el("zoomSliderWrap");
    if (wrap && window.ResizeObserver) {
        new ResizeObserver((entries) => {
            const h = entries[0].contentRect.height;
            if (h > 0) zoomSlider.style.width = h + "px";
        }).observe(wrap);
    }
}
