"""
frame_pipeline.py — Turning one thumbnail slot into finished pixels.

Owns the coordinate math between three spaces that are easy to confuse:

  cropped-frame   what the automatic pipeline sees (frame_*.jpg — overlay
                  bands removed). compute_geometry's crop_x/crop_y and every
                  detected face box live here.
  base            the restored pre-crop image, built from the FULL frame when
                  one exists, at the automatic scale. Its row 0 sits `y_off`
                  pixels ABOVE the cropped frame's row 0, so any coordinate
                  coming from cropped-frame space must add y_off before being
                  used here. Getting this wrong shifts the crop window up,
                  which shows as the subject sliding down in the output.
  zoomed          base multiplied by the manual fine-zoom. The crop window is
                  always exactly the target canvas's size in this space, and the
                  crop coordinates stored per frame and sent by the frontend
                  are in it.
"""

import os
import math
import threading
import logging

import cv2
import numpy as np

import face_restorer
import vram
from frame_extractor import full_frame_path, CROP_TOP_PCT, CROP_BOTTOM_PCT
from image_utils import imread
from logo_remover import bbox_in_scaled_coords, present_overlays, remove_logo
from subtitle_remover import remove_subtitles
from reframe_engine import (
    render_scaled, render_crop_filled, apply_dark_gradient,
    overpan_limits, target_w, target_h, MANUAL_ZOOM_MAX, MAX_SOURCE_UPSCALE,
)
from state import session, FrameRecord

log = logging.getLogger("uvicorn.error")

# Serializes calls into the restoration model itself — everything else in a
# request (cv2 crop/compose, the vary-frame candidate search) can run fully
# concurrently across threads, but nothing guarantees the model wrapper is
# safe to call from two threads at once, so only the actual inference step is
# funneled through this lock.
restore_lock = threading.Lock()

# Extra context, in ZOOMED pixels, extracted around the crop window before
# grading it. The classical pipeline's widest spatial kernel is the clarity
# unsharp mask (sigma 15, so ~3 sigma of real influence); without a margin,
# pixels near the window's edge would be filtered against nothing and the
# window's border would grade differently from its middle. Trimmed away
# afterwards — it exists only so every displayed pixel has full neighbourhood
# context.
POST_MARGIN_PX = 48

# Resolution the pre-crop drag preview (/frame-wide) is served at, relative to
# the base. The frontend displays it well under 1:1 (it's scaled into the
# canvas element, which is itself CSS-scaled below 1280px on most windows),
# and it is replaced by the real render the moment the user drops — so full
# resolution here bought nothing but a full-size grading pass and a much
# larger JPEG over the wire. Tone and color, which are what the preview needs
# to be honest about, are unaffected by resolution.
WIDE_PREVIEW_SCALE = 0.5

DEFAULT_FIDELITY = 0.60

# The three edit presets offered in the "Edit preset" panel, mapped to the
# GRADING intensity (see face_restorer.PostStats.at_intensity): "none"
# restores without any tone/color grading, "natural" grades about as much as
# the frame was measured to need, and "full" pushes past it. All three restore
# identically — deblurring, denoise and detail recovery don't scale with the
# dial, because a soft frame needs the same reconstruction whichever look the
# user picked.
#
# These used to be 0 / 0.35 / 1.0, and the top of that range was the measured
# correction itself. On already-graded source that correction is small, so the
# three buttons landed within about 2% of each other — measured end to end on
# real footage, six levels of 255 on skin between the two extremes, which is
# nothing anybody can see. A dial nobody can see is reported as a broken dial,
# and it was, three times.
#
# So the range is wider and centred differently. "natural" now sits AT the
# measured correction, which is what that word should have meant all along:
# what this photo needs, and no more. "full" applies the same correction
# roughly twice over, bounded by the ceilings the measurement itself respects
# (see at_intensity), so it reads as a deliberate heavy grade rather than as a
# slightly different one.
EDIT_PRESETS = {"none": 0.0, "natural": 1.0, "full": 2.1}
DEFAULT_EDIT_PRESET = "natural"


def compose(canvas: np.ndarray, flip_image: bool, flip_text: bool, gradient: bool = True) -> np.ndarray:
    """
    Presentation layer applied on top of a finished canvas: mirror the photo
    (flip_image), then lay the dark gradient on the side the text sits on
    (flip_text mirrors both, since the gradient exists to back the text) —
    unless `gradient` is off, in which case the flipped canvas is returned
    as-is.

    Kept separate from — and always after — restoration, so neither flag
    changes a single restored pixel: toggling one only re-runs this.
    """
    out = cv2.flip(canvas, 1) if flip_image else canvas
    return apply_dark_gradient(out, mirror=flip_text) if gradient else out


def full_frame_y_off(record: FrameRecord) -> int:
    """
    Vertical offset, in the base's pixel space, between the cropped frame's
    row 0 and the full uncropped frame's row 0 — i.e. how far down the
    cropped frame sits within the full frame (see frame_extractor.
    CROP_TOP_PCT). 0 when this frame has no full sibling.

    Read from the shape captured when the slot was created; this used to
    decode the full-resolution JPEG on every call purely to read its height.
    """
    if record.full_shape is None:
        return 0
    return int(round(int(record.full_shape[0] * CROP_TOP_PCT) * record.geometry["scale"]))


def default_crop_state(record: FrameRecord) -> dict:
    """The automatic rule-of-thirds window, in base coordinates, at zoom 1."""
    return {
        "crop_x": record.geometry["crop_x"],
        "crop_y": record.geometry["crop_y"] + full_frame_y_off(record),
        "zoom": 1.0,
    }


def crop_state(record: FrameRecord) -> dict:
    return record.crop_state or default_crop_state(record)


def base_shape(record: FrameRecord) -> tuple[int, int]:
    """
    The (h, w) the restored base will have, without decoding anything — the
    same source ensure_base picks (full frame when this slot has one) at the
    automatic scale. Lets the crop/zoom limits be computed for a photo whose
    base hasn't been built yet.
    """
    h, w = record.wide_shape[:2]
    scale = record.geometry["scale"]
    return int(h * scale), int(w * scale)


def face_anchor(record: FrameRecord) -> tuple[float, float]:
    """
    The detected face's center in this record's base coordinates (zoom 1).
    The one point every framing decision is expressed relative to: the
    automatic crop places it on the rule-of-thirds, the zoom slider holds it
    fixed, and a variation swap re-places it (see face_placement).
    """
    fx, fy, fw, fh = record.best_face
    scale = record.geometry["scale"]
    return (fx + fw / 2.0) * scale, (fy + fh / 2.0) * scale + full_frame_y_off(record)


def face_placement(record: FrameRecord) -> dict:
    """
    The frame's current framing expressed independently of its photo: the
    zoom, plus where the subject sits inside the window as a fraction of it.

    Stated this way it survives being carried onto a DIFFERENT photo (see
    crop_state_from_placement) — which is what /vary-frame needs, since the
    replacement is another moment of the same shot where the face has moved
    and is a slightly different size, so the raw crop coordinates would mean
    something else there.
    """
    state = crop_state(record)
    zoom = state["zoom"] or 1.0
    ax, ay = face_anchor(record)
    win_w, win_h = target_w() / zoom, target_h() / zoom
    return {
        "zoom": zoom,
        "frac_x": (ax - state["crop_x"] / zoom) / win_w,
        "frac_y": (ay - state["crop_y"] / zoom) / win_h,
    }


def crop_state_from_placement(record: FrameRecord, placement: dict) -> dict:
    """
    The window that reproduces `placement` on this record's CURRENT photo:
    same zoom (clamped to what this photo allows — the automatic scale, and
    so the manual headroom above it, is per-photo), with the new face sitting
    at the same fractional position in the window as the old one did.
    """
    base_h, base_w = base_shape(record)
    zoom = clamp_zoom(record, (base_h, base_w), placement["zoom"])
    ax, ay = face_anchor(record)
    win_w, win_h = target_w() / zoom, target_h() / zoom

    # Zoomed space, same as /reframe-frame stores and render_window reads.
    limits = overpan_limits(int(base_w * zoom), int(base_h * zoom))
    crop_x = int(round((ax - placement["frac_x"] * win_w) * zoom))
    crop_y = int(round((ay - placement["frac_y"] * win_h) * zoom))
    return {
        "crop_x": max(limits["min_x"], min(crop_x, limits["max_x"])),
        "crop_y": max(limits["min_y"], min(crop_y, limits["max_y"])),
        "zoom": zoom,
    }


def fit_zoom(base_w: int, base_h: int) -> float:
    """
    Smallest manual zoom allowed for a base of this size: the zoom at which
    the crop window contains the whole source frame. Any lower and the slider
    would only be growing autofilled void around an already fully visible
    frame.
    """
    return min(target_w() / base_w, target_h() / base_h)


def zoom_ceiling(scale: float) -> float:
    """
    Largest manual zoom allowed on top of an automatic `scale`, so total
    upscale of the original video pixels stays within MAX_SOURCE_UPSCALE.
    Never below 1.0 — the automatic framing must always remain reachable —
    and never above MANUAL_ZOOM_MAX.
    """
    if scale <= 0:
        return MANUAL_ZOOM_MAX
    return max(1.0, min(MANUAL_ZOOM_MAX, MAX_SOURCE_UPSCALE / scale))


def clamp_zoom(record: FrameRecord, shape: tuple, zoom: float) -> float:
    return max(fit_zoom(shape[1], shape[0]),
               min(zoom, zoom_ceiling(record.geometry["scale"])))


def source_is_full_frame(record: FrameRecord, read_sibling: bool) -> bool:
    """
    Whether the image build_photo_base is working from is in the UNCROPPED
    frame's coordinate space.

    Two ways to be: the full sibling was read for this slot (the ordinary
    thumbnail case), or the slot's own source is already the uncropped frame
    and has no sibling to read (a capture slot — see record.source_is_cropped).
    Everything holding a coordinate measured against one of frame_extractor's
    two rectangles has to ask this, and asking it in one place is what stops
    the logo boxes and the subtitle band answering differently.
    """
    return read_sibling or not record.source_is_cropped


def strip_overlays(scaled: np.ndarray, source: np.ndarray, scale: float, full_frame: bool) -> np.ndarray:
    """
    Paints out every burned-in logo that is actually on screen in `source` (a
    frame at its native resolution) from `scaled` (that same frame, multiplied
    by `scale`).

    Presence is judged on the unscaled source because that's the space the
    overlays were detected in, and `full_frame` picks which of the two
    coordinate spaces this photo is in — the uncropped frame or the
    top/bottom-cropped one.
    """
    overlays = session.logo_overlays_full if full_frame else session.logo_overlays_cropped
    present = present_overlays(source, overlays)
    if not present:
        return scaled
    return remove_logo(scaled, [bbox_in_scaled_coords(o.bbox, scale) for o in present])


def build_photo_base(record: FrameRecord, fidelity: float = DEFAULT_FIDELITY):
    """
    Restores one photo, caching nothing.

    ensure_base below is this plus the per-slot cache, and the split exists for
    the photos that are not slots: the extra copies of a multi-figure thumbnail
    are cut from other moments of the same shot (see vary.alternate_moments),
    which have no frame_id of their own and must not evict — or worse, be
    stored under — the slot's own base. What IS cached for them is the finished
    cutout, which is a megabyte against this object's hundreds (see
    session.cutouts).
    """
    intensity = EDIT_PRESETS.get(record.edit_preset, 1.0)

    full_path = full_frame_path(record.source_path)
    src = imread(full_path) if record.full_shape is not None else None
    from_full_frame = src is not None   # which coordinate space `src` is in, below
    if src is None:
        src = imread(record.source_path)
    if src is None:
        raise FileNotFoundError(record.source_path)

    scale = record.geometry["scale"]
    sh, sw = src.shape[:2]
    in_full_space = source_is_full_frame(record, from_full_frame)
    wide = render_scaled(src, scale, int(sw * scale), int(sh * scale))
    wide = strip_overlays(wide, src, scale, in_full_space)
    # Painted out here, in the same pass and for the same reason the logos
    # are: this is the one place a slot's photo becomes pixels, so a frame
    # cleaned here is cleaned in the grid, in the preview, in every reframe
    # and in the download, with nothing downstream having to know it happened.
    # Ahead of restoration rather than after it, so the model reconstructs the
    # filled-in area along with everything else instead of sharpening a patch
    # that was pasted in behind its back.
    #
    # Never on a photo the user supplied themselves: the band describes where
    # THIS VIDEO puts its captions, and an image that did not come out of the
    # video has no reason to have anything there. Nor on a slot whose photo was
    # cleaned when it was created — see FrameRecord.captions_removed, which is
    # what keeps this off the path of every preset switch and every Variation.
    if not record.uploaded and not record.captions_removed:
        wide = remove_subtitles(wide, src, scale, session.subtitle_bands,
                                CROP_TOP_PCT, CROP_BOTTOM_PCT, in_full_space)

    state = default_crop_state(record)
    ref_rect = (
        max(0, int(state["crop_x"])), max(0, int(state["crop_y"])),
        min(wide.shape[1], int(state["crop_x"]) + target_w()),
        min(wide.shape[0], int(state["crop_y"]) + target_h()),
    )

    with restore_lock:
        base = face_restorer.build_base(wide, ref_rect, fidelity=fidelity, intensity=intensity)

    # Outside the lock: this is a device synchronise, and no other thread
    # needs to wait behind it to reach the model. Threshold-guarded, so a run
    # of similarly-sized frames keeps reusing the pool — see vram.release.
    vram.release()
    return base


def ensure_base(frame_id: int, record: FrameRecord, fidelity: float = DEFAULT_FIDELITY):
    """
    The frame's restored base (see face_restorer.build_base), building and
    caching it on first use. This is the only slow step in the whole editing
    flow, and it runs at most once per frame per edit preset.

    Because callers run off the event loop thread, this can be executing for a
    frame_id at the same moment /vary-frame swaps that frame_id onto a
    different photo. The generation captured at entry guards against that: if
    it no longer matches once the work finishes, the result was computed for a
    photo the slot no longer points at, so it's returned to this caller but
    never cached — caching it would attach the wrong photo's restoration to
    the frame the user is now looking at.

    The cached base also carries the edit-preset intensity it was built with;
    a cache from before the user switched presets is stale, not reusable.
    """
    intensity = EDIT_PRESETS.get(record.edit_preset, 1.0)

    cached = session.restored.get(frame_id)
    if cached is not None and cached.intensity == intensity:
        return cached

    generation = record.generation

    # Whole-base, regardless of where the user later pans, instead of once per
    # drag only when the current window happened to intersect a logo; and the
    # grading parameters measured on the automatic framing's own window, a
    # fixed region, so panning can never change the grade. See
    # build_photo_base, and face_restorer's module header.
    base = build_photo_base(record, fidelity)

    if record.generation == generation:
        session.restored.put(frame_id, base)
    return base


def render_window(base, crop_x: float, crop_y: float, zoom: float,
                  frame_id: int | None = None) -> np.ndarray | None:
    """
    The finished target-sized canvas for a crop window, autofilling
    whatever the source doesn't cover.

    Only the window (plus POST_MARGIN_PX of context) is graded — not the whole
    pre-crop frame — which is what keeps the cost of a drag proportional to
    what's displayed rather than to the source's resolution. The grade itself
    is identical to any other window of this frame because the parameters were
    frozen when the base was built.

    Passing `frame_id` lets an immediately-repeated request for the same
    window reuse the previous result (see VideoSession.last_render) — which is
    what a flip or gradient toggle is: same crop, different presentation layer
    applied on top afterwards.

    That reuse is keyed on the BASE as well as the window, not the window
    alone. Switching edit preset rebuilds the base at a different intensity
    while leaving the crop exactly where it was, so a geometry-only key still
    matched and handed back the previous preset's pixels — the canvas only
    caught up once the user dragged, because that finally moved the window
    enough to miss the cache. The entry therefore records which base its
    canvas came from, weakly; see VideoSession.cached_render.
    """
    key = (round(crop_x), round(crop_y), round(zoom, 6))
    if frame_id is not None:
        cached = session.cached_render(frame_id, key, base)
        if cached is not None:
            return cached

    base_h, base_w = base.shape[:2]
    margin = max(1, math.ceil(POST_MARGIN_PX / max(zoom, 1e-6)))

    bx1 = max(0, math.floor(crop_x / zoom) - margin)
    by1 = max(0, math.floor(crop_y / zoom) - margin)
    bx2 = min(base_w, math.ceil((crop_x + target_w()) / zoom) + margin)
    by2 = min(base_h, math.ceil((crop_y + target_h()) / zoom) + margin)
    if bx2 <= bx1 or by2 <= by1:
        return None   # the window doesn't overlap the source at all

    region = face_restorer.render_region(base, (bx1, by1, bx2, by2), zoom=zoom)
    if region.size == 0:
        return None

    # Where the extracted region's own origin sits in zoomed space, so the
    # window can be addressed relative to it.
    canvas = render_crop_filled(
        region,
        int(round(crop_x - bx1 * zoom)),
        int(round(crop_y - by1 * zoom)),
    )
    if frame_id is not None and canvas is not None:
        session.remember_render(frame_id, key, base, canvas)
    return canvas


def render_current(frame_id: int, record: FrameRecord, fidelity: float = DEFAULT_FIDELITY) -> tuple:
    """(canvas, base) for the frame's currently stored crop window."""
    base = ensure_base(frame_id, record, fidelity)
    state = crop_state(record)
    canvas = render_window(base, state["crop_x"], state["crop_y"], state["zoom"], frame_id)
    return canvas, base


def wide_preview(frame_id: int, record: FrameRecord) -> dict:
    """
    The pre-crop view the frontend pans over while dragging, plus the window's
    current position within it.

    Served at WIDE_PREVIEW_SCALE; `preview_scale` tells the frontend how to
    map the (full-resolution) crop coordinates in this response onto the
    image's own pixels.
    """
    base = ensure_base(frame_id, record)
    base_h, base_w = base.shape[:2]

    graded = face_restorer.render_region(base, (0, 0, base_w, base_h), zoom=WIDE_PREVIEW_SCALE)
    state = crop_state(record)
    face_x, face_y = face_anchor(record)

    return {
        "image": graded,
        "preview_scale": WIDE_PREVIEW_SCALE,
        "scaled_w": base_w,
        "scaled_h": base_h,
        "crop_x": int(round(state["crop_x"])),
        "crop_y": int(round(state["crop_y"])),
        "zoom": state["zoom"],
        "target_w": target_w(),
        "target_h": target_h(),
        # Zoom anchor: the detected face's center, so zooming resizes around
        # the subject and leaves the composition intact. Anchoring on the
        # window's own center pushed the face out of frame as the window
        # shrank, because the automatic framing deliberately places it
        # off-center (rule of thirds, text zone on the left).
        "face_x": round(face_x, 2),
        "face_y": round(face_y, 2),
        # How far past the source's edges the crop may be dragged — the
        # uncovered strip gets autofilled by inpainting on drop.
        "overpan": overpan_limits(base_w, base_h),
        # Slider floor: the zoom at which the window shows this frame whole
        # (per-frame — depends on how far the automatic framing went).
        # Floored, not rounded: rounding up by even a fraction would leave a
        # sub-pixel sliver of the frame outside the window at the minimum.
        "zoom_min": int(fit_zoom(base_w, base_h) * 10000) / 10000,
        # The ceiling is per-frame too: how much manual upscale the source
        # still has left after the automatic framing spent its share.
        "zoom_max": round(zoom_ceiling(record.geometry["scale"]), 4),
    }


# Longest side the whole-frame render used for a cutout is produced at.
#
# The base can be far larger than this — the automatic framing routinely scales
# a 1080p source up — and none of that resolution survives: the subject is
# placed a few hundred pixels wide on a 1280x720 canvas, and the segmentation
# itself works at 320x320 whatever it is handed. What full resolution would buy
# is a grading pass over several times as many pixels, per cutout, for detail
# that is downsampled away on the very next line.
CUTOUT_MAX_SIDE = 2048


def render_full(base) -> tuple:
    """
    A whole restored photo, graded, at a bounded size — plus the factor it was
    rendered at, so coordinates in the base can be mapped onto it.

    The counterpart of render_window for the one caller that does not want a
    window: the cutout. A window is a composition decision about a photograph
    that is going to BE the thumbnail, and for a channel that only borrows the
    person out of the frame it is exactly the wrong rectangle — it is what
    saws the subject off at the knees. What that channel wants is as much of
    the person as the footage contains, and then its own placement decides how
    much of them the viewer sees.
    """
    base_h, base_w = base.shape[:2]
    zoom = min(1.0, CUTOUT_MAX_SIDE / max(base_w, base_h))
    return face_restorer.render_region(base, (0, 0, base_w, base_h), zoom=zoom), zoom


def plain_crop_path(frame_id: int) -> str:
    from state import TEMP_DIR
    return os.path.join(TEMP_DIR, f"frame_{frame_id}.jpg")


def build_crop_at(image: np.ndarray, record: FrameRecord, state: dict) -> np.ndarray | None:
    """
    The plain (unrestored) crop of a source photo at an ARBITRARY window —
    the cheap grid render, taken wherever the slot's crop state currently
    points instead of only at the automatic framing.

    `state` is in base coordinates (full-frame space, already multiplied by
    the zoom), the space /reframe-frame stores; `image` here is the CROPPED
    frame, whose row 0 sits y_off lower — hence the subtraction. Whatever the
    window covers beyond that image's own pixels is autofilled, exactly as
    the restored path does. At the automatic window this is a plain crop:
    that window is clamped inside the source, so nothing needs filling.
    """
    geometry = record.geometry
    zoom = state.get("zoom", 1.0) or 1.0
    scale = geometry["scale"] * zoom
    scaled = render_scaled(image, scale,
                           int(geometry["scaled_w"] * zoom), int(geometry["scaled_h"] * zoom))
    scaled = strip_overlays(scaled, image, scale, full_frame=False)
    return render_crop_filled(
        scaled,
        int(round(state["crop_x"])),
        int(round(state["crop_y"] - full_frame_y_off(record) * zoom)),
    )


def build_initial_crop(image: np.ndarray, record: FrameRecord) -> np.ndarray | None:
    """
    The plain (unrestored) automatic crop shown in the grid before a frame has
    ever been selected. Cheap by design — twenty of these are produced during
    /process-video, and restoring them all up front would make the initial
    wait many times longer for work the user may never look at.
    """
    return build_crop_at(image, record, default_crop_state(record))
