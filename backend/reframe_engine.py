import cv2
import numpy as np
from dataclasses import dataclass

from framing import active as framing_profile
from inpainter import inpaint
from region_segmenter import feather_mask
from image_utils import luma

# The finished frame's size, which is the run's and not this module's: 1280×720
# for a YouTube thumbnail, 540×960 for a Snapchat capture (see framing.py). The
# whole pipeline — reframe geometry, enhancement, autofill, frontend canvas and
# export — works in it, and the frontend's CW/CH is set from the same channel
# pack that sets this.
#
# Functions rather than module constants for the reason the three below are: a
# constant is bound once at import, so every module that had written
# `from reframe_engine import TARGET_W` would go on reporting 1280 for the life
# of the process no matter which channel the user picked.
def target_w() -> int:
    return framing_profile().canvas_w


def target_h() -> int:
    return framing_profile().canvas_h


# What the face fills, where it is placed, and how much has to stay clear
# beside it are no longer constants: they are the run's framing profile, which
# a channel can state for itself (see framing.py). These three names are kept
# because they are what the rest of the codebase has always asked for, and they
# now answer with the ACTIVE profile's numbers rather than with a fixed one.
#
# Functions, not module constants, for exactly that reason — a constant is
# bound once at import and would go on reporting the default for the life of
# the process, which is the failure this indirection exists to prevent.
def single_face_area_ratio() -> float:
    return framing_profile().face_area_ratio


def rule_of_thirds_x() -> float:
    return framing_profile().anchor_x


def rule_of_thirds_y() -> float:
    return framing_profile().anchor_y


def min_text_space_frac() -> float:
    return framing_profile().min_text_space


# Manual reframe used to hard-clamp the crop window inside the scaled source
# frame — the user could only drag as far as the original pixels allowed. Now
# the window may extend past the source's edges, and whatever the source
# doesn't cover is autofilled by inpainting (same LaMa core the logo remover
# uses, via inpainter.py). Bounded: generated content should extend a
# composition, not become it — and the larger the strip, the less real
# context anchors it, so quality falls off with distance from the edge.
OVERPAN_MAX_FRAC = 0.35  # how far past each source edge the crop may go, as a fraction of the target dimension

# Manual fine-zoom ceiling for the reframe UI's slider — multiplies the
# automatic scale. 1.0 = exactly the automatic framing; higher zooms in
# (window sees less of the source), lower zooms out (sees more, possibly
# past the edges, where the autofill takes over). The FLOOR is not a
# constant: it's the per-frame zoom at which the window contains the whole
# source frame (see api._fit_zoom) — any lower would only grow autofilled
# void around an already fully visible frame.
#
# This is a bound on the CONTROL, not on quality; MAX_SOURCE_UPSCALE below is
# what protects the pixels, and zoom_ceiling applies whichever of the two bites
# first. Keeping this at 2.0 meant it was the one that bit on a frame fetch,
# for no reason: a 1080x1920 source cropped to a 540x960 still is being scaled
# DOWN by half, so at 2.0 its pixels were not being stretched at all (total
# upscale 1.00x) and there was another 1.6 stops of headroom the guard would
# have allowed. Footage that sets its picture inside a border — a photo on
# card, a framed still — needs exactly those stops to crop the border away, and
# the control stopped short of them.
#
# A thumbnail channel is unaffected either way. Its automatic framing already
# spends the budget making a face fill a sixth of the canvas, so zoom_ceiling
# there comes back at 1.0 from the quality guard and never reaches this number.
MANUAL_ZOOM_MAX = 4.0

# Neither is the ceiling a constant in practice. Zooming in resamples the
# ORIGINAL video pixels up, and the automatic framing already spends part of
# that budget (making the face ~1/6 of the frame routinely scales a 1080p
# source above 1:1). A flat 2.0x manual zoom on top of that measured 2.2x
# total upscale on real footage, where luma laplacian variance — the same
# focus/detail proxy face_restorer uses — fell from 54 to 17: a visibly
# mushy frame that the restoration pass then sharpens into an artifact.
# Capping TOTAL source upscale instead keeps whatever manual room the
# frame's own automatic scale left over (a big face, downscaled to fit,
# still gets the full MANUAL_ZOOM_MAX), and never drops below 1.0 so the
# automatic framing is always reachable. See api._zoom_ceiling.
MAX_SOURCE_UPSCALE = 1.6  # measured: still crisp at ~1.65x, clearly soft by 2.2x
FILL_SEAM_OVERLAP_PX = 8  # dilate the fill mask this far into real pixels so the inpainter blends the seam
                          # instead of butting generated content against the source's literal edge row

# LaMa was trained for inpainting (holes surrounded by context), not
# outpainting (masks touching the image border) — on border-extension fills
# it regresses toward a washed-out gray that ignores the scene's ambient
# color and lighting (a generated gray slab next to a warm pink wall, on real
# footage). Structure/texture still come out fine; it's the low-frequency
# color field that goes wrong. So after inpainting, the fill's low-frequency
# component is replaced with that of a mirror-padded reference built from the
# real pixels: the mirror has exactly the neighboring environment's colors
# and lighting by construction, and taking only its low frequencies means no
# recognizable mirrored objects can appear — the fill keeps LaMa's generated
# structure, re-lit to match the surroundings.
FILL_HARMONIZE_SIGMA   = 40.0  # gaussian radius (px) defining "low frequency" — large enough that only
                               # ambient color/lighting transfers, none of the mirror's actual shapes
FILL_HARMONIZE_FEATHER = 15.0  # feather radius (px) on the harmonization weight so the correction fades
                               # in across the seam instead of stepping at it

# The harmonization reference used to be built by mirroring the ENTIRE
# visible region (see _mirror_pad) — for a large gap, that reflection reaches
# far enough into the frame to include content nowhere near the true edge
# (a face well inside the crop), and duplicated it into the fill: skin tone
# bleeding across the gap on real footage, not just ambient wall/background
# color. FILL_EDGE_BAND_FRAC bounds how deep into the frame (as a fraction of
# the target canvas) the reference is allowed to draw from — content beyond
# it is replaced with a reflection of its own near-edge band (see
# _edge_band_source) before mirroring, so the fill can only ever extend what
# is immediately at the border, never something from deeper inside the frame.
FILL_EDGE_BAND_FRAC = 0.08

# LaMa's own generated content in the fill area tends to carry visible
# grain/micro-texture — noticeably rougher than a real photo's naturally
# soft background, especially next to an already-smooth surface like a wall.
# Since this content is synthetic to begin with (extending the scene, not
# reproducing it), there's no real detail worth preserving there — a flatter,
# less textured continuation reads as more convincing background than fake
# fine detail. Blurred in over the same feathered fill_mask the harmonization
# uses, so it fades in exactly at the true source edge and never softens any
# real pixel.
FILL_SMOOTH_SIGMA = 10.0

# The autofill runs the inpainter at HALF the canvas's linear resolution and
# upsamples the result. This costs essentially nothing visually and roughly
# quarters the inpainting work — the single most expensive step of a drag
# that overpans, and (on a small-VRAM GPU or CPU-only machine) the one most
# likely to be slow or fail.
#
# The reason it's free: FILL_SMOOTH_SIGMA above already blurs the generated
# region with a 10px radius on purpose, because LaMa's own micro-texture
# reads as fake next to a real photo's smooth background. Detail finer than
# that radius is deliberately discarded moments after it's generated, so
# generating it at full resolution was paying for pixels the very next line
# throws away. Structure at the scale that survives the smoothing is
# preserved by the upsample. Only the FILL is downscaled-and-restored this
# way; every real source pixel stays untouched at full resolution (see the
# mask composite in inpainter.inpaint).
FILL_INPAINT_SCALE = 0.5

GRADIENT_STRENGTH_BASE    = 0.65   # peak (left-edge) darken multiplier, at GRADIENT_REFERENCE_LUMA
GRADIENT_REFERENCE_LUMA   = 130.0  # left-side (text-zone) mean luma this was validated against
GRADIENT_STRENGTH_MIN     = 0.35   # floor: a dark left side still gets some gradient, for consistent text contrast
GRADIENT_STRENGTH_MAX     = 0.85   # ceiling: never darken past this even on a very bright left side


@dataclass(frozen=True)
class Framing:
    """
    Everything the automatic framing decides for one (image, face) pair, in
    exact float arithmetic. THE single source of this math — see
    compute_framing.
    """
    scale: float
    scaled_w: float
    scaled_h: float
    crop_x: float        # clamped into the source
    crop_y: float
    ideal_crop_x: float  # what rule-of-thirds wanted, before clamping
    ideal_crop_y: float
    face_cx: float       # face center, in scaled coords
    face_cy: float
    face_w: float        # face width, in scaled coords
    valid: bool

    @property
    def text_space_frac(self) -> float:
        """Fraction of the canvas width left clear to the left of the face."""
        return (self.face_cx - self.crop_x - self.face_w / 2.0) / target_w()

    def source_rect(self) -> tuple[float, float, float, float]:
        """The source-image rectangle that maps onto the final canvas."""
        return (self.crop_x / self.scale, self.crop_y / self.scale,
                target_w() / self.scale, target_h() / self.scale)


def compute_scale(h: int, w: int, best_face: tuple[int, int, int, int], face_count: int = 1) -> float:
    """
    The zoom factor a source frame gets reframed at. Kept as its own function
    because it's the one piece of framing math that's meaningful on its own
    (quality_scorer's face_fill_penalty needs the scale without the rest).
    """
    fx, fy, fw, fh = best_face
    cover_scale = max(target_w() / w, target_h() / h)

    if face_count == 1:
        target_face_area = single_face_area_ratio() * target_w() * target_h()
        face_area = fw * fh
        face_scale = (target_face_area / face_area) ** 0.5 if face_area > 0 else cover_scale
        return max(face_scale, cover_scale)
    return cover_scale


def compute_framing(image_shape: tuple[int, int], best_face: tuple[int, int, int, int],
                    face_count: int = 1) -> Framing:
    """
    Scale + rule-of-thirds crop position for a source image, without rendering
    anything.

    This is the ONLY implementation of that math in the app. It used to exist
    three times — compute_geometry here (integer arithmetic), and
    quality_scorer's _simulate_reframe and reframability_score (float
    arithmetic) — which meant a frame's predicted score was computed from a
    subtly different framing than the one it would actually get. compute_scale
    had already been extracted for exactly that reason; the crop placement had
    not. Everything derives from this now, so the score can't disagree with
    the render.

    Exact floats throughout; compute_geometry rounds to pixels at the edge,
    the scorers use the float values directly.
    """
    h, w = image_shape[:2]
    fx, fy, fw, fh = best_face

    scale = compute_scale(h, w, best_face, face_count)
    scaled_w, scaled_h = w * scale, h * scale
    face_cx = (fx + fw / 2.0) * scale
    face_cy = (fy + fh / 2.0) * scale

    ideal_x = face_cx - target_w() * rule_of_thirds_x()
    ideal_y = face_cy - target_h() * rule_of_thirds_y()
    crop_x = max(0.0, min(ideal_x, scaled_w - target_w()))
    crop_y = max(0.0, min(ideal_y, scaled_h - target_h()))

    face_w = fw * scale
    text_space = (face_cx - crop_x - face_w / 2.0) / target_w()

    return Framing(
        scale=scale, scaled_w=scaled_w, scaled_h=scaled_h,
        crop_x=crop_x, crop_y=crop_y,
        ideal_crop_x=ideal_x, ideal_crop_y=ideal_y,
        face_cx=face_cx, face_cy=face_cy, face_w=face_w,
        valid=text_space >= min_text_space_frac(),
    )


def compute_geometry(image_shape: tuple[int, int], best_face: tuple[int, int, int, int],
                      face_count: int = 1) -> dict:
    """
    Pixel-rounded view of compute_framing, in the dict shape the render and
    manual-reframe paths have always consumed (they need integer crop
    coordinates and integer scaled dimensions to slice with).
    """
    f = compute_framing(image_shape, best_face, face_count)
    scaled_w, scaled_h = int(f.scaled_w), int(f.scaled_h)
    # Re-clamp in INTEGER space against the truncated dimensions the caller
    # will actually slice. Rounding the crop and truncating the size
    # independently can put the window one pixel past the buffer's last
    # column — render_crop then finds a short slice and returns None, which
    # silently drops the frame from the results. (Measured: two of six
    # selected frames vanished this way.)
    return {
        "scale": f.scale,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "crop_x": max(0, min(int(round(f.crop_x)), scaled_w - target_w())),
        "crop_y": max(0, min(int(round(f.crop_y)), scaled_h - target_h())),
        "valid": f.valid,
    }


def render_scaled(image: np.ndarray, scale: float, scaled_w: int, scaled_h: int) -> np.ndarray:
    """
    The full scaled (pre-crop) image a crop window is taken from — used for
    the manual-reframe pan preview.

    LANCZOS4 when scaling UP (the common case: making the face ~1/6 of the
    frame routinely upscales), INTER_AREA when scaling down. Lanczos on a
    downscale both costs more and aliases, because it samples a fixed small
    kernel instead of averaging the pixels being discarded; INTER_AREA is the
    correct — and cheaper — filter in that direction.
    """
    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (scaled_w, scaled_h), interpolation=interp)


def _adaptive_gradient_strength(canvas: np.ndarray, max_x: int, mirror: bool = False) -> float:
    """
    Scales GRADIENT_STRENGTH_BASE by how bright the text zone (the region the
    gradient actually covers) already is: at GRADIENT_REFERENCE_LUMA it's
    exactly the validated peak strength, a brighter text zone (less natural
    contrast for white text) gets darkened more, an already-dark one gets
    darkened less — capped so neither ends up over- or under-corrected.

    `mirror` measures the right side instead — the zone must be the one the
    gradient will actually cover, or a dark left/bright right frame would be
    graded by the wrong half.
    """
    zone = canvas[:, -max_x:] if mirror else canvas[:, :max_x]
    current = float(luma(zone).mean())
    if current < 1e-3:
        return GRADIENT_STRENGTH_MIN
    strength = GRADIENT_STRENGTH_BASE * (current / GRADIENT_REFERENCE_LUMA)
    return float(np.clip(strength, GRADIENT_STRENGTH_MIN, GRADIENT_STRENGTH_MAX))


def apply_dark_gradient(canvas: np.ndarray, mirror: bool = False) -> np.ndarray:
    """
    Dark gradient: full-height, never past 50% width — clears space for the
    text overlay. Peaks on the left by default; `mirror` peaks on the right
    instead, for the mirrored text layout (the gradient exists to back the
    text, so it has to follow it to the other side).

    Callers must apply this last, after any face restoration/enhancement —
    never before. Baking it into the canvas earlier means denoise/black-point/
    white-point/contrast/sharpen (see face_restorer.py) would all reprocess
    the gradient's own darkened pixels as if they were real image content
    (stretching, sharpening, and re-lighting a region that isn't part of the
    photo), and the strength computed here is measured from the *current*
    pixels — it needs to see the frame's actual post-enhancement brightness,
    not a pre-enhancement guess.
    """
    canvas_f = canvas.astype(np.float32)
    max_x = int(target_w() * 0.50)
    peak_strength = _adaptive_gradient_strength(canvas, max_x, mirror)
    x_norm = np.clip(np.arange(target_w(), dtype=np.float32) / max_x, 0.0, 1.0)
    ramp = peak_strength * (1.0 - x_norm)
    if mirror:
        ramp = ramp[::-1]
    canvas_f *= (1.0 - ramp.reshape(1, target_w(), 1))
    return canvas_f.clip(0, 255).astype(np.uint8)


def render_crop(scaled: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray | None:
    """Crop the target-sized window out of an already-scaled image. No gradient here — see apply_dark_gradient."""
    canvas = scaled[crop_y:crop_y + target_h(), crop_x:crop_x + target_w()]
    if canvas.shape[0] != target_h() or canvas.shape[1] != target_w():
        return None
    return canvas


def overpan_limits(scaled_w: int, scaled_h: int) -> dict:
    """
    The crop-position range the manual reframe accepts, including the
    beyond-the-source overpan margin. Shared by the /frame-wide endpoint (so
    the frontend clamps the drag to exactly what the backend will accept) and
    /reframe-frame's own clamp — the two must never compute this independently.
    """
    over_x = int(target_w() * OVERPAN_MAX_FRAC)
    over_y = int(target_h() * OVERPAN_MAX_FRAC)
    return {
        "min_x": -over_x,
        "max_x": scaled_w - target_w() + over_x,
        "min_y": -over_y,
        "max_y": scaled_h - target_h() + over_y,
    }


def _mirror_pad(region: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """
    copyMakeBorder(BORDER_REFLECT_101), chunked: reflection can only extend by
    (dimension - 1) per pass, so a pad wider than the region itself (possible
    when only a sliver of source overlaps a heavily-overpanned crop) is built
    up over multiple passes, each reflecting the already-padded result.
    """
    out = region
    while top or bottom or left or right:
        h, w = out.shape[:2]
        t, b = min(top, h - 1), min(bottom, h - 1)
        l, r = min(left, w - 1), min(right, w - 1)
        if max(t, b, l, r) == 0:  # degenerate 1px-wide region — replicate is all that's left
            return cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_REPLICATE)
        out = cv2.copyMakeBorder(out, t, b, l, r, cv2.BORDER_REFLECT_101)
        top, bottom, left, right = top - t, bottom - b, left - l, right - r
    return out


def _edge_band_source(
    real: np.ndarray, need_left: bool, need_right: bool, need_top: bool, need_bottom: bool,
    band_w: int, band_h: int,
) -> np.ndarray:
    """
    Returns a same-shape copy of `real` whose content beyond `band_w`/`band_h`
    from any edge that needs extending has been overwritten with a mirrored
    continuation of that edge's own near-border band — e.g. for a left gap,
    every column past band_w from the true left edge becomes a reflection of
    just that first band_w-wide strip, rather than the genuine (possibly
    face-containing) pixels further in.

    Used only to build the harmonization reference (see FILL_EDGE_BAND_FRAC);
    the true visible pixels in the output are untouched by this — they come
    from `real` directly, not from this doctored copy, since the caller only
    blends the reference into the GAP region (weight 0 over real pixels).
    Keeping the same shape is what lets the result feed straight into
    _mirror_pad's existing pad-to-the-target-canvas math unchanged.

    Horizontal (left/right) capping always wins when a corner needs both —
    computed independently from `real`, not chained. Chaining used to let the
    top/bottom step run second and take just the (by-then already
    horizontally-correct) top band_h-tall band and mirror it DOWN across the
    entire remaining height, discarding every row's own horizontally-mirrored
    color in favor of one thin band's — on real footage with a bright ceiling
    light sitting in that top band, this dragged the light's color down the
    whole column instead of filling it row-by-row from the left/right edge,
    which is what a left/right gap should always do.
    """
    rh, rw = real.shape[:2]

    x_capped = real
    bw = min(band_w, rw // 2 if (need_left and need_right) else rw)
    bw = max(bw, 1)
    if need_left and need_right and rw > 2 * bw:
        left_fill = _mirror_pad(real[:, :bw], 0, 0, 0, rw - bw)
        right_fill = _mirror_pad(real[:, -bw:], 0, 0, rw - bw, 0)
        mid = rw // 2
        x_capped = np.concatenate([left_fill[:, :mid], right_fill[:, mid:]], axis=1)
    elif need_left and rw > bw:
        x_capped = _mirror_pad(real[:, :bw], 0, 0, 0, rw - bw)
    elif need_right and rw > bw:
        x_capped = _mirror_pad(real[:, -bw:], 0, 0, rw - bw, 0)

    if need_left or need_right:
        return x_capped

    if not (need_top or need_bottom):
        return real

    bh = min(band_h, rh // 2 if (need_top and need_bottom) else rh)
    bh = max(bh, 1)
    if need_top and need_bottom and rh > 2 * bh:
        top_fill = _mirror_pad(real[:bh, :], 0, rh - bh, 0, 0)
        bottom_fill = _mirror_pad(real[-bh:, :], rh - bh, 0, 0, 0)
        mid = rh // 2
        return np.concatenate([top_fill[:mid, :], bottom_fill[mid:, :]], axis=0)
    elif need_top and rh > bh:
        return _mirror_pad(real[:bh, :], 0, rh - bh, 0, 0)
    elif need_bottom and rh > bh:
        return _mirror_pad(real[-bh:, :], rh - bh, 0, 0, 0)
    return real


def _directional_blur(img: np.ndarray, sigma: float, axis: str) -> np.ndarray:
    """
    Gaussian blur along only ONE axis — the other axis's kernel is pinned to
    1px (a no-op), so the blur can never mix information across it. Used to
    build the harmonization low-frequency reference for a single-edge gap
    (see render_crop_filled): a left/right gap's reference must only ever
    average WITHIN a row, never down a column, or a bright feature in one
    row's low-frequency content leaks into every other row's correction —
    which is exactly what let a ceiling light near the top of the frame get
    smeared down the entire height of a left/right gap in the ordinary
    (isotropic) blur this replaces for that case.
    """
    return cv2.GaussianBlur(img, (0, 1), sigmaX=sigma, sigmaY=0) if axis == "x" \
        else cv2.GaussianBlur(img, (1, 0), sigmaX=0, sigmaY=sigma)


def render_crop_filled(scaled: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray | None:
    """
    Like render_crop, but the window may extend past the scaled image's edges
    (within overpan_limits): the part the source covers is copied as-is, the
    uncovered remainder is inpainted from it, and the inpainted area's
    low-frequency color/lighting is harmonized with the surrounding scene
    (see the FILL_HARMONIZE_* constants block for why the raw inpaint isn't
    enough). Returns None only when the window doesn't overlap the source at
    all — nothing real to extend from.

    A single-edge gap (only left/right OR only top/bottom, never a corner)
    harmonizes with a DIRECTIONAL blur (see _directional_blur) instead of the
    ordinary 2D one: a generic inpaint isn't bound to either axis, and on
    real footage a bright feature (e.g. a ceiling light) near the top of the
    frame got smeared down the entire height of a left/right gap instead of
    extending row-by-row from the side. Comparing the inpaint's own
    low-frequency content against the mirror reference ROW-BY-ROW (for a
    left/right gap) or COLUMN-BY-COLUMN (top/bottom) — never blending
    either — corrects exactly that without discarding the inpaint's texture
    (an earlier version of this fix skipped inpainting entirely for this
    case and used the blurred mirror reflection as the fill outright, which
    read as an obviously flat, textureless patch whenever the real edge band
    itself had little texture to reflect). A corner gap (both a horizontal
    AND a vertical gap at once) keeps the plain isotropic blur — a genuinely
    2D problem no single axis resolves.
    """
    sh, sw = scaled.shape[:2]
    sx1, sy1 = max(crop_x, 0), max(crop_y, 0)
    sx2, sy2 = min(crop_x + target_w(), sw), min(crop_y + target_h(), sh)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    if sx2 - sx1 == target_w() and sy2 - sy1 == target_h():
        return render_crop(scaled, crop_x, crop_y)  # fully inside — no fill needed

    real = scaled[sy1:sy2, sx1:sx2]
    rh, rw = real.shape[:2]
    dx1, dy1 = sx1 - crop_x, sy1 - crop_y

    canvas = np.zeros((target_h(), target_w(), 3), dtype=scaled.dtype)
    fill_mask = np.full((target_h(), target_w()), 255, dtype=np.uint8)
    canvas[dy1:dy1 + rh, dx1:dx1 + rw] = real
    fill_mask[dy1:dy1 + rh, dx1:dx1 + rw] = 0

    k = 2 * FILL_SEAM_OVERLAP_PX + 1
    filled = inpaint(
        canvas, cv2.dilate(fill_mask, np.ones((k, k), np.uint8)),
        work_scale=FILL_INPAINT_SCALE,
    ).astype(np.float32)

    gap_top, gap_bottom = dy1, target_h() - dy1 - rh
    gap_left, gap_right = dx1, target_w() - dx1 - rw
    need_horiz = gap_left > 0 or gap_right > 0
    need_vert = gap_top > 0 or gap_bottom > 0

    edge_source = _edge_band_source(
        real, gap_left > 0, gap_right > 0, gap_top > 0, gap_bottom > 0,
        max(1, int(target_w() * FILL_EDGE_BAND_FRAC)), max(1, int(target_h() * FILL_EDGE_BAND_FRAC)),
    )
    mirror = _mirror_pad(edge_source, gap_top, gap_bottom, gap_left, gap_right)
    weight = feather_mask(fill_mask, radius=FILL_HARMONIZE_FEATHER).astype(np.float32)[:, :, np.newaxis] / 255.0

    if need_horiz != need_vert:  # exactly one axis — a straight edge, never a corner
        axis = "x" if need_horiz else "y"
        low_mirror = _directional_blur(mirror.astype(np.float32), FILL_HARMONIZE_SIGMA, axis)
        low_fill = _directional_blur(filled, FILL_HARMONIZE_SIGMA, axis)
    else:
        low_mirror = cv2.GaussianBlur(mirror.astype(np.float32), (0, 0), FILL_HARMONIZE_SIGMA)
        low_fill = cv2.GaussianBlur(filled, (0, 0), FILL_HARMONIZE_SIGMA)

    harmonized = filled + (low_mirror - low_fill) * weight

    # Flatten LaMa's own generated grain/texture — see FILL_SMOOTH_SIGMA.
    # Reuses the same feathered weight as the harmonization blend, so this
    # only ever softens generated pixels, fading out at the true source edge.
    # Small radius relative to the frame either way — safe to leave isotropic
    # even on a single-edge gap.
    smoothed = cv2.GaussianBlur(harmonized, (0, 0), FILL_SMOOTH_SIGMA)
    harmonized = harmonized * (1.0 - weight) + smoothed * weight

    return harmonized.clip(0, 255).astype(scaled.dtype)


def reframe(image: np.ndarray, best_face: tuple[int, int, int, int], face_count: int = 1) -> np.ndarray | None:
    """Returns the plain (un-gradiented) crop — the canonical frame stored on disk and fed to enhancement."""
    geo = compute_geometry(image.shape, best_face, face_count)
    if not geo["valid"]:
        return None
    scaled = render_scaled(image, geo["scale"], geo["scaled_w"], geo["scaled_h"])
    return render_crop(scaled, geo["crop_x"], geo["crop_y"])
