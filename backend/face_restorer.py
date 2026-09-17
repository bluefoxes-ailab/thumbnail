"""
face_restorer.py — Face restoration with GFPGAN.

    pip install gfpgan

If it is not installed restore_faces() leaves the image untouched by the GAN
and only the classical pipeline runs — /enhance-frame still responds, with
enhanced: false.

## What this file used to claim, and why it was wrong

It opened with "CodeFormer (preferred) or GFPGAN fallback" and told you to
`pip install codeformer-pytorch`. There is no such package on PyPI and there
never was, so `_try_codeformer` below has failed on every start this app has
ever had, its failure is logged at debug, and GFPGAN — the documented fallback
— is the only backend that has ever run. Nothing was broken by that; the
documentation simply described a machine nobody has.

Two things follow, and both are still true of the code as it stands:

  * FIDELITY_WEIGHT is dead. It is read in exactly one place, the CodeFormer
    constructor that never executes.
  * The `fidelity` that everything else passes around — /enhance-frame's
    parameter, frame_pipeline.DEFAULT_FIDELITY — never reaches the GAN. It
    controls the eye pullback and the drastic-change pullback below, which are
    blends of the restored face back toward the source. That is a real knob and
    it works; it is just not the identity-vs-quality dial the old header
    described.

CodeFormer was measured against GFPGAN on this app's own footage before this
comment was written, using the `codeformer` package (the real one, whose API is
different from the stub below: `CodeFormer(upscale=..., bg_upsampler=...)` and
`forward(pil_image, fidelity_weight=...)`, PIL in and PIL out, the weight per
call rather than per instance). It restores far more — mean absolute difference
from the source of 5-6/255 against GFPGAN's 0.94-0.98 on the same three faces —
and on a 191px face it rebuilds hair, brow and teeth that GFPGAN leaves as
mass. It was not adopted: at low fidelity weights it invents skin texture that
is not in the source, and this app would rather under-restore than invent. That
is a taste decision and it is allowed to be revisited; the numbers are here so
the next person does not have to measure it again.

── Structure ──────────────────────────────────────────────────────────────
This module is split into two phases, and the split matters:

  build_base()  runs ONCE per frame. Face restoration, the protection blends,
                the subject masks, and — the new part — MEASUREMENT of every
                adaptive parameter the classical pipeline needs.
  render()      runs per crop. Applies the classical pipeline to just the
                region being shown, using the parameters frozen by build_base.

Previously both happened together over the entire pre-crop frame, because
every adaptive stage measures the image it's given and two different crops of
one photo would otherwise be graded differently (a real bug: a heavily
overpanned crop's large dark fill area skewed the whole-frame statistics
enough to crush contrast well past what the same photo's other framings got).
Freezing the measurements achieves that same guarantee — every crop of a
frame is graded by identical numbers — without also forcing every pixel of a
~3.5 megapixel frame through a pipeline that only ever displays 0.9 of them.

The measurements are taken on a FIXED reference window (the automatic
rule-of-thirds framing), not on whatever the user is currently looking at, so
they cannot drift as the user pans. That window is also more representative
of the output than the whole frame was: it's the region the photo is actually
composed around, and it never includes autofilled void.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, replace
from scipy.interpolate import PchipInterpolator

from face_detector import detect_faces, detect_eyes
from region_segmenter import feather_mask, segment_regions, face_ellipse_mask
from image_utils import luma, stats_sample, STATS_PIXEL_BUDGET

# "uvicorn.error" rather than __name__: uvicorn installs handlers on its own
# loggers and leaves the root logger bare, so a __name__ logger's records
# propagate up to a root with nowhere to go and vanish. That silently ate the
# one message that says WHY the GAN didn't load — the failure looked
# identical to a machine that simply has no GPU.
log = logging.getLogger("uvicorn.error")

# Dead on every machine this has ever run on: its only reader is the
# CodeFormer constructor in _try_codeformer, and that package does not exist
# (see the header). Kept, rather than deleted, because it is also the default
# value of the `fidelity` argument build_base takes — so removing it means
# choosing a new default for the eye pullback, which is a behaviour change and
# not a tidy-up.
FIDELITY_WEIGHT = 0.40
EYE_PROTECT_FEATHER = 6.0    # gaussian feather radius (px) softening the eye-protection mask edges
EYE_DIFF_SATURATE = 35.0    # per-channel mean abs diff (0-255) at which deviation-protection saturates
EYE_DIFF_STRENGTH = 0.85    # max blend-toward-original weight the deviation signal can contribute

# ── How damaged is the source face? ────────────────────────────────────────
# Every protection below exists to catch the restorer doing something WRONG,
# and each one recognises "wrong" as "far from the source". That definition
# quietly assumes the source is worth staying close to. On a badly out-of-
# focus face it isn't: the restorer is supposed to depart from it — that
# departure IS the reconstruction — and protections tuned on mildly-soft
# footage read that legitimate rebuild as damage and undo it. Measured on
# this app's own frames: only 13% of the GAN's work survived on the softest
# frame against 19% on a moderately sharp one, i.e. the worse the input, the
# less reconstruction reached the output — backwards.
#
# So the damage level is measured up front and the protections are scaled by
# it. The metric is laplacian variance over the face crop divided by that
# crop's own luma variance, on a canonical-size resize:
#   - dividing by luma variance separates focus from contrast (a flat, hazy
#     but perfectly sharp face has low raw laplacian variance too),
#   - the resize makes it independent of how many pixels the face happens to
#     occupy (the same face at 200px and at 800px is equally in focus, but
#     per-pixel edge energy is not remotely the same).
# Thresholds are from the observed spread over this app's test footage:
# 0.017 for the visibly out-of-focus frames, 0.036 for one the restoration
# already handled well, up to 0.13 for the crispest.
FOCUS_CANON_PX = 256    # face crops are measured at this size, whatever their real size
FOCUS_SHARP    = 0.060  # normalised face focus at/above this -> undamaged source, protect it normally
FOCUS_BLURRED  = 0.020  # at/below this -> heavily damaged source, the restorer's departure is the point

# What the damage level does to the fixed eye pullback (`fidelity`): scales it
# down toward this fraction of itself. Pulling eyes back toward the source is
# protective when the source has real eyes to preserve; on a defocused face
# it just reinstates the blur over the single feature that most decides
# whether a thumbnail looks restored. Not taken to zero — deformation risk is
# highest at the eyes whatever the input.
EYE_FIDELITY_DAMAGE_FLOOR = 0.15

# GFPGAN/CodeFormer align the detected face to a canonical pose, restore that
# aligned crop, then warp it back into the frame — on an off-angle or
# partially-cropped face (common on real footage, not just studio portraits)
# that warp can land a little off from where the face actually sits, showing
# up as a visible ghosting/seam line across the cheek, temple, or jaw: real
# skin and restored skin both present, offset by a few pixels, in the same
# spot. Eye/hair deviation protection above only ever look inside their own
# zones, so a seam anywhere else on the face — which is most of it — had
# nothing catching it. SEAM_DIFF_SATURATE sits above the eye/hair thresholds
# because ordinary restoration (denoise, sharpen, contrast) shifts every face
# pixel by some moderate amount; only a genuinely drastic mismatch, the seam
# itself, should trip this. STRENGTH goes all the way to 1.0 (unlike eye/hair
# below) — a seam this real needs to fully disappear at full severity, not
# just fade to 10% of itself.
SEAM_DIFF_SATURATE = 40.0
SEAM_DIFF_STRENGTH = 1.0

# Both seam checks below fire on an ABSOLUTE amount of change, which is what
# made them scale backwards (see the damage block above): the blurrier the
# source, the more the restorer legitimately has to change, and the more of
# that change these two flagged as a seam. Measured on real frames, the edge
# check alone was reverting 87% of the GAN's work on a soft frame and fully
# reverting 64% of every pixel the GAN touched — a paste-back seam is a thin
# line, so a check that fires on two thirds of the restored area is not
# describing a seam any more.
#
# The fix is to judge a seam RELATIVE TO THIS FRAME'S OWN RESTORATION rather
# than against a fixed number. Within the restorer's footprint (the pixels it
# actually changed), the bulk of the signal is legitimate rebuild; a seam is
# an outlier against it. So the saturate point becomes
#     max(absolute floor, this frame's typical rebuild level * MULTIPLE)
# The absolute floor still governs a sharp source, where the restorer changes
# little and the typical level is far below it — those frames behave exactly
# as before. A soft source raises its own bar, so reconstruction survives
# while a genuine discontinuity, which is much stronger than the diffuse
# texture rebuild around it, still trips.
SEAM_SCALE_PERCENTILE = 75.0  # percentile of the footprint's own signal taken as "typical legitimate rebuild"
SEAM_SCALE_MULTIPLE   = 3.0   # a seam has to exceed that typical level by this much to count as one

# A paste-back seam isn't always a big color jump — sometimes it's a smooth,
# low-contrast tonal mismatch that a magnitude-only diff check (above) is too
# insensitive to catch without also tripping on ordinary restoration. What a
# seam always is, though, is a LINE THAT WASN'T THERE BEFORE: an edge in the
# restored output with no corresponding edge in the original at that same
# spot.
#
# A plain "how much did edge energy increase" test does NOT capture that,
# and measuring it (a synthetic already-sharp edge, amplified further, the
# way ordinary sharpening amplifies eyes/nose contours) is what caught it:
# the ABSOLUTE increase from strengthening an already-strong edge (e.g.
# 160 -> 200) can be just as large as a genuinely new edge appearing out of
# flat, edge-free skin (0 -> 40) — an absolute-difference test can't tell
# those apart. SEAM_EDGE_ALLOWANCE fixes this by scaling the allowed
# increase to the edge that was already there: `out` may exceed `image`'s
# edge energy by up to this multiple before anything gets flagged, so
# amplifying a real edge stays invisible to this check regardless of how
# strong that edge already was, while a seam — near-zero original edge
# energy, so any multiple of ~0 is still ~0 — trips it at almost any
# absolute strength.
SEAM_EDGE_ALLOWANCE = 2.2  # legitimate sharpening may strengthen an existing edge up to this multiple
SEAM_EDGE_SATURATE = 18.0
SEAM_EDGE_STRENGTH = 1.0
# Widens the (inherently knife-thin, 1-2px) detected edge into a soft few-px
# band, so the blend-back covers the seam's actual gradient, not just its
# sharpest core. Dilation, not a Gaussian blur: a thin, already-saturated
# spike survives a max-filter at full strength, where a blur of comparable
# radius would average it down against its (zero) surroundings and undo the
# saturation this was tuned against.
SEAM_EDGE_WIDEN_PX = 5
SEAM_EDGE_SOFTEN_SIGMA = 1.2  # small — just feathers the dilated mask's own now-hard edges

# WHERE a paste-back seam can be, which is the other half of telling it apart
# from reconstruction. The restorer replaces one contiguous region — its
# aligned crop warped back into place — and inside that region its output is
# self-consistent: a misaligned warp displaces the whole crop together, so
# the discontinuity it produces is at the region's BORDER, against the
# original pixels it meets there (the cheek/temple/jaw line in the comment
# above is exactly that border). Reconstruction, by contrast, is spread
# through the region's interior.
#
# Scoping the edge check to a band around that border is what finally
# separates the two. Rescaling the threshold alone (SEAM_SCALE_* above) was
# not enough on its own: SEAM_EDGE_WIDEN_PX dilates every flagged pixel by
# 5px, which is right for one thin line but turns scattered per-strand
# detections across a whole face into blanket coverage. Measured on real
# frames, the band takes the edge check from reverting 67-79% of the GAN's
# work down to 6-10%, while still covering where a seam actually lands.
#
# The region is taken from a blurred `diff` rather than from the raw
# changed-pixel set: the restorer's own footprint is speckled (in flat skin
# it moves many pixels by less than a level), so eroding it directly leaves
# nothing behind — blurring first turns that speckle into the solid area it
# represents.
SEAM_BAND_DENSITY_SIGMA_FRAC = 0.04  # blur radius that solidifies the footprint, as a fraction of face height
SEAM_BAND_INWARD_FRAC = 0.12         # how far the band reaches into the region, same units
SEAM_BAND_OUTWARD_PX = 9             # how far it reaches outside it — the region edge is already where diff fades out
SEAM_BAND_FEATHER_FRAC = 0.15        # gaussian feather on the band, as a fraction of its own width

# GFPGAN/CodeFormer are trained overwhelmingly on facial features (eyes, nose,
# mouth, skin) — hair is comparatively under-represented, and long/wavy/
# complex hair near the face-hair boundary often comes back blurred, smeared,
# or outright melted. There's no hair segmenter in this app, so this bounds a
# "hair-plausible" zone around the face bbox instead (wide/tall enough for
# temples, crown, and long hair falling past the jaw) and reuses the same
# deviation-based protection as eyes within it, excluding skin.
HAIR_ZONE_EXPAND_W    = 1.9   # zone width, as a multiple of the face bbox width
HAIR_ZONE_EXPAND_UP   = 1.0   # extra height above the face top, as a multiple of face bbox height
HAIR_ZONE_EXPAND_DOWN = 2.0   # extra height below the face bottom, as a multiple of face bbox height — long hair reaches past the jaw
HAIR_ZONE_FEATHER     = 10.0  # gaussian feather radius (px) softening the hair-zone mask edges
HAIR_DIFF_SATURATE = 40.0    # per-channel mean abs diff (0-255) at which hair deviation-protection saturates
HAIR_DIFF_STRENGTH = 0.85    # max blend-toward-original weight the hair deviation signal can contribute

# Color-value deviation (above) catches drastic distortion (wrong colors,
# warped shapes) but misses the *other* common failure mode: the restorer
# smoothing individual hair strands into a flat, waxy mass. A blurred patch
# of similarly-colored hair doesn't average out to a very different color
# from the source, so it barely moves HAIR_DIFF_SATURATE's raw color-diff
# metric — it needs its own signal that specifically measures lost fine
# texture, not color change.
HAIR_DETAIL_WINDOW        = 9     # local-std window size (px) used as the texture-energy metric
HAIR_DETAIL_LOSS_SATURATE = 3.0   # local texture-energy deficit (original minus restored) at which this saturates
HAIR_DETAIL_STRENGTH      = 0.9   # max blend-toward-original weight the hair detail-loss signal can contribute

# The protection above only *preserves* hair — it blends damaged pixels back
# toward the source, so the best hair can ever look is exactly how it arrived.
# The restorer, meanwhile, only ever processes its aligned crop around the
# face: long hair past the jaw is never restored at all, and the global
# sharpen pass at the end is weakest exactly when the restored face is
# crisp (it adapts to whole-frame focus). Net result on long-haired subjects:
# a sharp restored face framed by source-soft hair. This stage closes the gap
# from the other side — a strand-scale unsharp mask scoped to the same
# hair-plausible zone (minus the core face, intersected with the GrabCut
# foreground so the background behind the hair isn't sharpened along with it),
# reactive to how much strand texture the hair region already has: soft/
# blurry hair gets pushed harder, hair that arrived crisp is barely touched.
HAIR_ENHANCE_SIGMA         = 2.0   # gaussian radius (px) of the unsharp detail layer — strand scale
HAIR_ENHANCE_PRE_SIGMA     = 1.0   # pre-smooth radius (px) making the detail layer band-pass instead of
                                   # high-pass: pixel-scale sensor/compression noise sits below this scale,
                                   # real strands at/above HAIR_ENHANCE_SIGMA. Boosting a plain high-pass
                                   # layer amplified that noise into a visible fishnet/mesh texture across
                                   # soft out-of-focus hair (worst exactly where hair is softest, i.e.
                                   # where strength scales highest) — cutting the noise octave out of the
                                   # detail layer itself removes most of it at the source.
HAIR_ENHANCE_STRENGTH      = 0.6   # boost at HAIR_ENHANCE_REFERENCE_STD (see reference-anchored pattern below)
HAIR_ENHANCE_REFERENCE_STD = 6.0   # std of the hair-zone band-pass detail layer this was validated against
HAIR_ENHANCE_MIN           = 0.1
HAIR_ENHANCE_MAX           = 0.9   # tuned by eye on real long-hair frames: 1.2 still meshed residual noise
                                   # through the structure gate, 0.7 left soft hair visibly under-defined

# Per-pixel structure gate on the boost: even band-passed, noise energy in
# soft/out-of-focus hair survives enough to mesh when amplified — but it's
# weaker than real strand structure. Measured on real long-hair footage:
# noise-only hair regions had local detail energy (see _local_detail_energy)
# with median ~2.8 / p75 ~4.3, real strand regions p75 ~8.4 / p95 ~20. The
# gate zeroes the boost below the floor (noise territory), reaches full
# strength at the saturate point (unambiguous strands), linear in between —
# so amplification concentrates on structure that actually exists. The
# distributions genuinely overlap in the middle; faint strands near the floor
# lose some boost, which is the acceptable side of the trade (hair staying
# slightly soft beats hair turning into amplified-noise fishnet).
HAIR_ENHANCE_GATE_FLOOR    = 3.0   # local detail energy at/below this -> no boost (noise territory)
HAIR_ENHANCE_GATE_SATURATE = 8.0   # local detail energy at/above this -> full boost (unambiguous strands)

POST_BLACK_POINT = 4  # levels-style black point (0-255): input <= this crushes to 0, range above rescales
POST_WHITE_POINT = 14  # levels-style white point (0-255): input >= (255-this) blows out to 255, range below rescales
WHITE_POINT_BRIGHTNESS_RANGE = 15  # max points added (dark subject) / subtracted (bright subject) from
                                    # POST_WHITE_POINT, scaled continuously by the *subject's* (face/skin)
                                    # mean brightness — not the whole frame's. A dark background behind a
                                    # well-lit face isn't an underexposed photo in the sense that matters for
                                    # a face thumbnail, and a bright background behind an underexposed face
                                    # is. Falls back to whole-frame mean if no face was detected.

# Custom tone curve — (input, output) control points on the 0-255 luminance scale.
# Identity by default (no change); the highlight point is adaptively pulled down
# by _adaptive_curve_points (see below). Applied on luminance only (YCrCb Y
# channel) so color/saturation are untouched.
CURVE_FLATTEN_MAX      = 30.0   # max points the white end of the curve can be pulled down
CURVE_FLATTEN_PHIGH_LO = 150.0  # photo's own 99th-percentile luma below this -> no flatten needed
CURVE_FLATTEN_PHIGH_HI = 230.0  # at/above this -> full flatten (highlight-heavy photo, tame it)

# Shadow lift — a "fake HDR" tone-mapping move: compress the shadow range
# upward (lift a mid-shadow reference point) while leaving true black (0) and
# the midtone/highlight points untouched, so blacks don't wash out gray but
# the broader dark range isn't as crushed/heavy either. Distinct from
# POST_BLACK_POINT, which controls how hard the very deepest tones crush to
# pure black — this instead lifts the shadow *range above* that floor.
SHADOW_LIFT_MAX     = 22.0  # max points the shadow reference point gets lifted
SHADOW_LIFT_X       = 50.0  # input luma value used as the shadow-range reference/anchor point
SHADOW_LIFT_PLOW_LO = 4.0   # photo's own 1st-percentile luma at/below this -> full shadow lift (deep shadows)
SHADOW_LIFT_PLOW_HI = 40.0  # at/above this -> no shadow lift needed (shadows aren't that deep to begin with)

SHARPEN_SIGMA = 1.5  # gaussian blur radius (px) used to build the unsharp mask detail layer

# Reference values below are what each *_BASE constant was tuned/validated against
# (measured on real restored frames from this app's own test footage) — every
# adjustment in this pipeline is reactive rather than a flat percentage, since
# incoming frames span wildly different lighting/color/contrast/focus. A photo
# already at the reference level gets exactly the historically-validated amount;
# one further from it gets proportionally more or less, so a flat/hazy/dark/
# blurry frame isn't left undercorrected and an already-punchy/sharp one isn't
# pushed past the point of looking natural.
POST_CONTRAST          = 15.0  # percent bump, at CONTRAST_REFERENCE_STD
CONTRAST_REFERENCE_STD = 47.2  # luma std this was validated against
CONTRAST_ALPHA_MIN     = 1.0
CONTRAST_ALPHA_MAX     = 1.6   # lowered from 2.0 — see CONTRAST_PIVOT note below
CONTRAST_PIVOT         = 128.0  # mid-gray pivot the contrast stretch expands away from — see _apply_post_adjustments
# POST_CONTRAST/CONTRAST_ALPHA_MAX were tuned by eye against a bug where this
# stage multiplied luma from a zero pivot ("Y * alpha") instead of the
# midtone ("(Y - 128) * alpha + 128") — a brightness gain mislabeled as
# contrast, which happened to make dark photos look more "acceptable" by
# flattening everything toward white rather than genuinely adding contrast.
# Fixed to pivot at CONTRAST_PIVOT; measured on real dark footage that the
# old alpha values (up to 2.0, ~35% boost at reference) then crushed 10-24%
# of pixels to near-black even before any brightness adjustment on top —
# retuned down to what keeps a real dark scene's shadow crush in a
# comparable range to before this fix (~5-6% near-black, vs <0.1% pre-
# restoration) instead of the 10-24% the old values produced through the
# corrected formula.

POST_SATURATION        = 10     # percent bump, at SATURATION_REFERENCE
SATURATION_REFERENCE   = 0.332  # mean HSV-S (0-1) this was validated against
SATURATION_FACTOR_MIN  = 1.0
SATURATION_FACTOR_MAX  = 2.0

# Ceilings for the two grading stages that have no measurement cap of their own
# and for the tone curve's individual points, used only when the edit-preset
# dial is pushed past the measured amount (see PostStats.at_intensity). Each is
# roughly three times what the measurement itself produces on ordinary footage,
# which is enough headroom for a visibly heavier look and not enough for a
# posterised one.
BLACK_PUSH_CEILING     = 45.0
WHITE_POINT_CEILING    = 60.0
CURVE_POINT_CEILING    = 60.0   # how far a curve point may end up from the identity line

POST_BRIGHTNESS                   = -3    # percent luminance shift, at BRIGHTNESS_REFERENCE_SUBJECT_LUMA
                                           # (negative = darken; sets the direction for _adaptive_brightness_beta)
                                           # Lowered from -10 — combined with CONTRAST_PIVOT's now-correct shadow
                                           # spread, -10's beta (up to -27 for a bright subject) was crushing
                                           # legitimate shadow tones to near-black on top of what contrast alone
                                           # already did (measured 24% of a real dark scene's pixels driven to
                                           # near-black, vs ~6% with this value).
BRIGHTNESS_REFERENCE_SUBJECT_LUMA = 83.4  # subject (skin) mean luma this was validated against
BRIGHTNESS_BETA_MAX                = 255.0 * 0.30  # cap: never shift luminance by more than 30% of full range

POST_SHARPNESS          = 27.5  # percent-strength unsharp mask, at SHARPEN_REFERENCE_LAPVAR
SHARPEN_REFERENCE_LAPVAR = 10.8  # luma laplacian variance (focus proxy) this was validated against
SHARPEN_MIN              = 10
SHARPEN_MAX              = 55

LOCAL_CONTRAST_STRENGTH        = 0.18   # midtone/"clarity" strength on skin, at CLARITY_REFERENCE_DETAIL_STD
LOCAL_CONTRAST_SIGMA           = 15.0   # large-radius unsharp mask — isolates mid-frequency texture, not fine edges
CLARITY_REFERENCE_DETAIL_STD   = 12.14  # std of the skin-region detail layer this was validated against
CLARITY_STRENGTH_MIN           = 0.05
CLARITY_STRENGTH_MAX           = 0.40

DITHER_STRENGTH = 0.5  # +/- this many levels of uniform noise added right before the single final uint8
                       # cast — black_point/white_point/contrast all stretch the tonal range, which widens
                       # gaps between adjacent 8-bit levels and shows up as visible banding in smooth
                       # gradients (most noticeably in shadows, which start with the fewest distinct levels);
                       # a small dither breaks the sharp steps into imperceptible grain instead

NOISE_SATURATE    = 6.0  # estimated noise sigma (0-255 scale) at which denoise blend reaches full strength

# Face/skin: gentle filter, capped well below full strength — there's real texture
# (pores, wrinkles) worth preserving there, and the local-contrast/sharpen passes
# later expect it.
DENOISE_DIAMETER    = 7
DENOISE_SIGMA_COLOR = 50
DENOISE_SIGMA_SPACE = 50
DENOISE_FACE_MAX    = 0.6

# Background: stronger filter, can go all the way to fully denoised — no fine
# detail there worth protecting, so noise can be extinguished rather than just reduced.
DENOISE_BG_DIAMETER    = 15
DENOISE_BG_SIGMA_COLOR = 100
DENOISE_BG_SIGMA_SPACE = 100
DENOISE_BG_MAX         = 1.0

# The background filter above is deliberately destructive of fine detail, so
# it is computed on a half-resolution copy and upsampled: a d=15/sigmaSpace=100
# bilateral over a multi-megapixel float32 image was the single most expensive
# operation in this module, and halving each dimension quarters it. The
# half-resolution parameters below cover the same spatial neighbourhood.
# sigmaColor is NOT halved — it measures color distance, which doesn't scale
# with resolution.
DENOISE_BG_WORK_SCALE      = 0.5
DENOISE_BG_DIAMETER_HALF   = 7
DENOISE_BG_SIGMA_SPACE_HALF = 50

CHROMA_DENOISE_SIGMA    = 4.0   # gaussian blur radius (px) on Cr/Cb over skin/face
CHROMA_DENOISE_BG_SIGMA = 10.0  # larger radius over background — no color detail there worth protecting,
                                # so it can be smoothed harder than skin
CHROMA_DENOISE_MAX      = 1.0   # can reach full blur strength — color-channel resolution loss is far less
                                # perceptible than luminance loss, so fully smoothing chroma when noise is
                                # severe costs essentially nothing visually

# Subject masks (skin/hair/body) are inherently soft — they exist to be
# feathered by 10-45px and used as blend weights, never to resolve detail.
# Computing them at this width and upsampling is visually indistinguishable
# and avoids running GrabCut's upscale plus three large-sigma gaussians over
# a full multi-megapixel frame.
MASK_WORK_WIDTH = 1280

# Module-level state — populated lazily on first restore call
_restorer     = None
_backend_name = None   # "codeformer" | "gfpgan" | None


def _try_codeformer() -> bool:
    """
    Attempt to initialise CodeFormer. ALWAYS FAILS — see the module header.

    `codeformer-pytorch` is not a package that exists, so this raises
    ImportError on every call and the caller falls through to GFPGAN. It is
    left in place rather than deleted because it is the shape of the thing
    somebody would want if they ever revive this, but it is not a working
    starting point: the real package is `codeformer`, its class is
    `codeformer.CodeFormer`, it takes and returns PIL images rather than BGR
    arrays, and its fidelity weight is an argument to forward() rather than to
    the constructor. Reviving it means rewriting this function and adding a
    PIL round-trip in _run_restorer, not changing the import line.
    """
    global _restorer, _backend_name
    try:
        import torch
        from codeformer_pytorch import CodeFormerRestoration   # pip install codeformer-pytorch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _restorer = CodeFormerRestoration(
            upscale=1,
            fidelity_weight=FIDELITY_WEIGHT,
            device=device,
        )
        _backend_name = "codeformer"
        log.info("face_restorer: CodeFormer ready on %s", device)
        return True
    except Exception as e:
        log.debug("face_restorer: CodeFormer not available (%s)", e)
        return False


def _try_gfpgan() -> bool:
    """Attempt to initialise GFPGAN v1.4 as fallback."""
    global _restorer, _backend_name
    try:
        import torch
        from gfpgan import GFPGANer   # pip install gfpgan

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _restorer = GFPGANer(
            model_path=(
                "https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.4/GFPGANv1.4.pth"
            ),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        _backend_name = "gfpgan"
        log.info("face_restorer: GFPGAN v1.4 ready on %s", device)
        return True
    except Exception as e:
        # Warning, not debug: GFPGAN is the ONLY backend this app has ever
        # run on (CodeFormer above cannot load at all, so its failure stays
        # quiet). If this one fails, the whole
        # GAN stage is gone and the only symptom the user sees is "output
        # looks like the classical pipeline" — the traceback is the only
        # thing that says whether it's a missing install, missing weights, a
        # torchvision API mismatch, or an OOM.
        log.warning("face_restorer: GFPGAN not available (%s: %s)", type(e).__name__, e, exc_info=True)
        return False


def _ensure_loaded() -> bool:
    if _restorer is not None:
        return True
    if _backend_name is None:
        return _try_codeformer() or _try_gfpgan()
    return False


def preload() -> bool:
    """
    Load the restoration backend now rather than on the first /enhance-frame.
    Called at server startup: importing torch and loading weights takes long
    enough (tens of seconds cold) that paying it mid-interaction reads as the
    app having hung on the user's very first click.
    """
    return _ensure_loaded()


def backend_name() -> str | None:
    return _backend_name


def _run_restorer(image: np.ndarray) -> np.ndarray | None:
    """
    Restored copy of the WHOLE image, or None if restoration is unavailable or
    the model raised.

    One pass over the frame the caller gives it — no region scoping, no
    hardware-dependent precision, no second attempt on a different scope. The
    model decides for itself which faces it finds and rewrites only those
    aligned crops, leaving every other pixel byte-identical to the input,
    which is what the protection blends downstream rely on.
    """
    if not _ensure_loaded():
        return None

    try:
        if _backend_name == "codeformer":
            return _restorer.restore(image)
        if _backend_name == "gfpgan":
            _, _, out = _restorer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True,
            )
            return out
    except Exception as e:
        log.warning("face_restorer: restoration failed (%s)", e)
    return None


# ── Protection masks ───────────────────────────────────────────────────────
# All of these take the already-detected face list. They used to each call
# detect_faces() themselves, running the cascade three separate times over the
# same (large) frame for the same answer.

def _eye_protection_mask(image: np.ndarray, faces: list) -> np.ndarray:
    """
    Per-pixel mask (float32, 0..1) marking detected eye regions, feathered at the edges.

    GFPGAN/CodeFormer occasionally hallucinate a deformed eye shape — most often with
    glasses or a slight off-angle — since the eye region is small and detail-dense.
    Blending the restored output back toward the original source there (weighted by
    `fidelity`) removes the artifact without touching the restoration anywhere else
    on the face.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for face in faces:
        for (ex, ey, ew, eh) in detect_eyes(image, face):
            cx, cy = ex + ew // 2, ey + eh // 2
            axes = (int(ew * 0.75), int(eh * 0.85))
            cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    if not mask.any():
        return np.zeros((h, w), dtype=np.float32)

    return feather_mask(mask, radius=EYE_PROTECT_FEATHER).astype(np.float32) / 255.0


def _eye_deviation_protection(diff: np.ndarray, eye_mask: np.ndarray) -> np.ndarray:
    """
    Extra protection weight (0..1) proportional to how much the restorer actually
    changed each eye pixel vs the source.

    Glasses are the single biggest trigger of GFPGAN eye deformation — lens
    reflections and thin frames confuse its face-alignment step — but thin wire
    frames don't reliably show up in edge/cascade-based glasses detectors (tested:
    nose-bridge edge density and the eye_tree_eyeglasses cascade both missed them
    on real footage). Measuring the actual pixel deviation sidesteps that: whatever
    the cause, a large local change *is* the deformation, so pulling back toward
    the original in proportion to that deviation catches glasses artifacts (and
    any other localized hallucination) without needing to classify the cause.

    `diff` is the shared per-pixel mean absolute difference between source and
    restored — computed once by the caller, since the seam and hair checks
    below need the identical array.
    """
    severity = np.clip(diff / EYE_DIFF_SATURATE, 0.0, 1.0)
    return severity * eye_mask * EYE_DIFF_STRENGTH


def _seam_saturate(signal: np.ndarray, footprint: np.ndarray, floor: float) -> float:
    """
    The level at which a seam check should reach full strength on THIS frame:
    the higher of its fixed floor and this frame's own typical rebuild level
    (see the SEAM_SCALE_* comment).

    `footprint` marks the pixels the restorer actually changed — the only
    place the question is meaningful. Everything outside it is byte-identical
    to the source and would just drag the percentile to zero.
    """
    inside = signal[footprint]
    if inside.size < 100:
        return floor
    # Strided like every other whole-image statistic here (see
    # image_utils.stats_sample) — that helper takes a 2D image, and boolean
    # indexing has already flattened this one.
    step = max(1, inside.size // STATS_PIXEL_BUDGET)
    typical = float(np.percentile(inside[::step], SEAM_SCALE_PERCENTILE))
    return max(floor, typical * SEAM_SCALE_MULTIPLE)


def _seam_deviation_protection(diff: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """
    Extra protection weight (0..1), same cause-agnostic deviation approach as
    _eye_deviation_protection but with no zone mask — catches the restorer's
    own face-alignment paste-back seam (see the SEAM_DIFF_* comment), which
    can land anywhere on the face, not just eyes or hair.

    The saturate point is this frame's own (see _seam_saturate), so a face
    the restorer had to rebuild heavily isn't flagged for the rebuild itself.

    No mask needed to keep this from bleeding into the background: GFPGAN
    runs with bg_upsampler=None (see _try_gfpgan) and CodeFormer pastes its
    restored crop back onto the untouched original background, so background
    pixels come back numerically identical to `image` either way — a large
    diff here can only mean the restorer actually touched that pixel.
    """
    saturate = _seam_saturate(diff, footprint, SEAM_DIFF_SATURATE)
    return np.clip(diff / saturate, 0.0, 1.0) * SEAM_DIFF_STRENGTH


def _paste_border_band(diff: np.ndarray, footprint: np.ndarray, faces: list) -> np.ndarray:
    """
    Soft mask (0..1) covering the border of the region the restorer replaced —
    where a misaligned paste-back shows up as a seam (see the SEAM_BAND_*
    comment). Everything well inside that region, where reconstruction lives,
    is left out.

    Widths scale with the subject's face, so the band means the same thing on
    a close-up as on a wide shot. With no face detected, the footprint's own
    extent stands in.
    """
    if not footprint.any():
        return np.zeros(diff.shape, dtype=np.float32)

    faces = [f for f in faces if f[3] > 0]
    face_h = max((f[3] for f in faces), default=0) or float(np.sqrt(footprint.sum()))

    sigma = max(3.0, face_h * SEAM_BAND_DENSITY_SIGMA_FRAC)
    region = (cv2.GaussianBlur(diff, (0, 0), sigma) > TOUCHED_DIFF).astype(np.uint8)

    inward = max(3, int(face_h * SEAM_BAND_INWARD_FRAC)) | 1
    inner = cv2.erode(region, np.ones((inward, inward), np.uint8))
    outer = cv2.dilate(region, np.ones((SEAM_BAND_OUTWARD_PX,) * 2, np.uint8))
    band = (outer - inner).astype(np.float32)
    return cv2.GaussianBlur(band, (0, 0), max(2.0, inward * SEAM_BAND_FEATHER_FRAC))


def _seam_edge_protection(gray_image: np.ndarray, gray_out: np.ndarray,
                          footprint: np.ndarray, band: np.ndarray) -> np.ndarray:
    """
    Extra protection weight (0..1) where the restorer introduced edge energy
    well beyond what amplifying an already-existing edge there can explain
    (see SEAM_EDGE_ALLOWANCE) — catches seams _seam_deviation_protection
    misses (a smooth tonal mismatch too subtle to trip a magnitude check, but
    still a line that wasn't there before).

    SEAM_EDGE_ALLOWANCE scales the allowance to the edge that was already
    there, which covers a sharpened source but not a rebuilt one: on an
    out-of-focus face there is no edge to scale against, so every recovered
    eyelash and beard hair reads as "a line that wasn't there". Two things
    separate those cases — a saturate point taken from the frame's own rebuild
    level (see _seam_saturate), and `band`, which restricts the whole check to
    where a seam can physically be (see _paste_border_band).
    """
    edge_image = np.abs(cv2.Laplacian(gray_image, cv2.CV_32F, ksize=3))
    edge_out = np.abs(cv2.Laplacian(gray_out, cv2.CV_32F, ksize=3))
    excess_edge = np.clip(edge_out - edge_image * SEAM_EDGE_ALLOWANCE, 0, None)
    saturate = _seam_saturate(excess_edge, footprint, SEAM_EDGE_SATURATE)
    severity = np.clip(excess_edge / saturate, 0.0, 1.0)
    severity = cv2.dilate(severity, np.ones((SEAM_EDGE_WIDEN_PX, SEAM_EDGE_WIDEN_PX), np.uint8))
    severity = cv2.GaussianBlur(severity, (0, 0), SEAM_EDGE_SOFTEN_SIGMA)
    return severity * band * SEAM_EDGE_STRENGTH


def _hair_zone_mask(image_shape: tuple[int, int], face: tuple[int, int, int, int]) -> np.ndarray:
    """
    Generous region where hair plausibly falls: the face bounding box expanded
    outward (wide enough for temples/sides, tall enough above for the crown
    and below for long hair reaching past the jaw onto the shoulders). Not a
    hair *segmentation* — just a bound on where the restorer's own crop could
    plausibly have touched hair, used to scope _hair_deviation_protection.
    """
    h, w = image_shape[:2]
    fx, fy, fw, fh = face
    cx = fx + fw / 2.0
    zone_w = fw * HAIR_ZONE_EXPAND_W
    zone_top = fy - fh * HAIR_ZONE_EXPAND_UP
    zone_bottom = fy + fh + fh * HAIR_ZONE_EXPAND_DOWN

    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = max(0, int(cx - zone_w / 2))
    x1 = min(w, int(cx + zone_w / 2))
    y0 = max(0, int(zone_top))
    y1 = min(h, int(zone_bottom))
    mask[y0:y1, x0:x1] = 255
    return mask


def _local_detail_energy(gray: np.ndarray, window: int = HAIR_DETAIL_WINDOW) -> np.ndarray:
    """
    Per-pixel map of how much fine local texture exists, via local standard
    deviation over a small window (box-filter mean-of-squares minus square-of-
    mean — the standard fast way to get this without a per-pixel sliding-window
    loop): flat/smoothed areas score low, strand-level hair texture scores
    high. Tried a high-pass-then-blur version first, but real footage's hair
    texture energy on that scale was too small and spiky to threshold
    reliably; local std over a window aggregates the same neighborhood more
    robustly.
    """
    mean = cv2.boxFilter(gray, -1, (window, window))
    mean_sq = cv2.boxFilter(gray * gray, -1, (window, window))
    variance = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(variance)


def _hair_deviation_protection(diff: np.ndarray, gray_image: np.ndarray, gray_out: np.ndarray,
                               hair_protect_mask: np.ndarray) -> np.ndarray:
    """
    Extra protection weight (0..1), scoped to the face-adjacent "hair zone"
    (see _hair_zone_mask) minus the core face, proportional to how much the
    restorer damaged that area — by either of two signals:

      1. Raw color deviation (as in _eye_deviation_protection) — catches
         drastic distortion/wrong colors, warped shapes.
      2. Local texture-energy loss — catches the restorer smoothing individual
         hair strands into a flat, waxy mass, which (1) mostly misses: a
         blurred patch of similarly-colored hair doesn't average out to a very
         different color, it just loses fine detail. Compares how much
         high-frequency texture existed at each pixel *before* vs *after*;
         wherever restoration removed real texture that was there, that's the
         "melted" signature regardless of hairstyle or what caused it.

    Whichever signal is stronger at a given pixel drives the blend-back —
    same cause-agnostic philosophy as eye protection, extended to a failure
    mode eye protection's metric alone doesn't catch.

    `hair_protect_mask` excludes the core face via the geometric face ellipse
    (face_ellipse_mask), not the color-based skin_mask used elsewhere in this
    module: skin_mask's YCrCb threshold routinely also matches brown/black
    hair (tried this first — measured skin_mask covering 69% of a real
    photo's hair region in testing), which would gut this exact protection for
    the most common hair colors. The ellipse is a rougher fit to the face than
    skin_mask, but doesn't depend on color at all, so it can't misclassify
    hair as face.
    """
    if not hair_protect_mask.any():
        return np.zeros(diff.shape, dtype=np.float32)

    color_severity = np.clip(diff / HAIR_DIFF_SATURATE, 0.0, 1.0) * HAIR_DIFF_STRENGTH

    detail_loss = np.clip(_local_detail_energy(gray_image) - _local_detail_energy(gray_out), 0, None)
    detail_severity = np.clip(detail_loss / HAIR_DETAIL_LOSS_SATURATE, 0.0, 1.0) * HAIR_DETAIL_STRENGTH

    return np.maximum(color_severity, detail_severity) * hair_protect_mask


def _face_damage(image: np.ndarray, faces: list) -> float:
    """
    How badly out of focus the subject's face is, 0 (sharp) to 1 (heavily
    blurred) — see the FOCUS_* block for what the metric is and why it's
    normalised the way it is.

    Measured on the LARGEST detected face: that's the subject the thumbnail
    is about, and it's the one whose reconstruction the protections either
    let through or throw away. A sharp bystander in the background shouldn't
    make a blurred subject look undamaged.

    Returns 0 when no face was found — with nothing identified to reconstruct,
    there's no reason to relax any protection.
    """
    faces = [f for f in faces if f[2] > 0 and f[3] > 0]
    if not faces:
        return 0.0

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    h, w = image.shape[:2]
    crop = image[max(0, fy):min(h, fy + fh), max(0, fx):min(w, fx + fw)]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return 0.0

    crop = cv2.resize(crop, (FOCUS_CANON_PX, FOCUS_CANON_PX), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    variance = float(gray.var())
    if variance < 1e-3:
        return 0.0

    focus = float(cv2.Laplacian(gray, cv2.CV_32F).var()) / variance
    return float(np.clip((FOCUS_SHARP - focus) / (FOCUS_SHARP - FOCUS_BLURRED), 0.0, 1.0))


def _subject_masks(image: np.ndarray, faces: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One segmentation pass over all detected faces, returning three feathered
    float32 (0..1) masks:

      - skin_mask: skin/face pixels — scopes local contrast, denoise weighting
        and every subject-brightness measurement to the subject, not the
        background.
      - hair_mask: the hair-plausible zone (_hair_zone_mask) minus the core
        face ellipse, intersected with the GrabCut foreground — scopes
        _apply_hair_enhance. The GrabCut intersection is what keeps the
        rectangular zone honest: the zone alone is mostly background on a
        short-haired subject, but background isn't foreground, so it drops
        out.
      - hair_protect_mask: the same zone minus the face, WITHOUT the
        foreground intersection — what _hair_deviation_protection is scoped
        to. Face exclusion uses the geometric ellipse rather than skin_mask
        for the reason documented in _hair_deviation_protection.

    Computed at MASK_WORK_WIDTH and upscaled: these are blend weights that
    get feathered by 10-45px regardless, so full-resolution segmentation
    bought nothing and cost several large-sigma gaussians over the whole
    frame.
    """
    h, w = image.shape[:2]
    scale = min(1.0, MASK_WORK_WIDTH / w)
    if scale < 1.0:
        work_w, work_h = max(1, int(w * scale)), max(1, int(h * scale))
        work = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        work_faces = [tuple(int(round(v * scale)) for v in f) for f in faces]
    else:
        work, work_faces = image, faces

    wh, ww = work.shape[:2]
    skin = np.zeros((wh, ww), dtype=np.float32)
    hair = np.zeros((wh, ww), dtype=np.float32)
    hair_protect = np.zeros((wh, ww), dtype=np.float32)

    feather = HAIR_ZONE_FEATHER * max(scale, 0.25)
    for face in work_faces:
        if face[2] <= 0 or face[3] <= 0:
            continue
        regions = segment_regions(work, face)
        skin = np.maximum(skin, regions["skin_mask"].astype(np.float32) / 255.0)

        zone_f = feather_mask(_hair_zone_mask(work.shape, face), radius=feather).astype(np.float32) / 255.0
        face_f = feather_mask(face_ellipse_mask(work.shape, face), radius=feather).astype(np.float32) / 255.0
        scoped = zone_f * (1.0 - face_f)
        body_f = regions["body_mask"].astype(np.float32) / 255.0
        hair = np.maximum(hair, scoped * body_f)
        hair_protect = np.maximum(hair_protect, scoped)

    if scale < 1.0:
        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)
        hair = cv2.resize(hair, (w, h), interpolation=cv2.INTER_LINEAR)
        hair_protect = cv2.resize(hair_protect, (w, h), interpolation=cv2.INTER_LINEAR)
    return skin, hair, hair_protect


# ── Adaptive measurements ──────────────────────────────────────────────────
# Each returns ONE scalar describing the image it's given. Whole-image
# statistics are taken on a strided sample (see image_utils.stats_sample) —
# a 1st/99th percentile over a quarter-million pixels is stable well below a
# tonal level, and computing it over every pixel of a multi-megapixel frame
# cost an order of magnitude more for the same number. Statistics that are
# resolution-dependent (noise sigma, laplacian variance, detail-layer energy)
# are NOT sampled: striding breaks the pixel neighbourhood they measure.

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 100:
        return float(values.mean())
    return float((values * weights).sum() / total)


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Fast noise-sigma estimate (Immerkaer 1996): convolve with a kernel that
    cancels flat regions and linear gradients, leaving mostly noise energy in
    the response, then scale that response into a sigma estimate. Cheap (one
    3x3 convolution) and doesn't need a reference/clean image to compare against.
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray, -1, kernel)
    h, w = gray.shape
    if w <= 2 or h <= 2:
        return 0.0
    return float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def _noise_severity(image: np.ndarray) -> float:
    return float(np.clip(_estimate_noise_sigma(luma(image)) / NOISE_SATURATE, 0.0, 1.0))


def _adaptive_black_push(y: np.ndarray) -> float:
    """
    How much of POST_BLACK_POINT this photo can actually take.

    Takes the luminance plane directly rather than a BGR image: cv2's YCrCb Y
    channel IS BT.601 luma, so the measurement pass — which already has Y in
    hand between stages — never needs to derive it again.

    Capped to a fraction of this photo's own shadow headroom (own 1st-
    percentile luma): tried applying the full requested push uncapped, but on
    a photo whose shadows already sit near black (a fair chunk of this app's
    real source footage — 10 of the 34 images in the test set have their own
    10th-percentile luma at 0, i.e. already-black content before any push at
    all), pushing the same fixed amount crushes a large share of the frame to
    a flat, detail-less block — measured 40% of pixels driven to 0 on one
    such test photo with no cap, vs 16% (all pre-existing black, none newly
    crushed) with this cap back in place. A photo with real headroom below
    its shadows still gets the full requested depth.
    """
    if POST_BLACK_POINT <= 0:
        return 0.0
    p_low = float(np.percentile(stats_sample(y), 1))
    return min(float(POST_BLACK_POINT), p_low * 0.8)


def _adaptive_white_point_amount(y: np.ndarray, skin_mask: np.ndarray) -> float:
    """
    Scales POST_WHITE_POINT by the *subject's* (face/skin) mean brightness —
    histogram-based, like the original version, but measured over the skin
    region rather than the whole frame. At subject mean 128 the request is
    unchanged, at subject mean 0 (dark subject) it's boosted by the full
    WHITE_POINT_BRIGHTNESS_RANGE, at subject mean 255 (bright subject) it's cut
    by the same amount. Falls back to the whole-frame mean if no face was
    detected (skin_mask empty). This is the only proportionality applied to
    the request — _apply_white_point no longer caps how much of it actually
    lands based on the photo's own headroom, so whatever this returns is
    applied in full.
    """
    subject_luma = _weighted_mean(y, skin_mask)
    adjustment = -WHITE_POINT_BRIGHTNESS_RANGE * (subject_luma - 128.0) / 128.0
    return max(0.0, POST_WHITE_POINT + adjustment)


def _adaptive_curve_points(y: np.ndarray) -> list[tuple[float, float]]:
    """
    Two independent adaptive moves on top of an otherwise-identity curve:

      - Highlight flatten: scales by how much highlight content the photo's
        own 99th percentile already has — little (below CURVE_FLATTEN_PHIGH_LO)
        -> identity, no flatten needed. A lot (at/above CURVE_FLATTEN_PHIGH_HI)
        -> full CURVE_FLATTEN_MAX pulled off the white point, taming a
        highlight-heavy photo.
      - Shadow lift: scales by how deep the photo's own 1st-percentile luma
        already is — at/below SHADOW_LIFT_PLOW_LO (deep shadows) -> full
        SHADOW_LIFT_MAX added at the SHADOW_LIFT_X reference point; at/above
        SHADOW_LIFT_PLOW_HI (shadows aren't that deep to begin with) ->
        identity, no lift needed. True black (0) and the midtone/highlight
        points are never touched by either move.

    Both linear in between their own thresholds.
    """
    p_low, p_high = np.percentile(stats_sample(y), [1, 99])

    flatten_norm = np.clip((float(p_high) - CURVE_FLATTEN_PHIGH_LO) / (CURVE_FLATTEN_PHIGH_HI - CURVE_FLATTEN_PHIGH_LO), 0.0, 1.0)
    flatten = CURVE_FLATTEN_MAX * flatten_norm

    lift_norm = np.clip((SHADOW_LIFT_PLOW_HI - float(p_low)) / (SHADOW_LIFT_PLOW_HI - SHADOW_LIFT_PLOW_LO), 0.0, 1.0)
    lift = SHADOW_LIFT_MAX * lift_norm

    return [(0, 0), (SHADOW_LIFT_X, SHADOW_LIFT_X + lift), (128, 128), (192, 192), (255, 255.0 - flatten)]


def _adaptive_contrast_alpha(y: np.ndarray) -> float:
    """
    Scales POST_CONTRAST's multiplicative stretch inversely with the photo's own
    luma std: at CONTRAST_REFERENCE_STD it's exactly the validated POST_CONTRAST
    boost, flatter/hazier photos (lower std) get more, already-punchy photos
    (higher std) get less — avoiding blown-out clipping on top of contrast that
    was already fine.
    """
    current_std = float(stats_sample(y).std())
    if current_std < 1e-3:
        return CONTRAST_ALPHA_MAX
    boost = (POST_CONTRAST / 100.0) * (CONTRAST_REFERENCE_STD / current_std)
    return float(np.clip(1.0 + boost, CONTRAST_ALPHA_MIN, CONTRAST_ALPHA_MAX))


def _adaptive_brightness_beta(y: np.ndarray, skin_mask: np.ndarray) -> float:
    """
    Scales POST_BRIGHTNESS's additive shift by the *subject's* (face/skin)
    mean brightness — not the whole frame's, same reasoning as
    _adaptive_white_point_amount — in whichever direction actually makes
    sense for the subject's own exposure:

      - POST_BRIGHTNESS > 0 (brighten): a subject darker than
        BRIGHTNESS_REFERENCE_SUBJECT_LUMA needs more lift, one already
        brighter needs less — ratio is REFERENCE/subject_mean.
      - POST_BRIGHTNESS < 0 (darken): a subject brighter than reference has
        headroom to tame, one already darker doesn't need darkening further
        — ratio is subject_mean/REFERENCE, the other way around.

    Using the same (REFERENCE/subject_mean) ratio regardless of sign — the
    previous version of this function — meant that when POST_BRIGHTNESS was
    negative, an already-dark subject (small subject_mean, so the ratio is
    large) got pushed toward *more* darkening instead of less, making the
    darkest photos in a batch the ones darkened the most.

    Capped at +/- BRIGHTNESS_BETA_MAX so a single stage can't run away; the
    clip range follows POST_BRIGHTNESS's sign so the result never crosses
    back past zero into the opposite direction.
    """
    subject_mean = _weighted_mean(y, skin_mask)

    lo, hi = (0.0, BRIGHTNESS_BETA_MAX) if POST_BRIGHTNESS >= 0 else (-BRIGHTNESS_BETA_MAX, 0.0)

    if subject_mean < 1e-3:
        # Near-black subject: max lift if brightening, but *no* extra push if
        # darkening — it has nothing left to give.
        return hi if POST_BRIGHTNESS >= 0 else 0.0

    beta_at_reference = POST_BRIGHTNESS / 100.0 * 255.0
    ratio = (
        BRIGHTNESS_REFERENCE_SUBJECT_LUMA / subject_mean if POST_BRIGHTNESS >= 0
        else subject_mean / BRIGHTNESS_REFERENCE_SUBJECT_LUMA
    )
    return float(np.clip(beta_at_reference * ratio, lo, hi))


def _adaptive_saturation_factor(image: np.ndarray) -> float:
    """
    Scales POST_SATURATION's multiplier inversely with the photo's own mean
    HSV saturation: at SATURATION_REFERENCE it's exactly the validated boost,
    a muted/desaturated photo gets more, an already-vivid one gets less —
    avoiding oversaturation on footage that didn't need it.
    """
    hsv = cv2.cvtColor(stats_sample(image) / 255.0, cv2.COLOR_BGR2HSV)
    current_s = float(hsv[:, :, 1].mean())
    if current_s < 1e-3:
        return SATURATION_FACTOR_MAX
    boost = (POST_SATURATION / 100.0) * (SATURATION_REFERENCE / current_s)
    return float(np.clip(1.0 + boost, SATURATION_FACTOR_MIN, SATURATION_FACTOR_MAX))


def _adaptive_clarity_strength(image: np.ndarray, skin_mask: np.ndarray) -> float:
    """
    Local-contrast strength, reactive to how much mid-frequency detail the
    skin region already has (already-textured/sharp skin gets less, flat/soft
    skin gets more), anchored at LOCAL_CONTRAST_STRENGTH /
    CLARITY_REFERENCE_DETAIL_STD.
    """
    if skin_mask.sum() <= 0:
        return 0.0
    detail = luma(image) - luma(cv2.GaussianBlur(image, (0, 0), LOCAL_CONTRAST_SIGMA))
    current_std = float(np.sqrt(_weighted_mean(detail ** 2, skin_mask)))
    if current_std < 1e-3:
        return CLARITY_STRENGTH_MAX
    strength = LOCAL_CONTRAST_STRENGTH * (CLARITY_REFERENCE_DETAIL_STD / current_std)
    return float(np.clip(strength, CLARITY_STRENGTH_MIN, CLARITY_STRENGTH_MAX))


def _adaptive_hair_strength(image: np.ndarray, hair_mask: np.ndarray) -> float:
    """
    Hair-enhancement strength: scales inversely with how much strand-scale
    texture the hair already has, so soft/blurred hair is pushed toward the
    validated reference crispness while hair that arrived sharp is left
    essentially alone.
    """
    if float(hair_mask.sum()) < 100:
        return 0.0
    smooth = cv2.GaussianBlur(image, (0, 0), HAIR_ENHANCE_PRE_SIGMA)
    detail = smooth - cv2.GaussianBlur(smooth, (0, 0), HAIR_ENHANCE_SIGMA)
    current_std = float(np.sqrt(_weighted_mean(luma(detail) ** 2, hair_mask)))
    if current_std < 1e-3:
        return HAIR_ENHANCE_MAX
    strength = HAIR_ENHANCE_STRENGTH * (HAIR_ENHANCE_REFERENCE_STD / current_std)
    return float(np.clip(strength, HAIR_ENHANCE_MIN, HAIR_ENHANCE_MAX))


def _adaptive_sharpen_amount(image: np.ndarray) -> float:
    """
    Scales POST_SHARPNESS inversely with the photo's own luma laplacian
    variance (a standard focus-sharpness proxy, already used elsewhere in this
    app's frame-selection scoring): at SHARPEN_REFERENCE_LAPVAR it's exactly the
    validated unsharp-mask strength, a soft/out-of-focus frame gets pushed
    toward SHARPEN_MAX, an already-crisp frame gets pulled back toward
    SHARPEN_MIN — sharpening an already-sharp face otherwise just adds halos
    and noise without recovering any real detail.
    """
    lap_var = float(cv2.Laplacian(luma(image), cv2.CV_32F).var())
    if lap_var < 1e-3:
        return SHARPEN_MAX / 100.0
    amount = POST_SHARPNESS * (SHARPEN_REFERENCE_LAPVAR / lap_var)
    return float(np.clip(amount, SHARPEN_MIN, SHARPEN_MAX)) / 100.0


# ── Pixel stages ───────────────────────────────────────────────────────────

def _build_curve_lut(points: list[tuple[float, float]]) -> np.ndarray:
    """Monotonic (PCHIP) interpolation through the given control points -> a 256-entry float32 LUT."""
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    curve = PchipInterpolator(xs, ys)
    return np.clip(curve(np.arange(256)), 0, 255).astype(np.float32)


def _apply_denoise(image: np.ndarray, skin_mask: np.ndarray, severity: float) -> np.ndarray:
    """
    Edge-preserving denoise, blended in proportional to this photo's own
    estimated noise level and weighted by region: a clean, low-noise frame is
    left essentially untouched either way, but a noisy one gets a stronger
    filter and a higher blend cap over the background (up to fully denoised —
    no fine detail there worth protecting) than over skin/face (capped well
    below full, gentler filter, so texture and the later local-contrast/
    sharpen passes aren't undone). Runs first, before any tonal stretch
    (black_point/white_point/contrast all amplify existing noise proportional
    to how hard they stretch).

    Both filters run on uint8, not float32: OpenCV has a dedicated 8-bit
    bilateral path that is several times faster than its generic float one,
    and the input at this point is a freshly-decoded 8-bit image anyway — the
    float32 chain begins with this function's own output.
    """
    src = image.astype(np.float32)
    if severity <= 0.01:
        return src

    src_u8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

    face_denoised = cv2.bilateralFilter(
        src_u8, d=DENOISE_DIAMETER, sigmaColor=DENOISE_SIGMA_COLOR, sigmaSpace=DENOISE_SIGMA_SPACE,
    ).astype(np.float32)

    # Background: half resolution, then upsampled — see DENOISE_BG_WORK_SCALE.
    h, w = src_u8.shape[:2]
    sw, sh = max(1, int(w * DENOISE_BG_WORK_SCALE)), max(1, int(h * DENOISE_BG_WORK_SCALE))
    small = cv2.resize(src_u8, (sw, sh), interpolation=cv2.INTER_AREA)
    bg_small = cv2.bilateralFilter(
        small, d=DENOISE_BG_DIAMETER_HALF,
        sigmaColor=DENOISE_BG_SIGMA_COLOR, sigmaSpace=DENOISE_BG_SIGMA_SPACE_HALF,
    )
    bg_denoised = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    bg_weight = (1.0 - skin_mask)[:, :, np.newaxis]
    denoised = face_denoised * (1.0 - bg_weight) + bg_denoised * bg_weight

    max_strength = DENOISE_FACE_MAX + (DENOISE_BG_MAX - DENOISE_FACE_MAX) * bg_weight
    blend = severity * max_strength
    return src * (1.0 - blend) + denoised * blend


def _apply_tone_and_contrast(image: np.ndarray, black_push: float, white_point: float,
                             curve_points: list, alpha: float, beta: float) -> np.ndarray:
    """
    Black point, white point, tone curve, contrast and brightness — all four
    stages in ONE luminance round trip.

    Every one of them operates on the YCrCb Y channel only (never on the raw
    B/G/R channels, which would scale the ratio between channels and shift
    color instead of just adjusting tone). They used to be four separate
    functions, each converting BGR->YCrCb, touching Y, and converting back:
    eight full-image color conversions on float32 to accomplish four
    single-channel operations. Fusing them is arithmetically identical — each
    stage still sees exactly what the previous one produced — at a quarter of
    the conversion cost.

    Contrast pivots around CONTRAST_PIVOT (mid-gray), not zero: "Y * alpha"
    only ever *increases* every value (for alpha > 1, alpha*Y > Y whenever
    Y > 0) — it's a brightness gain wearing a contrast label, and at the
    alpha a flat/hazy photo can pull (up to CONTRAST_ALPHA_MAX) it blows
    bright/mid tones toward white long before it does anything useful to
    shadows. "(Y - pivot) * alpha + pivot" stretches shadows and highlights
    apart from the midtone symmetrically instead, which is what "contrast"
    is actually supposed to mean.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]

    if black_push > 0:
        y = np.clip((y - black_push) * (255.0 / (255.0 - black_push)), 0, 255)

    if white_point > 0:
        # Numerical guard only (avoid a zero/negative denominator). The full
        # requested push is always applied — not capped to this photo's own
        # highlight headroom, so a photo whose highlights are already near
        # white gets blown out over the same top range as any other photo.
        threshold = 255.0 - min(white_point, 254.0)
        y = np.clip(y * (255.0 / threshold), 0, 255)

    lut = _build_curve_lut(curve_points)
    # np.interp (not cv2.LUT, which requires uint8) so the curve applies to
    # the exact float32 luma value instead of rounding to the nearest 8-bit
    # level first.
    y = np.interp(y, np.arange(256, dtype=np.float32), lut).astype(np.float32)

    ycrcb[:, :, 0] = np.clip((y - CONTRAST_PIVOT) * alpha + CONTRAST_PIVOT + beta, 0, 255)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _apply_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0 + 1e-6:
        return image
    hsv = cv2.cvtColor(image / 255.0, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 1)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255.0, 0, 255)


def _apply_local_contrast(image: np.ndarray, skin_mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Midtone/local contrast ("clarity") — a large-radius unsharp mask that boosts
    texture definition (pores, wrinkles, fabric weave) without stretching the
    image's overall tonal histogram the way POST_CONTRAST's linear stretch does.
    Restricted to skin/face via `skin_mask` so background texture isn't also
    sharpened. Operates on and returns float32 — no intermediate uint8 rounding.
    """
    if strength <= 0 or not skin_mask.any():
        return image
    detail = image - cv2.GaussianBlur(image, (0, 0), LOCAL_CONTRAST_SIGMA)
    a = (skin_mask * strength)[:, :, np.newaxis]
    return np.clip(image + detail * a, 0, 255)


def _apply_hair_enhance(image: np.ndarray, hair_mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Strand-scale band-pass boost scoped to the hair zone (see the
    HAIR_ENHANCE_* constants block for why hair needs its own enhancement
    pass at all, and for the band-pass/gate rationale). On top of the overall
    strength, a per-pixel structure gate concentrates the boost where real
    strand structure exists and zeroes it over noise-only softness. Amplifies
    texture that exists rather than inventing strands — a fully melted patch
    has little detail to amplify, which is also what keeps the pass safe from
    hallucination. Operates on and returns float32.
    """
    if strength <= 0 or float(hair_mask.sum()) < 100:
        return image

    smooth = cv2.GaussianBlur(image, (0, 0), HAIR_ENHANCE_PRE_SIGMA)
    detail = smooth - cv2.GaussianBlur(smooth, (0, 0), HAIR_ENHANCE_SIGMA)

    energy = _local_detail_energy(luma(smooth))
    gate = np.clip(
        (energy - HAIR_ENHANCE_GATE_FLOOR) / (HAIR_ENHANCE_GATE_SATURATE - HAIR_ENHANCE_GATE_FLOOR),
        0.0, 1.0,
    )

    a = (hair_mask * strength * gate)[:, :, np.newaxis]
    return np.clip(image + detail * a, 0, 255)


def _apply_chroma_denoise(image: np.ndarray, skin_mask: np.ndarray, severity: float) -> np.ndarray:
    """
    Final chroma-only denoise — blurs just the color channels (Cr/Cb), leaving
    luminance (and the sharpening/local-contrast work already done on it)
    untouched. Background gets a larger blur radius than skin/face — there's
    no color detail worth protecting there, so it can be smoothed harder.

    Colored noise is the most visible residual by the end of the chain: the
    saturation boost amplifies whatever chroma noise survived the first
    (luminance-focused) denoise pass at the start, and contrast/local-contrast/
    sharpening all boost high-frequency content generally, including that
    residual. The severity is re-measured on the fully processed image rather
    than reusing the initial estimate, since it reflects how much those
    amplifying stages actually made it worse. Human vision is far less
    sensitive to chroma resolution than luminance resolution, so smoothing
    color here costs essentially nothing perceptually.
    """
    if severity <= 0.01:
        return image

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]

    face_cr = cv2.GaussianBlur(cr, (0, 0), CHROMA_DENOISE_SIGMA)
    face_cb = cv2.GaussianBlur(cb, (0, 0), CHROMA_DENOISE_SIGMA)
    bg_cr = cv2.GaussianBlur(cr, (0, 0), CHROMA_DENOISE_BG_SIGMA)
    bg_cb = cv2.GaussianBlur(cb, (0, 0), CHROMA_DENOISE_BG_SIGMA)

    bg_weight = 1.0 - skin_mask
    blurred_cr = face_cr * (1.0 - bg_weight) + bg_cr * bg_weight
    blurred_cb = face_cb * (1.0 - bg_weight) + bg_cb * bg_weight

    blend = severity * CHROMA_DENOISE_MAX
    ycrcb[:, :, 1] = cr * (1.0 - blend) + blurred_cr * blend
    ycrcb[:, :, 2] = cb * (1.0 - blend) + blurred_cb * blend
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ── Measure / apply ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PostStats:
    """
    Every adaptive parameter the classical pipeline needs, measured once on a
    frame's reference window and then frozen. Two crops of the same frame
    graded with the same PostStats are guaranteed to match.
    """
    noise_severity: float
    black_push: float
    white_point: float
    curve_points: list
    contrast_alpha: float
    brightness_beta: float
    saturation_factor: float
    clarity_strength: float
    hair_strength: float
    sharpen_amount: float
    chroma_severity: float

    def at_intensity(self, intensity: float) -> "PostStats":
        """
        The same measurements with only the GRADING scaled — restoration is
        left at full strength.

        This is what the edit presets move. They used to be applied as a
        final per-pixel blend of the whole result back toward the untouched
        source, which pulled back everything at once: at the default
        "natural" (0.35) the frame kept only a third of its deblurring,
        denoise and detail recovery, so a soft, slightly out-of-focus source
        came out looking barely reconstructed. Measured on a soft frame:
        sharpness rose 1.26x at that preset against 2.06x at full.

        Restoration isn't a matter of taste — a blurry frame needs the same
        reconstruction regardless of how punchy the user wants the picture.
        Tone is the part that's a preference, so the dial moves only that:
        black/white point, the tone curve, contrast, brightness and saturation
        each slide from doing nothing toward their measured amount, while
        denoise, clarity, hair enhancement, sharpening and the GAN blend stay
        exactly where they were measured.

        ## Why `intensity` may exceed 1

        Because the measured amount is not, on its own, a range anybody can
        see. What measure_post_stats decides is what THIS photo needs to look
        correct, and on already-graded source — a comedy special, a finished
        broadcast — that is a small correction: measured on real footage,
        contrast 1.20, white point 23, a shadow lift of 22 points. Sliding
        from none to all of that moves skin by about six levels of 255, which
        is 2%. Three buttons two percent apart are three buttons nobody can
        tell apart, and the middle one being called "Heavy edit" made that a
        promise the app was not keeping.

        So the top of the range now pushes PAST what was measured (see
        EDIT_PRESETS): 1.0 is still exactly the correction the frame asked
        for, and above it the same correction is simply applied harder. The
        stretch is bounded by the very ceilings measure_post_stats respects,
        so an already-punchy source cannot be amplified into one — a flat,
        hazy photo that measured near CONTRAST_ALPHA_MAX has almost no room
        above it and barely moves, while the finished-looking footage this
        exists for has all of it.
        """
        t = max(0.0, float(intensity))
        if t == 1.0:
            return self

        def toward(value, neutral, ceiling):
            """`value` scaled by t about `neutral`, never past `ceiling`."""
            scaled = neutral + (value - neutral) * t
            return min(scaled, ceiling) if value >= neutral else max(scaled, -ceiling)

        return replace(
            self,
            black_push=toward(self.black_push, 0.0, BLACK_PUSH_CEILING),
            white_point=toward(self.white_point, 0.0, WHITE_POINT_CEILING),
            # Toward — or past — the identity curve (output == input), point by
            # point, with each point held inside the same span the curve's own
            # measurement is allowed to move it.
            curve_points=[(x, x + toward(y - x, 0.0, CURVE_POINT_CEILING))
                          for x, y in self.curve_points],
            contrast_alpha=toward(self.contrast_alpha, 1.0, CONTRAST_ALPHA_MAX),
            brightness_beta=toward(self.brightness_beta, 0.0, BRIGHTNESS_BETA_MAX),
            saturation_factor=toward(self.saturation_factor, 1.0, SATURATION_FACTOR_MAX),
        )


def measure_post_stats(image: np.ndarray, skin_mask: np.ndarray, hair_mask: np.ndarray) -> PostStats:
    """
    Runs the classical pipeline over `image` purely to record what each
    adaptive stage decides. Callers pass a fixed reference window of the frame
    (see this module's header) rather than the whole thing, so the numbers are
    both cheap to obtain and stable no matter where the user later pans.

    The stages must run in order because each one measures what the previous
    produced — the white point depends on the denoised image, contrast on the
    tone-curved one, and so on. That ordering is the reason this can't be
    collapsed into a set of independent measurements of the input.
    """
    noise_severity = _noise_severity(image)
    out = _apply_denoise(image, skin_mask, noise_severity)

    # The tonal half of the chain is measured and applied inside ONE YCrCb
    # round trip, stage by stage on the Y plane — exactly mirroring what
    # _apply_tone_and_contrast will later do in one shot, so the numbers
    # recorded here are the numbers that stage would have measured for
    # itself.
    ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]

    black_push = _adaptive_black_push(y)
    if black_push > 0:
        y = np.clip((y - black_push) * (255.0 / (255.0 - black_push)), 0, 255)

    white_point = _adaptive_white_point_amount(y, skin_mask)
    if white_point > 0:
        y = np.clip(y * (255.0 / (255.0 - min(white_point, 254.0))), 0, 255)

    curve_points = _adaptive_curve_points(y)
    lut = _build_curve_lut(curve_points)
    y = np.interp(y, np.arange(256, dtype=np.float32), lut).astype(np.float32)

    alpha = _adaptive_contrast_alpha(y)
    beta = _adaptive_brightness_beta(y, skin_mask)

    ycrcb[:, :, 0] = np.clip((y - CONTRAST_PIVOT) * alpha + CONTRAST_PIVOT + beta, 0, 255)
    contrasted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    sat_factor = _adaptive_saturation_factor(contrasted)
    saturated = _apply_saturation(contrasted, sat_factor)

    clarity = _adaptive_clarity_strength(saturated, skin_mask)
    clarified = _apply_local_contrast(saturated, skin_mask, clarity)

    hair_strength = _adaptive_hair_strength(clarified, hair_mask)
    haired = _apply_hair_enhance(clarified, hair_mask, hair_strength)

    sharpen = _adaptive_sharpen_amount(haired)
    sharpened = np.clip(haired + (haired - cv2.GaussianBlur(haired, (0, 0), SHARPEN_SIGMA)) * sharpen, 0, 255)

    chroma_severity = _noise_severity(sharpened)

    return PostStats(
        noise_severity=noise_severity,
        black_push=black_push,
        white_point=white_point,
        curve_points=curve_points,
        contrast_alpha=alpha,
        brightness_beta=beta,
        saturation_factor=sat_factor,
        clarity_strength=clarity,
        hair_strength=hair_strength,
        sharpen_amount=sharpen,
        chroma_severity=chroma_severity,
    )


def apply_post_adjustments(image: np.ndarray, skin_mask: np.ndarray, hair_mask: np.ndarray,
                           stats: PostStats) -> np.ndarray:
    """
    The denoise/black-point/white-point/tone-curve/+contrast/+saturation/
    +brightness/+local-contrast/+hair/+sharpness/chroma-denoise chain, applied
    with parameters already decided by measure_post_stats. No white-balance/
    color-cast correction — every stage here only touches luminance/contrast/
    saturation/detail, never shifts the B/G/R color balance.

    Runs entirely in float32 from the first stage to the last, converting to
    uint8 exactly once at the end (with a small dither). Each stage previously
    rounded its output to uint8 before the next stage read it — compounding
    quantization error across ~8 sequential stages, which showed up as visible
    banding in smooth gradients (most noticeable in shadows, which have the
    fewest distinct source levels to begin with). Chaining in float32 removes
    that compounding; the dither masks whatever banding is inherent to the
    black_point/white_point/contrast stretches themselves.
    """
    out = _apply_denoise(image, skin_mask, stats.noise_severity)
    out = _apply_tone_and_contrast(
        out, stats.black_push, stats.white_point, stats.curve_points,
        stats.contrast_alpha, stats.brightness_beta,
    )
    out = _apply_saturation(out, stats.saturation_factor)
    out = _apply_local_contrast(out, skin_mask, stats.clarity_strength)
    out = _apply_hair_enhance(out, hair_mask, stats.hair_strength)

    detail = out - cv2.GaussianBlur(out, (0, 0), SHARPEN_SIGMA)
    out = np.clip(out + detail * stats.sharpen_amount, 0, 255)

    out = _apply_chroma_denoise(out, skin_mask, stats.chroma_severity)

    dither = np.random.default_rng().uniform(
        -DITHER_STRENGTH, DITHER_STRENGTH, size=out.shape,
    ).astype(np.float32)
    return np.clip(out + dither, 0, 255).astype(np.uint8)


# ── Public API ─────────────────────────────────────────────────────────────

def _store_weights(mask: np.ndarray) -> np.ndarray:
    """
    A 0..1 float32 blend weight, packed to uint8 for storage on a RestoredBase.

    Costs a quarter of the memory, which matters because these are held at the
    frame's full pre-crop resolution for as long as the base is cached: at
    float32 the two stored masks together outweighed the restored pixels they
    describe (24MB of mask against 9MB of image on a 2300x1300 base).

    The precision isn't missed. Every one of these starts life as a uint8
    (region_segmenter's masks and feather_mask's output, both divided by 255
    in _subject_masks), so 8 bits is what they were built from, and what they
    weight is a feathered blend. Measured against the float32 masks on the
    same base, holding everything else fixed: no pixel moves by more than ONE
    level of 255, on 0.04% of them.

    Rounded rather than truncated — truncation biases every weight downward by
    half a step, which doubled that mean error (0.0008 levels/pixel against
    0.0004) for nothing.
    """
    return np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8)


def _load_weights(mask: np.ndarray) -> np.ndarray:
    """The inverse of _store_weights — back to the 0..1 float32 the pixel math wants."""
    return mask.astype(np.float32) / 255.0


@dataclass
class RestoredBase:
    """
    Everything about a frame that is computed once and shared by every crop of
    it: the GAN-restored (and protection-blended) pixels, the subject masks,
    and the frozen grading parameters.

    `skin_mask`/`hair_mask` are uint8 (see _store_weights), not the float32
    they're computed and consumed as — render_region converts the slice it
    needs, which is a fraction of the frame.
    """
    restored: np.ndarray
    skin_mask: np.ndarray
    hair_mask: np.ndarray
    stats: PostStats
    intensity: float
    was_enhanced: bool
    backend: str | None

    @property
    def shape(self) -> tuple:
        return self.restored.shape


TOUCHED_DIFF = 1.0  # per-pixel mean abs diff above which the restorer is considered to have changed a pixel


def _log_restoration(diff: np.ndarray, weight: np.ndarray, faces: list, intensity: float,
                     damage: float, eye_fidelity: float) -> None:
    """
    One line per frame saying how much of the GAN's work actually reaches the
    output.

    "The restoration isn't visible" has several quite different causes that
    look identical in the result: the model changed almost nothing, the
    protection blends reverted what it did, or the edit preset's intensity
    diluted the grading afterwards. Each has its own fix and there was no way
    to tell them apart from the image alone — this reports them as numbers
    instead, alongside the measured source damage the protections scale by, so
    an under-restored frame can be traced to whichever of the two it was.

    Note `intensity` scales the GRADING only, so it can no longer suppress
    restoration — it's reported for context, not as a factor on the survival
    figure.
    """
    touched = diff > TOUCHED_DIFF
    share = float(touched.mean())
    if not touched.any():
        log.info("restore: backend=%s ran but changed nothing (faces=%d)", _backend_name, len(faces))
        return

    kept = 1.0 - float(weight[touched].mean())
    log.info(
        "restore: backend=%s faces=%d damage=%.2f (eye fidelity %.2f) | "
        "changed %.1f%% of the frame by %.1f levels | %.0f%% survives protection | "
        "grading intensity %.2f",
        _backend_name, len(faces), damage, eye_fidelity,
        share * 100, float(diff[touched].mean()), kept * 100, intensity,
    )


def build_base(image: np.ndarray, ref_rect: tuple[int, int, int, int] | None = None,
               fidelity: float = FIDELITY_WEIGHT, intensity: float = 1.0) -> RestoredBase:
    """
    The once-per-frame half of restoration (see this module's header).

    `fidelity` (0..1) controls how strongly the eye region is pulled back toward
    the original around detected eyes — higher fidelity means eyes stay closer to
    the source (safer, less deformation risk), lower fidelity leaves GFPGAN's raw
    output untouched there. On top of that, any eye pixel the restorer changed
    drastically (the actual signature of a deformation, regardless of cause —
    glasses being the most common trigger) gets pulled back further, independent
    of `fidelity`. The same drastic-change pullback also applies around the
    face (see _hair_deviation_protection) — hair (long, wavy, or otherwise
    detail-dense) is comparatively under-represented in what these restorers
    were trained on and often comes back blurred or melted. Since protection
    can only preserve hair (never improve it) and the restorer's crop never
    reaches long hair at all, the post pass also runs a hair-scoped texture
    enhancement (see _apply_hair_enhance) so hair keeps up with the restored
    face instead of staying source-soft next to it. Two more unscoped checks
    cover everywhere else on the face, catching the restorer's own alignment
    paste-back seam — a visible ghosting line where the restored crop was
    warped back slightly off from the original, which can show up on a
    cheek, temple, or jaw, nowhere eye/hair protection would ever look:
    _seam_deviation_protection for a seam with a real color jump, and
    _seam_edge_protection for one that's too smooth/subtle to trip a
    magnitude check but still a line that wasn't in the source.

    `intensity` (0..1) is the frontend's editing-preset dial, and it applies
    to the GRADING ONLY — see PostStats.at_intensity. Restoration (the GAN
    pass here, plus denoise/clarity/hair/sharpening downstream) runs at full
    strength at every intensity including 0, because a soft or noisy frame
    needs the same reconstruction no matter how punchy the user wants the
    picture; only tone/color is a preference. Intensity 0 therefore means
    "restore, don't grade", NOT "do nothing" — this function used to
    short-circuit and return the untouched image there, which made the
    lowest preset a pure upscale+crop with no deblurring at all, exactly the
    thing the preset dial was reworked to stop doing.

    `ref_rect` is the (x1, y1, x2, y2) window the grading parameters are
    measured on. Defaults to the whole image when omitted.
    """
    faces = detect_faces(image)
    skin_mask, hair_mask, hair_protect_mask = _subject_masks(image, faces)

    out = _run_restorer(image)
    was_enhanced = out is not None and out.shape == image.shape
    if not was_enhanced:
        out = image

    if was_enhanced:
        src_f = image.astype(np.float32)
        out_f = out.astype(np.float32)
        # One shared difference map: eye, seam and hair protection all reduce
        # to "how far did this pixel move", and each used to recompute it.
        diff = np.abs(src_f - out_f).mean(axis=2)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Where the restorer actually did something. Both seam checks take
        # their scale from this region rather than from a fixed constant, so
        # a face that needed heavy reconstruction isn't flagged for having
        # received it (see the SEAM_SCALE_* comment).
        footprint = diff > TOUCHED_DIFF

        # A blurred source is not worth staying close to — see the FOCUS_*
        # block. This is the one protection that fires on position rather
        # than on evidence of damage (it pulls the eyes back by a fixed
        # amount whatever the restorer did), so it's the one that has to be
        # told how much the source is worth.
        damage = _face_damage(image, faces)
        eye_fidelity = np.clip(fidelity, 0.0, 1.0) * (
            1.0 - damage * (1.0 - EYE_FIDELITY_DAMAGE_FLOOR)
        )

        eye_mask = _eye_protection_mask(image, faces)
        weight = np.maximum.reduce([
            eye_fidelity * eye_mask,
            _eye_deviation_protection(diff, eye_mask),
            _hair_deviation_protection(diff, gray_image, gray_out, hair_protect_mask),
            _seam_deviation_protection(diff, footprint),
            _seam_edge_protection(gray_image, gray_out, footprint,
                                  _paste_border_band(diff, footprint, faces)),
        ])[:, :, np.newaxis]
        restored = np.clip(out_f * (1.0 - weight) + src_f * weight, 0, 255).astype(np.uint8)
        _log_restoration(diff, weight[:, :, 0], faces, intensity, damage, eye_fidelity)
    else:
        log.info("restore: GAN did not run (backend=%s, faces detected=%d)", _backend_name, len(faces))
        restored = image

    if ref_rect is None:
        ref = restored
        ref_skin, ref_hair = skin_mask, hair_mask
    else:
        x1, y1, x2, y2 = ref_rect
        ref = restored[y1:y2, x1:x2]
        ref_skin, ref_hair = skin_mask[y1:y2, x1:x2], hair_mask[y1:y2, x1:x2]
        if ref.size == 0:
            ref, ref_skin, ref_hair = restored, skin_mask, hair_mask

    # Measured on the float32 masks this function computed, so the grading
    # parameters are bit-for-bit what they were before they started being
    # stored packed.
    stats = measure_post_stats(ref.astype(np.float32), ref_skin, ref_hair)

    return RestoredBase(
        restored=restored,
        skin_mask=_store_weights(skin_mask), hair_mask=_store_weights(hair_mask),
        stats=stats, intensity=intensity,
        was_enhanced=was_enhanced, backend=_backend_name if was_enhanced else None,
    )


def render_region(base: RestoredBase, rect: tuple[int, int, int, int], zoom: float = 1.0) -> np.ndarray:
    """
    The finished (graded, intensity-blended) pixels for one region of a frame.

    `rect` is (x1, y1, x2, y2) in the base's own coordinates; `zoom` resizes
    the extracted region afterwards. Only this region is ever pushed through
    the pixel pipeline — the frozen stats (see PostStats) are what guarantee
    it comes out identical to any other region of the same frame.
    """
    x1, y1, x2, y2 = rect
    region = base.restored[y1:y2, x1:x2]
    if region.size == 0:
        return region

    if zoom != 1.0:
        rh, rw = region.shape[:2]
        size = (max(1, int(round(rw * zoom))), max(1, int(round(rh * zoom))))
        interp = cv2.INTER_LANCZOS4 if zoom > 1.0 else cv2.INTER_AREA
        region = cv2.resize(region, size, interpolation=interp)

    def _mask(m):
        # Sliced and resized while still uint8 (cheaper, and the resize is the
        # same bilinear either way), then unpacked — so only the window's own
        # pixels are ever expanded to float32, never the whole frame's.
        sub = m[y1:y2, x1:x2]
        if zoom != 1.0:
            sub = cv2.resize(sub, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_LINEAR)
        return _load_weights(sub)

    # The preset scales the grading only — restoration runs at full strength
    # whatever the preset says. See PostStats.at_intensity.
    return apply_post_adjustments(
        region.astype(np.float32), _mask(base.skin_mask), _mask(base.hair_mask),
        base.stats.at_intensity(base.intensity),
    )
