"""
framing.py — What "a good frame" means, as a set of numbers a channel can
change.

Everything the selector and the reframe geometry used to treat as universal
truth — the face fills a sixth of the canvas, it sits on the right-hand
rule-of-thirds line, an eighth of the width has to stay clear to its left — was
in fact ONE channel's idea of a thumbnail, written into the pipeline. It is a
good idea for a channel whose thumbnail is the photograph with a title beside
it, which every channel here was until now.

Laugh Society 1 is not that. Its thumbnail is a fixed backdrop with the person
cut out of the photo and placed on it, so the photo behind the subject is
thrown away before anyone sees it. Two of those three numbers then mean the
opposite of what they say: clear space beside the face is worth nothing (there
is no text going there, and no photo left to put it on), and a face filling a
sixth of the canvas is a head-and-shoulders close-up when what the cutout wants
is the person from about the waist up.

So the numbers moved here, where a channel pack can state its own and the rest
of the backend goes on reading one thing. See `framing` in
frontend/content/channels/laugh-society-1/channel.json for a pack that does,
and content/README.md for the key.

The canvas the frames are rendered ONTO is here for the same reason. It was
1280x720 written into reframe_engine as a constant, which is the YouTube
thumbnail and was every channel this app had. The Snapchat pack's frames are
540x960, and every number that decides a crop — the cover scale, the anchor,
the overpan margin, the autofill band — is expressed against that canvas. A
second constant somewhere else would have to agree with this one forever; one
profile per run cannot disagree with itself.

## One profile per run, not per frame

The frontend's channel selection is per FRAME (see the note at the top of
js/channels.js). This is not, and cannot be: frame selection happens once, over
the whole video, before a single thumbnail exists to have a channel. So the
channel picked BEFORE processing is what a run is framed for, and it is the one
the frames are born branded with; changing a later frame's brand changes how it
is drawn, not which moment of the video it came from.

That is also why this is module state rather than a parameter threaded through
compute_framing: the same profile has to answer for the scoring pass, the
initial crops, and every /vary-frame and /upload-frame the user makes
afterwards, and a run is one video for one channel.
"""

import logging
from dataclasses import dataclass, field

log = logging.getLogger("uvicorn.error")

# The defaults, which are the numbers the pipeline carried inline before this
# module existed. A channel that says nothing about framing gets exactly the
# behaviour it had.
DEFAULT_FACE_AREA_RATIO = 1 / 6   # share of the canvas the face fills when alone
DEFAULT_ANCHOR_X = 0.667          # where the face is placed, as a fraction of the canvas
DEFAULT_ANCHOR_Y = 0.333
DEFAULT_MIN_TEXT_SPACE = 0.12     # width that must stay clear to the face's left, or the framing is unusable
DEFAULT_CANVAS_W = 1280           # the canvas every crop is rendered onto...
DEFAULT_CANVAS_H = 720            # ...which is the YouTube thumbnail unless a pack says otherwise

# Bounds, applied to whatever a pack sends. This is data arriving over HTTP
# from a file a user can edit by hand, and every one of these numbers divides
# or multiplies something: a zero face ratio is a division by zero inside
# compute_scale, and an anchor outside the canvas is a crop window that can
# never be satisfied. Clamped rather than rejected — a pack with a silly number
# in it should cost that channel its intended framing, not the run.
FACE_AREA_RATIO_RANGE = (1 / 400, 1 / 2)
ANCHOR_RANGE = (0.0, 1.0)
MIN_TEXT_SPACE_RANGE = (0.0, 0.45)
# The canvas is allocated, sliced and inpainted at this size on every render,
# so both ends matter: below the low end the gradient ramp and the autofill
# edge band round down to nothing, and above the high end a single frame's
# restoration pass is large enough to exhaust VRAM on its own.
CANVAS_SIDE_RANGE = (160, 4096)


def _clamp(value, low, high, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:   # NaN, which compares false against everything
        return fallback
    return max(low, min(number, high))


def _clamp_int(value, low, high, fallback):
    """`_clamp` for a pixel count — same treatment, rounded to a whole pixel."""
    return int(round(_clamp(value, low, high, fallback)))


@dataclass(frozen=True)
class FramingProfile:
    """
    How one channel wants its frames chosen and cropped.

    `weights` is a PARTIAL override of quality_scorer's own weights, not a
    replacement: a channel says which of the six criteria it does not care
    about (or cares more about), and the rest keep their relative importance,
    renormalised. Stating the whole set would mean every channel restating five
    numbers it has no opinion on, and a change to the shared scoring would then
    reach none of them.
    """
    # Whether a face that would fill much more of the CROP than this profile
    # intends is penalised for it (see quality_scorer.face_fill_penalty).
    #
    # True everywhere, because for a channel that composites the photograph an
    # inherently close-up shot really is a worse thumbnail: it blows past the
    # target face size with no way to zoom out, and the penalty is what lets
    # the selector prefer a better-composed alternative.
    #
    # False for a channel that never shows the crop. A cutout channel cuts its
    # subject out of the WHOLE photo (see api._cutout_png, which renders the
    # full frame rather than the crop window) and rescales them onto a backdrop
    # of its own, so a close-up is not a composition failure there — it is more
    # pixels on the only thing that survives. Penalising it costs exactly the
    # frames such a channel most wants.
    #
    # Measured on a stage set with two camera positions: with the penalty on,
    # the sharpest frames of two of the five character clusters scored 0.000
    # and the grid came out with 7 of 20 thumbnails below the sharpness floor.
    # With it off, 2 of 20, and the median sharpness rose from 49 to 59.
    face_fill: bool = True

    face_area_ratio: float = DEFAULT_FACE_AREA_RATIO
    anchor_x: float = DEFAULT_ANCHOR_X
    anchor_y: float = DEFAULT_ANCHOR_Y
    min_text_space: float = DEFAULT_MIN_TEXT_SPACE
    weights: dict = field(default_factory=dict)
    # The size of the finished frame. Read through reframe_engine.target_w /
    # target_h, which is what the whole backend asks — nothing outside this
    # module names these two fields.
    canvas_w: int = DEFAULT_CANVAS_W
    canvas_h: int = DEFAULT_CANVAS_H

    @classmethod
    def from_request(cls, spec: dict | None) -> "FramingProfile":
        """
        A profile from what the frontend sent, with every number clamped and
        anything unrecognised ignored.
        """
        if not isinstance(spec, dict):
            return cls()
        weights = spec.get("weights")
        return cls(
            face_area_ratio=_clamp(spec.get("face_area_ratio"), *FACE_AREA_RATIO_RANGE,
                                   DEFAULT_FACE_AREA_RATIO),
            anchor_x=_clamp(spec.get("anchor_x"), *ANCHOR_RANGE, DEFAULT_ANCHOR_X),
            anchor_y=_clamp(spec.get("anchor_y"), *ANCHOR_RANGE, DEFAULT_ANCHOR_Y),
            min_text_space=_clamp(spec.get("min_text_space"), *MIN_TEXT_SPACE_RANGE,
                                  DEFAULT_MIN_TEXT_SPACE),
            # Negative weights would let one criterion cancel another out, and
            # the normalisation below assumes a non-negative total.
            weights={str(k): max(0.0, float(v)) for k, v in (weights or {}).items()
                     if isinstance(v, (int, float))},
            # Anything but an explicit false leaves the penalty on, so a pack
            # with a typo in this key keeps the behaviour every channel has had.
            face_fill=spec.get("face_fill") is not False,
            canvas_w=_clamp_int(spec.get("canvas_w"), *CANVAS_SIDE_RANGE, DEFAULT_CANVAS_W),
            canvas_h=_clamp_int(spec.get("canvas_h"), *CANVAS_SIDE_RANGE, DEFAULT_CANVAS_H),
        )

    def is_default(self) -> bool:
        return self == FramingProfile()


_active = FramingProfile()


def active() -> FramingProfile:
    """The profile every framing decision in this process is currently made under."""
    return _active


def set_active(profile: FramingProfile) -> None:
    global _active
    _active = profile
    log.info("framing: %dx%d canvas, face fills 1/%.0f of it, anchored at (%.3f, %.3f), "
             "clear space required %.0f%%%s",
             profile.canvas_w, profile.canvas_h,
             1 / profile.face_area_ratio, profile.anchor_x, profile.anchor_y,
             profile.min_text_space * 100,
             f", weights {profile.weights}" if profile.weights else "")


def reset() -> None:
    """Back to the shipped numbers — what a run that named no channel is framed by."""
    set_active(FramingProfile())
