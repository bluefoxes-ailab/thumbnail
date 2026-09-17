"""
frame_grab.py — Picking N frames out of a video when nobody is the subject.

The rest of the pipeline chooses thumbnails: it finds faces, works out who is
in the video, and hands back twenty moments of the most interesting people in
it. That is the right answer for a thumbnail and the wrong one for a frame
fetcher, where the ask is "give me forty usable stills, spread across this
video, in the order they happen". Half the frames will have nobody in them at
all, and a face-shaped selector answers that request with nothing (the run
dies at "no suitable frames found — no faces detected").

So this module is the other selector. It shares the extraction and the whole
rendering path with the thumbnail one and differs in exactly two ways:

  * what a good frame IS. There is no face to measure, so quality is judged on
    the picture as a whole — is it in focus, is it smeared by motion, is there
    anything in it at all.
  * how the survivors are SPREAD. Score order would happily return forty
    frames of the same static shot, so the video is cut into as many equal
    slices as there are frames wanted and each slice contributes its best.

## The second half: what a measurement cannot see

Focus, motion and contrast are properties of the pixels, and a frame can be
perfect on all three and still be one nobody would use — because the person in
it has their eyes shut, or because it is the same shot as the frame beside it
with nothing to tell them apart. Neither is visible to a per-frame measurement:
the first needs to know there is a face and what it is doing, and the second is
not about the frame at all but about its NEIGHBOURS.

So selection runs in two halves. The measurement picks a spread of usable
frames, and then two passes go back over that set with the candidates that were
not chosen still to hand, and substitute:

  spread_scenes  two adjacent slots showing the same scene are one slot's worth
                 of information in two. The later one is swapped for the
                 nearest following moment that is a different scene.
  open_eyes      a slot with a face whose eyes are shut is swapped for the
                 nearest moment of the SAME scene where they are open.

In that order, and the order matters: the scene pass moves a slot to a
different moment of the video, and the eye pass then fixes the expression
wherever the scene pass left it. Running them the other way round would fix an
expression on a frame about to be thrown away.

Both are substitutions, never insertions or deletions: the grid keeps exactly
the frames it was asked for, and a pass that cannot find a better candidate
leaves the one it has.

## Why the thresholds are relative as well as absolute

Laplacian variance is not comparable between videos. A clean 4K interview and
a compressed, grainy phone clip sit an order of magnitude apart while both
being perfectly in focus, so a single absolute floor either passes everything
on the first or rejects everything on the second.

Both floors are therefore applied: an absolute one, which catches a video that
is soft from end to end, and one expressed as a fraction of this video's own
median, which catches the frames that are soft FOR THIS VIDEO. A frame has to
clear both to be called sharp.

## Why measurements are taken at a fixed size

Laplacian variance and gradient variance both scale with resolution — the same
shot measures several times sharper at 4K than at 720p purely because there
are more, smaller pixel differences. Everything here is measured on the frame
downscaled to a fixed analysis size, so one set of numbers means the same
thing whatever the source was shot on.
"""

import os
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from image_utils import imread, color_hist, hist_similarity
from face_detector import detect_faces, eyes_open
from frame_extractor import (
    MIN_INTERVAL, MAX_INTERVAL, get_video_duration, extract_window, full_frame_path,
)
from state import TEMP_DIR

log = logging.getLogger("uvicorn.error")

# Long side every frame is measured at. See the module header: the thresholds
# below are meaningless without it.
ANALYSIS_SIDE = 640

# Focus, as the variance of the Laplacian. STRICT is what a frame has to clear
# to be picked in the ordinary way; FLOOR is the point below which a frame is
# not worth handing over at all — no amount of restoration puts back detail
# that was never recorded, which is exactly what the request asks to avoid.
FOCUS_STRICT = 60.0
FOCUS_FLOOR = 25.0
# ...and the same two as a fraction of this video's own median focus.
FOCUS_STRICT_RELATIVE = 0.45
FOCUS_FLOOR_RELATIVE = 0.22
# Where the focus sub-score stops improving. Past this a frame is simply
# sharp, and the difference between "sharp" and "very sharp" should not
# outweigh a frame being in the right part of the video.
FOCUS_SATURATE = 400.0

# Motion blur, as the ratio between the weaker and the stronger directional
# gradient variance — a pan or a fast subject smears one axis and leaves the
# other, so the ratio collapses. Deliberately lenient as a reject: a picket
# fence, a venetian blind or a skyline of verticals is directionally lopsided
# while being perfectly sharp, so this mostly RANKS and only rejects the
# frames that are lopsided and soft together.
MOTION_STRICT = 0.20
MOTION_FLOOR = 0.08

# Contrast, as the standard deviation of luma. This is the fade-to-black, the
# white flash and the single-colour title card — frames that are technically
# sharp, carry no picture, and are the most obviously useless thing a frame
# fetcher can hand back.
CONTRAST_STRICT = 12.0
CONTRAST_FLOOR = 6.0

# How the three combine into the number that ranks candidates inside one slice
# of the video. Focus dominates because it is the one that decides whether a
# frame is usable at all; contrast is a tiebreak more than a criterion.
GRAB_WEIGHTS = {"focus": 0.60, "motion": 0.25, "contrast": 0.15}

# Contrast at which the contrast sub-score saturates. Well above the reject
# threshold — this is "a normal picture", not "an extraordinary one".
CONTRAST_SATURATE = 60.0

# Same reasoning and the same cap as frame_selector's pool: OpenCV releases
# the GIL in imread/Laplacian/Sobel, and each worker holds one decoded frame.
MAX_SCORING_WORKERS = 8

# How much of a slice's width has to separate two frames chosen out of the
# same neighbourhood, once the one-per-slice pass starts backfilling. Without
# it a video whose only sharp footage is one long static shot returns forty
# consecutive samples of it — technically forty frames, in practice one.
BACKFILL_MIN_GAP_FRAC = 0.5

# ── Telling one scene from another ────────────────────────────────────────
# Correlation between two frames' whole-frame colour histograms above which
# they are called the same scene. Whole-frame, not the face region the
# thumbnail selector compares: there is often no face here, and what is being
# asked is "is this the same shot", which is a question about the whole
# picture.
#
# 0.90 is deliberately not near 1.0. Two samples a second apart in one
# continuous shot are never identical — people move, the camera drifts — and a
# threshold that demanded they be would call every pair distinct and the pass
# would do nothing. Measured on test footage: samples within one shot ran
# 0.93-1.00, samples either side of a cut 0.10-0.72.
SCENE_SAME_CORRELATION = 0.90

# The histogram is taken at this size. Colour distribution survives a heavy
# downscale intact, and it makes the signature cheap enough to compute for
# every candidate rather than only for the chosen ones — which is what lets
# the substitution passes reach for a frame nobody had measured yet.
SIGNATURE_SIDE = 128

# ── Eyes ──────────────────────────────────────────────────────────────────
# How far either side of a slot to look for the same moment with the eyes
# open, in seconds. A blink is a fifth of a second and a sentence is a few, so
# this is generous; what stops it wandering is not the radius but the
# same-scene test, which every candidate has to pass.
EYE_SEARCH_SECONDS = 3.0
# ...and how many candidates to actually open and scan before giving up.
# Nearest-first, so the ones examined are the ones most likely to be the same
# moment; the cap is there because face detection runs at native resolution
# (see face_detector) and a slot in a long static shot could otherwise have
# dozens of neighbours to work through.
EYE_SEARCH_MAX_TRIES = 8

# When the sample has no open-eyed neighbour to offer, a window of the video
# either side of the slot is extracted fresh, this densely, and searched
# instead.
#
# It is needed because of the arithmetic of a blink. The whole-video sample is
# spread to cover the running time — a second or more between frames on
# anything long — and a blink lasts about a fifth of one. A slot that landed
# on a blink therefore has, on average, NO other sample of that same blink to
# be moved to: its neighbours in the sample are seconds away, which is a
# different moment of the shot and often a different shot. Sampling at
# EYE_WINDOW_INTERVAL puts a dozen frames inside the blink's own neighbourhood,
# where the eyes are open in nearly all of them.
#
# Same mechanism /vary-frame uses to find another take of one shot, and for the
# same reason — see vary.dense_pool.
EYE_WINDOW_SECONDS = 1.2
EYE_WINDOW_INTERVAL = 0.12
# ...and how many of that window's frames to actually open. Nearest-first, so
# these are the closest moments to the one the slot already holds.
EYE_WINDOW_MAX_TRIES = 12
# ...and how sharp a frame out of that window has to be, relative to the slot
# it would replace. Not an absolute floor: the window is inside one shot, so
# the honest comparison is against the frame already in hand, and a blink
# fixed at the cost of a visibly softer picture is not a fix.
DENSE_FOCUS_FLOOR_FRAC = 0.75

# What a frame's people are doing, as far as this can tell.
EYES_NO_FACE = "no-face"   # nobody in shot — nothing to check, and most frames
EYES_OPEN = "open"         # a face, and two plausible open eyes on it
EYES_SHUT = "shut"         # a face, and no such pair — blinking, squinting or turned away

# How many candidates to extract per frame the user asked for. The selection
# below can only choose from what was sampled, and it throws away everything
# soft, so the sample has to be several times the answer. Three is what leaves
# a slice with something to offer after the blurred and the black frames in it
# are gone.
CANDIDATES_PER_FRAME = 3


# Furthest apart this selection will ever sample. The thumbnail pipeline caps
# at MAX_INTERVAL (1s) because it has to catch a character who is on screen for
# three seconds; this is not looking for anybody, it is looking for a spread,
# and sampling an hour-long video every second to hand back forty frames is
# 3600 extractions to throw 3560 of them away.
CAPTURE_MAX_INTERVAL = 15.0


def sample_interval(video_path: str, count: int) -> float:
    """
    How densely to extract for a capture of `count` frames.

    Enough candidates that every slice of the video has something to offer
    after the blurred and the blank frames in it are rejected, and no more
    than that — see CANDIDATES_PER_FRAME and CAPTURE_MAX_INTERVAL. A video
    whose duration cannot be read falls back to the thumbnail pipeline's own
    densest sampling, which is never too sparse, only slower than it needed to
    be.
    """
    duration = get_video_duration(video_path)
    if not duration or duration <= 0:
        return MAX_INTERVAL
    wanted = duration / max(1, count * CANDIDATES_PER_FRAME)
    return max(MIN_INTERVAL, min(CAPTURE_MAX_INTERVAL, wanted))


@dataclass
class Grab:
    """One candidate frame, measured."""
    path: str
    timestamp: float     # seconds into the source video
    focus: float
    motion: float
    contrast: float
    score: float
    sharp: bool          # cleared the strict floors, not just the relaxed ones
    # Whole-frame colour histogram, for telling this frame's scene from
    # another's (see SCENE_SAME_CORRELATION). Computed for every candidate
    # while the frame is already decoded, because the substitution passes draw
    # from candidates that were never chosen and would otherwise have to go
    # back to disk for each one they consider.
    signature: np.ndarray | None = None


def _worker_count() -> int:
    return max(1, min(MAX_SCORING_WORKERS, (os.cpu_count() or 4)))


def _analysis_gray(image: np.ndarray) -> np.ndarray:
    """The frame as grayscale at ANALYSIS_SIDE, which is where every number below is measured."""
    h, w = image.shape[:2]
    scale = ANALYSIS_SIDE / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def measure(image: np.ndarray) -> tuple[float, float, float]:
    """
    (focus, motion, contrast) for a whole frame.

    `motion` is the statistic quality_scorer.motion_blur_penalty computes
    around a face, applied to the entire picture instead — there being no face
    here to centre it on, and the smear of a camera move being a property of
    the frame rather than of any one region of it.
    """
    gray = _analysis_gray(image)

    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x, var_y = gx.var(), gy.var()
    motion = 0.0 if var_x + var_y < 1e-6 else float(min(var_x, var_y) / max(var_x, var_y))

    return focus, motion, contrast


def _grab_score(focus: float, motion: float, contrast: float) -> float:
    return (
        min(focus / FOCUS_SATURATE, 1.0) * GRAB_WEIGHTS["focus"]
        + motion * GRAB_WEIGHTS["motion"]
        + min(contrast / CONTRAST_SATURATE, 1.0) * GRAB_WEIGHTS["contrast"]
    )


def scene_signature(image: np.ndarray) -> np.ndarray:
    """This frame's colour distribution, for comparing one shot against another."""
    h, w = image.shape[:2]
    scale = SIGNATURE_SIDE / max(1, max(h, w))
    small = (cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else image)
    return color_hist(small)


def same_scene(a: Grab, b: Grab) -> bool:
    """Whether two candidates are the same shot of the same thing."""
    if a.signature is None or b.signature is None:
        return False
    return hist_similarity(a.signature, b.signature) >= SCENE_SAME_CORRELATION


def eye_state(image: np.ndarray) -> str:
    """
    EYES_NO_FACE, EYES_OPEN or EYES_SHUT for a whole frame.

    The largest face, because on a still the subject is the person nearest the
    camera — the same thing every other framing decision in the app assumes.

    "Shut" is the honest name for what the third answer means, but it covers
    more than a blink: face_detector.eyes_open also refuses a profile, where
    only one eye is visible, and a face too small or too soft for the cascade
    to resolve. That is the right reading here in every case. This verdict is
    only ever used to go LOOKING for something better, and a frame where the
    eyes cannot be seen to be open is one a better alternative would improve.
    """
    faces = detect_faces(image)
    if not faces:
        return EYES_NO_FACE
    return EYES_OPEN if eyes_open(image, max(faces, key=lambda f: f[2] * f[3])) else EYES_SHUT


def _measure_one(item: tuple[str, float]) -> Grab | None:
    path, timestamp = item
    image = imread(path)
    if image is None:
        return None
    focus, motion, contrast = measure(image)
    return Grab(path=path, timestamp=timestamp, focus=focus, motion=motion, contrast=contrast,
                score=_grab_score(focus, motion, contrast), sharp=False,
                signature=scene_signature(image))


def measure_frames(paths: list[str], interval: float) -> list[Grab]:
    """
    Every extracted frame, measured, in the order it appears in the video.

    OpenCV's internal parallelism is switched off for the duration for the
    reason frame_selector.score_frames switches it off: the outer pool already
    keeps every core busy, and stacking the two makes eight workers fight over
    sixteen threads each.
    """
    items = [(path, index * interval) for index, path in enumerate(paths)]

    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=_worker_count()) as pool:
            measured = list(pool.map(_measure_one, items))
    finally:
        cv2.setNumThreads(previous_threads)

    return [g for g in measured if g is not None]


def _apply_floors(grabs: list[Grab]) -> list[Grab]:
    """
    Marks each candidate sharp / usable / unusable, and drops the unusable.

    Both focus thresholds are the greater of an absolute number and a fraction
    of this video's median — see the module header for why neither works
    alone.
    """
    if not grabs:
        return []

    median_focus = float(np.median([g.focus for g in grabs]))
    strict_focus = max(FOCUS_STRICT, median_focus * FOCUS_STRICT_RELATIVE)
    floor_focus = max(FOCUS_FLOOR, median_focus * FOCUS_FLOOR_RELATIVE)

    kept = []
    for g in grabs:
        if g.focus < floor_focus or g.motion < MOTION_FLOOR or g.contrast < CONTRAST_FLOOR:
            continue
        g.sharp = (g.focus >= strict_focus and g.motion >= MOTION_STRICT
                   and g.contrast >= CONTRAST_STRICT)
        kept.append(g)

    log.info("capture: %d of %d frames usable (%d of them sharp); "
             "median focus %.0f, strict floor %.0f, reject floor %.0f",
             len(kept), len(grabs), sum(1 for g in kept if g.sharp),
             median_focus, strict_focus, floor_focus)
    return kept


def _best_per_slice(usable: list[Grab], count: int, span: float) -> tuple[list[Grab], list[Grab]]:
    """
    The best sharp frame from each of `count` equal slices of the video, and
    everything not taken.

    One per slice rather than the top `count` by score, because score order on
    a video with one well-lit static shot in it returns that shot forty times.
    A slice with nothing sharp in it contributes nothing here and is made up
    for by the backfill.
    """
    width = span / count if count else span
    by_slice: dict[int, Grab] = {}
    for g in usable:
        if not g.sharp:
            continue
        index = min(count - 1, int(g.timestamp / width)) if width > 0 else 0
        best = by_slice.get(index)
        if best is None or g.score > best.score:
            by_slice[index] = g

    chosen = list(by_slice.values())
    taken = {id(g) for g in chosen}
    return chosen, [g for g in usable if id(g) not in taken]


def _backfill(chosen: list[Grab], leftovers: list[Grab], count: int, span: float) -> list[Grab]:
    """
    Tops the selection up to `count` from what the per-slice pass did not take,
    best first, keeping the frames apart.

    Sharp candidates are exhausted before the merely usable ones are touched,
    so a run only reaches into the soft frames when the alternative is handing
    back fewer frames than were asked for. The spacing rule is dropped once
    nothing satisfies it, for the same reason: a video whose usable footage is
    all in one place should still fill the grid.
    """
    gap = (span / count) * BACKFILL_MIN_GAP_FRAC if count else 0.0

    for pool in ([g for g in leftovers if g.sharp], [g for g in leftovers if not g.sharp]):
        for enforce_gap in (True, False):
            for g in sorted(pool, key=lambda c: c.score, reverse=True):
                if len(chosen) >= count:
                    return chosen
                if g in chosen:
                    continue
                if enforce_gap and any(abs(g.timestamp - c.timestamp) < gap for c in chosen):
                    continue
                chosen.append(g)
    return chosen


def spread_scenes(chosen: list[Grab], candidates: list[Grab]) -> int:
    """
    Replaces any slot showing the same scene as the slot before it with the
    nearest FOLLOWING moment that shows a different one. Returns how many were
    swapped; `chosen` must be in time order and is modified in place.

    Two adjacent slots of the same shot are one slot's worth of information
    printed twice, and on a video with a long static stretch in it the
    one-per-slice rule produces plenty of them: the slices are equal lengths of
    TIME, and a video does not hand out its scene changes evenly.

    The replacement is looked for AFTER the offending slot and BEFORE the next
    one, which is what "the nearest following scene" means here and is also
    what keeps the grid in order without re-sorting it: the slot stays in its
    own stretch of the video, it just moves to the first moment in that stretch
    with something new in it. A slot with nothing new after it keeps what it
    has — a video that really is one continuous shot has no better answer, and
    inventing one by reaching further would only take another slot's frame.

    The candidate also has to differ from the slot AFTER it, not just the one
    before. Without that, a slot sandwiched between two halves of the same shot
    can be "fixed" into the very scene its other neighbour already shows,
    which trades one duplicate for another.
    """
    taken = {g.path for g in chosen}
    swapped = 0

    for i in range(1, len(chosen)):
        previous, current = chosen[i - 1], chosen[i]
        if not same_scene(previous, current):
            continue

        following = chosen[i + 1] if i + 1 < len(chosen) else None
        limit = following.timestamp if following else float("inf")

        replacement = next(
            (c for c in candidates
             if c.path not in taken
             and current.timestamp < c.timestamp < limit
             and not same_scene(c, previous)
             and (following is None or not same_scene(c, following))),
            None,
        )
        if replacement is None:
            continue

        taken.discard(current.path)
        taken.add(replacement.path)
        chosen[i] = replacement
        swapped += 1

    return swapped


def _first_open_eyed(slot: Grab, candidates: list[Grab], taken: set) -> Grab | None:
    """
    The nearest moment of `slot`'s own scene, within EYE_SEARCH_SECONDS, whose
    subject has their eyes open. None when there isn't one.

    Nearest first and same scene only, because this is meant to fix an
    expression and not to change the shot: the frame that comes back has to be
    the same moment of the same thing, or the slot has silently become a
    different picture. The scene test is what enforces that; the radius only
    bounds how far it is worth looking.
    """
    near = sorted(
        (c for c in candidates
         if c.path not in taken
         and abs(c.timestamp - slot.timestamp) <= EYE_SEARCH_SECONDS
         and same_scene(c, slot)),
        key=lambda c: abs(c.timestamp - slot.timestamp),
    )
    for candidate in near[:EYE_SEARCH_MAX_TRIES]:
        image = imread(candidate.path)
        if image is None:
            continue
        if eye_state(image) == EYES_OPEN:
            return candidate
    return None


def _eye_state_of(grab: Grab) -> str:
    """eye_state for a candidate on disk. A frame that will not open is nobody's problem here."""
    image = imread(grab.path)
    return EYES_NO_FACE if image is None else eye_state(image)


def _dense_open_eyed(slot: Grab, video_path: str, index: int) -> Grab | None:
    """
    The nearest moment to `slot`, out of a freshly-extracted dense window of
    the video around it, whose subject has their eyes open. None when the
    window has none either.

    The same-scene test still applies, and matters more here than in the
    sparse search: a window of a second either side can still cross a cut, and
    a slot that came back as a different shot has stopped being the moment the
    selection chose.
    """
    if not video_path:
        return None

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(TEMP_DIR, f"eyes_{video_id}_{index}")
    try:
        window = extract_window(video_path, slot.timestamp, EYE_WINDOW_SECONDS,
                                EYE_WINDOW_INTERVAL, out_dir)
    except Exception as e:
        log.warning("capture: could not extract an eye-search window at %s (%s)",
                    timecode(slot.timestamp), e)
        return None

    # extract_window hands back the overlay-cropped frames; a capture slot is
    # made of the uncropped ones, so every path is taken to its full sibling
    # (see api's capture path for why).
    start = max(0.0, slot.timestamp - EYE_WINDOW_SECONDS)
    paths = []
    for i, cropped in enumerate(window):
        full = full_frame_path(cropped)
        paths.append((full if os.path.exists(full) else cropped,
                      start + i * EYE_WINDOW_INTERVAL))
    paths.sort(key=lambda pt: abs(pt[1] - slot.timestamp))

    for path, timestamp in paths[:EYE_WINDOW_MAX_TRIES]:
        image = imread(path)
        if image is None:
            continue
        focus, motion, contrast = measure(image)
        candidate = Grab(path=path, timestamp=timestamp, focus=focus, motion=motion,
                         contrast=contrast, score=_grab_score(focus, motion, contrast),
                         sharp=False, signature=scene_signature(image))
        # Everything the slot it replaces had to clear: the same shot, and not
        # a smeared or out-of-focus frame. An open-eyed blur is not an
        # improvement on a sharp blink.
        if not same_scene(candidate, slot):
            continue
        if candidate.focus < slot.focus * DENSE_FOCUS_FLOOR_FRAC:
            continue
        if eye_state(image) == EYES_OPEN:
            return candidate
    return None


def open_eyes(chosen: list[Grab], candidates: list[Grab], video_path: str = "") -> int:
    """
    Replaces any slot whose subject has their eyes shut with the nearest
    moment of the same scene where they are open. Returns how many were
    swapped; `chosen` is modified in place.

    Most frames never reach the search at all: the verdict on the slots
    themselves is taken in parallel, and the overwhelming majority come back
    EYES_NO_FACE (a frame fetch is mostly not portraits) or EYES_OPEN. Only
    the remainder pay for a hunt, and each hunt opens at most
    EYE_SEARCH_MAX_TRIES frames.

    Parallel for the reason frame_selector's scoring pass is: face detection
    runs at native resolution by design (see face_detector), which is the
    expensive part, and it releases the GIL. OpenCV's own threading is turned
    off around it so the two levels of parallelism do not fight, exactly as
    the other passes do it.
    """
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=_worker_count()) as pool:
            states = list(pool.map(_eye_state_of, chosen))
    finally:
        cv2.setNumThreads(previous_threads)

    taken = {g.path for g in chosen}
    swapped = 0
    for i, state in enumerate(states):
        if state != EYES_SHUT:
            continue
        # The sample first, because it costs nothing: those frames are already
        # measured and on disk. Only a slot the sample cannot help pays for an
        # extraction of its own.
        replacement = (_first_open_eyed(chosen[i], candidates, taken)
                       or _dense_open_eyed(chosen[i], video_path, i))
        if replacement is None:
            continue
        taken.discard(chosen[i].path)
        taken.add(replacement.path)
        chosen[i] = replacement
        swapped += 1

    faces = sum(1 for s in states if s != EYES_NO_FACE)
    shut = sum(1 for s in states if s == EYES_SHUT)
    log.info("capture: %d of %d frames have a face in them; %d had their eyes shut, %d opened",
             faces, len(states), shut, swapped)
    return swapped


def select(paths: list[str], interval: float, count: int, video_path: str = "") -> list[Grab]:
    """
    `count` frames from the extracted sample, quality-filtered and spread
    across the video, in the order they occur in it.

    Returns fewer than asked for when the video genuinely has fewer usable
    frames than that. That is deliberate: padding the answer out with frames
    the filters just rejected would hand back exactly the smeared, out-of-focus
    stills this selection exists to keep out, and a short grid is a far more
    honest answer than a full one the user has to weed.
    """
    measured = measure_frames(paths, interval)
    usable = _apply_floors(measured)
    if not usable:
        return []

    # The sampled span, not the video's own duration: a frame can only be
    # chosen from what was extracted, so the slices have to divide the same
    # thing the candidates lie in.
    span = max(interval, len(paths) * interval)

    chosen, leftovers = _best_per_slice(usable, count, span)
    chosen = _backfill(chosen, leftovers, count, span)
    chosen.sort(key=lambda g: g.timestamp)

    # The second half of selection — see the module header. Both draw from
    # every usable candidate, not just the ones the first half passed over,
    # and both leave the grid exactly the size it already is.
    duplicates = spread_scenes(chosen, usable)
    blinks = open_eyes(chosen, usable, video_path)
    log.info("capture: %d frames selected of %d asked for, spanning %s to %s "
             "(%d moved off a repeated scene, %d moved off a blink)",
             len(chosen), count,
             timecode(chosen[0].timestamp), timecode(chosen[-1].timestamp),
             duplicates, blinks)
    return chosen


def timecode(seconds: float) -> str:
    """
    A timestamp as the app records it internally: "00m12s34" — minutes,
    seconds, hundredths. Never shown to anyone; it exists so that "which
    moment of the video is frame 4" has an answer in the log and in the frame
    record (see api's capture path).
    """
    total = max(0.0, seconds)
    minutes = int(total // 60)
    secs = int(total % 60)
    hundredths = int(round((total - int(total)) * 100)) % 100
    return f"{minutes:02d}m{secs:02d}s{hundredths:02d}"
