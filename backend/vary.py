"""
vary.py — Swapping one slot onto a nearby moment of the same shot.

Lets the user dodge a closed-eyes frame or fish for a better expression on a
frame they otherwise like, without hunting through the whole grid for an
alternative of the same character.

There are two pickers here, because the two kinds of slot mean different
things by "another moment of this one".

pick_variation is the thumbnail one, and it is built on faces: it holds the
character fixed and changes the expression, and every test it applies —
identity, outfit, pose, the face-region histogram — exists to make sure the
person who comes back is the person who was there. That is the right question
for a thumbnail, whose whole subject is a person.

pick_capture_variation is the frame-fetcher one (see frame_grab). Most of
those frames have nobody in them at all, so a face-anchored search finds
nothing to hold fixed and refuses every candidate. What it holds fixed instead
is the SCENE, and what it looks for is a different moment of it that is at
least as sharp — preferring, when there is a person in shot, one with their
eyes open, which is the same preference the selection pass applies (see
frame_grab.open_eyes).
"""

import os
import glob
import random
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from frame_extractor import extract_window, full_frame_path
from quality_scorer import score_frame
from character_identifier import face_descriptor
from image_utils import face_region, color_hist, hist_diff, hist_similarity, imread
from state import session, TEMP_DIR
import frame_grab

log = logging.getLogger("uvicorn.error")

# /vary-frame draws candidates from a DENSE, freshly-extracted micro-sample
# of the source video, not from /process-video's sparse whole-video pool
# (0.3-1.5s apart, spread across the whole video — a fixed number of steps
# through it can span many seconds, occasionally crossing a scene cut, and on
# some frames left as few as two usable options within reach). Sampling
# tightly around the origin frame's own timestamp instead — a couple of
# seconds, well inside a single shot on virtually all real footage — gives
# dozens of genuinely close-in-time options and keeps that count roughly
# constant regardless of how sparse or dense /process-video's own sample
# happened to be for this particular video.
VARY_WINDOW_SECONDS = 2.0     # +/- real-time radius around the origin frame's timestamp
VARY_SAMPLE_INTERVAL = 0.15   # sampling interval within that window

# Color-histogram difference (1 - correlation) allowed between the origin
# frame and a candidate variation, checked BOTH over the whole frame (catches
# a background/scene change) and over just the face region (catches a
# character change even when the background happens to be similarly
# colored — the whole-frame check alone missed this). A candidate is judged
# by whichever of the two is worse. This is the preferred cap, not a hard
# wall — see the tiered search below: it's only ever relaxed as a last
# resort, after every candidate that stays under it has been exhausted.
VARY_MAX_VISUAL_DIFF = 0.05

MAX_CANDIDATE_WORKERS = 8


def dense_pool(frame_id: int) -> list[str]:
    """
    Densely-sampled candidate paths from a +/-VARY_WINDOW_SECONDS window
    around this frame_id's ORIGIN timestamp (its very first pick from
    /process-video, not its current one — anchoring on origin, not current,
    is what keeps repeated clicks from drifting: re-centering on wherever the
    last click landed turned a run of clicks into a random walk that could
    wander toward the edge of the video or into a sparse stretch), in random
    order. Built once per frame_id via a fresh ffmpeg pass over the source
    video and cached — every later Variation click on this frame reuses the
    same pool instead of paying for another extraction.
    """
    cached = session.vary_pools.get(frame_id)
    if cached is not None:
        pool = list(cached)
        random.shuffle(pool)
        return pool

    if not session.video_path or not session.extract_interval:
        return []

    record = session.frames[frame_id]
    global_dir = os.path.dirname(record.origin_path)
    all_frames = sorted(glob.glob(os.path.join(global_dir, "frame_*.jpg")))
    if record.origin_path not in all_frames:
        return []
    origin_time = all_frames.index(record.origin_path) * session.extract_interval

    # Scoped by video, not just frame_id: frame_ids reset to 0, 1, 2... for
    # every new video, so "vary_{frame_id}" alone reused the SAME on-disk
    # directory across videos, and extract_window's own reuse check then
    # served a later video's frame_id 0 the PREVIOUS video's leftover files.
    video_id = os.path.splitext(os.path.basename(session.video_path))[0]
    out_dir = os.path.join(TEMP_DIR, f"vary_{video_id}_{frame_id}")
    try:
        pool = extract_window(
            session.video_path, origin_time, VARY_WINDOW_SECONDS, VARY_SAMPLE_INTERVAL, out_dir,
        )
    except Exception as e:
        log.warning("vary: window extraction failed (%s)", e)
        pool = []

    session.vary_pools[frame_id] = pool
    shuffled = list(pool)
    random.shuffle(shuffled)
    return shuffled


def _visual_diff(origin_img, origin_hist_whole, origin_hist_face, cand_img, cand_face) -> float:
    """
    How different a candidate looks from the origin frame, as the WORSE of
    two histogram comparisons: whole-frame (catches a background/scene
    change) and face-region only (catches a character change even when the
    background happens to be similarly colored, which the whole-frame check
    alone let through in testing).
    """
    whole = hist_diff(origin_hist_whole, color_hist(cand_img))
    face = hist_diff(origin_hist_face, color_hist(face_region(cand_img, cand_face)))
    return max(whole, face)


# How coarse a picture has to get before two moments of the same shot can be
# compared as POSES rather than as photographs.
#
# 64x36 is the frame at a five-hundredth of its area: a face is four pixels
# across and an expression is gone, an arm is a smear a few pixels wide, and
# what survives is where the mass of the person is and which way it is leaning.
# That is the right level of detail for the question being asked — two frames
# of a comedian at a microphone differ in almost nothing else, and a measure
# that could still see the face would answer "different" to every blink.
POSE_SIGNATURE_SIZE = (64, 36)


def _pose_signature(img) -> np.ndarray:
    """
    A frame reduced to the arrangement of light and dark in it, normalised.

    Contrast-normalised rather than raw, so that a moment lit a stop brighter
    than another is not a different pose on that account alone — the stage
    lighting on this footage swings a long way between bits, and it is the
    LAYOUT that has to carry the comparison.
    """
    small = cv2.resize(img, POSE_SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype("float32")
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def _pose_diff(a: np.ndarray, b: np.ndarray) -> float:
    """How differently two frames are arranged. 0 is the same photograph."""
    return float(np.abs(a - b).mean())


# The chest, as multiples of the detected face box, measured downward from the
# top of it. 1.15 clears the chin; 2.2 stops above the waist.
#
# Both ends matter and the lower one was learnt the hard way: taken down to
# 2.9 the window lands on a standing performer's HIP, and a shot of the same
# man in the same shirt then reads as a different outfit because one sample was
# of his trousers. What is wanted is a patch that is his shirt in every shot he
# appears in, whatever distance the camera is at.
OUTFIT_BOX = (1.1, 1.15, 2.2)   # half-width, top, bottom — all in face heights/widths

# How different two chest patches have to look before they are called two
# outfits. Colour-histogram difference, the same scale as VARY_MAX_VISUAL_DIFF.
#
# Measured on this footage: the same grey shirt across the whole special
# compares between 0.03 and 0.24 from shot to shot, and against a different
# person's clothes from 0.68 up. 0.45 sits in the gap with room on both sides.
#
# It is a PREFERENCE and never a filter, which is what makes a number in that
# gap safe enough. Called wrongly in one direction it passes over one moment in
# favour of another that is also the subject; called wrongly in the other it
# lets two similar outfits sit on one thumbnail, which is what happens anyway
# on the videos that only contain one.
OUTFIT_MIN_DIFF = 0.45

# How close two faces have to descriptor-match before a candidate is allowed in
# on identity alone, past the same-scene test.
#
# This is the door that makes different clothes POSSIBLE at all. The same-scene
# test is deliberately tight (VARY_MAX_VISUAL_DIFF, 0.05) and a compilation's
# other episode — different set, different lighting, different shirt — is
# nowhere near passing it, so without this every alternate comes from the
# segment the frame itself is in and every figure wears the same thing by
# construction.
#
# Set conservatively, and the reason is in the measurement: on this footage the
# same performer's faces compare from 0.44 apart, with a tail out past 1.2 on
# the wide shots where the face is sixty pixels and blurred, while the nearest
# DIFFERENT person compares 1.19. Those distributions touch. character_
# identifier built this descriptor to cluster a pool, not to answer yes or no
# about one pair, and its own header says it is biased toward over-splitting.
#
# So the threshold is put well below where the two meet rather than between
# them. What that costs is recall — a genuinely different outfit in a blurry
# wide shot is turned away — and what it buys is that the failure this cannot
# have does not happen: three figures on one thumbnail who are three different
# people. Fewer outfits is a disappointment; the wrong person is a wrong
# thumbnail.
IDENTITY_MAX_DIST = 0.85


def _outfit_signature(img, face: tuple[int, int, int, int]):
    """
    A colour signature for what the subject is WEARING — the chest, as a
    histogram. None when the box falls outside the frame.

    The chest rather than the whole figure, because the whole figure carries
    the stage behind it and the stage is the thing that is the same all night.
    """
    fx, fy, fw, fh = face
    half, top, bottom = OUTFIT_BOX
    ih, iw = img.shape[:2]
    cx = fx + fw // 2
    x1, x2 = max(0, cx - int(half * fw)), min(iw, cx + int(half * fw))
    y1, y2 = min(ih, fy + int(top * fh)), min(ih, fy + int(bottom * fh))
    patch = img[y1:y2, x1:x2]
    return color_hist(patch) if patch.size else None


def _outfit_diff(a, b) -> float:
    """How differently two moments are dressed. 0 when one of them cannot be measured."""
    return hist_diff(a, b) if a is not None and b is not None else 0.0


def _same_person(candidate: dict, origin_identity) -> bool:
    """
    Whether a candidate that failed the same-scene test is nonetheless
    unmistakably the same person — see IDENTITY_MAX_DIST.
    """
    if origin_identity is None or candidate.get("identity") is None:
        return False
    return float(np.linalg.norm(candidate["identity"] - origin_identity)) <= IDENTITY_MAX_DIST


def _evaluate(path: str, origin_img, origin_hist_whole, origin_hist_face,
              used_paths: set, recent: set) -> dict | None:
    img = imread(path)
    if img is None:
        return None

    result = score_frame(img)
    if result is None or not result["framing"].valid:
        return None

    return {
        "path": path,
        "face": result["best_face"],
        "face_count": result["face_count"],
        "pose": _pose_signature(img),
        "outfit": _outfit_signature(img, result["best_face"]),
        "identity": face_descriptor(img, result["best_face"]),
        "on_diff": _visual_diff(
            origin_img, origin_hist_whole, origin_hist_face, img, result["best_face"],
        ) <= VARY_MAX_VISUAL_DIFF,
        "used": path in used_paths,
        "recent": path in recent,
    }


# How far apart, in the video's own running time, two figures on one thumbnail
# have to be cut from — as a fraction of the whole video.
#
# The first version of this drew from the same +/-2 second window Variation
# does, and the three figures came out near-identical: two seconds of a person
# talking is one pose from three angles, and a thumbnail showing the subject
# three times wants three things they DID. Five percent of a twenty-minute
# special is a minute apart, which is a different bit and usually a different
# position on the stage.
#
# A fraction rather than a number of seconds, so it means the same thing on a
# two-minute clip as on an hour-long one.
#
# This is necessary and it is not sufficient, which took a second look at the
# canvas to see. A minute apart is a different MOMENT; it is very often the
# same PICTURE, because a stand-up special is one man at one microphone and he
# returns to the same stance all night. Time separation cannot tell those
# apart. `_by_distinctness` is what does.
MOMENT_SEPARATION_FRAC = 0.05

# How far either side of a target position the search may wander to find a
# frame with a usable face in it, as a fraction of the video. Well under the
# separation above, so a candidate can never drift close enough to its
# neighbour to defeat the point of the spacing.
MOMENT_SEARCH_FRAC = 0.015

# How many alternates are offered at most. The extra figures a slot shows use
# the first few; Variation walks the rest, which is what gives that button
# something to do on a thumbnail whose figures are already spread across the
# video (see api._cutout_sync).
MAX_ALTERNATE_MOMENTS = 8


def _video_frames(record) -> list[str]:
    """Every frame extracted from this video, in time order."""
    return sorted(glob.glob(os.path.join(os.path.dirname(record.origin_path), "frame_*.jpg")))


def alternate_moments(frame_id: int, count: int) -> list[dict]:
    """
    Up to `count` OTHER moments of this video to cut extra figures from — the
    same person, spread across the running time, in a fixed order.

    Drawn from the whole video rather than from Variation's dense window (see
    MOMENT_SEPARATION_FRAC), and each pick is checked to be the same
    person/scene by the same visual-difference test Variation uses. That test
    is the thing standing between "the subject three times" and "three
    different people", and on a single-location shoot — which is what this
    channel is for — it holds across the whole video.

    Returned most DISTINCT first rather than in the order the search found them
    (see `_by_distinctness`), so the second and third figures of a thumbnail are
    the two moments least like the frame's own and least like each other — the
    spacing alone does not deliver that on footage where the subject stands in
    one place.

    Deterministic, and that is not a nicety: a thumbnail carrying three figures
    re-cuts all three whenever the edit preset changes, and a shuffled search
    would put three different people on the canvas each time.

    Returns fewer than asked — or none at all, on an uploaded still that has no
    video behind it — rather than inventing something. The caller draws the
    frame's own photo again in that case, which is visibly a repeat rather than
    visibly wrong.
    """
    if count <= 0:
        return []
    record = session.frames.get(frame_id)
    if record is None or record.uploaded:
        return []

    cached = session.alternates.get(frame_id)
    if cached is not None:
        return cached[:count]

    frames = _video_frames(record)
    if record.origin_path not in frames or len(frames) < 3:
        # Named rather than returned silently. An empty answer here is not a
        # neutral outcome: it is what makes a three-figure thumbnail draw one
        # photograph three times with a Variation button that cannot move it,
        # and from the outside that is indistinguishable from a bug in either
        # of those features. Chased once from the symptom end, it took a
        # session's worth of guessing to get back to this line.
        log.info("alternates: none for frame %s — origin %s, %d frame_*.jpg beside it%s",
                 frame_id, os.path.basename(record.origin_path), len(frames),
                 "" if record.origin_path in frames else " (origin NOT among them)")
        return []
    here = frames.index(record.origin_path)

    origin_img = imread(record.origin_path)
    if origin_img is None:
        return []
    origin_hist_whole = color_hist(origin_img)
    origin_hist_face = color_hist(face_region(origin_img, record.origin_face))
    origin_identity = face_descriptor(origin_img, record.origin_face)

    step = max(1, int(round(len(frames) * MOMENT_SEPARATION_FRAC)))
    reach = max(1, int(round(len(frames) * MOMENT_SEARCH_FRAC)))

    # Targets alternate forward and back at growing multiples of the spacing,
    # so the first figures come from either side of the frame's own moment
    # rather than marching off in one direction — and so a frame near the start
    # or the end of a video still has somewhere to look.
    targets = []
    for k in range(1, MAX_ALTERNATE_MOMENTS + 2):
        for direction in (1, -1):
            at = here + direction * k * step
            if 0 <= at < len(frames):
                targets.append(at)

    picked, taken = [], []
    for target in targets:
        if len(picked) >= MAX_ALTERNATE_MOMENTS:
            break
        # Outward from the target: the nearest frame to it with a usable face.
        order = sorted(range(max(0, target - reach), min(len(frames), target + reach + 1)),
                       key=lambda at: abs(at - target))
        for at in order:
            if any(abs(at - other) < step for other in taken + [here]):
                continue
            candidate = _evaluate(frames[at], origin_img, origin_hist_whole,
                                  origin_hist_face, set(), set())
            if candidate is None:
                continue
            # Same shot, OR a face this certainly is the subject's. The second
            # arm is what lets a compilation offer the same person in a
            # different shirt: those moments come from another episode and fail
            # the same-scene test by a mile, and turning them all away is what
            # made every figure on a multi-figure thumbnail identically dressed.
            # See IDENTITY_MAX_DIST for why that door is only open this far.
            if not candidate["on_diff"] and not _same_person(candidate, origin_identity):
                continue
            picked.append(candidate)
            taken.append(at)
            break

    # Nothing survived the same-scene / same-person test. Rather than hand back
    # an empty list — which draws the frame's own photo again for every extra
    # figure and leaves Variation with nowhere to step — take the same targets
    # again on spacing alone.
    #
    # The strict test is there to stop a compilation putting three different
    # people on one thumbnail, and relaxing it is a real cost. It is worth
    # paying because the alternative is not "no wrong figure", it is "the same
    # figure three times", which is visibly broken on every such slot. A
    # different moment of the same video is the likelier reading of a frame the
    # scene test merely could not confirm — a lighting change, a cutaway, a
    # face the detector saw at a bad angle — and every one of these is still a
    # frame with a usable face in it, still spaced across the running time, and
    # still something the user can walk past with Variation.
    if not picked:
        relaxed = []
        for target in targets:
            if len(relaxed) >= MAX_ALTERNATE_MOMENTS:
                break
            order = sorted(range(max(0, target - reach), min(len(frames), target + reach + 1)),
                           key=lambda at: abs(at - target))
            for at in order:
                if any(abs(at - other) < step for other in taken + [here]):
                    continue
                candidate = _evaluate(frames[at], origin_img, origin_hist_whole,
                                      origin_hist_face, set(), set())
                if candidate is None:
                    continue
                relaxed.append(candidate)
                taken.append(at)
                break
        if relaxed:
            log.info("alternates: frame %s found %d only by dropping the same-scene test",
                     frame_id, len(relaxed))
        else:
            log.info("alternates: frame %s found none at all — %d frames, step %d, reach %d",
                     frame_id, len(frames), step, reach)
        picked = relaxed

    picked = _by_distinctness(picked, _pose_signature(origin_img),
                              _outfit_signature(origin_img, record.origin_face))
    for candidate in picked:
        # Measured, used, and not worth carrying in the cache.
        for key in ("pose", "outfit", "identity"):
            candidate.pop(key, None)

    session.alternates[frame_id] = picked
    return picked[:count]


def _by_distinctness(candidates: list[dict], origin_pose: np.ndarray,
                     origin_outfit) -> list[dict]:
    """
    The same moments, in the order that shows the subject looking the most
    different — a different outfit first, and failing that, doing the most
    different thing.

    The spacing above buys separation in TIME, and that was assumed to buy
    separation in what the person is doing. On this footage it does not. A
    stand-up special is a man standing at a microphone for an hour: two moments
    a minute apart are a minute apart and identical, and the picks that came
    back nearest each target measured 0.10 from the origin on the scale
    `_pose_diff` uses, where two genuinely different bits measure 0.4. The
    thumbnails showed the subject twice and the two were the same photograph as
    far as anyone looking could tell — which takes the one thing a multi-figure
    option is FOR and throws it away.

    So the set is ORDERED rather than filtered. A floor was tried first and is
    the wrong tool: at a threshold strict enough to mean anything, one origin
    frame in this video was left with a single alternate and a three-figure
    slot had nothing to put in its third place. What a video contains is what
    it contains, and a rule that answers "nothing, then" is worse than the
    duplicates it was meant to prevent.

    Ordered farthest-first: each place goes to whichever moment is least like
    everything already chosen, the origin included. That puts the most
    different thing the video has into the second figure, the most different
    thing from BOTH into the third, and leaves the near-duplicates at the back
    where only a user clicking Variation repeatedly will ever reach them — and
    by then they are asking to see everything there is.

    Clothes come first in that ranking, ahead of pose, because a user handed a
    compilation is watching the same person in five different shirts and a
    thumbnail carrying three figures in three of them is the thing that look is
    for — where the same shirt three times reads as one photograph pasted down
    three times however differently the arms are held. A moment nobody else in
    the set is dressed like wins the next place outright; among moments that
    are all dressed alike, or all different, pose decides.

    Preferred and not required: on a single-shoot video — one stage, one
    evening, one shirt — no candidate is dressed differently from any other,
    every one of them ties on the first term, and the whole ordering falls
    through to pose exactly as it would have without this. Repeating an outfit
    is a fine outcome; having nothing to put in the third figure is not.

    Nothing is dropped and nothing is random: `remaining` is walked in the
    order the search produced it and ties go to the earlier entry, so the same
    video gives the same three figures every time it is loaded — which it has
    to, since a slot re-cuts all of them whenever the edit preset changes.
    """
    ordered = []
    poses, outfits = [origin_pose], [origin_outfit]
    remaining = list(candidates)
    while remaining:
        def unlike(c):
            # A pair of ranks, compared in order: "is anybody already chosen
            # wearing this", then "how differently is it arranged".
            fresh = min(_outfit_diff(c["outfit"], o) for o in outfits) >= OUTFIT_MIN_DIFF
            return (fresh, min(_pose_diff(c["pose"], p) for p in poses))

        best = max(remaining, key=unlike)
        remaining.remove(best)
        poses.append(best["pose"])
        outfits.append(best["outfit"])
        ordered.append(best)
    return ordered


def pick_variation(frame_id: int) -> dict | None:
    """
    Picks the next candidate through progressively looser tiers rather than
    a single hard filter — a Variation click must always produce SOMETHING,
    never a dead-end error, even after the user has clicked through most of
    what the dense window has to offer:

      1. Not shown elsewhere in the grid, not in this frame's own recent
         history, and close enough to origin (both whole-frame and
         face-region diff <= VARY_MAX_VISUAL_DIFF). The ideal case.
      2. Same as (1) but allows repeats from this frame's own history —
         "return something already seen before" is explicitly fine (an
         inevitable outcome if the user just keeps clicking) as long as it's
         still visibly the same shot/character.
      3. Same diff requirement, but also allows a photo currently in use by
         another frame_id (rare — only matters on a very short window).
      4. Whatever's left with a usable face, ignoring the diff cap. Last
         resort; only reached if literally nothing nearby stayed on-shot.

    All four draw from the SAME pre-scored pool (computed once — detection
    and the diff math are the expensive part, not the tier logic), so this
    costs one pass over the window regardless of which tier ends up used.
    That pass is spread across threads, since each candidate is independent.
    """
    record = session.frames.get(frame_id)
    if record is None:
        return None

    # Compared against ORIGIN, not the current photo — otherwise each click's
    # diff check only bounds the step from the last hop, and repeated clicks
    # could drift further from the original pick than that check ever
    # intended to allow.
    origin_img = imread(record.origin_path)
    if origin_img is None:
        return None
    origin_hist_whole = color_hist(origin_img)
    origin_hist_face = color_hist(face_region(origin_img, record.origin_face))

    used_paths = session.used_source_paths()
    recent = set(record.vary_history)
    pool = dense_pool(frame_id)
    if not pool:
        return None

    workers = max(1, min(MAX_CANDIDATE_WORKERS, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored = [
            c for c in executor.map(
                lambda p: _evaluate(p, origin_img, origin_hist_whole, origin_hist_face, used_paths, recent),
                pool,
            )
            if c is not None
        ]

    tiers = [
        lambda c: c["on_diff"] and not c["used"] and not c["recent"],
        lambda c: c["on_diff"] and not c["used"],
        lambda c: c["on_diff"],
        lambda c: True,
    ]
    return next((c for tier in tiers for c in scored if tier(c)), None)


# ── The frame fetcher's variation ─────────────────────────────────────────

# How far a capture slot's candidate may drift from the frame it replaces, as
# a fraction of that frame's focus. A variation is meant to be a different
# moment, not a worse picture.
CAPTURE_FOCUS_FLOOR_FRAC = 0.75

# How many candidates to open and check the eyes of before settling for one
# whose eyes were never confirmed. Face detection runs at native resolution
# (see face_detector), so this bounds what a click can cost; the list is in
# preference order, so the ones examined are the ones most worth examining.
CAPTURE_EYE_TRIES = 10

# How different from the photo on screen a candidate has to be, as a fraction
# of the best difference available, to count as a real move.
#
# It is a filter and not an ordering, and the difference matters. Ordering by
# it alone is deterministic: the frame furthest from the one you are looking at
# is a fixed answer, so once the history stopped blocking a photo it came
# straight back to the top and the button walked the same short loop for ever
# — six photos out of twenty-one available, measured. Filtering instead and
# then choosing at random among what is left keeps the guarantee that a click
# visibly does something, without making it the same something.
CAPTURE_MOVE_ENOUGH_FRAC = 0.5


def _capture_pool(record, frame_id: int) -> list[str]:
    """
    Densely-sampled paths from a window around this slot's ORIGIN moment, as
    the uncropped frames a capture slot is made of.

    dense_pool cannot answer this. It locates the origin by looking
    `record.origin_path` up in the whole-video sample's `frame_*.jpg` listing,
    and a capture slot's origin is a `full_*.jpg` that is not in that listing
    — so it returns nothing at all. It does not need to look: a capture slot
    records the moment it came from (see FrameRecord.origin_timestamp).
    """
    if not session.video_path or record.origin_timestamp is None:
        return []

    cached = session.vary_pools.get(frame_id)
    if cached is None:
        video_id = os.path.splitext(os.path.basename(session.video_path))[0]
        out_dir = os.path.join(TEMP_DIR, f"vary_{video_id}_{frame_id}")
        try:
            cached = extract_window(
                session.video_path, record.origin_timestamp,
                VARY_WINDOW_SECONDS, VARY_SAMPLE_INTERVAL, out_dir,
            )
        except Exception as e:
            log.warning("vary: capture window extraction failed (%s)", e)
            cached = []
        session.vary_pools[frame_id] = cached

    # The uncropped sibling of each, which is what this slot's photos are.
    out = []
    for cropped in cached:
        full = full_frame_path(cropped)
        out.append(full if os.path.exists(full) else cropped)
    return out


def pick_capture_variation(frame_id: int) -> dict | None:
    """
    Another moment of this capture slot's own shot: same scene, no softer, and
    with the subject's eyes open where there is a subject.

    Tiered the way pick_variation is tiered, and for the same reason — a
    Variation click has to produce something rather than an error, however
    many times it is pressed:

      1. same scene, sharp enough, not already in the grid, not one of this
         slot's own recent answers, and eyes open if anyone is in shot
      2. ...allowing this slot's own history back in
      3. ...allowing a moment another slot is already using
      4. ...dropping the eyes requirement, which is a preference and not a
         requirement: a shot where the subject is turned away has no
         open-eyed frame in it at all, and the click must still work

    Within a tier the candidates are ordered by how UNLIKE the photo being
    replaced they are, so a click visibly moves and a second click moves
    again, rather than returning something a frame away from where it started.
    """
    record = session.frames.get(frame_id)
    if record is None:
        return None

    origin = imread(record.origin_path)
    current = imread(record.source_path)
    if origin is None or current is None:
        return None

    pool = _capture_pool(record, frame_id)
    if not pool:
        return None

    origin_signature = frame_grab.scene_signature(origin)
    origin_focus, _, _ = frame_grab.measure(origin)
    current_signature = frame_grab.scene_signature(current)
    # Whether anyone is in shot at all is asked once, of the frame being
    # replaced. A shot with no people in it has no eyes to prefer, and asking
    # per candidate would pay for face detection over the whole window.
    wants_eyes = frame_grab.eye_state(current) != frame_grab.EYES_NO_FACE

    used = session.used_source_paths()
    recent = set(record.vary_history)
    start = max(0.0, record.origin_timestamp - VARY_WINDOW_SECONDS)

    current_raw = record.source_raw_path or record.source_path

    def evaluate(item):
        index, path = item
        if path == current_raw:
            return None
        image = imread(path)
        if image is None:
            return None
        focus, _, _ = frame_grab.measure(image)
        if focus < origin_focus * CAPTURE_FOCUS_FLOOR_FRAC:
            return None
        signature = frame_grab.scene_signature(image)
        if hist_similarity(signature, origin_signature) < frame_grab.SCENE_SAME_CORRELATION:
            return None
        return {
            "path": path,
            "timestamp": start + index * VARY_SAMPLE_INTERVAL,
            "used": path in used,
            "recent": path in recent,
            # Distance from the photo on screen, which is what makes a click
            # visibly do something.
            "moved": 1.0 - hist_similarity(signature, current_signature),
        }

    workers = max(1, min(MAX_CANDIDATE_WORKERS, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored = [c for c in executor.map(evaluate, enumerate(pool)) if c is not None]
    if not scored:
        return None

    # Everything that moves the picture enough to be worth showing, in no
    # particular order — see CAPTURE_MOVE_ENOUGH_FRAC. Shuffled for the reason
    # dense_pool shuffles its own: a deterministic best answer makes repeated
    # clicks a loop rather than a search.
    best_move = max(c["moved"] for c in scored)
    scored = [c for c in scored if c["moved"] >= best_move * CAPTURE_MOVE_ENOUGH_FRAC] or scored
    random.shuffle(scored)

    def first_with_eyes(candidates):
        for candidate in candidates[:CAPTURE_EYE_TRIES]:
            image = imread(candidate["path"])
            if image is not None and frame_grab.eye_state(image) == frame_grab.EYES_OPEN:
                return candidate
        return None

    tiers = [
        lambda c: not c["used"] and not c["recent"],
        lambda c: not c["used"],
        lambda c: True,
    ]
    if wants_eyes:
        for tier in tiers:
            found = first_with_eyes([c for c in scored if tier(c)])
            if found is not None:
                return found
    return next((c for tier in tiers for c in scored if tier(c)), None)
