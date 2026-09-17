"""
frame_selector.py — Scores every extracted frame once, in parallel, and
deduplicates the survivors.

This used to be two full passes over the whole video (a "strict" one and a
"relaxed" one), each re-reading and re-scoring the same files, plus a third
read per surviving candidate to build its dedup histogram. The passes only
ever differed by a sharpness threshold, so everything is now produced by a
single pass at the lower floor and partitioned afterwards by the caller.
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from quality_scorer import score_faces, SHARPNESS_FLOOR_RELAXED
from character_identifier import face_descriptor
from image_utils import face_region, color_hist, hist_similarity, imread

# Above this correlation between two frames' face-region histograms, the
# later one is treated as a duplicate of the earlier.
DUPLICATE_HIST_CORRELATION = 0.95

# ...and the same question asked of the face DESCRIPTOR instead — the equalised
# grey crop plus tone vector the character clusterer is built on (see
# character_identifier.face_descriptor). Above this cosine, same moment.
#
# A second measure rather than a replacement, because the two fail in opposite
# places. The histogram is a colour statistic: across people on a single-
# location shoot it cannot tell two faces apart, which is why deduplication is
# per-character. Within one person it has the milder version of the same
# problem, and on some footage that version is total. Measured on a stand-up
# special — one performer, one costume, one light, a locked-off camera — 100%
# of candidate pairs inside every character cluster cleared 0.95, including the
# two most different frames in the whole video (0.965). The pool collapsed to
# one frame per character and only POOL_FLOOR_PER_CHARACTER kept the run alive:
# 3 characters x a floor of 4 = the 12 thumbnails the user got, where the grid
# asks for 20.
#
# The descriptor separates the same footage about five times as well: pairwise
# spread 0.13-0.17 against the histogram's 0.03, median 0.95 against 0.99. What
# moves it is the shape of the face, which is what an expression IS.
#
# 0.97 is where it was set. On that video it rejects roughly the closest
# quarter of pairs and leaves a pool of 34 for a 20-slot grid — enough headroom
# for the candidates the reframe later throws out, without keeping frames a
# viewer would call the same photograph. At 0.96 the pool is 23, which fills
# the grid with almost nothing spare; at 0.99 nothing is rejected at all.
DUPLICATE_DESCRIPTOR_COSINE = 0.97

# OpenCV releases the GIL inside imread/detectMultiScale/Canny/Sobel, so
# threads genuinely run in parallel here. Capped because each worker holds a
# decoded frame while it works — unbounded workers on 4K footage is a lot of
# resident memory for no extra throughput past the core count.
MAX_SCORING_WORKERS = 8


def _worker_count() -> int:
    return max(1, min(MAX_SCORING_WORKERS, (os.cpu_count() or 4)))


def _score_one(path: str) -> list[dict]:
    """
    Scores a single frame — one candidate per usable face in it, not just the
    largest (see quality_scorer.score_faces) — and returns everything later
    stages need, so nothing has to open this file again.

    This also computes the dedup histogram AND the character descriptor here,
    while the image is already decoded, then drops the image. The previous
    design read each candidate a second time for the histogram and then held
    EVERY candidate's decoded frame in a dict for the descriptor pass — up to
    400 full-resolution frames at once, which is ~2.5GB at 1080p and ~10GB at
    4K.

    Both are computed per FACE, and `key` identifies the candidate as (path,
    face) rather than by path alone: two people in one frame are two separate
    candidates that must not shadow each other in the selector's bookkeeping.
    """
    img = imread(path)
    if img is None:
        return []

    results = score_faces(img, min_sharpness=SHARPNESS_FLOOR_RELAXED)
    for result in results:
        face = result["best_face"]
        result["path"] = path
        result["key"] = (path, face)
        result["hist"] = color_hist(face_region(img, face))
        result["descriptor"] = face_descriptor(img, face)
    return results


def score_frames(frame_paths: list[str]) -> list[dict]:
    """
    Every usable face of every frame, scored, in the order the paths were
    given.

    OpenCV's own internal parallelism is switched off for the duration.
    Frames are already being processed one per worker thread, so leaving it on
    means each of those workers also fans every cascade scan and filter out
    across all cores — 8 workers times 16 OpenCV threads competing for the
    same cores, which is slower than either level of parallelism alone.

    It also makes the results reproducible. detectMultiScale grazes a
    different set of detections run-to-run when it runs multi-threaded (~4% of
    frames on this app's test footage), because the order overlapping
    candidate rectangles get grouped isn't fixed. That was true before this
    pass was parallelised at all — it just means two runs over the same video
    could pick slightly different frames. Pinning it to one thread here costs
    nothing (the outer pool is what provides the parallelism) and removes that
    source of drift.
    """
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=_worker_count()) as pool:
            per_frame = list(pool.map(_score_one, frame_paths))
    finally:
        cv2.setNumThreads(previous_threads)
    return [r for frame_results in per_frame for r in frame_results]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two face descriptors (1 = identical)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


# The two ways of asking whether two candidates are the same moment: which
# field carries the evidence, how to compare two of them, and above what value
# the later one is a duplicate. Named so a channel can choose (see
# api.SelectionRequest) without anything here knowing what a channel is.
DEDUPE_MODES = {
    "tone":       ("hist",       hist_similarity, DUPLICATE_HIST_CORRELATION),
    "expression": ("descriptor", _cosine,         DUPLICATE_DESCRIPTOR_COSINE),
}
DEFAULT_DEDUPE_MODE = "tone"


def deduplicate(candidates: list[dict], top_n: int, min_keep: int = 0,
                mode: str = DEFAULT_DEDUPE_MODE) -> list[dict]:
    """
    Keeps the highest-scoring frames, dropping any whose face region looks
    like one already kept.

    `mode` picks what "looks like" measures — see DEDUPE_MODES. "tone" is the
    colour histogram every channel has always used; "expression" is the face
    descriptor, for a channel whose thumbnails are one performer in one setup,
    where colour has nothing left to say (see DUPLICATE_DESCRIPTOR_COSINE).

    Either way this is applied in SCORE ORDER, so quality decides who is kept
    and the measure only decides who is redundant. Nothing here can promote a
    soft frame over a sharp one.

    Call this with ONE character's candidates at a time (see api._build_pool).
    The test is a correlation between face-region color histograms, which
    within a person means "same expression, same moment" — but across people
    on a single-location shoot it mostly measures skin and set lighting, and
    happily calls two different faces duplicates. Fed the whole video at once,
    in score order, it deletes the lower-scoring characters outright.

    Deduplication is on the FACE region, not the whole frame — same face
    expression = duplicate, regardless of background. Full-frame histograms
    incorrectly dedup frames with the same background but different
    expressions (different character moments).

    Two faces detected in the SAME frame are never duplicates of each other,
    whatever their histograms say — they are two different people, sat in the
    same light against the same background, which is exactly the case an 8x8x8
    color histogram of the face region correlates hardest on. Without that
    exemption this pass could drop a character before clustering ever saw
    them, and no amount of coverage guarantees downstream can recover a
    candidate that isn't in the pool.

    Sequential by necessity: whether a frame is a duplicate depends on which
    higher-scoring frames were already accepted, so this can't be split
    across workers the way scoring can.
    """
    field, alike, threshold = DEDUPE_MODES.get(mode, DEDUPE_MODES[DEFAULT_DEDUPE_MODE])
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)

    selected: list[dict] = []
    kept: list[tuple[str, np.ndarray]] = []
    rejected: list[dict] = []
    for c in ordered:
        if len(selected) >= top_n:
            break
        h = c.get(field)
        # A candidate with no descriptor (the detector found a face the
        # descriptor could not crop) is kept rather than dropped: it is a real
        # frame, and "unmeasurable" is not "duplicate".
        if h is None:
            selected.append(c)
            continue
        if any(path != c["path"] and alike(h, prev) > threshold for path, prev in kept):
            rejected.append(c)
            continue
        selected.append(c)
        kept.append((c["path"], h))

    if len(selected) < min_keep and rejected:
        selected += _most_distinct(rejected, [h for _, h in kept],
                                   min(min_keep, top_n) - len(selected), mode)
    return selected


def _most_distinct(candidates: list[dict], kept_hists: list[np.ndarray], count: int,
                   mode: str = DEFAULT_DEDUPE_MODE) -> list[dict]:
    """
    The `count` candidates least like each other and least like `kept_hists`,
    chosen greedily (farthest-first).

    The duplicate threshold is a fixed number applied to a statistic whose
    spread depends entirely on the footage. On a character filmed in one
    unbroken setup — same seat, same light, same distance — genuinely
    different moments can all sit above it, and the whole character collapses
    to one or two survivors. Measured on the reported video: one character
    went from 101 usable candidates to 4.

    So rather than loosen the threshold globally (which would let real
    duplicates through everywhere), a character that came out under its floor
    gets topped back up with the most different-looking frames it has left.
    Worst frames available, but the alternative is a character the user cannot
    choose from.
    """
    field, alike, _ = DEDUPE_MODES.get(mode, DEDUPE_MODES[DEFAULT_DEDUPE_MODE])
    picked: list[dict] = []
    hists = list(kept_hists)
    # Indices, not the dicts themselves: a candidate holds numpy arrays, so
    # list.remove's equality test raises rather than matching.
    remaining = set(range(len(candidates)))
    while remaining and len(picked) < count:
        best = min(remaining, key=lambda i: max(
            (alike(candidates[i][field], h) for h in hists
             if candidates[i].get(field) is not None), default=0.0))
        remaining.discard(best)
        picked.append(candidates[best])
        if candidates[best].get(field) is not None:
            hists.append(candidates[best][field])
    return picked
