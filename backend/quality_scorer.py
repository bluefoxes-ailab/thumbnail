import cv2
import numpy as np

from framing import active as framing_profile
from face_detector import detect_faces, face_sharpness, eyes_open
from reframe_engine import (
    target_w, target_h, single_face_area_ratio, compute_framing,
)

# What a frame is judged on, and how much each criterion counts. These are the
# DEFAULTS: a channel may state that one of them means nothing to it (see
# framing.FramingProfile.weights), which is not a hypothetical — a channel
# whose thumbnail replaces the photo's background with a fixed backdrop has no
# use whatever for "is there clear space beside the face", because there is no
# text going there and no photo left to put it on.
WEIGHTS = {
    "sharpness":     0.28,
    "left_space":    0.22,  # measured on the simulated reframe's text zone
    "exposure":      0.15,  # face brightness quality
    "reframability": 0.13,  # how cleanly the anchor is achieved without clamping
    "motion_blur":   0.12,
    "face_size":     0.10,
}


def weights() -> dict:
    """
    The weights this run scores with — WEIGHTS, with the active profile's
    overrides applied and the whole set renormalised back to 1.

    Renormalised rather than left to sum to whatever it lands on, because the
    total is what every score is expressed against: zeroing left_space without
    it would not redistribute that 0.22, it would simply make every frame in
    the video score 22% lower than before, and the sharpness threshold, the
    dedup floor and the coverage passes would all go on comparing those
    numbers to each other as though nothing had happened. Spreading it in
    proportion is the honest reading of "this channel does not care about
    that": the criteria it DOES care about keep their relative importance and
    share out what was freed.
    """
    overrides = framing_profile().weights
    if not overrides:
        return WEIGHTS
    merged = {**WEIGHTS, **{k: v for k, v in overrides.items() if k in WEIGHTS}}
    total = sum(merged.values())
    if total <= 0:
        return WEIGHTS   # a pack that zeroed everything has said nothing usable
    return {k: v / total for k, v in merged.items()}


SHARPNESS_MIN = 20.0
SHARPNESS_FLOOR_RELAXED = 8.0
SHARPNESS_SATURATE = 300.0

# cover_scale (the minimum zoom needed to fill the target canvas) is a hard
# floor — it can't be reduced further without leaving empty canvas. On an
# inherently close-up shot (face already large in the source), that floor
# alone can push the face well past the profile's ideal fill, with
# no way to "zoom out" to compensate. face_fill_penalty (below) multiplies
# the total score down in that case, so the selector prefers a better-composed
# alternative frame when the candidate pool has one, rather than the geometry
# math trying (and failing) to fix something only framing/scale can't.
FACE_FILL_TOLERANCE  = 1.5  # no penalty until the final fill exceeds this multiple of the target ratio
FACE_FILL_SATURATE   = 3.0  # multiple of the target ratio at which the penalty caps out (score -> 0)


def reframability_score(framing) -> float:
    """
    Reward frames where the profile's anchor placement is unconstrained.
    When the desired crop would go out of bounds and gets clamped, the face
    drifts from the ideal position — the larger the drift, the lower the score.

    Reads the clamped-vs-ideal pair straight off the Framing the real reframe
    will use (see reframe_engine.compute_framing), instead of re-deriving it.
    """
    x_err = abs(framing.crop_x - framing.ideal_crop_x) / target_w()
    y_err = abs(framing.crop_y - framing.ideal_crop_y) / target_h()
    total_err = (x_err + y_err) / 2.0
    return float(max(0.0, 1.0 - total_err * 3.0))


def left_space_score(image: np.ndarray, framing) -> float:
    """
    Score the 'text area' — left 40% — of the SIMULATED reframed output.
    Low edge density there means the final thumbnail will have clean space for text.

    The simulated rectangle comes from the same Framing the real reframe
    uses, so what's measured here is exactly what will be rendered.
    """
    if not framing.valid:
        return 0.0

    src_x, src_y, src_w, src_h = framing.source_rect()
    h, w = image.shape[:2]

    # Left 40% of the reframed canvas in source coordinates
    text_w = src_w * 0.40
    x1 = max(0, int(src_x))
    y1 = max(0, int(src_y))
    x2 = min(w, int(src_x + text_w))
    y2 = min(h, int(src_y + src_h))

    if x2 <= x1 or y2 <= y1:
        return 0.5

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.5

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    return float(max(0.0, 1.0 - edge_density * 5.0))


def exposure_score(image: np.ndarray, face: tuple) -> float:
    """
    Well-exposed faces (mean brightness 80-190) score highest.
    Too dark or blown-out faces are penalised linearly.
    """
    x, y, w, h = face
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.5
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    if 80.0 <= mean <= 190.0:
        return 1.0
    elif mean < 80.0:
        return max(0.0, mean / 80.0)
    else:
        return max(0.0, (255.0 - mean) / 65.0)


def face_size_score(face: tuple, image_shape: tuple) -> float:
    """
    Larger face in source = less upscaling needed = less quality loss.
    Normalised 0-1, saturating at 25% of the frame area.
    """
    h, w = image_shape[:2]
    face_area_ratio = (face[2] * face[3]) / (w * h)
    return float(min(face_area_ratio / 0.25, 1.0))


def face_fill_penalty(face: tuple, scale: float) -> float:
    """
    Multiplicative penalty (1.0 = none) for a face that would fill much more
    of the final canvas than the active framing profile intends, even at
    cover_scale — the minimum zoom needed to fill the target canvas, which
    can't be reduced further. An inherently close-up shot (face already large
    in the source) blows past the target with no way to compensate by
    zooming out, so this downranks the frame instead, letting the selector
    prefer a better-composed alternative when the candidate pool has one.

    Switched off entirely by a channel that never shows the crop — see
    `face_fill` in framing.FramingProfile. There the close-up it is built to
    demote is the frame with the most resolution on the subject, which is the
    only part that reaches the canvas.
    """
    if not framing_profile().face_fill:
        return 1.0

    final_ratio = (face[2] * scale) * (face[3] * scale) / (target_w() * target_h())
    relative = final_ratio / single_face_area_ratio()
    if relative <= FACE_FILL_TOLERANCE:
        return 1.0
    span = FACE_FILL_SATURATE - FACE_FILL_TOLERANCE
    over = (relative - FACE_FILL_TOLERANCE) / span
    return float(max(0.0, 1.0 - over))


def sharpness_score(sharpness: float) -> float:
    return float(min(sharpness / SHARPNESS_SATURATE, 1.0))


def motion_blur_penalty(image: np.ndarray, face: tuple) -> float:
    """
    Directional gradient variance ratio on a 2× padded region around the face.
    Using a wider area gives a more stable blur estimate than the tight face ROI.
    """
    fx, fy, fw, fh = face
    pad_x = fw // 2
    pad_y = fh // 2
    ih, iw = image.shape[:2]
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y)
    x2 = min(iw, fx + fw + pad_x)
    y2 = min(ih, fy + fh + pad_y)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = gx.var()
    var_y = gy.var()

    if var_x + var_y < 1e-6:
        return 0.0

    return float(min(var_x, var_y) / max(var_x, var_y))


# Most faces considered per frame, largest first. Every extra face costs a
# full Haar eye-cascade scan, and past a handful the rest are background
# extras too small to reframe a thumbnail around anyway.
MAX_FACES_PER_FRAME = 4


def _score_face(image: np.ndarray, face: tuple, faces: list, framing_face_count: int,
                min_sharpness: float) -> dict | None:
    """
    Scores ONE face of an already-detected frame, treating it as the subject.
    None if that face is unusable (eyes closed, or below `min_sharpness`).

    `framing_face_count` is what the framing math is told (see
    reframe_engine.compute_scale — at 1 it zooms until the face fills
    the profile's face area ratio, above 1 it stays at cover scale to keep the group
    in shot), which is not always len(faces); see score_faces.
    """
    # Sharpness first: it's one Laplacian over the face box, where eyes_open
    # runs a full Haar cascade. Both are hard rejects, so ordering changes no
    # outcome — only how much work a rejected frame costs.
    sharp = face_sharpness(image, face)
    if sharp < min_sharpness:
        return None

    if not eyes_open(image, face):
        return None

    best_face = face
    face_count = framing_face_count
    framing = compute_framing(image.shape, best_face, face_count)

    s_face_size    = face_size_score(best_face, image.shape)
    s_sharpness    = sharpness_score(sharp)
    s_motion       = motion_blur_penalty(image, best_face)
    s_exposure     = exposure_score(image, best_face)
    s_left_space   = left_space_score(image, framing)
    s_reframe      = reframability_score(framing)
    s_fill_penalty = face_fill_penalty(best_face, framing.scale)

    w = weights()
    total = (
        s_face_size  * w["face_size"] +
        s_sharpness  * w["sharpness"] +
        s_motion     * w["motion_blur"] +
        s_exposure   * w["exposure"] +
        s_left_space * w["left_space"] +
        s_reframe    * w["reframability"]
    ) * s_fill_penalty

    return {
        "faces": faces,
        "best_face": best_face,
        "face_count": face_count,
        "sharpness": sharp,
        "score": total,
        "framing": framing,   # the exact framing this score was predicted from — reused by the renderer
        "score_breakdown": {
            "face_size":       round(s_face_size, 3),
            "sharpness":       round(s_sharpness, 3),
            "motion_blur":     round(s_motion, 3),
            "exposure":        round(s_exposure, 3),
            "left_space":      round(s_left_space, 3),
            "face_fill_penalty": round(s_fill_penalty, 3),
            "reframability": round(s_reframe, 3),
        },
    }


def score_faces(image: np.ndarray, min_sharpness: float = SHARPNESS_FLOOR_RELAXED) -> list[dict]:
    """
    One scored candidate per usable face in the frame, largest first.

    Selection used to see only the biggest face in each frame. Anyone who
    shares the shot with someone closer to camera therefore produced no
    candidate at all from that frame — no descriptor, so not even a cluster —
    and a character who is mostly filmed in two-shots could be absent from the
    entire pool no matter how the selector was tuned afterwards.

    Face detection, the expensive part, still runs exactly once per frame;
    what repeats per face is the eye cascade and the cheap sub-scores.

    Secondary faces are framed as subjects in their own right (face_count=1),
    not at the group's cover scale: a candidate that exists specifically to
    give that person a slot is worthless if it renders them small and
    off-centre. The largest face keeps the real count, so the frames the
    pipeline already picked are framed exactly as before.
    """
    faces = detect_faces(image)
    if not faces:
        return []

    ordered = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:MAX_FACES_PER_FRAME]
    face_count = len(faces)

    results = []
    for i, face in enumerate(ordered):
        scored = _score_face(image, face, faces, face_count if i == 0 else 1, min_sharpness)
        if scored is not None:
            results.append(scored)
    return results


def score_frame(image: np.ndarray, min_sharpness: float = SHARPNESS_FLOOR_RELAXED) -> dict | None:
    """
    Scores one frame on its largest face, or returns None if it's unusable (no
    face, eyes closed, or below `min_sharpness`). The single-subject view of
    score_faces, kept for /vary-frame — that search is looking for another
    take of one specific shot, not for new characters.

    There is no strict/relaxed switch anymore. The two used to be separate
    full passes over the whole video — the relaxed one re-running face
    detection, eye detection and every sub-score on the same frames the
    strict one had already rejected, which is most of them. The only thing
    that actually differed between the passes was the sharpness floor, so a
    single pass at the LOWER floor now produces both sets: callers split the
    results on SHARPNESS_MIN afterwards (see frame_selector.score_frames),
    for the same partition at half the work.
    """
    faces = detect_faces(image)
    if not faces:
        return None
    return _score_face(image, max(faces, key=lambda f: f[2] * f[3]), faces, len(faces), min_sharpness)
