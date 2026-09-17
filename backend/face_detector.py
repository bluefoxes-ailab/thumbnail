import threading

import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

# cv2.CascadeClassifier is NOT thread-safe: detectMultiScale mutates
# per-detection scale data held on the classifier object itself, so two
# threads scanning through one shared instance corrupt each other's state.
# It surfaces as an assertion failure deep in cascadedetect.hpp
# ("0 <= scaleIdx && scaleIdx < scaleData->size()"), intermittently — which is
# exactly what scoring frames across a thread pool started hitting. Each
# thread gets its own classifier instead; they're cheap to construct and the
# XML is parsed once per worker, not per call.
_local = threading.local()


def _cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    if not hasattr(_local, "face"):
        _local.face = cv2.CascadeClassifier(CASCADE_PATH)
        _local.eye = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    return _local.face, _local.eye

MIN_FACE_PX = 60

# Detection deliberately runs at the frame's NATIVE resolution.
#
# Running the cascade on a downscaled copy and scaling the boxes back is
# tempting — detectMultiScale is the single most expensive operation in the
# selection pass and its cost is linear in pixel count. It was implemented and
# measured, and it does not merely lose a little recall: on this app's own
# test footage, detecting at 960px changed the face COUNT on 23% of frames and
# changed which face was the largest — the one every framing decision is built
# on — on 10% of them. One frame gained a detection at 960px that native
# resolution does not see at all, and because it was the bigger box it took
# over as the subject and reframed the whole thumbnail around it.
#
# Detection results are the product here, not an intermediate: a cheaper scan
# that silently reframes one thumbnail in ten is not a speedup, it's a
# different app. The selection pass gets its speed from scoring each frame
# once instead of twice and from doing it across threads (see frame_selector),
# neither of which changes a single output.
def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes in `image`'s own coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _cascades()[0].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def face_sharpness(image: np.ndarray, face: tuple[int, int, int, int]) -> float:
    """
    Laplacian variance over the face box. Must be measured on full-resolution
    pixels: this statistic is strongly resolution-dependent, and every
    threshold built on it (quality_scorer.SHARPNESS_MIN and friends) was
    calibrated against native-resolution frames.
    """
    x, y, w, h = face
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _eye_boxes(image: np.ndarray, face: tuple[int, int, int, int]):
    """Raw eye detections within the upper 55% of a face, in ROI-local coords."""
    x, y, w, h = face
    search_h = int(h * 0.55)
    roi = image[y:y + search_h, x:x + w]
    if roi.size == 0:
        return [], search_h
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    min_eye_px = max(12, int(w * 0.12))
    eyes = _cascades()[1].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6,
        minSize=(min_eye_px, min_eye_px),
    )
    return eyes, search_h


def detect_eyes(image: np.ndarray, face: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Eye boxes (full-image coords) found in the upper 55% of a face region."""
    x, y, _, _ = face
    eyes, _ = _eye_boxes(image, face)
    return [(x + int(ex), y + int(ey), int(ew), int(eh)) for (ex, ey, ew, eh) in eyes]


def eyes_open(image: np.ndarray, face: tuple[int, int, int, int]) -> bool:
    """
    Returns True only when two geometrically plausible open eyes are detected.

    Three checks in order:
    1. Frontality — faces with w/h < 0.60 are profile/near-profile; only one eye
       is visible so reject immediately without running the cascade.
    2. Cascade detection — Haar eye cascade only fires on open eyes; closed/squinted
       eyes produce no match.
    3. Geometric validation — the two strongest detections must be on opposite
       horizontal halves of the face (horiz separation ≥ 25% of face width) and
       at similar heights (vertical difference ≤ 20% of search region height).
       This rejects eyebrow+eye or shadow+eye pairings that a bare count check misses.
    """
    x, y, w, h = face

    # 1. Profile / near-profile rejection
    if w / h < 0.60:
        return False

    # 2. Eye cascade — search upper 55% of face where eyes actually are
    eyes, search_h = _eye_boxes(image, face)
    if len(eyes) < 2 or search_h <= 0:
        return False

    # 3. Geometric validation — keep the 2 largest (most confident) detections
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = sorted(
        eyes, key=lambda e: e[2] * e[3], reverse=True
    )[:2]

    cx1, cy1 = ex1 + ew1 / 2.0, ey1 + eh1 / 2.0
    cx2, cy2 = ex2 + ew2 / 2.0, ey2 + eh2 / 2.0

    # Must be on clearly opposite sides of the face (rejects profile + false second eye)
    if abs(cx1 - cx2) / w < 0.25:
        return False

    # Must be at similar heights (rejects eye + nostril / eye + shadow below)
    if abs(cy1 - cy2) / search_h > 0.20:
        return False

    return True
