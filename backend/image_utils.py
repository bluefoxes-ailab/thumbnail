"""
image_utils.py — Small image primitives that were previously copy-pasted
across modules. Each one existed in two or three places with slightly
different arithmetic; a single definition is what keeps them from drifting
apart again (the same reasoning reframe_engine.compute_scale is shared by the
real reframe and the scorer's simulation of it).
"""

import os

import cv2
import numpy as np

# Face-region padding, as a fraction of the face box's larger side. Used by
# frame_selector's dedup histogram and by the /vary-frame visual-diff check —
# these MUST stay identical: vary's "is this still the same shot" test is
# calibrated against the same crop the dedup pass uses.
FACE_REGION_PAD_FRAC = 1.0 / 3.0

# Histogram shape used everywhere a color histogram is compared in this app.
HIST_BINS = [8, 8, 8]
HIST_RANGES = [0, 256, 0, 256, 0, 256]

# Target pixel budget for statistics-only measurements (percentiles, means,
# stds). Every adaptive stage in face_restorer reduces a whole image to ONE
# scalar; computing that over 3.5 megapixels instead of ~0.25 costs ~14x more
# for a number that doesn't measurably move (a 1st/99th percentile over 250k
# samples is stable to well under a tonal level). Stages whose statistic is
# resolution-dependent — noise sigma, laplacian variance — must NOT use this;
# they measure on the real pixels (see face_restorer's measurement pass).
STATS_PIXEL_BUDGET = 250_000


# ── File I/O ──────────────────────────────────────────────────────────────
# cv2.imread/imwrite must NOT be called directly anywhere in this app: on
# Windows OpenCV opens the file through the narrow (ANSI) API, so any path
# holding a character outside the machine's codepage fails to open. A Windows
# install in Portuguese puts the desktop at "C:\Users\<user>\Área de
# Trabalho", which is enough to break every read — and imread reports it by
# returning None, exactly like a corrupt file, so the failure surfaced far
# downstream as "no faces detected" on frames that were sitting on disk
# perfectly intact.
#
# Going through numpy keeps the path handling in Python (which is
# Unicode-correct) and hands OpenCV only an in-memory buffer.


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Unicode-path-safe cv2.imread. Returns None when the file can't be read/decoded."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path: str, image: np.ndarray, params: list[int] | None = None) -> bool:
    """Unicode-path-safe cv2.imwrite. Returns True on success, like cv2's own."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


def luma(image: np.ndarray) -> np.ndarray:
    """
    BT.601 luma from a BGR image, as float32 — for statistics, not display.

    Cheaper than a cvtColor round-trip when only the luminance plane is
    wanted, and identical in intent to the YCrCb Y channel used elsewhere.
    """
    img = image if image.dtype == np.float32 else image.astype(np.float32)
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def stats_sample(image: np.ndarray, budget: int = STATS_PIXEL_BUDGET) -> np.ndarray:
    """
    A strided subsample of `image` holding roughly `budget` pixels, for
    statistics only (see STATS_PIXEL_BUDGET). Returns the image itself when
    it's already at or under budget, so small inputs pay nothing.

    Strided rather than resized: a resize would low-pass the data and shift
    exactly the noise/detail statistics some callers care about, whereas
    taking every Nth pixel leaves each sampled pixel's own value untouched.
    """
    h, w = image.shape[:2]
    total = h * w
    if total <= budget:
        return image
    step = int(np.ceil((total / budget) ** 0.5))
    return image[::step, ::step]


def face_region(image: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    """
    The face box padded by FACE_REGION_PAD_FRAC of its larger side, clipped to
    the image. Falls back to the whole image if the crop comes out empty.
    """
    fx, fy, fw, fh = face
    pad = int(max(fw, fh) * FACE_REGION_PAD_FRAC)
    ih, iw = image.shape[:2]
    x1, y1 = max(0, fx - pad), max(0, fy - pad)
    x2, y2 = min(iw, fx + fw + pad), min(ih, fy + fh + pad)
    region = image[y1:y2, x1:x2]
    return region if region.size > 0 else image


def color_hist(region: np.ndarray) -> np.ndarray:
    """Normalized 8x8x8 BGR histogram — the app's one comparable color signature."""
    h = cv2.calcHist([region], [0, 1, 2], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(h, h)
    return h


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two color_hist outputs (1 = identical)."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    """1 - hist_similarity (0 = identical, 1 = unrelated)."""
    return 1.0 - hist_similarity(a, b)


def scale_box(box: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    """Scales an (x, y, w, h) box by `factor`, rounding to ints."""
    x, y, w, h = box
    return (int(round(x * factor)), int(round(y * factor)),
            int(round(w * factor)), int(round(h * factor)))


def expand_box(box: tuple[int, int, int, int], factor: float, shape: tuple[int, int],
               min_size: int = 0) -> tuple[int, int, int, int]:
    """
    Grows an (x, y, w, h) box around its own center by `factor` (and to at
    least `min_size` per side), clipped to an image of `shape`. Returns
    (x1, y1, x2, y2) — corner form, ready to slice with.
    """
    x, y, w, h = box
    ih, iw = shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    half_w = max(w * factor, min_size) / 2.0
    half_h = max(h * factor, min_size) / 2.0
    x1 = int(max(0, round(cx - half_w)))
    y1 = int(max(0, round(cy - half_h)))
    x2 = int(min(iw, round(cx + half_w)))
    y2 = int(min(ih, round(cy + half_h)))
    return x1, y1, x2, y2
