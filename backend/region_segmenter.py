import cv2
import numpy as np


def feather_mask(mask: np.ndarray, radius: float = 18.0) -> np.ndarray:
    """
    Soften a hard 0/255 mask into a smooth gradient at its edges. Without this,
    any imprecision in the segmentation boundary (especially from the
    downscaled GrabCut) shows up as a visible halo/silhouette outline once the
    mask is used to blend two differently-graded versions of the image.
    """
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=radius)


def face_ellipse_mask(image_shape: tuple[int, int], face: tuple[int, int, int, int]) -> np.ndarray:
    """Elliptical mask inscribed in the face bounding box — closer to real face shape than a rectangle."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy, fw, fh = face
    cx, cy = fx + fw // 2, fy + fh // 2
    axes = (int(fw * 0.55), int(fh * 0.62))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


def skin_mask_in_region(image: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """
    YCrCb skin-color threshold, restricted to a region of interest so background
    objects that happen to be skin-toned (wood, walls) aren't picked up.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin = cv2.inRange(ycrcb, lower, upper)
    return cv2.bitwise_and(skin, region_mask)


def person_region_rect(image_shape: tuple[int, int], face: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Rough body region for GrabCut seeding: face expanded upward slightly and down to frame bottom."""
    h, w = image_shape[:2]
    fx, fy, fw, fh = face
    cx = fx + fw // 2

    rect_w = int(fw * 3.2)
    x0 = max(0, cx - rect_w // 2)
    x1 = min(w, cx + rect_w // 2)
    y0 = max(0, int(fy - fh * 0.4))
    y1 = h

    return (x0, y0, x1 - x0, y1 - y0)


GRABCUT_WORK_WIDTH = 320  # GrabCut runs on a downscaled copy — full-res is far too slow for per-frame use


def foreground_mask_grabcut(image: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    """
    Separate the person (face + body/torso) from the background using GrabCut,
    seeded with a rectangle around the face and the area below it. Runs on a
    small downscaled copy (GrabCut's cost scales with pixel count) and the
    resulting mask is upscaled back to the original resolution. Falls back to
    the seed rectangle on failure (e.g. degenerate image).
    """
    h, w = image.shape[:2]
    rect = person_region_rect((h, w), face)

    work_scale = min(1.0, GRABCUT_WORK_WIDTH / w)
    work_w, work_h = max(1, int(w * work_scale)), max(1, int(h * work_scale))
    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    small_rect = (
        int(rect[0] * work_scale), int(rect[1] * work_scale),
        max(1, int(rect[2] * work_scale)), max(1, int(rect[3] * work_scale)),
    )

    mask = np.zeros((work_h, work_w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask, small_rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        small_result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if small_result.sum() == 0:
            raise ValueError("empty foreground")
        # Linear upscale (not nearest) so the low-res mask's blocky edges
        # become a soft gradient instead of a jagged silhouette outline.
        return cv2.resize(small_result, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        fallback = np.zeros((h, w), dtype=np.uint8)
        x, y, rw, rh = rect
        fallback[y:y + rh, x:x + rw] = 255
        return fallback


def segment_regions(image: np.ndarray, face: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """
    Lightweight region segmentation using Haar-detected face + GrabCut, since
    MediaPipe/YOLOv8-seg aren't available in this environment. Returns:
      - face_mask: elliptical face region
      - skin_mask: skin-toned pixels within the person region (face + body)
      - body_mask: GrabCut foreground (face + torso), includes face_mask
      - background_mask: everything outside body_mask
    """
    face_mask = face_ellipse_mask(image.shape, face)
    body_mask = foreground_mask_grabcut(image, face)
    body_mask = cv2.bitwise_or(body_mask, face_mask)  # guarantee face is always inside body

    # Skin detection uses the hard (pre-feather) body mask so color thresholding
    # isn't diluted by soft edge values, then gets its own feather afterward.
    skin_mask = skin_mask_in_region(image, body_mask)

    face_mask = feather_mask(face_mask, radius=12)
    skin_mask = feather_mask(skin_mask, radius=10)
    body_mask = feather_mask(body_mask, radius=45)
    background_mask = (255 - body_mask).astype(np.uint8)

    return {
        "face_mask": face_mask,
        "skin_mask": skin_mask,
        "body_mask": body_mask,
        "background_mask": background_mask,
    }
