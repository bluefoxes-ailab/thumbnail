"""
inpainter.py — Shared inpainting core: LaMa when available (reconstructs
surrounding texture/pattern), OpenCV Telea as fallback (smudges neighboring
color/gradient inward — fine on plain backgrounds, visibly softer on textured
ones). Used by logo_remover (painting over a static logo remnant),
subtitle_remover (painting out burned-in subtitles) and the manual-reframe
autofill (extending the image where the user dragged the crop window past the
source frame's edge).

LaMa loads lazily on first use and is cached at module level; a failed load is
remembered so every subsequent call doesn't retry the import.
"""

import cv2
import numpy as np
import logging

import vram

log = logging.getLogger("uvicorn.error")  # see the note in face_restorer.py

_lama = None
_lama_load_failed = False

# Below this many pixels a work_scale request is ignored — the model has a
# minimum useful context size, and shrinking an already-small region buys
# nothing while measurably degrading it.
MIN_SCALED_INPAINT_PIXELS = 256 * 256

# How much surrounding picture each box is inpainted against. LaMa processes
# its input at full resolution with no internal tiling, and a heavily
# zoomed-in reframe's pre-crop image can run into the tens of thousands of
# pixels per side: handing it the whole frame tried to allocate ~90GB and
# crashed with a CUDA OOM in testing. A generous fixed margin gives the model
# plenty of texture to reconstruct from however large the frame is.
DEFAULT_CONTEXT_MARGIN_PX = 250


def _ensure_lama_loaded() -> bool:
    global _lama, _lama_load_failed
    if _lama is not None:
        return True
    if _lama_load_failed:
        return False
    try:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
        return True
    except Exception as e:
        log.warning("inpainter: LaMa unavailable, falling back to OpenCV inpainting (%s)", e)
        _lama_load_failed = True
        return False


def preload() -> bool:
    """
    Force the LaMa load now instead of on the first inpaint. Called at server
    startup so the user doesn't pay a multi-second model load in the middle of
    their first overpanned drag. Returns whether it's available.
    """
    return _ensure_lama_loaded()


def is_available() -> bool:
    """Whether LaMa is loaded/loadable — False means every call falls back to Telea."""
    return _lama is not None or not _lama_load_failed


def _run_model(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if not _ensure_lama_loaded():
        return None
    try:
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = _lama(Image.fromarray(rgb), Image.fromarray(mask))
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("inpainter: LaMa inference failed, falling back to OpenCV inpainting (%s)", e)
        return None
    finally:
        # LaMa is handed a differently-shaped region on almost every call (a
        # logo's ROI, an overpan strip), and each new shape reserves blocks
        # the previous one's can't satisfy. Threshold-guarded so a run of
        # similar sizes still reuses the pool — see vram.release.
        vram.release()


def inpaint(image: np.ndarray, mask: np.ndarray, work_scale: float = 1.0) -> np.ndarray:
    """
    Fill `mask`'s nonzero pixels of BGR `image` with plausible content.

    LaMa processes its input at full resolution with no internal tiling —
    callers handing it very large images (tens of thousands of pixels per
    side) must bound the region themselves first (see logo_remover's
    ROI/context-margin logic, and the CUDA OOM history documented there).

    `work_scale` < 1 runs the model on a downscaled copy and upsamples the
    generated content back. Inference cost scales with pixel count, so 0.5 is
    roughly a 4x saving — worth taking wherever the caller is going to
    low-pass the fill anyway (see reframe_engine.FILL_INPAINT_SCALE). Real
    (unmasked) pixels never go through the round trip: they're composited
    back from the original at full resolution below, so downscaling can only
    ever affect generated content.
    """
    result = None
    scaled = work_scale < 1.0 and image.shape[0] * image.shape[1] * work_scale ** 2 >= MIN_SCALED_INPAINT_PIXELS

    if scaled:
        h, w = image.shape[:2]
        sw, sh = max(1, int(w * work_scale)), max(1, int(h * work_scale))
        small_img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        # Nearest on the mask, then re-threshold: an averaging filter would
        # produce partial values that quietly shrink the masked region at its
        # own border, leaving a rim of un-inpainted original inside the hole.
        small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        out = _run_model(small_img, small_mask)
        if out is not None:
            result = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = _run_model(image, mask)

    if result is None:
        result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # LaMa can return a size rounded to its own internal stride — resize back
    # to the exact input shape before handing the result to the caller.
    if result.shape[:2] != image.shape[:2]:
        result = cv2.resize(result, (image.shape[1], image.shape[0]))

    # LaMa reconstructs its ENTIRE input end-to-end, not just the masked
    # pixels — and the stride resize above interpolates every pixel, mask or
    # not. Callers pass a generous context margin around the actual mask so
    # the model has real surrounding texture to work from, on the assumption
    # (per this function's own contract above) that only mask pixels come
    # back changed. Without this, that context margin's own outer edge —
    # which can sit well away from the masked region, anywhere the caller's
    # ROI happened to land — showed up as a sharp rectangular seam wherever
    # it fell, including straight across a face nowhere near the actual mask.
    # Compositing back through the mask makes the contract true instead of
    # assumed: everywhere mask==0 is guaranteed byte-identical to the input.
    # This is also what makes work_scale safe — every real pixel below comes
    # from `image` at full resolution, never from the upsampled round trip.
    mask_bool = mask.astype(bool)
    result[~mask_bool] = image[~mask_bool]
    return result



def inpaint_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]],
                  pad: int = 0, context: int = DEFAULT_CONTEXT_MARGIN_PX) -> np.ndarray:
    """
    Paints out each (x, y, w, h) box, in this image's own coordinates,
    grown by `pad` on every side and reconstructed from `context` pixels of
    picture around it.

    Rectangles rather than the shape of whatever is inside them, in both
    callers, and for the same reason: a mask cut tightly around a logo's
    glyphs or a subtitle's letterforms leaves the anti-aliasing, the outline
    and the drop shadow standing, which reads as a ghost of the thing that was
    supposedly removed. The box takes the halo with it.

    Returns the image unchanged when there is nothing to paint, and never
    modifies the input.
    """
    if not boxes:
        return image

    ih, iw = image.shape[:2]
    out = image.copy()

    for x, y, w, h in boxes:
        bx1, by1 = max(0, x - pad), max(0, y - pad)
        bx2, by2 = min(iw, x + w + pad), min(ih, y + h + pad)
        if bx2 <= bx1 or by2 <= by1:
            continue

        rx1, ry1 = max(0, bx1 - context), max(0, by1 - context)
        rx2, ry2 = min(iw, bx2 + context), min(ih, by2 + context)

        roi = out[ry1:ry2, rx1:rx2]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        mask[by1 - ry1:by2 - ry1, bx1 - rx1:bx2 - rx1] = 255
        out[ry1:ry2, rx1:rx2] = inpaint(roi, mask)

    return out
