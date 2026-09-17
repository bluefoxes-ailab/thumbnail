"""
text_detector.py — Where the lettering is in a frame.

One question, asked of a model rather than of a threshold: which rectangles of
this picture are text. subtitle_remover uses it twice — once over a sample of
the video to work out where that video keeps putting its captions, and again on
every frame it renders to find the words actually on screen at that moment.

## Why a model, when everything else here is classical

Because the classical version was written first, measured, and did not work.

Morphology plus Otsu finds text-shaped structure very well; what it cannot do
is tell that structure apart from a picture that happens to have the same
statistics. Measured on real footage — a close, brightly lit shot of a person
in sequins and denim — the distributions for "a line of caption" and "a fold of
fabric" overlapped almost exactly on every property tried: contrast, brightness,
the swing of the column means, and the width, height and horizontal position of
the box. There was no threshold in there to find, and the version that shipped
on those thresholds removed two captions cleanly out of six, smeared the
picture on three, and missed one.

The same six frames through this model: six captions, six tight boxes, nothing
on the fabric. It also finds the small agency credit in the bottom corner, which
the classical pass could only have reached by lowering its minimum text height
to the point where it fired on everything.

## What it costs

2.4 MB, downloaded once, and about 0.12s per frame on the CPU. That is nothing
beside the restoration pass it runs alongside, and next to the 176 MB U2-Net
this app already fetches it is not worth discussing.

The model is PP-OCRv3's English text detector, run through OpenCV's own DB
wrapper — so there is no new runtime dependency, only a file. `cv2.dnn` is
already present wherever OpenCV is.
"""

import os
import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("uvicorn.error")  # see the note in face_restorer.py

# The OpenCV model zoo's own copy. Fetched from the LFS media endpoint and not
# from the ordinary raw one, which for a zoo file answers with a git-lfs
# pointer — 132 bytes of text that loads as a corrupt model.
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx")
MODEL_NAME = "text_detection_en_ppocrv3.onnx"

# Enough to reject a truncated download or an LFS pointer served in place of
# the file. See background_remover for why this is worth a constant.
MODEL_MIN_BYTES = 2 * 1024 * 1024

# Beside the app's own code, in the directory updater.KEEP_ACROSS_UPDATES
# already names, so a patch replacing app/ wholesale does not throw it away.
MODEL_DIR = Path(__file__).resolve().parent / "models"

# What the network is fed. DB is fully convolutional but its input has to be a
# multiple of 32, and it is calibrated around this size; a frame is letterboxed
# into it rather than squashed, so the aspect ratio the text was set at
# survives (see _detect).
INPUT_SIZE = 736

# PP-OCRv3's own preprocessing: ImageNet channel means, and a plain 1/255
# scale.
MEAN = (122.67891434, 116.66876762, 104.00698793)
SCALE = 1.0 / 255.0

# DB's two thresholds and its dilation. The defaults from OpenCV's own demo,
# and left there deliberately: they were measured on this app's footage as
# finding every caption with no false box on a busy frame, and a value tuned
# past that would be tuned to one video.
BINARY_THRESHOLD = 0.3
POLYGON_THRESHOLD = 0.5
UNCLIP_RATIO = 2.0
MAX_CANDIDATES = 200

_model = None
_load_failed = False
# One thread builds the model; the rest wait rather than racing to download the
# same file into the same place. Same arrangement as background_remover's.
_load_lock = threading.Lock()


def model_path() -> Path:
    return MODEL_DIR / MODEL_NAME


def _download(dest: Path) -> bool:
    """
    Fetches the model once, through a .part file so an interrupted download can
    never be mistaken for a finished one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("text detection: downloading %s (~2 MB, once)", MODEL_NAME)
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Thumbnail-Maker"})
        with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as f:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                f.write(block)
        if tmp.stat().st_size < MODEL_MIN_BYTES:
            raise OSError(f"downloaded only {tmp.stat().st_size} bytes")
        os.replace(tmp, dest)
        return True
    except Exception as e:
        log.warning("text detection: could not download %s (%s)", MODEL_NAME, e)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _ensure_loaded(download: bool = True) -> bool:
    global _model, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False

    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False

        path = model_path()
        if not path.exists():
            if not download or not _download(path):
                _load_failed = True
                return False
        try:
            model = cv2.dnn_TextDetectionModel_DB(str(path))
            model.setBinaryThreshold(BINARY_THRESHOLD)
            model.setPolygonThreshold(POLYGON_THRESHOLD)
            model.setUnclipRatio(UNCLIP_RATIO)
            model.setMaxCandidates(MAX_CANDIDATES)
            model.setInputParams(SCALE, (INPUT_SIZE, INPUT_SIZE), MEAN)
            _model = model
            return True
        except Exception as e:
            log.warning("text detection: model at %s would not load (%s)", path, e)
            _load_failed = True
            return False


def preload(download: bool = True) -> bool:
    """
    Build the model now rather than inside the first frame that needs it.
    Called at server startup, alongside the other three. Returns whether text
    detection is available at all.
    """
    return _ensure_loaded(download)


def is_available() -> bool:
    """
    Whether text detection can run. False means subtitle removal does nothing
    at all — see the header of subtitle_remover for why there is no classical
    fallback.
    """
    return _model is not None or not _load_failed


def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """
    The frame fitted into the square the network takes, with grey padding
    rather than a squash, plus what it takes to map a box back.

    Padding, not resizing to the square, because these frames are 9:16 and the
    network's input is 1:1: squashed to fit, a line of caption is stretched to
    two and a half times its width and stops being the shape of the type the
    model was trained on.

    On this app's own footage it makes no measurable difference — captions came
    out 39 of 40 either way, and the small corner credit 37 of 40 either way.
    It is kept because it is the version whose reasoning holds for footage that
    is not 9:16, where the squash would be more violent than anything measured
    here, and because it costs one resize.
    """
    h, w = image.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=image.dtype)
    off_x, off_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas, scale, off_x, off_y


def detect(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Every region of `image` the model reads as text, as (x, y, w, h) boxes in
    the image's own coordinates.

    Empty when the model is unavailable, which is the same answer it gives for
    a frame with no text in it — the caller treats both as "nothing to remove",
    and the reason it is unavailable has already been logged once.
    """
    if not _ensure_loaded():
        return []

    canvas, scale, off_x, off_y = _letterbox(image)
    try:
        polygons, _ = _model.detect(canvas)
    except Exception as e:
        log.warning("text detection: detect failed (%s)", e)
        return []

    h, w = image.shape[:2]
    boxes = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        # Back out of the letterbox, then out of the resize.
        x0, x1 = (x0 - off_x) / scale, (x1 - off_x) / scale
        y0, y1 = (y0 - off_y) / scale, (y1 - off_y) / scale
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes
