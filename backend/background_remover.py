"""
background_remover.py — Cutting the person out of a frame, so a channel can
put them on a backdrop of its own.

One channel (Laugh Society 1) does not use the photograph as its background at
all: the thumbnail is a fixed stage image with the subject standing on it. That
needs an alpha channel, which nothing else in this pipeline produces — the
whole of the rest of it works in three-channel BGR, from the crop through the
restoration to the export.

## Two models, because the question has two halves

WHERE the person is, and WHERE EXACTLY their hair ends, are different
questions, and the model that is good at one is bad at the other.

U2-Net answers the second. It is a SALIENT-OBJECT network: it finds what the
picture is about and traces it to the strand. What it does not have is any
notion of a person — handed a lit stage it returns the spotlight too, because
a spotlight is unquestionably what that picture is about. Measured on real
footage: a stage lamp beside the performer came back joined to his forearm
across a hundred pixels of shared edge, which no amount of splitting the mask
into regions can undo, because it genuinely is one region.

So a person-segmentation network answers the first. torchvision's LR-ASPP,
trained on COCO/VOC, is asked only which pixels are a PERSON, and its answer
gates U2-Net's. It is coarse — it will not find hair — but it does not have to:
it is dilated and used as "may this pixel belong to the subject at all", and
U2-Net decides the edge inside it. Measured on the same footage: the lamp goes,
and outstretched hands and a held microphone stay (99.7% and 97.8% of the
matte kept on the two frames that have them, against 87% and 80% for the
classical segmenter, which ate the hands).

It costs 12 MB of weights and about 30ms a frame, and torch is already here.

## And why the saliency question gets asked twice

U2-Net resizes whatever it is handed to 320x320. On the wide stage shots this
channel is for, that is the difference between a cutout and a shirt: a
performer 350px across in a 1920px frame arrives at the network sixty pixels
wide, and sixty pixels is not enough to have a head in it. Measured on this
footage, the shirt came back at a saliency of 1.00 and the head, the hair, the
trousers and the strip of chest behind the microphone stand came back at 0.00
— not faint, absent. Nothing downstream can recover values like those. The
person prior cannot either: it is a gate, and a gate only takes away.

So the pass is taken a second time on the subject's own region — the part of
the prior the detected face is in, padded — and the two are combined with a
maximum. Cropped that way the same performer arrives nearly full height and
comes back whole. Across this video's stage frames it takes the ones that lost
the head from five in sixty-seven to none, for one more inference (~300ms,
paid once per photo and cached with it). See `_saliency`.

## And so, in the end, does the person question

For the same reason, discovered later and the harder way. The prior runs at
PERSON_INPUT_SIDE over the whole frame, and a coarse prior is fine for "is
there a person here" and useless for "is this two-pixel gap between his knees a
person". So the stand behind a performer, the strip of poster wall showing
between his legs, the stage floor under him: salient to U2-Net, joined to the
silhouette, and inside the prior's own dilation, which is all three of the ways
a pixel can be dropped, failing at once. It reaches the canvas as a slab of
scenery growing out of the subject, which is the one artefact a user reports
unprompted.

So the prior is asked a second time on the same crop, combined the same way,
for about 30ms on top of the 1.3 seconds the two saliency passes already cost.
See `_person_prior` — and PERSON_RECT_FRAC, which is the other half of that
story and was the larger half.

## Which saliency model, and why it is not a pip package

U2-Net, the salient-object network `rembg` is built around, run directly
through onnxruntime rather than through rembg itself. rembg pins old numpy,
opencv and pillow in its own requirements and downgrades all three on a plain
install — the same trap simple-lama-inpainting sets, documented in
requirements.txt — and everything it would give us beyond the model is a
resize, a normalise and a transpose, which are the forty lines below. So the
dependency is onnxruntime alone, and the weights are the one file
(`models/u2net.onnx`, ~176 MB) fetched from rembg's own release.

## What happens when it is not there

The same thing that happens when LaMa is not there: something worse, rather
than nothing. `region_segmenter.foreground_mask_grabcut` already separates a
person from a background using the detected face as a seed, and it needs no
model and no GPU. It is visibly coarser — GrabCut has no idea what hair is —
but a cutout with a rough edge is a thumbnail, and no cutout is a channel that
does not work. Which of the two ran is reported, never guessed at.

## Where the mask is measured

On the FINISHED window — the restored, graded, cropped 1280x720 canvas the user
is looking at — and not on the raw frame before it.

That is the opposite of the order a reader might expect ("cut it out, then
enhance it"), and it is deliberate. The editing pipeline is defined over BGR
frames end to end: it measures its grading parameters from the window's own
pixels, denoises, deblurs and sharpens, and there is nowhere in it for an alpha
channel to travel. Segmenting first would mean either carrying a fourth channel
through every stage that has no concept of one, or grading a frame with a hole
punched in it — where the black void is then part of what the black-point,
contrast and clarity passes measure themselves against. Segmenting last leaves
the pipeline exactly as it is for every channel, and hands the matting a
cleaner, sharper, better-exposed image than the raw frame was, which is the
input it wants.
"""

import os
import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from region_segmenter import feather_mask, foreground_mask_grabcut, person_region_rect

log = logging.getLogger("uvicorn.error")  # see the note in face_restorer.py

# rembg's own published weights, from the release its downloader points at.
# Named here rather than left to a library so that the installer, a patch and
# the running app all fetch the same file from the same place — see
# installer/README.md and the `models` block in patch.json.
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MODEL_NAME = "u2net.onnx"

# Roughly what the file weighs. Used only to reject a truncated download — a
# half-written .onnx loads as a corrupt model with a stack trace nobody can act
# on, where "it downloaded short, delete it and try again" is actionable.
MODEL_MIN_BYTES = 100 * 1024 * 1024

# Beside the app's own code, NOT in the user's home cache, for the reason
# installer/README.md gives about torch's: an uninstall that deletes the
# install folder should take everything it downloaded with it and nothing
# else. `backend/models` is named in updater.KEEP_ACROSS_UPDATES so a patch
# replacing app/ wholesale does not throw 176 MB away.
MODEL_DIR = Path(__file__).resolve().parent / "models"

# What U2-Net was trained at. The network is fully convolutional and will
# accept other sizes, but its saliency is calibrated for this one.
INPUT_SIZE = 320

# ImageNet channel statistics, in RGB order — U2-Net's own preprocessing,
# applied after the image is scaled by its maximum rather than by 255. The
# division by max is not a mistake copied from somewhere: it is what the
# reference implementation does, and a mask produced by dividing by 255
# instead comes out measurably flatter on a dark frame.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# How hard the mask's own edge is softened before it is used as alpha. Small:
# this is a matte on a foreground the viewer is looking straight at, so the aim
# is to take the staircase off a boundary the network already placed well, not
# to blur the person's outline into the backdrop.
ALPHA_FEATHER_PX = 1.2

# Below this the network's output is background, above it foreground, with the
# band between the two left as partial alpha — which is where hair and motion
# blur live. Rescaling that band to the full 0-255 range (rather than keeping
# U2-Net's raw output, which is confident almost everywhere) is what stops the
# cutout carrying a faint rectangular veil of near-black background around it.
ALPHA_FLOOR, ALPHA_CEILING = 0.12, 0.85

# Alpha at or above which the first pass counts as "the subject is somewhere
# around here", when the person prior found nobody and its own answer has to
# stand in as the prior for the refinement crop (see _saliency). Half, because
# the question is only where to point a crop: below it the map is as likely to
# be a bright patch of curtain as a person.
ALPHA_PRIOR_MIN = 128

# The person-segmentation half (see this module's header). Its weights come
# down through torch hub into TORCH_HOME, the same place the inpainter's do —
# see MODELS_DIRNAME in installer/engine.py.
PERSON_CLASS = 15          # "person" in the VOC class list these weights are trained on
PERSON_INPUT_SIDE = 520    # longest side the prior is computed at; it only has to be roughly right

# How far the person prior is grown before it is used as a gate.
#
# It is a coarse mask being used to answer a coarse question — "may this pixel
# be part of the subject at all" — and everything fine about the edge comes
# from U2-Net inside it. So the dilation only has to cover what a semantic
# segmentation routinely misses at a person's outline: hair, a raised finger,
# the rim of a shoulder against a dark background.
#
# Bounded by what it must NOT reach. Measured on the footage this was tuned
# against: at 3% a stage lamp standing just behind the performer's shoulder was
# still inside the dilation and came through as an arc of itself, and the
# region test could not drop it because the dilation had joined it to him. At
# 2% the lamp is outside and the hair is still inside, which is the whole
# window this number has to fit through.
PERSON_GATE_FRAC = 0.02

# How far the prior reaches INSIDE the body region — the column around and
# below the face where a semantic segmentation is least trustworthy about a
# person's own outline (see person_region_rect, and the note in
# _isolate_subject).
#
# This used to be a blanket pass: everything inside that column was immune to
# the gate. The reason was real — the prior dropped a performer's light
# trousers against a dark stage and the gate cut him off at the waist — but the
# remedy was far larger than the problem. The column runs from just above the
# face to the BOTTOM OF THE FRAME, so it pardons not only the trousers but the
# stage floor behind them, the stand between his feet and the strip of poster
# wall showing between his knees. Measured on the stage set this was tuned
# against: of the scenery that survived the gate, on all twenty frames, 100% of
# it was inside this rectangle.
#
# So the exemption is local instead of blanket. The prior's failure is at the
# EDGES of a body it half-recognises, which is a local failure; a wider
# dilation inside the column answers it, and a blanket pass answers something
# nobody asked. At 3% the trousers are still safe — the body measure is
# unchanged or better on every one of the twenty frames — and the scenery left
# inside the silhouette falls from 4.0% of the cutout to 1.9%.
#
# Dropping the exemption altogether measured better still (1.2%, and no frame
# over 5%). It is kept anyway, at this width, because the failure it guards
# against was observed and one stage set cannot prove it impossible on another.
# That is the whole of the case for the 0.7 points; a reader with a second set
# in hand should feel free to revisit it.
PERSON_RECT_FRAC = 0.03

# How much room is left around the subject when the saliency pass is taken a
# second time, on them alone — as a fraction of the frame's long side.
#
# Enough that the network is still looking at a person standing in a picture
# rather than at a picture OF a person: cropped to the silhouette exactly,
# U2-Net has no background left to call background, and the mask spreads into
# whatever is touching the subject at the edges of the crop.
SUBJECT_CROP_PAD_FRAC = 0.06

# ...and how much smaller than the frame that crop has to be for the second
# pass to be worth taking. At 0.85 a subject filling most of the frame is
# already being seen at full size and the pass would be the same computation
# twice; a stage two-shot or an audience cutaway, where the prior spans
# everything, falls out here too.
SUBJECT_CROP_MAX_FRAC = 0.85

_person_model = None
_person_failed = False

_session = None
_load_failed = False
# Serialises the model LOAD, not inference: two requests arriving together on
# an install that has yet to download the weights would otherwise both start
# the same 176 MB download into the same file.
_load_lock = threading.Lock()


def model_path() -> Path:
    return MODEL_DIR / MODEL_NAME


def _download(dest: Path) -> bool:
    """
    Fetches the weights once, through a .part file so an interrupted download
    can never be mistaken for a finished one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("background removal: downloading %s (~176 MB, once)", MODEL_NAME)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=300) as response, open(tmp, "wb") as f:
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
        log.warning("background removal: could not download %s (%s)", MODEL_NAME, e)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _ensure_loaded(download: bool = True) -> bool:
    global _session, _load_failed
    if _session is not None:
        return True
    if _load_failed:
        return False

    with _load_lock:
        if _session is not None:
            return True
        if _load_failed:
            return False
        try:
            import onnxruntime
        except Exception as e:
            log.warning("background removal: onnxruntime unavailable, falling back to "
                        "the classical segmenter (%s)", e)
            _load_failed = True
            return False

        path = model_path()
        if not path.exists() and not (download and _download(path)):
            _load_failed = True
            return False

        try:
            # CPU explicitly. The network is a few hundred milliseconds at
            # 320x320 on any CPU this app runs on, and the GPU is where the
            # restoration model lives. A second session on it buys nothing
            # measurable and competes for VRAM with the one stage of the
            # pipeline that is genuinely tight for it (see vram.py).
            _session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"])
            log.info("background removal: U2-Net ready (%s)", path)
            return True
        except Exception as e:
            log.warning("background removal: %s could not be loaded, falling back to "
                        "the classical segmenter (%s)", MODEL_NAME, e)
            _load_failed = True
            return False


def preload(download: bool = True) -> bool:
    """
    Loads (and, if missing, fetches) both models now rather than inside the
    user's first cutout. Called from the server's startup warm-up and by the
    installer, exactly as face_restorer.preload and inpainter.preload are.

    Answers for the saliency model alone, because that is the one the feature
    cannot degrade past: without the person prior a cutout comes out with some
    of the scenery in it, which is worse but is still a cutout.
    """
    ready = _ensure_loaded(download=download)
    if download:
        _ensure_person_model()
    return ready


def is_available() -> bool:
    """Whether the network is loaded or still loadable — False means every call falls back."""
    return _session is not None or not _load_failed


def backend_name() -> str:
    """What produced the last mask, for the log and the response header."""
    saliency = "u2net" if _session is not None else "grabcut"
    return f"{saliency}+person" if _person_model is not None else saliency


def _ensure_person_model():
    """
    torchvision's LR-ASPP, loaded once. None when torch is not usable here, in
    which case the cutout is U2-Net's alone — which is what it was before this
    existed, scenery and all.
    """
    global _person_model, _person_failed
    if _person_model is not None or _person_failed:
        return _person_model
    with _load_lock:
        if _person_model is not None or _person_failed:
            return _person_model
        try:
            import torch
            from torchvision.models.segmentation import (
                lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights,
            )
            model = lraspp_mobilenet_v3_large(
                weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).eval()
            _person_model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            log.info("background removal: person segmentation ready (%s)",
                     next(_person_model.parameters()).device)
        except Exception as e:
            log.warning("background removal: person segmentation unavailable, the cutout "
                        "will be saliency alone (%s)", e)
            _person_failed = True
    return _person_model


def person_mask(image: np.ndarray) -> np.ndarray | None:
    """
    Which pixels of a BGR frame are a person, coarsely. None when the model is
    not available.

    Computed at a fixed size rather than the frame's own: this is a prior that
    gets dilated by 3% of the frame before anything reads it, so resolution
    past a few hundred pixels buys nothing but time.
    """
    model = _ensure_person_model()
    if model is None:
        return None
    try:
        import torch
        h, w = image.shape[:2]
        scale = PERSON_INPUT_SIDE / max(h, w)
        small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1))[None].to(
            next(model.parameters()).device)
        with torch.no_grad():
            label = model(tensor)["out"][0].argmax(0).byte().cpu().numpy()
        return cv2.resize((label == PERSON_CLASS).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        log.warning("background removal: person segmentation failed (%s)", e)
        return None


def _person_prior(image: np.ndarray,
                  face: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """
    Which pixels are a person — asked twice, at a resolution that can see the
    gaps between two legs.

    person_mask runs at PERSON_INPUT_SIDE over the WHOLE frame, and on the wide
    stage shots this channel exists for that is short by an order of magnitude:
    a performer 350px wide in a 2048px frame reaches the network about ninety
    pixels tall, and the gap between his knees is two pixels across. A prior
    that cannot resolve that gap cannot be asked to remove what is standing in
    it — and nothing downstream can either, because the region split sees one
    blob (it IS one blob) and the gate sees a person pixel within reach (there
    is one, on both sides).

    That is the failure a user reports as "it took the background with it", and
    it survived every earlier attempt at the gate untouched: a stand behind a
    performer is salient, joined to the silhouette, and inside the prior's own
    dilation, so not one of the three things that can drop a pixel applied to
    it.

    So the pass is taken again on the subject's own region, reusing the crop
    _saliency already computes for the saliency network and combined the same
    way. It costs one more LR-ASPP inference, about 30ms, paid once per photo
    and cached with it. Measured on twenty frames of one stage set: the scenery
    left inside the silhouette falls from 4.0% of the cutout to 1.9%, the worst
    frame from 17.3% to 5.5%, and the frames carrying more than a twentieth of
    themselves in stage from six to one — with the body measure unchanged or
    better on every single frame, because a sharper statement of where a person
    is takes nothing away from the person.

    Combined with a maximum rather than replacing the coarse pass, so this can
    only ever ADD: whatever the whole-frame pass found outside the crop — a
    second performer, a hand thrown out past the box — is still in the answer.
    """
    coarse = person_mask(image)
    if coarse is None or not coarse.any():
        return coarse

    box = _subject_crop(image.shape, coarse, face)
    if box is None:
        return coarse
    x1, y1, x2, y2 = box
    close = person_mask(image[y1:y2, x1:x2])
    if close is None:
        return coarse

    out = coarse.copy()
    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], close)
    return out


def _run_model(image: np.ndarray) -> np.ndarray | None:
    """U2-Net's saliency map for a BGR image, at the image's own size. None if it could not run."""
    if not _ensure_loaded():
        return None
    try:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

        x = small.astype(np.float32)
        peak = float(x.max())
        x = x / peak if peak > 0 else x
        x = (x - MEAN) / STD
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        name = _session.get_inputs()[0].name
        # The network has several side outputs, deep-supervision style; the
        # first is the fused one and the only one worth reading.
        pred = _session.run(None, {name: x})[0][0, 0]

        low, high = float(pred.min()), float(pred.max())
        pred = (pred - low) / (high - low) if high > low else np.zeros_like(pred)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        log.warning("background removal: inference failed, falling back to the "
                    "classical segmenter (%s)", e)
        return None


def _stretch(pred: np.ndarray) -> np.ndarray:
    """The saliency map as 0-255 alpha, with the confident ends pushed to the ends."""
    clipped = np.clip((pred - ALPHA_FLOOR) / (ALPHA_CEILING - ALPHA_FLOOR), 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def _subject_crop(shape: tuple, people: np.ndarray,
                  face: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    """
    The part of the frame THIS subject is in, padded — or None when that is
    most of the frame anyway.

    The subject's own part of the prior, not the prior as a whole, and on the
    shots this matters most for those are very different rectangles. A wide
    house shot has an audience across the bottom of the frame: the prior marks
    every one of them, its overall box is the whole picture, and a second pass
    taken on that is the first pass again. Narrowed to the region the face is
    in, the same frame gives a crop around one performer.
    """
    ys, xs = np.where(people > 0)
    if not len(ys):
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(people, 8)
    mine = _region_with_face(labels, count, face)
    if not mine and count > 1:
        # No face to point at the subject, so take the biggest person in the
        # frame instead of all of them.
        #
        # Without this the crop is skipped exactly when it matters most.
        # `_region_with_face` answers 0 whenever there is no face, the box is
        # then drawn round every person the prior found — performer and
        # audience alike — it spans the picture, and it fails the size test
        # below. And a face is precisely what the detector does NOT return when
        # the subject is small and far away, which is the same shot whose
        # subject most needs a second pass.
        #
        # Measured on this footage before the fallback existed: the frames that
        # fell through gave the network 39 to 53 pixels across a subject that
        # is then drawn 662px tall on the canvas — about one mask pixel for
        # every fourteen on screen, which is a visibly stepped outline. The
        # frames that did crop got 300.
        #
        # The largest region is the right guess for the same reason the framing
        # picks a medium shot: this channel's subject is a performer on a
        # stage, and an audience is many small regions rather than one big one.
        # It is only a guess, but a crop round the wrong person is still a crop
        # at full resolution, and everything downstream — the person gate, the
        # region split, the subject test — still runs on the result. The pass
        # is also combined with a maximum, so a wrong guess can only fail to
        # add rather than take anything away.
        mine = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if mine:
        ys, xs = np.where(labels == mine)
    h, w = shape[:2]
    pad = int(SUBJECT_CROP_PAD_FRAC * max(h, w))
    x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    if (x2 - x1) * (y2 - y1) > SUBJECT_CROP_MAX_FRAC * w * h:
        return None
    return (x1, y1, x2, y2)


def _saliency(image: np.ndarray, people: np.ndarray | None,
              face: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """
    U2-Net's alpha for the frame, measured where the network can actually see
    the subject.

    U2-Net resizes whatever it is handed to 320x320 (see INPUT_SIZE). That is
    fine when the subject is what the frame is mostly of, and it is the whole
    problem when they are not: on the wide stage shots this channel exists for,
    a performer 350px across in a 1920px frame arrives at the network about
    sixty pixels wide, and sixty pixels is not enough to have a head in it.

    What comes back in that case is not a rough person — it is the LIT PART of
    one. Measured on this footage: the pale shirt came back whole and
    confident, and the head, the hair, the dark trousers and the strip of chest
    behind the microphone stand came back as background, at a raw saliency of
    0.00 where the shirt was 1.00. No threshold reaches values like those, and
    the person prior cannot put them back either — it is a gate, and a gate
    only takes away. The cutout that reached the canvas was a shirt with nobody
    in it.

    So the pass is taken again on the subject's own region, which is the one
    thing that changes the network's answer: the same performer, cropped to
    where the prior says the people are, arrives at 320x320 nearly full height,
    and comes back complete — head, arms, legs, and the bar across the chest
    filled in.

    The two are combined with a maximum rather than the crop replacing the
    frame, so this can only ever ADD. Anything the whole-frame pass found
    outside the crop — a hand thrown out past where the prior saw a person — is
    still in the answer, and every pixel both passes agree on keeps the higher
    confidence of the two. What the crop adds is still subject to everything
    downstream: the person gate, the region split and the subject test all run
    on the result exactly as before.
    """
    pred = _run_model(image)
    if pred is None:
        return None
    alpha = _stretch(pred)

    # Which mask says where to look for the second pass. The person prior when
    # it found anything, and the first pass's own answer when it did not.
    #
    # That fallback is not a nicety. The prior is DeepLab at PERSON_INPUT_SIDE
    # over the whole frame, so a performer 150px wide in a 1920px frame reaches
    # it about forty pixels across, and it comes back EMPTY — measured, 0.0%
    # of the frame on the wide shots in this footage. Returning here on that
    # basis skipped the refinement pass exactly on the frames whose subject is
    # smallest, which are the frames that need it most: they were left with 39
    # to 53 mask pixels across a figure then drawn 662px tall on the canvas,
    # about one mask pixel per fourteen on screen, and the outline arrived
    # visibly stepped.
    #
    # U2-Net's own first pass is a usable stand-in for the prior here. It is
    # coarse — that is the whole problem being solved — but it is coarse about
    # WHERE, and where is all a crop needs. Everything downstream still runs on
    # the result, and the two passes are combined with a maximum, so a crop
    # taken around the wrong thing can only fail to add.
    prior = people if (people is not None and people.any()) else None
    if prior is None:
        prior = (alpha >= ALPHA_PRIOR_MIN).astype(np.uint8)
        if not prior.any():
            return alpha

    box = _subject_crop(image.shape, prior, face)
    if box is None:
        return alpha
    x1, y1, x2, y2 = box
    close = _run_model(image[y1:y2, x1:x2])
    if close is None:
        return alpha

    near = np.zeros_like(alpha)
    near[y1:y2, x1:x2] = _stretch(close)
    return np.maximum(alpha, near)


# Alpha at or above which a pixel counts as solidly part of a subject when
# the mask is being split into subjects. Half, deliberately: this is a question
# about which blob is which person, and an edge pixel belongs to whichever blob
# it is on the edge OF.
SUBJECT_ALPHA_MIN = 128

# The widest bridge between the subject and something else that gets cut, as a
# fraction of the frame's short side.
#
# Splitting the mask into regions and keeping the one with the face in it does
# nothing at all when the scenery is TOUCHING the person, which on a stage is
# most of the time: a spotlight beside a performer's head, a mic stand across a
# shoulder, and the whole thing labels as one region. Measured on real footage,
# a stage lamp joined the subject through a few dozen pixels of shared edge —
# so the regions are found on an OPENED mask, where a join that thin has
# already been eroded away and the lamp is its own region again.
#
# Small on purpose. This is a bridge-cutter, not a de-noiser: at much more than
# this it starts severing the subject's own thin parts — a raised arm, a
# microphone held away from the body — into separate regions that then get
# thrown away with the scenery.
SUBJECT_OPEN_FRAC = 0.012

# ...and how far the kept region is grown back before it is used to gate the
# alpha, as a fraction of the same short side.
#
# Wider than the opening, so everything the opening ate off the subject's own
# edge comes back — hair, fingers, the rim of a shoulder — and narrow enough
# that what was cut away stays cut away. It is also what removes the faint
# leftovers the region test never sees: a burned-in watermark, a stage light's
# glow, anything the network marked weakly enough to sit below the region
# threshold but above nothing. Those are not connected to the subject and are
# not within reach of them, so the gate clears them.
SUBJECT_GATE_FRAC = 0.02

# How far from the subject a separate region may sit and still be kept when a
# frame is cut WITH ITS PROPS, as a fraction of the short side.
#
# The exception this exists for is a ventriloquist. The whole pipeline above is
# built on one sentence — the thumbnail is a person, and everything that is not
# a person is the stage — and for that act the sentence is false: the dummy is
# not a person by any measure the person prior applies, it is the other half of
# the act, and a cutout that removes it removes the subject of the video. The
# same is true of anything else held out from the body far enough for the
# region split to see daylight between the two: an instrument, a mask on a
# stick, a second puppet.
#
# So `keep_props` turns the person prior off for that frame and lets the region
# split keep the subject's own region plus everything within this of it. Three
# percent of the short side is about twenty pixels on the frames this runs on:
# enough to bridge the gap the opening leaves between a hand and what is in it,
# and nowhere near enough to reach the stage lighting, which is what the prior
# was removing.
#
# It is off by default and per frame, because it is a licence rather than an
# improvement, and the size of the licence has now been measured. `keep_props`
# does not merely widen the region test — it turns the person prior OFF, and
# the prior is the only thing here that knows a spotlight is not a person. Over
# the twenty frames of one stage set (JAM Comedy, a 2048px source), the share
# of the cutout that is not the subject runs at 9.1% with props kept against
# 5.4% without, the worst frame at 49.8% against 15.0%, and four frames carry
# more than a tenth of themselves in stage against two — one of them a slab of
# the poster wall with no person in it, another an audience member's head
# standing on the performer's legs.
#
# What it did NOT cost on that footage is the thing it exists for: not one
# microphone and not one hand was lost with the prior on, because a mic held to
# the mouth sits well inside PERSON_GATE_FRAC of the person holding it. The
# licence is for the act whose other half is genuinely a separate object at
# arm's length — the dummy, the instrument, the mask on a stick — and it is
# the user who says which frame that is, one frame at a time (see CUTOUT_MODES
# in frontend/js/editor.js).
PROP_REACH_FRAC = 0.03


def _region_with_face(labels: np.ndarray, count: int,
                      face: tuple[int, int, int, int] | None) -> int:
    """
    Which labelled region the subject's face is in, or 0 for "no answer".

    Decided by a vote over the face BOX rather than by the pixel at its centre.
    A centre pixel can land in a gap — a fringe, an open mouth, a pair of
    glasses — and one unlucky pixel would then name the wrong region entirely.
    """
    if face is None:
        return 0
    x, y, w, h = face
    ih, iw = labels.shape[:2]
    box = labels[max(0, y):min(ih, y + h), max(0, x):min(iw, x + w)]
    if not box.size:
        return 0
    votes = np.bincount(box.ravel(), minlength=count)
    votes[0] = 0   # background is not a candidate however much of the box it covers
    return int(votes.argmax()) if votes.max() > 0 else 0


def _odd(value: int) -> int:
    """The nearest odd number at or above `value`, and at least 3 — a kernel size."""
    return max(3, int(value) | 1)


def _isolate_subject(alpha: np.ndarray, face: tuple[int, int, int, int] | None,
                     people: np.ndarray | None = None,
                     keep_props: bool = False) -> np.ndarray:
    """
    Keeps the subject and clears everything else — the person the frame is
    about, and none of the stage they are standing on.

    U2-Net is a SALIENT-OBJECT model, not a person detector. Handed a two-shot
    it returns both people, because both of them are what the picture is about;
    handed a lit stage it returns the lamp as well, for the same reason. Both
    are the right answer to the question it was asked and the wrong one for a
    channel whose thumbnail is one person on a backdrop of its own.

    The frame already knows which figure it is about: the face its whole
    framing was computed around. So the mask is split into regions, the one
    that face sits in is the subject, and the rest go.

    Split on an OPENED copy of the mask (see SUBJECT_OPEN_FRAC), because on a
    stage the scenery is usually touching the person and an un-opened mask has
    them as one region. The region that survives is then grown back (see
    SUBJECT_GATE_FRAC) and used as a gate on the ORIGINAL alpha, so the
    subject's soft edge is the network's and not this function's — the opening
    and the dilation decide only WHAT is kept, never how its edge looks.

    Which region is the subject's is decided by a vote over the face BOX rather
    than by the pixel at its centre. A centre pixel can land in a gap — a
    fringe, an open mouth, a pair of glasses — and one unlucky pixel would then
    throw the whole subject away in favour of the lamp beside them.

    With no face to go on, the largest region is kept, which is the same guess
    the framing itself makes about who the subject is.

    `keep_props` is the exception to all of it, and it is asked for a frame at
    a time by the user rather than decided here (see PROP_REACH_FRAC). It turns
    the person prior off and widens the region test to everything sitting near
    the subject, for the acts where the thing the video is about is not a
    person — a ventriloquist's dummy above all, which no person model has any
    reason to call anybody and which every rule above is therefore built to
    delete.
    """
    short = min(alpha.shape[:2])

    # What is a PERSON at all, before anything asks which person. This is what
    # takes the stage out of the cutout — a spotlight is salient and is not
    # anybody, and no amount of reasoning about the shape of the mask can tell
    # the two apart once they touch. Dilated, because the prior is coarse and
    # what it must not do is trim the subject's own edge; U2-Net decides that,
    # inside here.
    #
    # UNIONED with a plain geometric body region, and that is not belt and
    # braces — it is the whole reason this is usable. A semantic segmentation
    # is confident about a person's outline and unreliable about the parts of
    # them that do not look like the training data: measured on this footage,
    # it dropped a performer's light trousers against a dark stage entirely,
    # and the gate then cut him off at the waist. A third of his silhouette,
    # gone, for a prior that was only ever meant to remove the scenery.
    #
    # The body region is where a person MUST be if their face is here: a column
    # around and below it, down to the bottom of the frame (see
    # region_segmenter.person_region_rect). Inside it the prior is given a WIDER
    # reach rather than being switched off, and that distinction is the whole of
    # this paragraph — see PERSON_RECT_FRAC. A torso, a waistband or a hand held
    # at the hip is safe because it is within that wider reach of something the
    # model did recognise; the stage floor two metres behind him is not, and
    # used to be pardoned by the very same rule. Outside the column the prior
    # still rules at its own width, which is where the stage lighting always is
    # — and where an outstretched arm is too, which is exactly the case the
    # model IS reliable on.
    if people is not None and people.any() and not keep_props:
        gate_k = _odd(short * PERSON_GATE_FRAC)
        reachable = cv2.dilate(people, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gate_k, gate_k)))
        if face is not None:
            bx, by, bw, bh = person_region_rect(alpha.shape, face)
            rect_k = _odd(short * PERSON_RECT_FRAC)
            wide = cv2.dilate(people, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rect_k, rect_k)))
            reachable = reachable.copy()
            reachable[by:by + bh, bx:bx + bw] = np.maximum(
                reachable[by:by + bh, bx:bx + bw], wide[by:by + bh, bx:bx + bw])
        alpha = np.where(reachable > 0, alpha, 0).astype(np.uint8)

    solid = (alpha >= SUBJECT_ALPHA_MIN).astype(np.uint8)

    open_k = _odd(short * SUBJECT_OPEN_FRAC)
    opened = cv2.morphologyEx(
        solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    # A subject thin enough for the opening to erase entirely — a distant
    # figure, an arm and nothing else — is one this cannot help with, and
    # gating on an empty region would return an empty cutout.
    if not opened.any():
        opened = solid

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return alpha

    keep = _region_with_face(labels, count, face)
    if keep == 0:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    subject = (labels == keep).astype(np.uint8)
    if keep_props:
        # Everything the subject is holding, sitting beside, or joined to
        # through a gap the opening cut. Found by asking which regions fall
        # inside the subject's own reach rather than by measuring distances
        # pair by pair: one dilation and one pass of the labels answers it for
        # all of them at once.
        near = cv2.dilate(subject, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(short * PROP_REACH_FRAC),) * 2))
        for label in np.unique(labels[near > 0]):
            if label:
                subject |= (labels == label).astype(np.uint8)

    gate_k = _odd(short * SUBJECT_GATE_FRAC)
    gate = cv2.dilate(subject, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gate_k, gate_k)))
    return np.where(gate > 0, alpha, 0).astype(np.uint8)


def alpha_for(image: np.ndarray, face: tuple[int, int, int, int] | None = None,
              keep_props: bool = False) -> np.ndarray:
    """
    The chosen person's alpha channel for a BGR frame: 255 where they are, 0
    where everything else is, and the values between reserved for the edge.

    `face` says which person that is (see _isolate_subject), and is also what
    seeds the fallback segmenter. Without one, and with neither network,
    everything is foreground rather than nothing: an uncut frame on the
    backdrop is wrong in a way the user can see and fix, where an empty frame
    looks like the app lost their picture.

    Isolated BEFORE the feather, never after. Feathering first would blur the
    two subjects' edges toward each other, and a region test run on that can
    find them joined by a corridor of faint alpha that is not in either of
    them.
    """
    people = _person_prior(image, face)
    alpha = _saliency(image, people, face)
    if alpha is not None:
        return feather_mask(_isolate_subject(alpha, face, people, keep_props),
                            radius=ALPHA_FEATHER_PX)

    if face is None and people is None:
        return np.full(image.shape[:2], 255, dtype=np.uint8)
    if face is None:
        return feather_mask(people * 255, radius=ALPHA_FEATHER_PX)
    # GrabCut is seeded from this same face, but its seed rectangle reaches
    # three face-widths wide and down to the bottom of the frame, so it picks
    # up whoever is standing next to the subject just as readily.
    return feather_mask(
        _isolate_subject(foreground_mask_grabcut(image, face), face, people, keep_props),
        radius=ALPHA_FEATHER_PX)


# How much of the frame the subject is allowed to be. A mask covering
# essentially all of it means nothing was found to remove; one covering
# essentially none means no subject was found. Neither is a cutout, and both
# are better reported than silently composited.
MIN_SUBJECT_FRAC, MAX_SUBJECT_FRAC = 0.01, 0.995

# Alpha at or above which a pixel counts as part of the subject when the cutout
# is trimmed to its own bounds. Above the feather's own faintest values, so a
# soft edge does not enlarge the box it is measured into.
TRIM_ALPHA_MIN = 12


def cutout(image: np.ndarray, face: tuple[int, int, int, int] | None = None,
           keep_props: bool = False) -> dict | None:
    """
    A BGRA image of just the person, trimmed to their own bounding box, plus
    where in the source that box sat.

    Trimmed rather than returned canvas-sized, because the caller places this
    on a different canvas: it has to know how wide the PERSON is to fit them
    into the space a channel gives them, and a 1280x720 image that is mostly
    transparent answers a different question. The offsets come back so the trim
    can still be described in the frame's own coordinates.

    `keep_props` cuts the subject WITH whatever is beside them rather than the
    person alone — the exception for an act whose other half is not a person.
    See PROP_REACH_FRAC.

    None when what came back is not a cutout — see MIN/MAX_SUBJECT_FRAC.
    """
    alpha = alpha_for(image, face, keep_props)
    h, w = alpha.shape[:2]

    solid = alpha >= TRIM_ALPHA_MIN
    covered = float(solid.sum()) / float(h * w)
    if not (MIN_SUBJECT_FRAC <= covered <= MAX_SUBJECT_FRAC):
        log.info("background removal: no usable subject (%.1f%% of the frame)", covered * 100)
        return None

    ys, xs = np.where(solid)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    bgra = np.dstack([image[y1:y2, x1:x2], alpha[y1:y2, x1:x2]])
    return {
        "image": bgra,
        "x": x1, "y": y1,
        "source_w": w, "source_h": h,
        "backend": backend_name(),
    }
