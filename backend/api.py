import os
import sys
import json
import base64
import shutil
import asyncio
import logging
import threading
from typing import Callable, NamedTuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

import appinfo
import background_remover
import face_restorer
import framing
import inpainter
import vram
import vary as vary_module
from video_downloader import (
    download_video, expected_stem, uploaded_video_path, upload_stem, UploadWriter,
)
from frame_extractor import extract_frames, full_frame_path, CROP_TOP_PCT, CROP_BOTTOM_PCT
import frame_grab
import transcriber
import narrator
from image_utils import imread, imwrite
from frame_selector import score_frames, deduplicate, DEDUPE_MODES, DEFAULT_DEDUPE_MODE
from quality_scorer import SHARPNESS_MIN
from reframe_engine import (
    compute_geometry, overpan_limits, compute_scale,
    rule_of_thirds_x, rule_of_thirds_y, target_w, target_h,
)
from face_detector import detect_faces
from character_identifier import cluster_characters
from logo_remover import detect_static_overlays, overlays_to_cropped_coords
import subtitle_remover
import text_detector
from state import session, prune_temp, TEMP_DIR, FrameRecord
import frame_pipeline as pipeline
from frame_pipeline import (
    compose, crop_state, default_crop_state, clamp_zoom, ensure_base, build_photo_base,
    render_window, render_current, render_full, wide_preview, build_initial_crop,
    build_crop_at, face_anchor, face_placement, crop_state_from_placement,
    EDIT_PRESETS, DEFAULT_EDIT_PRESET, DEFAULT_FIDELITY,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    # The frontend is served from a different port, so every response header
    # it needs to READ has to be listed here — allow_headers governs the
    # request side only, and without this the browser hands JavaScript a
    # response whose custom headers are all silently null. That matters now
    # that the editing endpoints return raw JPEG bytes with their metadata
    # alongside: x-frame-meta carries the entire drag geometry.
    expose_headers=["x-frame-meta", "enhanced", "backend", "frame_id"],
)

# "uvicorn.error" is uvicorn's own already-configured console logger — using it
# means these lines show up in the server output without this module having to
# call basicConfig (which would fight uvicorn's handler setup).
log = logging.getLogger("uvicorn.error")

TARGET_FRAME_COUNT = 20
MIN_FRAMES_PER_CHARACTER = 2
CANDIDATE_POOL_SIZE = 600           # how many deduplicated candidates the selection passes draw from
MIN_POOL_PER_CHARACTER = 40         # ...and the floor each character keeps of that, however many there are
# Candidates every character reaches the selector with, even when the
# duplicate test would have left it fewer (see frame_selector._most_distinct).
# Above MIN_FRAMES_PER_CHARACTER on purpose — some candidates are still lost
# afterwards to framing that can't be reframed into a thumbnail.
POOL_FLOOR_PER_CHARACTER = 2 * MIN_FRAMES_PER_CHARACTER

# A capture run's frame count, clamped on arrival the way every other number a
# channel pack sends is. The floor is the Snapchat pack's own default and the
# smallest grid that request asks for; the ceiling is where the up-front
# restoration pass (every frame, before the user has clicked anything) stops
# being something anyone waits through.
CAPTURE_COUNT_RANGE = (1, 400)
DEFAULT_CAPTURE_COUNT = 40

_progress_lock = threading.Lock()
_progress = {"stage": "idle", "percent": 0, "status": "idle", "detail": None}

# Held for as long as a pipeline run is in flight — see the note on
# /process-video. Also what makes "is a run happening right now" a single
# unambiguous fact rather than something inferred from the progress numbers.
_process_lock = threading.Lock()

# The finished payload, waiting to be collected by /process-video/result.
_process_result: dict | None = None


@app.on_event("startup")
def _startup() -> None:
    """
    Pay the model-loading cost now rather than inside the user's first click.
    Importing torch and pulling GFPGAN's weights into memory takes tens of
    seconds cold — long enough that, lazily, the first /enhance-frame looked
    like the app had hung.
    """
    def warm():
        if face_restorer.preload():
            log.info("face restoration ready (%s)", face_restorer.backend_name())
        else:
            log.info("face restoration unavailable — classical pipeline only")
        log.info("inpainting: %s", "LaMa ready" if inpainter.preload() else "LaMa unavailable, using Telea")
        # Downloads its weights on a machine that has not got them yet, which
        # is why this is in the warm-up thread and not in a request: the file
        # is 176 MB and the alternative is paying for it inside the click that
        # first needs a cutout. Until it lands, cutouts fall back to the
        # classical segmenter rather than failing.
        log.info("background removal: %s",
                 "U2-Net ready" if background_remover.preload()
                 else "U2-Net unavailable, using the classical segmenter")
        # 2 MB, and the only thing standing between a burned-in caption and
        # the frames the user downloads: without it subtitle_remover does
        # nothing at all, deliberately (see its header).
        log.info("text detection: %s",
                 "ready" if text_detector.preload()
                 else "unavailable, so burned-in captions will be left alone")
        # Last of the five, and the only one a run can be well under way
        # without: it is read once per video rather than once per frame, so
        # the cost of it landing late is a caption that appears a moment after
        # the grid does — not a frame drawn wrong. It is here all the same
        # because the weights are a few hundred megabytes on a machine that
        # has never fetched them, and paying for that inside a run is what
        # this whole warm-up exists to avoid.
        log.info("transcription: %s",
                 "Whisper ready" if transcriber.preload()
                 else "unavailable, so captured frames will arrive without captions")
        # Sixth, last, and by some distance the largest thing this thread
        # fetches — about 3 GB on a machine that has never had it. Here for
        # the reason all five above are here, only more so: paying for that
        # download inside the run that first needs it would stall a capture
        # for as long as the rest of the run takes. Until it lands, the
        # channel that asks for a rewrite gets the transcript as spoken.
        log.info("narration: %s",
                 "Llama ready" if narrator.preload()
                 else "unavailable, so captions will not be rewritten")

    appinfo.log_startup()
    threading.Thread(target=warm, name="model-preload", daemon=True).start()


@app.get("/app/info")
async def app_info():
    """
    What version is running, and what this install knows about updates.

    Read off disk, with no network call: the page asks for this on every load,
    and a page load is not a reason to contact an update server. The check
    that does that runs once, in the launcher, before either server is up —
    see installer/updater.py.
    """
    return appinfo.as_dict()


def _set_progress(stage: str, percent: int) -> None:
    with _progress_lock:
        _progress["stage"] = stage
        _progress["percent"] = percent


def _set_status(status: str, detail: str | None = None) -> None:
    """
    Whether a run is idle/running/done/error — the part of the progress
    payload the client makes control-flow decisions on, as opposed to the
    stage/percent it merely displays.
    """
    with _progress_lock:
        _progress["status"] = status
        _progress["detail"] = detail


@app.get("/process-video/progress")
async def get_progress():
    with _progress_lock:
        return dict(_progress)


def img_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def jpeg_response(img: np.ndarray, quality: int = 85, **headers) -> Response:
    """
    A finished canvas, as raw JPEG bytes rather than base64 inside JSON.

    Base64 costs a third more bytes on the wire plus an encode on the server
    and a decode in the browser, per interaction, for every drag, zoom, flip
    and preset switch. The few scalars that travelled alongside the image
    (whether restoration ran, which backend) ride in headers instead.
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "Could not encode frame")
    return Response(
        content=buf.tobytes(), media_type="image/jpeg",
        headers={k: str(v) for k, v in headers.items()},
    )


def _record(frame_id: int) -> FrameRecord:
    record = session.frames.get(frame_id)
    if record is None:
        raise HTTPException(404, f"Frame {frame_id} not found — run /process-video first")
    return record


# ── /upload-video ──────────────────────────────────────────────────────────

# How much of the request body is held in memory at a time. These are whole
# videos: reading one into a bytes object to write it out afterwards would mean
# a gigabyte of RSS for a gigabyte of video, on a machine already holding GAN
# weights on the GPU.
UPLOAD_CHUNK_BYTES = 1024 * 1024


@app.post("/upload-video")
async def upload_video(request: Request, name: str = ""):
    """
    Takes a video off the user's own machine and puts it where the pipeline
    already looks, then answers with the id to start /process-video with.

    The body is the file's raw bytes rather than a multipart form. There is
    exactly one thing being sent and no fields to go alongside it, so multipart
    would buy nothing but a parser — and python-multipart is a dependency this
    project does not otherwise carry, which would mean adding it to
    requirements.txt and to the installer's own list for a payload that is
    already a single stream. The filename rides in the query string instead;
    it is read for its extension and nothing else.

    Two uploads at once, or an upload overlapping the start of a run, is not
    defended against: the staging file lives in TEMP_DIR and a run beginning
    mid-upload prunes it. Reaching that needs two browser tabs driving one
    local single-user tool, and the cost of the collision is an upload that
    has to be redone.
    """
    try:
        writer = UploadWriter(name)
    except ValueError as e:
        # 415 rather than 400: the request is well formed, it is the media type
        # that this endpoint won't take.
        raise HTTPException(415, str(e))

    try:
        async for chunk in request.stream():
            writer.write(chunk)
    except Exception as e:
        # A browser tab closed mid-upload lands here, and the partial file has
        # to go with it — nothing else would ever collect it, since it never
        # gets the name prune_temp would recognise.
        writer.abort()
        log.warning("upload of %r failed after %d bytes: %s", name, writer.size, e)
        raise HTTPException(500, f"Upload failed: {e}")

    try:
        upload_id = writer.finish()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        writer.abort()
        raise HTTPException(500, f"Could not store the upload: {e}")

    log.info("upload: %r (%.1f MB) stored as %s",
             name, writer.size / (1024 * 1024), upload_id)
    return {"upload_id": upload_id, "size": writer.size}


# ── /process-video ─────────────────────────────────────────────────────────

class FramingRequest(BaseModel):
    """
    How this run wants its frames chosen and cropped — the selected channel's
    `framing` block, sent as numbers rather than as a channel name.

    A brand name would mean the backend holding a second copy of the channel
    list, in a language that cannot read the packs, going stale the first time
    somebody drops a folder into <install>/content. Numbers keep the packs the
    only place a channel is described. Every field is optional and every one is
    clamped on arrival — see framing.FramingProfile.
    """
    face_area_ratio: float | None = None
    anchor_x: float | None = None
    anchor_y: float | None = None
    min_text_space: float | None = None
    weights: dict[str, float] | None = None
    # Whether a face that overfills the CROP is penalised for it. False for a
    # channel that never shows the crop — see framing.FramingProfile.face_fill.
    face_fill: bool | None = None
    # The size of the finished frame. Here rather than in a block of its own
    # because on this side it IS framing: the cover scale, the anchor, the
    # overpan margin and the autofill band are all expressed against it, and
    # one profile per run is what stops those four disagreeing (see framing.py).
    canvas_w: int | None = None
    canvas_h: int | None = None


class SelectionRequest(BaseModel):
    """
    How this run should decide that two candidate frames are the same moment.

    "tone" is the colour histogram of the face region and is what every channel
    has always used. "expression" is the face descriptor — the equalised grey
    crop the character clusterer is built on — and exists because the histogram
    has a footage type it cannot read at all.

    That type is exactly what the Laugh Society channels are for: one performer,
    one costume, one lighting setup, a locked-off camera. The face histogram is
    then mostly skin under an unchanging light, and it reports every frame as a
    copy of every other. Measured on a stand-up special, 100% of candidate pairs
    inside every character cluster cleared the duplicate threshold — the two
    most different frames in eight and a half minutes included. The pool
    collapsed to one frame per character, the emergency floor topped each back
    up to four, and the grid came out with 12 thumbnails instead of 20.

    The descriptor separates the same footage about five times as well, because
    what moves it is the shape of the face rather than its colour, and the shape
    of a face is what an expression is.

    It is not the default. On a video with cuts, locations and wardrobe changes
    the histogram is the cheaper answer to the same question and has never been
    the thing that failed — so this is a channel's own statement, and a channel
    that says nothing keeps the behaviour it always had.
    """
    dedupe: str = DEFAULT_DEDUPE_MODE


class CleanupRequest(BaseModel):
    """
    Whether this run should paint burned-in overlays out of its frames.

    Both passes exist because this app is fed YouTube videos, and a great many
    of them carry a channel bug in a corner and captions burned across the
    bottom. Every channel that composites the PHOTOGRAPH wants them gone: an
    overlay left in is an overlay in the finished thumbnail, and there is no
    later stage that could remove it.

    One channel does not composite the photograph. Laugh Society cuts the
    person out and stands them on a backdrop of its own — the frame is where
    the subject comes from, not where the picture comes from — so every pixel
    these passes could clean is a pixel that gets thrown away anyway. The
    upside is zero and the downside is not: both passes are INPAINTERS, and an
    inpainter aimed at a detection that is really part of the set will paint
    over whatever is standing in front of it.

    Measured on this app's own footage, a stage with neon signage on the back
    wall: the text detector read the signage as captions and declared a band
    from 0.297 to 0.611 of the frame height — a third of the picture, through
    the middle — and the subtitle remover then erased 30% of the frame,
    including the performer's head and shoulders. What reached the canvas was
    a flat grey wall with an arm beside it, and the face the pipeline had
    correctly detected on the ORIGINAL frame now pointed at nothing, which is
    what took the cutout apart downstream.

    So it is a channel's own statement, not a global. Absent means true, which
    is what every run made before this field existed did.
    """
    overlays: bool = True
    subtitles: bool = True


class CaptureRequest(BaseModel):
    """
    A run that fetches frames instead of building thumbnails — the selected
    channel's `capture` block, sent as a count.

    Present only for a channel that declares one (see `capture` in
    content/README.md). Its absence is what every run this app made before the
    Snapchat pack existed looks like, and is the thumbnail pipeline.

    The output SIZE is not here: it rides in `framing` as canvas_w/canvas_h,
    because it is read by the same geometry every crop in the run goes through
    and there is one profile per run for exactly that reason (see framing.py).
    """
    count: int = DEFAULT_CAPTURE_COUNT
    # Whether each frame comes back with the sentence that was being said over
    # it (see transcriber.py, and `capture.captions` in content/README.md).
    #
    # The pack's, not the run's: a channel that puts words on its frames does
    # it to every frame of every video, and a channel that does not has no use
    # for a transcription it would pay for and throw away. Default off, which
    # is what every capture run before this feature existed looks like.
    captions: bool = False
    # Whether that caption is the transcript as spoken or the transcript
    # rewritten as third-person report (see narrator.py, and `capture.rewrite`
    # in content/README.md).
    #
    # The pack's, like `captions`, and for the same reason: a channel that
    # reports its captions does it to every frame of every video. It is also
    # meaningless on its own — there is nothing to rewrite without a
    # transcript — so the run below only reaches for it when `captions` is
    # set, rather than treating the pair as two independent switches.
    rewrite: bool = False


class ProcessRequest(BaseModel):
    # Exactly one of these. The link field keeps its original name even though
    # it stopped being Drive-only a long time ago — renaming it would break
    # every older client for no gain here.
    google_drive_url: str | None = None
    upload_id: str | None = None
    # Absent for a run started with no channel picked, which is every run this
    # app made before channels could ask for their own framing.
    framing: FramingRequest | None = None
    # Absent for every channel that makes thumbnails, which is all of them but
    # one — see CaptureRequest.
    capture: CaptureRequest | None = None
    # Absent for every channel that keeps the photograph, which is all of them
    # but two — see CleanupRequest.
    cleanup: CleanupRequest | None = None
    # Absent for every channel whose footage the colour histogram can read,
    # which is all of them but two — see SelectionRequest.
    selection: SelectionRequest | None = None


class VideoSource(NamedTuple):
    """
    Where a run's video comes from, and how to get it onto disk.

    A link and an upload differ in exactly three ways — what the file will be
    called, what to say while it is being fetched, and what fetching it means
    — and in no way at all after that. Naming those three differences here is
    what lets _process_video_sync stay a single path: it asks the source for a
    file and, from that line on, cannot tell which kind it was handed.
    """
    stem: str | None            # filename stem when knowable up front, for prune_temp
    stage: str                  # progress wording while fetch() runs
    failure: str                # how to label a fetch() that raises
    fetch: Callable[[], str]    # -> path to the video on disk


def _source_for(req: ProcessRequest) -> VideoSource:
    if req.upload_id and req.google_drive_url:
        raise HTTPException(400, "Give a link or an uploaded file, not both")

    if req.upload_id:
        upload_id = req.upload_id
        try:
            # Resolved HERE, not just inside the run: the run opens by resetting
            # the session and pruning the previous video's cache, so an id that
            # is malformed or has already expired would cost the user
            # everything they had loaded before anyone noticed. fetch() checks
            # again rather than reusing this answer — a file can still go
            # missing in between, and that path has to stay honest.
            uploaded_video_path(upload_id)
            stem = upload_stem(upload_id)
        except ValueError as e:
            raise HTTPException(400, f"Upload unavailable: {e}")
        return VideoSource(
            # Known without touching the disk, exactly as expected_stem is for
            # a link — so the previous video's cache is cleared BEFORE this one
            # is read rather than after, which is what bounds peak disk use.
            stem=stem,
            stage="Reading uploaded video and extracting frames...",
            failure="Upload unavailable",
            fetch=lambda: uploaded_video_path(upload_id),
        )

    url = (req.google_drive_url or "").strip()
    if not url:
        raise HTTPException(400, "Paste a link, or upload a video file")
    return VideoSource(
        stem=expected_stem(url),
        stage="Downloading video and extracting frames...",
        failure="Download failed",
        fetch=lambda: download_video(url),
    )


def _build_pool(scored: list[dict], mode: str = DEFAULT_DEDUPE_MODE) -> list[dict]:
    """
    Narrows every scored candidate down to the pool the selection passes draw
    from, deduplicating each character SEPARATELY.

    Deduplication used to run across the whole video at once, in score order,
    and it was where the missing characters actually died — not in selection,
    which was already trying to protect them. Its "is this a duplicate" test is
    a correlation between two face-region color histograms, and on a video shot
    in one location that statistic barely separates people at all: measured on
    the reported video, 9.2% of RANDOM candidate pairs cleared the 0.95
    duplicate threshold, and 8.2% of pairs more than a minute apart — different
    shots, different people — did too. Same set, same lighting, same wardrobe,
    and the face box is mostly skin either way.

    Score order then decided who paid for it. The highest-scoring character
    goes in first and every later frame is tested against what is already
    kept, so the best-lit subject filled the pool and the others were
    discarded as "duplicates" of him: 473 candidates became 131, roughly
    three-quarters of them one man, before clustering had even run. Selection
    cannot guarantee coverage of characters that are no longer in its input.

    Per character, that same test means what it was designed to mean — two
    frames of one person's face, near-identical expression — so it is applied
    within each cluster instead, under a per-character budget. What a
    character keeps now depends on how varied their own frames are, not on how
    they scored against somebody else.

    The two rounds inside each character are the sharpness partition (see
    SHARPNESS_MIN): the sharp frames get first claim on the budget, then the
    softer ones are re-mined for whatever distinct moments the first round
    didn't already cover. That ordering is what puts quality ahead of variety —
    a soft frame is only ever reached once the sharp ones have run out of
    distinct moments to offer, whichever measure `mode` uses to decide that.
    """
    by_character: dict[int, list[dict]] = {}
    for c in scored:
        by_character.setdefault(c["character_id"], []).append(c)

    budget = max(MIN_POOL_PER_CHARACTER, CANDIDATE_POOL_SIZE // max(1, len(by_character)))

    pool: list[dict] = []
    for members in by_character.values():
        round_one = deduplicate([c for c in members if c["sharpness"] >= SHARPNESS_MIN],
                                budget // 2, mode=mode)
        picked = {c["key"] for c in round_one}
        # The floor is what turns "at least two options per character" into a
        # guarantee rather than a hope: whatever the duplicate test thinks,
        # this character reaches the selector with enough candidates to fill
        # its slots, with headroom for ones the reframe later rejects. Only
        # the second round carries it, so the first still reports honestly how
        # much genuine variety the sharp frames had.
        round_two = deduplicate([c for c in members if c["key"] not in picked], budget // 2,
                                min_keep=POOL_FLOOR_PER_CHARACTER - len(round_one),
                                mode=mode)
        pool += round_one + round_two
    return pool


def _select_frames(pool: list[dict]) -> list[dict]:
    """
    Chooses which candidates become thumbnail slots, guaranteeing character
    coverage before filling the rest.

    Breadth strictly before depth: every character gets its first slot before
    any character gets a second, and every character gets its second before
    the score-ranked fill starts. The old order ran one character to its full
    quota before moving to the next, which is fine while slots last and total
    starvation when they don't.

    No cluster is written off as noise anymore either. A cluster of two used
    to be dismissed as a misdetection and excluded from the guarantee — but
    "only a couple of usable frames in the whole video" is also the exact
    signature of someone who was on screen for three seconds, which is the
    case this is supposed to protect.
    """
    groups: dict[int, list[dict]] = {}
    for c in pool:
        groups.setdefault(c["character_id"], []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: c["score"], reverse=True)

    # Process characters in order of their best frame's score — the most
    # prominent/likely-main-subject character gets first pick.
    priority = sorted(groups, key=lambda cid: groups[cid][0]["score"], reverse=True)

    # The grid grows rather than dropping a character, on the rare video whose
    # cast alone would fill it (clustering caps out at MAX_CLUSTERS, so this
    # normally leaves TARGET_FRAME_COUNT untouched).
    target = max(TARGET_FRAME_COUNT, MIN_FRAMES_PER_CHARACTER * len(groups))

    chosen: list[dict] = []
    taken: set[tuple] = set()

    def take(c: dict) -> bool:
        if c["key"] in taken or not c["framing"].valid:
            return False
        chosen.append(c)
        taken.add(c["key"])
        return True

    # Pass A: MIN_FRAMES_PER_CHARACTER slots per character, one round at a
    # time, so a character can only lose its Nth slot to another character's
    # Nth — never to someone else's (N+1)th.
    for _ in range(MIN_FRAMES_PER_CHARACTER):
        for cid in priority:
            if len(chosen) >= target:
                break
            for c in groups[cid]:
                if take(c):
                    break

    # Pass B: fill remaining slots round-robin across characters (by current
    # count, fewest first) rather than pure global score — this prevents one
    # dominant character (more screen time / more usable frames) from
    # consuming every leftover slot when other characters' frames fail.
    if len(chosen) < target:
        counts = {cid: 0 for cid in groups}
        for c in chosen:
            counts[c["character_id"]] += 1

        exhausted: set[int] = set()
        while len(chosen) < target and len(exhausted) < len(groups):
            cid = min(
                (c for c in groups if c not in exhausted),
                key=lambda c: (counts[c], -groups[c][0]["score"]),
            )
            candidate = next(
                (c for c in groups[cid] if c["key"] not in taken and c["framing"].valid), None,
            )
            if candidate is None:
                exhausted.add(cid)
                continue
            if take(candidate):
                counts[cid] += 1

    # Pass C: if every character's pool is exhausted but slots remain, fall
    # back to any leftover candidate by score.
    if len(chosen) < target:
        for c in sorted(pool, key=lambda c: c["score"], reverse=True):
            if len(chosen) >= target:
                break
            take(c)

    return chosen


def _cleaned_source(image, path: str, tag: str):
    """
    This frame with its burned-in captions painted out, written beside the
    original, as (image, path). The original, untouched, when the video has no
    captions in it — which is most videos.

    Done HERE, once, rather than inside build_photo_base where it started.
    Two things follow, and both were the reason for moving it.

    The frame is clean before anything is drawn from it. The grid card, the
    preview and the download all read one file, so there is no stage at which
    a caption is on screen — and no moment when a card can be offered for
    editing while the words are still on it, which is exactly what happened
    when removal lived in the enhance pass, since the enhance is what turns a
    card from grey to ready.

    And it happens once per photo. Inside build_photo_base it ran again on
    every preset switch, every Variation and every rebuild after a reframe,
    measured at 2.3 seconds each, for a result identical to the one before.
    """
    if not session.subtitle_bands:
        return image, path

    cleaned = subtitle_remover.remove_subtitles(
        image.copy(), image, 1.0, session.subtitle_bands,
        CROP_TOP_PCT, CROP_BOTTOM_PCT, from_full_frame=True)
    if cleaned is image:
        return image, path       # nothing on this particular frame

    stem = os.path.splitext(os.path.basename(session.video_path or "video"))[0]
    out = os.path.join(TEMP_DIR, f"clean_{stem}_{tag}.jpg")
    imwrite(out, cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cleaned, out


def _capture_frames_sync(video_path: str, count: int, captions: bool = False,
                         rewrite: bool = False) -> dict:
    """
    The frame-fetcher run: `count` usable stills, spread across the video and
    handed back in the order they occur in it.

    Everything downstream of this — restoration, the edit presets, zoom,
    manual reframing, download — is the same code every thumbnail goes
    through. A capture slot is an ordinary FrameRecord; what differs is
    entirely upstream, and it is three things:

      * WHICH frames. frame_grab picks them on whole-picture quality and
        temporal spread, because most of them have nobody in them and the
        face-based selector answers a video like that with nothing at all.
      * WHICH pixels. The slot points at the UNCROPPED frame, not the
        overlay-cropped one. Trimming the top 8% and bottom 24% is right when
        the frame is raw material for a thumbnail with a title over it, and
        wrong when the frame IS the deliverable — a third of the picture, cut
        off before the user ever sees it. That also means no burned-in logo
        detection: those boxes are found in the cropped frames' coordinate
        space, and there is nothing here for them to describe.
      * WHAT ORDER. Slot 0 is the earliest moment in the video, slot 1 the
        next, and so on to the end — which is what makes "frame 4" a name the
        user can rely on, and what the grid, the numbering and the filenames
        in the download all inherit for free.

    `captions` adds a fourth thing, and it is the only one of the four that is
    not about pixels: each slot also comes back with the sentence that was
    being said over it. The moment a slot is cut from is already known (the
    `timestamp` below), so all this needs is the same video's words with the
    seconds they were spoken at — see transcriber.py, which produces exactly
    that and answers captions_for the whole grid at once.

    `rewrite` is a fifth thing, and it happens entirely between those two: the
    sentences are handed to a language model and come back as third-person
    report before any frame is asked what was being said over it. It changes
    the words and not the timeline — every sentence keeps the seconds it
    already covered — so captions_for below cannot tell the difference, and
    neither can anything downstream of it. See narrator.py.
    """
    _set_progress("Extracting frames...", 20)
    try:
        frame_paths, interval = extract_frames(video_path, frame_grab.sample_interval(video_path, count))
    except Exception as e:
        raise HTTPException(500, f"Frame extraction failed: {e}")
    session.extract_interval = interval

    if not frame_paths:
        raise HTTPException(500, "No frames extracted")

    # The uncropped original of each sample, falling back to the cropped one
    # for anything ffmpeg wrote only half of — a frame is better cropped than
    # missing, and both are the same picture at the same moment.
    sources = [full_frame_path(p) for p in frame_paths]
    sources = [full if os.path.exists(full) else cropped
               for cropped, full in zip(frame_paths, sources)]

    # Burned-in subtitles. Read before the frames are chosen rather than after,
    # so the answer is in place by the time the first slot is rendered — and
    # on the uncropped frames, which is both what these slots are made of and
    # where a subtitle low in the picture actually survives.
    _set_progress("Checking for subtitles...", 35)
    session.subtitle_bands = subtitle_remover.detect_bands(sources)

    _set_progress("Checking frame quality...", 40)
    grabs = frame_grab.select(sources, interval, count, video_path)
    if not grabs:
        raise HTTPException(422, "No usable frames found — everything sampled was out of focus, "
                                 "smeared by motion, or blank")

    # After the frames are chosen, not before: this is the one pass in the run
    # that would have been paid for even if the video turned out to have no
    # usable frame in it at all, and the 422 above is where that is found out.
    #
    # Empty for a channel that does not ask for captions, and empty again for
    # anything that went wrong inside it — a silent video, a missing model.
    # captions_for then answers "" for every frame, which is a grid of stills
    # with nothing written on them: exactly what this run produced before
    # there were captions at all.
    sentences = []
    if captions:
        _set_progress("Transcribing the voice track...", 55)
        sentences = transcriber.transcribe(video_path)

        # ...and, for a channel that asks for it, the same sentences written
        # as report rather than as speech. Inside the `captions` branch and
        # not beside it: this rewrites a transcript, so without one there is
        # nothing here to do.
        #
        # It answers with what it was given whenever it cannot do better —
        # no weights on this machine, a generation that came back malformed —
        # so there is no failure to test for here. The frames get the spoken
        # transcript instead, which is what the channel next door produces.
        #
        # The detected language travels with them. Whisper worked it out from
        # this same audio, and it is what keeps the rewrite from coming back
        # as a translation — see the language rule in narrator.SYSTEM, and
        # transcriber.detected_language for why nothing else in the app knows
        # what language a video is in.
        if rewrite and sentences:
            _set_progress("Rewriting the transcript...", 60)
            # Told how many stills it is writing for, and it is `grabs` rather
            # than `count`: the request is what the user asked for and this is
            # what the video actually yielded, after the blurred and blank
            # frames were rejected. Writing forty captions for a run that
            # produced thirty-one leaves nine of them with no frame to appear
            # on. See narrator._group.
            sentences = narrator.rewrite(sentences, transcriber.detected_language(),
                                         frames=len(grabs))

    # Every slot's caption, worked out for the grid rather than per frame, so
    # that a sentence several frames landed on is split between them instead
    # of repeated on all of them (see transcriber.captions_for). Which is why
    # it cannot live inside the loop below: the answer for one frame depends
    # on which other frames claimed the same sentence.
    #
    # Indexed by the GRAB and not by the finished slot. A grab that fails to
    # decode or to crop below is dropped from `results`, and its piece of a
    # split caption goes with it — one gap in a sentence spread over three
    # stills, rather than every later frame in the run captioned with the
    # words of the one before it.
    frame_captions = transcriber.captions_for(sentences, [g.timestamp for g in grabs])

    _set_progress("Preparing frames...", 65)
    results = []
    for slot, grab in enumerate(grabs):
        image = imread(grab.path)
        if image is None:
            continue

        frame_id = len(results)
        # Before the geometry is worked out, so every measurement and every
        # pixel from here on belongs to the frame the user will actually get.
        image, source_path = _cleaned_source(image, grab.path, str(frame_id))
        best_face, face_count = _centred_subject(image)
        record = FrameRecord(
            source_path=source_path,
            origin_path=source_path,
            origin_face=best_face,
            best_face=best_face,
            face_count=face_count,
            source_shape=image.shape,
            # No uncropped sibling to reach for: this slot already IS the
            # uncropped frame, and full_frame_path would only point back at
            # itself. Leaving it None is also what keeps full_frame_y_off at
            # zero, so the centred window above stays centred.
            full_shape=None,
            # ...which is a different statement from the one above, and both
            # are needed: full_shape says there is no sibling, this says which
            # rectangle the source itself is. Everything measured against one
            # of frame_extractor's two outputs — the logo boxes, the subtitle
            # band — is placed with it (see frame_pipeline.source_is_full_frame).
            source_is_cropped=False,
            source_raw_path=grab.path,
            captions_removed=source_path != grab.path,
            geometry=compute_geometry(image.shape, best_face, face_count),
            edit_preset=DEFAULT_EDIT_PRESET,
            timestamp=grab.timestamp,
            origin_timestamp=grab.timestamp,
        )
        canvas = build_initial_crop(image, record)
        if canvas is None:
            continue

        record.crop_state = default_crop_state(record)
        session.frames[frame_id] = record

        imwrite(pipeline.plain_crop_path(frame_id), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append({
            "frame":    img_to_base64(compose(canvas, False, False, False)),
            "frame_id": frame_id,
            # Carried to the frontend but never drawn there — see
            # FrameRecord.timestamp. The seconds are for anything that wants to
            # compute with them; the timecode is the same instant written the
            # way the log writes it.
            "timestamp": round(grab.timestamp, 3),
            "timecode":  frame_grab.timecode(grab.timestamp),
            "score":     round(grab.score, 4),
            # What was being said at that instant, or "" — for a channel that
            # asks for none, for a silent video, for a frame that fell in a
            # gap too wide to be called the same moment, and for one whose
            # sentence had no room left to be split any further (see
            # transcriber.captions_for). The frontend fills the frame's title
            # with it, so "" is simply an empty text box the user can type
            # into rather than a state anything has to test for.
            "caption":   frame_captions[slot],
        })
        _set_progress("Cleaning and preparing frames..." if session.subtitle_bands
                      else "Preparing frames...",
                      65 + int(30 * len(results) / max(1, len(grabs))))

    if not results:
        raise HTTPException(422, "No usable frames found — none of them could be reframed")

    log.info("capture: %s", ", ".join(
        f"frame {n + 1} at {r['timecode']}" for n, r in enumerate(results)))

    # What the frames actually came back carrying, said once and plainly.
    #
    # `rewrite` is what the CHANNEL asked for and `narrator.loaded()` is what
    # happened, and the two coming apart is the one failure in this run that
    # looks like success: every frame is captioned, every caption reads, and
    # every one of them is the speech the video contains rather than the
    # report the channel is built on. It cost a user a whole afternoon of
    # looking at the packs for a fault that was never in them, so it is a
    # warning at the top of the log and a flag in the payload rather than a
    # line buried in a debug message.
    rewritten = bool(rewrite and narrator.loaded())
    if captions:
        log.info("capture: %d of %d frames captioned, %s",
                 sum(1 for r in results if r["caption"]), len(results),
                 "rewritten as report" if rewritten
                 else "the transcript as spoken" if not rewrite
                 else "NOT REWRITTEN - the narration model did not load")
    if rewrite and not rewritten:
        log.warning("capture: this channel asks for rewritten captions and the "
                    "narration model is not available on this machine, so the "
                    "frames carry the transcript as spoken. See the "
                    "'narration:' line printed when the server started.")

    vram.release(force=True)
    _set_progress("Done", 100)
    # `rewritten` travels with the frames so the browser can say what the user
    # is looking at (see main.runPipeline). A channel that asked for a
    # rewrite and did not get one is the only case worth a word on screen, and
    # it is the case the user cannot otherwise tell from a working run.
    return {"frames": results, "count": len(results), "capture": True,
            "rewritten": rewritten}


def _process_video_sync(source: VideoSource, capture: CaptureRequest | None = None,
                        cleanup: CleanupRequest | None = None,
                        selection: SelectionRequest | None = None) -> dict:
    session.reset()
    # The previous video's restored bases have just been dropped, so this is
    # the moment their GPU memory can actually go back to the driver — see
    # vram.py for what was accumulating and why nothing released it.
    vram.release(force=True)

    # Everything from previous videos goes BEFORE the download, not after —
    # see state.prune_temp. When the URL doesn't tell us what the file will be
    # called (expected_stem returns None), pruning waits until it does, which
    # is the old behaviour.
    keep_stem = source.stem
    if keep_stem:
        prune_temp(keep_stem=keep_stem)

    _set_progress(source.stage, 5)
    try:
        video_path = source.fetch()
    except Exception as e:
        raise HTTPException(400, f"{source.failure}: {e}")
    session.video_path = video_path

    if not keep_stem:
        prune_temp(keep_stem=os.path.splitext(os.path.basename(video_path))[0])

    # The fork, and the only one: everything above is getting a video onto
    # disk, which both kinds of run need and neither does differently.
    if capture is not None:
        return _capture_frames_sync(video_path, capture.count, capture.captions,
                                    capture.rewrite)

    _set_progress(source.stage, 30)
    try:
        frame_paths, interval = extract_frames(video_path)
    except Exception as e:
        raise HTTPException(500, f"Frame extraction failed: {e}")
    session.extract_interval = interval

    if not frame_paths:
        raise HTTPException(500, "No frames extracted")

    # Logo detection runs on the FULL frames only; the cropped-frame boxes are
    # derived from that one result (see logo_remover.overlays_to_cropped_coords),
    # instead of scanning the video twice.
    full_paths = [p for p in map(full_frame_path, frame_paths) if os.path.exists(p)]
    probe = imread(full_paths[0]) if full_paths else None
    # Skipped outright for a channel that throws the photograph away — see
    # CleanupRequest. Skipped rather than detected-and-ignored because the
    # session lists ARE the switch: build_photo_base reads them and does
    # nothing when they are empty, so leaving them empty is how the whole
    # downstream stays untouched without a flag threaded through it.
    want = cleanup or CleanupRequest()
    session.logo_overlays_full = (
        detect_static_overlays(full_paths) if (want.overlays and full_paths) else [])
    session.logo_overlays_cropped = (
        overlays_to_cropped_coords(session.logo_overlays_full, probe.shape[0])
        if (probe is not None and session.logo_overlays_full) else []
    )
    log.info("logo detection: %s full-frame boxes=%s, cropped-frame boxes=%s (%d frames)",
             "on" if want.overlays else "OFF for this channel",
             [o.bbox for o in session.logo_overlays_full],
             [o.bbox for o in session.logo_overlays_cropped], len(frame_paths))

    # Burned-in subtitles, on the same sample and for the same reason the
    # logos are found here: it is a fact about the video, established once,
    # and every frame the run goes on to render is cleaned against it (see
    # frame_pipeline.build_photo_base). Most videos have none, and for those
    # this costs one pass over two dozen frames and nothing afterwards.
    _set_progress("Checking for subtitles...", 42)
    session.subtitle_bands = (
        subtitle_remover.detect_bands(full_paths or frame_paths) if want.subtitles else [])
    if not want.subtitles:
        log.info("subtitle detection: OFF for this channel")

    _set_progress("Selecting best frames...", 45)
    # ONE scoring pass over the video, at the relaxed sharpness floor; the
    # strict/relaxed split is a partition of these results, not a second pass.
    scored = score_frames(frame_paths)
    if not scored:
        raise HTTPException(422, "No suitable frames found (no faces detected)")

    _set_progress("Detecting characters...", 60)
    # Identify distinct characters via a lightweight face descriptor +
    # clustering, so both narrowing and selection can keep every character
    # instead of one person dominating. The descriptors were computed during
    # scoring, while each frame was already decoded.
    #
    # This runs BEFORE the pool is narrowed, not after. Clustering the
    # survivors of a whole-video deduplication meant clustering a set the
    # dominant character had already crowded everyone else out of — the
    # characters worth protecting were missing from the input, so no amount of
    # clustering or selection could put them back (see _build_pool).
    num_characters = cluster_characters(scored)

    _set_progress("Selecting best frames...", 70)
    # Clamped here rather than trusted, for the reason every framing number is:
    # this arrives from a JSON file a user can edit by hand, and an unknown
    # name would otherwise reach deduplicate and silently mean "tone".
    mode = (selection.dedupe if selection else DEFAULT_DEDUPE_MODE)
    if mode not in DEDUPE_MODES:
        log.warning("selection: unknown dedupe mode %r, using %r", mode, DEFAULT_DEDUPE_MODE)
        mode = DEFAULT_DEDUPE_MODE
    pool = _build_pool(scored, mode)
    if not pool:
        raise HTTPException(422, "No suitable frames found (no faces detected)")
    log.info("candidates: %d scored -> %d pooled across %d character clusters (dedupe: %s)",
             len(scored), len(pool), num_characters, mode)

    _set_progress("Generating thumbnails...", 80)
    results = []
    for c in _select_frames(pool):
        image = imread(c["path"])
        if image is None:
            continue

        frame_id = len(results)
        full_path = full_frame_path(c["path"])
        full_probe = imread(full_path) if os.path.exists(full_path) else None

        record = FrameRecord(
            source_path=c["path"],
            origin_path=c["path"],
            origin_face=c["best_face"],
            best_face=c["best_face"],
            face_count=c["face_count"],
            source_shape=image.shape,
            full_shape=full_probe.shape if full_probe is not None else None,
            geometry=compute_geometry(image.shape, c["best_face"], c["face_count"]),
            edit_preset=DEFAULT_EDIT_PRESET,
        )
        canvas = build_initial_crop(image, record)
        if canvas is None:
            continue

        record.crop_state = default_crop_state(record)
        session.frames[frame_id] = record

        imwrite(pipeline.plain_crop_path(frame_id), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append({
            "frame":           img_to_base64(compose(canvas, False, False, True)),
            "frame_id":        frame_id,
            "score":           round(c["score"], 4),
            "character_id":    c["character_id"],
            "score_breakdown": c["score_breakdown"],
        })
        _set_progress("Generating thumbnails...", 80 + int(15 * len(results) / TARGET_FRAME_COUNT))

    if not results:
        raise HTTPException(422, "No suitable frames found (no faces detected, or none could be reframed)")

    # Nothing is in flight at this point and the user is about to start
    # editing, so whatever the twenty initial crops left in the allocator can
    # go back rather than sitting reserved for the rest of the session.
    vram.release(force=True)

    _set_progress("Done", 100)
    return {"frames": results, "count": len(results), "characters_detected": num_characters}


def _run_pipeline(source: VideoSource, capture: CaptureRequest | None,
                  cleanup: CleanupRequest | None = None,
                  selection: SelectionRequest | None = None) -> None:
    """Body of a background run: leaves its outcome where /result can find it."""
    global _process_result
    try:
        _process_result = _process_video_sync(source, capture, cleanup, selection)
    except HTTPException as e:
        _set_progress("Error", 0)
        _set_status("error", str(e.detail))
    except Exception as e:
        log.exception("process-video failed")
        _set_progress("Error", 0)
        _set_status("error", f"{type(e).__name__}: {e}")
    else:
        _set_status("done")
    finally:
        _process_lock.release()


@app.post("/process-video")
async def process_video(req: ProcessRequest):
    """
    Starts the pipeline and returns immediately; the client follows it via
    /process-video/progress and collects the frames from
    /process-video/result.

    This used to hold the request open for the whole run and answer with the
    frames. That works on localhost and nowhere else: a full video takes
    minutes end to end, and every tunnel and reverse proxy in front of this
    app cuts a request that idles that long (measured on both localtunnel and
    a Cloudflare quick tunnel — the browser got an error page at the scoring
    stage while the pipeline ran happily to completion behind it). The
    progress endpoint was already being polled throughout, so the client had
    a live channel the whole time; the result just had nowhere to arrive
    through. Now every request finishes in milliseconds and nothing is left
    for a proxy to time out.

    One run at a time: _process_video_sync opens with session.reset(), so a
    second run starting while the first is going wipes the state the first is
    midway through writing, and both then interleave frame records and temp
    files in the one global session. Easy to trigger by accident, since a
    pipeline that looks stuck invites a second click.
    """
    global _process_result
    # Resolved before the lock is taken, not after: _run_pipeline's finally is
    # what releases it, so a request rejected up here would strand the lock and
    # every later run would answer 409 for the life of the process.
    source = _source_for(req)

    # Clamped here rather than trusted, for the reason every framing number is:
    # this arrives from a JSON file on disk that a user can edit by hand, and
    # it decides how many restorations the machine is about to be asked for.
    capture = None
    if req.capture is not None:
        low, high = CAPTURE_COUNT_RANGE
        capture = CaptureRequest(count=max(low, min(int(req.capture.count), high)),
                                 captions=bool(req.capture.captions),
                                 rewrite=bool(req.capture.rewrite))

    if not _process_lock.acquire(blocking=False):
        raise HTTPException(409, "A video is already being processed — wait for it to finish.")

    # Before the thread starts, not inside it: everything the run touches —
    # the scoring pass, the initial crops, and every later /vary-frame and
    # /upload-frame on these slots — reads this, and a profile applied a beat
    # after the worker began would frame the first frames by the last run's
    # channel. A request with no channel picked resets it to the shipped
    # numbers rather than inheriting the previous run's.
    framing.set_active(framing.FramingProfile.from_request(
        req.framing.model_dump(exclude_none=True) if req.framing else None))

    _process_result = None
    _set_progress(source.stage, 0)
    _set_status("running")
    threading.Thread(
        target=_run_pipeline, args=(source, capture, req.cleanup, req.selection),
        name="process-video", daemon=True,
    ).start()
    return JSONResponse({"status": "started"}, status_code=202)


@app.get("/process-video/result")
async def get_process_result():
    """The finished run's frames. 409 while one is still in flight."""
    with _progress_lock:
        status, detail = _progress["status"], _progress["detail"]

    if status == "error":
        raise HTTPException(500, detail or "Processing failed")
    if status != "done" or _process_result is None:
        raise HTTPException(409, "No finished run to collect")
    return JSONResponse(_process_result)


# ── Editing endpoints ──────────────────────────────────────────────────────

class EnhanceRequest(BaseModel):
    frame_id: int
    fidelity: float = DEFAULT_FIDELITY
    flip_image: bool = False   # mirror the photo
    flip_text: bool = False    # mirror the text layout (gradient follows it)
    gradient: bool = True      # dark gradient behind the text — see compose
    preset: str = DEFAULT_EDIT_PRESET


def _enhance_sync(req: EnhanceRequest):
    record = _record(req.frame_id)
    record.edit_preset = req.preset if req.preset in EDIT_PRESETS else DEFAULT_EDIT_PRESET

    canvas, base = render_current(req.frame_id, record, req.fidelity)
    if canvas is None:
        raise HTTPException(500, "Stored crop position no longer valid for this frame")

    return jpeg_response(
        compose(canvas, req.flip_image, req.flip_text, req.gradient),
        enhanced=int(base.was_enhanced), backend=base.backend or "",
    )


@app.post("/enhance-frame")
async def enhance_frame(req: EnhanceRequest):
    """
    Crop+compose this frame's CURRENT position out of its restored base,
    building that base first (see frame_pipeline.ensure_base) if this frame
    hasn't been touched yet — the only step here that can take more than a
    moment. Every later call for the same frame_id reuses it.

    Runs off the event loop thread specifically so this potentially
    multi-second work doesn't block other requests — /vary-frame in
    particular needs to be dispatched immediately even while an enhance pass
    for the frame it's about to replace is still running.
    """
    return await asyncio.to_thread(_enhance_sync, req)


# ── /cutout-frame ──────────────────────────────────────────────────────────

class CutoutRequest(BaseModel):
    frame_id: int
    fidelity: float = DEFAULT_FIDELITY
    preset: str = DEFAULT_EDIT_PRESET
    # WHICH moment this figure is cut from. 0 is the frame's own photo; 1 and
    # up index the other moments of the video (see vary.alternate_moments), so
    # a thumbnail showing the subject three times shows three different poses
    # rather than three prints of one photograph.
    #
    # Unbounded upward, and WRAPS: the frontend's Variation button walks a
    # single figure through the alternates one at a time, and a counter that
    # ran off the end would leave that button doing nothing after a few
    # clicks. See `copiesBySlot` in js/channels.js for where the first few
    # come from.
    copy: int = 0
    # Cut the subject WITH whatever is beside them, instead of the person
    # alone. The exception for an act whose other half is not a person — a
    # ventriloquist's dummy, an instrument, a mask held out on a stick — which
    # the person prior has no reason to keep and every reason to delete. Per
    # frame and off by default, because it is a licence rather than a better
    # setting, and measurably so: it also turns the person prior off, which on
    # a lit stage costs about twice as much scenery as it saves props. The
    # numbers are at background_remover.PROP_REACH_FRAC; the button that sends
    # this is CUTOUT_MODES in frontend/js/editor.js.
    keep_props: bool = False


def _record_for_photo(record: FrameRecord, moment: dict) -> FrameRecord:
    """
    A stand-in slot for one of the OTHER photos a thumbnail's extra figures are
    cut from.

    Everything downstream of here — the restoration, the grading, the geometry
    — is written against a FrameRecord, and this photo has no slot of its own.
    Rather than teaching all of it about a second kind of subject, the photo is
    handed the same shape the real slots have. It is deliberately never stored
    in the session: nothing may look it up later, and nothing should be able
    to reframe or vary it.
    """
    image = imread(moment["path"])
    if image is None:
        raise HTTPException(500, "Could not read the frame a second figure was to be cut from")
    full_path = full_frame_path(moment["path"])
    full_probe = imread(full_path) if os.path.exists(full_path) else None
    return FrameRecord(
        source_path=moment["path"], origin_path=moment["path"],
        origin_face=moment["face"], best_face=moment["face"],
        face_count=moment["face_count"], source_shape=image.shape,
        full_shape=full_probe.shape if full_probe is not None else None,
        geometry=compute_geometry(image.shape, moment["face"], moment["face_count"]),
        edit_preset=record.edit_preset,
    )


def _cutout_png(record: FrameRecord, fidelity: float, own_slot: int | None,
                keep_props: bool = False) -> tuple[bytes, dict]:
    """
    The finished cutout for one photo, as PNG bytes plus what it came out as.

    Cached on the PHOTO, the edit preset and whether props are kept (see
    session.cutouts), which is
    what makes the background warm-up worth doing: the grid fills itself in
    once and every later click on those frames is served from here. It also
    means a slot and one of its own extra figures can never pay twice for the
    same photograph.

    `own_slot` is the frame_id when this photo IS a slot's own, so its restored
    base goes in the shared per-slot cache that /enhance-frame reads. For an
    extra figure it is None: that photo is not a slot, and its base is used
    once and dropped — the megabyte of PNG is what is worth keeping, not the
    hundreds of megabytes it was made from (see build_photo_base).
    """
    # `keep_props` is in the key because it changes the PIXELS, not just what
    # is asked for: the same photo cut with and without it are two different
    # cutouts, and a cache that could not tell them apart would hand the user
    # back the one they just pressed the button to get away from.
    key = (record.source_path, record.edit_preset, keep_props)
    hit = session.cutouts.get(key)
    if hit is not None:
        return hit

    base = (ensure_base(own_slot, record, fidelity) if own_slot is not None
            else build_photo_base(record, fidelity))
    canvas, zoom = render_full(base)
    if canvas is None or canvas.size == 0:
        raise HTTPException(500, "This frame's photo could not be rendered")

    # The detected face, in that render's own coordinates, purely as the
    # fallback segmenter's seed and as the answer to WHICH person this photo is
    # about when there are several (see background_remover._isolate_subject).
    result = background_remover.cutout(
        canvas, _face_in_render(record, canvas.shape, zoom), keep_props)
    if result is None:
        raise HTTPException(
            422, "No subject could be separated from the background in this frame")

    ok, buf = cv2.imencode(".png", result["image"])
    if not ok:
        raise HTTPException(500, "Could not encode the cutout")
    meta = {k: result[k] for k in ("x", "y", "source_w", "source_h", "backend")}
    meta["w"] = int(result["image"].shape[1])
    meta["h"] = int(result["image"].shape[0])

    entry = (buf.tobytes(), meta)
    session.cutouts.put(key, entry)
    return entry


def _cutout_sync(req: CutoutRequest):
    record = _record(req.frame_id)
    record.edit_preset = req.preset if req.preset in EDIT_PRESETS else DEFAULT_EDIT_PRESET

    copy = max(0, req.copy)
    # The whole set is asked for at once and one of it taken, so figure 2 is
    # the same photo whether the thumbnail carries two figures or three: a run
    # of them is one evenly-spread set, not several independent picks that
    # could land on the same instant.
    moments = (vary_module.alternate_moments(req.frame_id, vary_module.MAX_ALTERNATE_MOMENTS)
               if copy else [])
    if copy and moments:
        # Wrapped, so Variation can keep walking a figure forward for as long
        # as the user keeps clicking (see CutoutRequest.copy).
        at = (copy - 1) % len(moments)
        png, meta = _cutout_png(_record_for_photo(record, moments[at]), req.fidelity,
                                own_slot=None, keep_props=req.keep_props)
        meta["copy"] = at + 1
    else:
        # Either this figure IS the frame's own photo, or there is nothing else
        # on this video to use — an uploaded still, a video with no other
        # usable face in it. Said plainly in the header rather than passed off
        # as another moment: a visible repeat is a thing the user can see and
        # move, where an empty figure looks like the app lost one.
        png, meta = _cutout_png(record, req.fidelity, own_slot=req.frame_id,
                                keep_props=req.keep_props)
        meta["copy"] = 0
    meta["moments"] = len(moments)

    return Response(
        content=png, media_type="image/png",
        headers={
            "x-frame-meta": json.dumps(meta, separators=(",", ":")),
            "backend": str(meta.get("backend", "")),
        },
    )


def _face_in_render(record: FrameRecord, shape: tuple, zoom: float) -> tuple | None:
    """
    Where this photo's detected face lands in a whole-frame render taken at
    `zoom`, or None if it somehow falls outside it.

    The face is in BASE coordinates (face_anchor), and the render is the base
    multiplied by `zoom`, so the whole mapping is that one factor — there is no
    crop window in it, which is the point.
    """
    scale = record.geometry["scale"] * zoom
    ax, ay = face_anchor(record)
    cx, cy = ax * zoom, ay * zoom
    fw = record.best_face[2] * scale
    fh = record.best_face[3] * scale
    h, w = shape[:2]
    x, y = int(cx - fw / 2), int(cy - fh / 2)
    if fw < 1 or fh < 1 or x + fw <= 0 or y + fh <= 0 or x >= w or y >= h:
        return None
    return (max(0, x), max(0, y), int(min(fw, w - max(0, x))), int(min(fh, h - max(0, y))))


@app.post("/cutout-frame")
async def cutout_frame(req: CutoutRequest):
    """
    One of this thumbnail's figures, with the background removed, as a PNG
    trimmed to the person's own bounds.

    For a channel whose thumbnail is a backdrop of its own with the subject
    placed on it rather than the photograph itself — the frontend composes them
    (see js/compose.js). `copy` picks WHICH moment: 0 is this frame's own
    photo, and a thumbnail that shows the subject more than once asks for 1 and
    2 as well, which come from minutes away in the same video. The response
    says how many alternates exist, so the caller can walk a single figure
    through them one at a time — which is what Variation does on this channel,
    to one figure rather than to the whole arrangement.

    Cut from the whole photo rather than from the crop window, so the person
    comes across as complete as the footage has them — see _cutout_png.

    The mirror is NOT here. A flipped figure is the same cutout drawn the other
    way round, which the canvas does for free and per figure; asking the
    backend for it would mean a second restoration to produce pixels the
    frontend already has.
    """
    return await asyncio.to_thread(_cutout_sync, req)


class ComposeRequest(BaseModel):
    frame_id: int
    flip_image: bool = False
    flip_text: bool = False
    gradient: bool = True


def _compose_sync(req: ComposeRequest):
    record = _record(req.frame_id)

    base = session.restored.get(req.frame_id)
    if base is not None:
        state = crop_state(record)
        canvas = render_window(base, state["crop_x"], state["crop_y"], state["zoom"], req.frame_id)
        enhanced = True
    else:
        path = pipeline.plain_crop_path(req.frame_id)
        canvas = imread(path) if os.path.exists(path) else None
        enhanced = False

    if canvas is None:
        raise HTTPException(404, f"Frame {req.frame_id} not found — run /process-video first")

    return jpeg_response(
        compose(canvas, req.flip_image, req.flip_text, req.gradient),
        enhanced=int(enhanced),
    )


@app.post("/compose-frame")
async def compose_frame(req: ComposeRequest):
    """
    Re-apply just the presentation layer (flips + gradient) to a frame's
    CURRENT crop — no re-restoration. Uses the cached restored base when one
    exists, so toggling a flip is instant; falls back to the frame's initial
    plain automatic crop on disk if this frame has never been enhanced yet,
    and reports which one it used so the frontend knows whether an enhance
    pass is still owed.
    """
    return await asyncio.to_thread(_compose_sync, req)


def _frame_wide_sync(frame_id: int):
    record = _record(frame_id)
    view = wide_preview(frame_id, record)
    image = view.pop("image")
    # Geometry rides in a header so the image itself can stay raw JPEG bytes.
    return jpeg_response(image, quality=80, **{
        "x-frame-meta": json.dumps(view, separators=(",", ":")),
    })


@app.get("/frame-wide/{frame_id}")
async def frame_wide(frame_id: int):
    """
    The pre-crop view a frame's crop window is taken from, plus that window's
    default position within it — lets the frontend show what's beyond the
    current target-sized frame while the user drags to manually
    reframe. Served as JPEG bytes with the geometry in an x-frame-meta
    header. Restoration runs here at most once per frame_id.
    """
    return await asyncio.to_thread(_frame_wide_sync, frame_id)


class ReframeRequest(BaseModel):
    frame_id: int
    crop_x: int    # window top-left in the zoom-1 base space /frame-wide describes
    crop_y: int
    zoom: float = 1.0  # manual fine-zoom multiplier on the automatic scale
    flip_image: bool = False
    flip_text: bool = False
    gradient: bool = True


def _reframe_sync(req: ReframeRequest):
    record = _record(req.frame_id)
    generation = record.generation

    base = ensure_base(req.frame_id, record)
    zoom = clamp_zoom(record, base.shape, req.zoom)

    # The incoming crop coordinates are in the zoom-1 space the frontend was
    # served to drag over — multiplying by the zoom maps them into the zoomed
    # view, where the window is TARGET-sized again.
    limits = overpan_limits(int(base.shape[1] * zoom), int(base.shape[0] * zoom))
    crop_x = max(limits["min_x"], min(int(round(req.crop_x * zoom)), limits["max_x"]))
    crop_y = max(limits["min_y"], min(int(round(req.crop_y * zoom)), limits["max_y"]))

    canvas = render_window(base, crop_x, crop_y, zoom, req.frame_id)
    if canvas is None:
        raise HTTPException(422, "Invalid reframe position")

    # Only commit this crop position if a /vary-frame swap hasn't landed on
    # this frame_id while this request was computing — otherwise a reframe
    # that started against the OLD source would overwrite the new source's
    # crop state with coordinates that don't apply to it.
    if record.generation == generation:
        record.crop_state = {"crop_x": crop_x, "crop_y": crop_y, "zoom": zoom}

    return jpeg_response(
        compose(canvas, req.flip_image, req.flip_text, req.gradient),
        enhanced=int(base.was_enhanced), backend=base.backend or "",
    )


@app.post("/reframe-frame")
async def reframe_frame(req: ReframeRequest):
    """
    Re-crop a frame's restored base at a manually-chosen position/zoom, and
    remember that position for subsequent /enhance-frame and /compose-frame
    calls. Restoration only runs here if this is the very first crop, zoom, or
    enhance requested for this frame_id.
    """
    return await asyncio.to_thread(_reframe_sync, req)


class VaryRequest(BaseModel):
    frame_id: int
    flip_image: bool = False
    flip_text: bool = False
    gradient: bool = True


def _vary_sync(req: VaryRequest):
    record = _record(req.frame_id)

    # Variation means "another moment of this same shot", and it finds one by
    # walking the frames extracted either side of this one in the video. An
    # uploaded still is not in that sequence and has no neighbours in it, so
    # the search would quietly hand back a video frame - throwing away the
    # picture the user just put here, through a button that promises the
    # opposite of that.
    if record.uploaded:
        raise HTTPException(
            409, "This frame is an image you uploaded, so there is no nearby "
                 "moment to vary. Upload a different image to change it.")

    # Two searches, because the two kinds of slot mean different things by
    # "another moment of this". A thumbnail is a person, so its search is
    # built on faces: same character, same shot, a different expression. A
    # capture frame is a picture, and most of them have nobody in them at all
    # — that search would find no face to anchor on and refuse every
    # candidate. See vary.pick_capture_variation.
    capture_slot = record.origin_timestamp is not None
    picked = (vary_module.pick_capture_variation(req.frame_id) if capture_slot
              else vary_module.pick_variation(req.frame_id))
    if picked is None:
        raise HTTPException(
            422, "No usable frame found for this frame at all — the extracted sequence may have no other frames nearby",
        )

    image = imread(picked["path"])
    if image is None:
        raise HTTPException(500, "Could not read the chosen variation frame")

    # A capture frame is framed on nothing in particular — the picture is the
    # product, so it is fitted to the canvas centred, exactly as it was when
    # the slot was created (see _centred_subject).
    #
    # The replacement comes straight out of the video, so it still carries the
    # video's captions; cleaned here for the reason the slot's first photo is
    # cleaned when it is created, and before the framing is measured off it.
    was_cleaned = False
    if capture_slot:
        raw_path = picked["path"]
        image, clean_path = _cleaned_source(
            image, raw_path, f"v{req.frame_id}_{record.generation + 1}")
        # _cleaned_source hands back the path it was given when there was
        # nothing on this frame to remove, so this is the only honest test of
        # whether a cleaned copy actually exists.
        was_cleaned = clean_path != raw_path
        best_face, face_count = _centred_subject(image)
        picked = {**picked, "path": clean_path, "face": best_face, "face_count": face_count}

    full_path = full_frame_path(picked["path"])
    full_probe = imread(full_path) if os.path.exists(full_path) else None

    # A framing the user set up by hand carries across the swap — losing the
    # zoom and pan they just dialled in is exactly what makes a Variation
    # click feel destructive, when its whole point is a fresh expression
    # without redoing the work. Measured BEFORE the record moves onto the new
    # photo, and as a face-relative placement rather than raw coordinates
    # (see face_placement): the replacement is another moment of the same
    # shot, so the subject has drifted and changed size a little, and it's
    # the subject's position in the window the user chose — not the pixel
    # offset. A frame still on its automatic framing has nothing to inherit
    # and keeps getting the new photo's own automatic framing.
    #
    # Flips and the gradient live on the frontend's frame object and ride in
    # on the request, so they carry over on their own.
    placement = face_placement(record) if crop_state(record) != default_crop_state(record) else None

    # Record the photo being replaced so it's deprioritized until
    # VARY_HISTORY_SIZE other clicks push it back out.
    # The frame it came OUT of, not the cleaned copy: this is what the search's
    # candidates are called, and comparing the two names was what left the
    # history inert (see FrameRecord.source_raw_path).
    record.vary_history.append(record.source_raw_path or record.source_path)

    record.source_path = picked["path"]
    record.source_raw_path = raw_path if capture_slot else picked["path"]
    record.captions_removed = was_cleaned
    # The recorded moment follows the photo — see FrameRecord.timestamp.
    if picked.get("timestamp") is not None:
        record.timestamp = picked["timestamp"]
    record.best_face = picked["face"]
    record.face_count = picked["face_count"]
    record.source_shape = image.shape
    # A capture slot's photo is the uncropped frame both before and after the
    # swap: extract_window writes the same pair of outputs the whole-video
    # pass does, and the full one is what this slot is made of. Saying so is
    # what keeps the subtitle band and the logo boxes placed correctly on it
    # (see frame_pipeline.source_is_full_frame).
    record.full_shape = None if capture_slot else (
        full_probe.shape if full_probe is not None else None)
    record.geometry = compute_geometry(image.shape, picked["face"], picked["face_count"])
    record.crop_state = (
        crop_state_from_placement(record, placement) if placement else default_crop_state(record)
    )

    # Rendered at the inherited window, not the automatic one — otherwise this
    # response would flash the automatic crop until the enhance pass that
    # follows it re-rendered the frame where it actually belongs.
    canvas = build_crop_at(image, record, record.crop_state)
    if canvas is None:
        raise HTTPException(422, "Chosen variation frame could not be reframed")

    # This slot now points at a different photo — the restored base and crop
    # position were only meaningful for the photo they were built from, and
    # anything still in flight for the OLD photo must not write itself back.
    session.invalidate(req.frame_id)
    session.bump_generation(req.frame_id)
    imwrite(pipeline.plain_crop_path(req.frame_id), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return jpeg_response(
        compose(canvas, req.flip_image, req.flip_text, req.gradient),
        frame_id=req.frame_id,
    )


@app.post("/vary-frame")
async def vary_frame(req: VaryRequest):
    """
    Swaps this frame_id's underlying source photo for a nearby one — a few
    frames forward or back, direction and distance random within a fixed
    window around the frame's ORIGIN photo (see vary.pick_variation) — and
    reframes it through the exact same geometry pipeline a fresh frame gets
    in /process-video. Never dead-ends: see that function's tiered search for
    how it degrades rather than erroring out just because the window's best
    options have already been clicked through.
    """
    return await asyncio.to_thread(_vary_sync, req)


# -- /upload-frame ----------------------------------------------------------

# What a replacement frame may arrive as. cv2.imdecode is what actually reads
# it, so this is the list of things that decoder handles and nothing more.
FRAME_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Longest side an uploaded frame is kept at. A phone screenshot or a still off
# a 6K master arrives far larger than anything the pipeline ever sees from a
# video, and every pixel of it is carried through a GAN restoration pass on
# the GPU. Downscaling first costs nothing visible - the window that survives
# is 1280x720 - and keeps an upload from being the one thing that exhausts
# VRAM.
MAX_UPLOADED_FRAME_SIDE = 4096


def _centred_subject(image: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    """
    A stand-in subject box placed where the rule-of-thirds anchor resolves to a
    CENTRED window, so the picture is simply fitted to the canvas instead of
    being shoved to one side by an anchor meant for a subject that isn't there.

    Reported as zero faces, which is also what stops compute_scale from zooming
    in on the stand-in.
    """
    h, w = image.shape[:2]
    scale = compute_scale(h, w, (0, 0, 1, 1), face_count=0)
    centred_x = (w * scale - target_w()) / 2 + target_w() * rule_of_thirds_x()
    centred_y = (h * scale - target_h()) / 2 + target_h() * rule_of_thirds_y()
    return (int(centred_x / scale), int(centred_y / scale), 1, 1), 0


def _pick_subject(image: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    """
    The face an uploaded frame should be framed around, and how many it has.

    The largest face, because on a thumbnail the subject is the person nearest
    the camera, and that is the same thing the automatic framing already
    assumes about a video frame.

    With no face at all - a wide shot, a graphic - it falls back to the centred
    stand-in above.
    """
    faces = detect_faces(image)
    if faces:
        return max(faces, key=lambda f: f[2] * f[3]), len(faces)
    return _centred_subject(image)


def _upload_frame_sync(frame_id: int, data: bytes, flip_image: bool,
                       flip_text: bool, gradient: bool):
    record = _record(frame_id)

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "That file could not be read as an image")

    h, w = image.shape[:2]
    if max(h, w) > MAX_UPLOADED_FRAME_SIDE:
        k = MAX_UPLOADED_FRAME_SIDE / max(h, w)
        image = cv2.resize(image, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)

    # On disk, because a slot is a PATH from here on: every later restoration,
    # preset switch and pan re-reads the source rather than holding a decoded
    # copy (see frame_pipeline.ensure_base). Named per slot and per upload, so
    # replacing a frame twice cannot leave the second render reading the
    # first upload's file.
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"upload_frame_{frame_id}_{record.generation + 1}.jpg")
    imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    best_face, face_count = _pick_subject(image)

    # Everything a slot knows about its photo, replaced together. Deliberately
    # NOT inherited: the manual framing, which /vary-frame does carry across.
    # A variation is another moment of the same shot, so a window the user
    # dialled in still means something on it; an unrelated picture shares
    # nothing with the framing chosen for the old one, and the honest starting
    # point is the automatic framing this image would have been given had it
    # come out of the video.
    record.source_path = path
    # The slot no longer came out of the video at all, so what it was "made
    # from" is this file — otherwise it would go on reporting the frame it
    # replaced as still in use (see FrameRecord.source_raw_path).
    record.source_raw_path = path
    record.best_face = best_face
    record.face_count = face_count
    record.source_shape = image.shape
    # An upload has no uncropped sibling on disk - nothing extracted it, so
    # there is no wider version of it to reach for (see ensure_base).
    record.full_shape = None
    record.geometry = compute_geometry(image.shape, best_face, face_count)
    record.crop_state = default_crop_state(record)
    record.uploaded = True

    canvas = build_crop_at(image, record, record.crop_state)
    if canvas is None:
        raise HTTPException(422, "That image could not be reframed to a thumbnail")

    # Same reasoning as /vary-frame: the slot now points at a different photo,
    # so the restored base and any render still in flight for the old one are
    # not merely stale but wrong for what the user is looking at.
    session.invalidate(frame_id)
    session.bump_generation(frame_id)
    imwrite(pipeline.plain_crop_path(frame_id), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return jpeg_response(compose(canvas, flip_image, flip_text, gradient), frame_id=frame_id)


@app.post("/upload-frame")
async def upload_frame(request: Request, frame_id: int, flip_image: bool = False,
                       flip_text: bool = False, gradient: bool = True):
    """
    Replaces one thumbnail slot's photo with an image of the user's own, and
    hands back the same kind of render every other editing endpoint does.

    From the moment this returns, the slot is not special. It carries the same
    FrameRecord fields a slot built from the video carries, so restoration,
    the edit presets, manual reframing, zoom and both flips all reach it
    through their existing endpoints with no idea an upload was involved -
    which is the whole point, and why this replaces the slot's photo rather
    than introducing a second kind of layer. Text and image overlays are drawn
    on the frontend canvas over whatever the backend returns, so they go on
    sitting above it for free.

    The body is the image's raw bytes, as /upload-video takes a video's: one
    payload, no accompanying fields, so multipart would add a parser and a
    dependency to wrap a stream that is already a stream. The slot and the
    presentation flags ride in the query string, the way they ride in the JSON
    body of the sibling editing endpoints.
    """
    data = await request.body()
    if not data:
        raise HTTPException(400, "No image was uploaded")
    return await asyncio.to_thread(
        _upload_frame_sync, frame_id, data, flip_image, flip_text, gradient)


@app.post("/cleanup")
async def cleanup():
    session.reset()
    # Part of the run, exactly as the frames are: leaving it set would frame
    # the next video by a channel nobody has picked yet.
    framing.reset()
    vram.release(force=True)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
