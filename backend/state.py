"""
state.py — The one video the app is currently working on.

Everything here used to live as nine separate module-level dictionaries in
api.py, each keyed by frame_id and each cleared by hand at the start of every
/process-video. Grouping them into one object makes the lifetime obvious (a
session is created per video and replaced wholesale), makes "what does the app
remember about a frame" answerable by reading one dataclass, and removes the
class of bug where a new endpoint forgets to clear one of the nine.

Still a single global session — this is a local, single-user tool and every
endpoint addresses "the current video". The difference is that it's now one
named thing instead of nine anonymous ones.
"""

import os
import glob
import shutil
import logging
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass, field

log = logging.getLogger("uvicorn.error")

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

# How many frames' restored bitmaps stay resident. Each holds the GAN-restored
# pre-crop image (and, at the "natural" preset, the untouched source to blend
# back toward) — roughly 10-20MB per frame at 1080p and several times that at
# 4K. Unbounded, twenty selected frames alone reached hundreds of megabytes
# and never released any of it for the life of the process. Evicting the
# least-recently-used one costs a rebuild only if the user returns to it.
RESTORED_CACHE_SIZE = 6

# How many finished cutout PNGs stay resident, across every frame and every
# photo a frame's extra copies are cut from. Each is around a megabyte, and the
# grid warms all of them in the background as soon as it appears (see
# editor.warmCutouts) — so this is what keeps that warm-up from being paid for
# twice the moment the user starts clicking through the frames it just filled.
# Comfortably more than twenty slots plus their extra copies.
CUTOUT_CACHE_SIZE = 64

# How many of a frame_id's most recently shown variation photos stay off
# limits for future picks — without this, a small candidate window can start
# cycling through the same 2-3 options (1, 2, 1, 2, 3, 4, 2, 1... was observed
# in testing). At 4, an option can't resurface until at least 4 other
# Variation clicks on this frame have gone by.
VARY_HISTORY_SIZE = 4


@dataclass
class FrameRecord:
    """
    Everything known about one thumbnail slot.

    `source_shape`/`full_shape`/`geometry` are captured when the slot is
    created. They exist so the manual-reframe endpoints never have to decode
    the source JPEG again: every drag, zoom, flip and preset switch used to
    call imread() on the full-resolution frame purely to read `.shape` and
    recompute the same geometry from it — a complete JPEG decode per
    interaction, for numbers that cannot change while the slot points at the
    same photo.
    """
    source_path: str
    origin_path: str          # never touched by /vary-frame — the frame's very first pick
    origin_face: tuple        # ditto — the face variations are diffed against
    best_face: tuple
    face_count: int
    source_shape: tuple
    full_shape: tuple | None  # None when this frame has no uncropped sibling on disk
    geometry: dict
    crop_state: dict = field(default_factory=dict)
    edit_preset: str = "natural"
    # Bumped only when /vary-frame swaps this slot onto a different photo.
    # Enhance/reframe run off the event loop thread so a Variation click never
    # queues behind them — which means one can be mid-flight, working from the
    # OLD photo's geometry, when a swap lands. Each captures this counter
    # before starting and refuses to write its result back if it moved on,
    # rather than clobbering the new photo's cache with the old photo's work.
    generation: int = 0
    vary_history: deque = field(default_factory=lambda: deque(maxlen=VARY_HISTORY_SIZE))
    # True once the user has replaced this slot's photo with one of their own.
    # The slot behaves identically from here on — same restoration, presets,
    # reframing and flips — with one exception: Variation searches the frames
    # extracted either side of this one IN THE VIDEO, and an uploaded still
    # has no such neighbours, so it is refused rather than silently throwing
    # the upload away for a video frame.
    uploaded: bool = False
    # Which of frame_extractor's two outputs `source_path` points at. True for
    # every slot the thumbnail pipeline makes — those are the overlay-cropped
    # frames — and False for a capture slot, whose source IS the uncropped
    # frame (see api's capture path).
    #
    # It is not the same question as "was a full sibling read", which is what
    # build_photo_base already worked out for itself: a capture slot has no
    # sibling to read and is nonetheless in full-frame space. Anything holding
    # a coordinate measured against one of the two rectangles — the logo boxes,
    # the subtitle band — needs this one to place it.
    source_is_cropped: bool = True
    # Which frame of the video this slot's photo was made FROM, before the
    # captions were painted out of it. The same as source_path whenever there
    # was nothing to remove, and on every thumbnail slot.
    #
    # It exists because cleaning gives a photo a second identity, and the
    # Variation search only knows the first one. Its candidates are frames
    # straight out of the video, so a slot that reports itself by its cleaned
    # filename matches nothing in that pool: "am I already showing this",
    # "is this one another slot has", and "have I just come from this" all
    # quietly answered no. The search then ranked purely by how far a
    # candidate sat from the photo on screen, and the frame furthest from the
    # one you just moved to is the one you just moved from — so it handed
    # back the previous photo, every time, for ever.
    source_raw_path: str = ""
    # True once this slot's photo has had its burned-in captions painted out
    # and the cleaned copy written to disk (see api._cleaned_source).
    #
    # It exists to stop the work happening twice. Removal used to live inside
    # build_photo_base, which meant it ran again on every preset switch, every
    # Variation and every rebuild after a reframe — 2.3 seconds each time, for
    # a result identical to the one before. Doing it once, when the slot is
    # created, also settles WHEN: the grid, the preview and every later render
    # all read the same already-clean file, so there is no window in which a
    # caption is on screen.
    captions_removed: bool = False
    # When in the source video this frame was taken from, in seconds. Only a
    # capture run fills it in (see api's capture path and frame_grab), and
    # nothing on screen ever shows it: it is recorded so that "which moment is
    # frame 4" has an answer, for the features that will want one.
    #
    # It FOLLOWS the photo. A Variation click moves the slot onto a different
    # moment, and the recorded answer to "when is this frame" has to be the
    # moment the user is actually looking at.
    timestamp: float | None = None
    # ...and this one does not. It is where the slot was first placed, and it
    # is what every Variation search is centred on — re-centring on wherever
    # the last click landed turns a run of clicks into a random walk that
    # wanders out of the shot, which is the same reason vary.dense_pool
    # anchors on origin_path rather than source_path.
    origin_timestamp: float | None = None

    @property
    def wide_shape(self) -> tuple:
        """Shape of the image the restored base is built from (full sibling when present)."""
        return self.full_shape or self.source_shape


class _LRU(OrderedDict):
    """Plain size-bounded LRU — evicts the least recently *used*, not inserted."""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        if key not in self:
            return default
        self.move_to_end(key)
        return self[key]

    def put(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.maxsize:
            evicted, _ = self.popitem(last=False)
            log.debug("state: evicted restored base for frame %s", evicted)


@dataclass
class VideoSession:
    """The current video and everything derived from it."""
    video_path: str | None = None
    extract_interval: float | None = None

    # Every burned-in logo overlay this video carries (a video can change its
    # channel bug partway through — see logo_remover), in full-frame and in
    # cropped-frame coordinates. Detected once per video; empty when it has
    # none. Which of them is actually on screen is a per-frame question, asked
    # at render time via logo_remover.present_overlays.
    logo_overlays_full: list = field(default_factory=list)
    logo_overlays_cropped: list = field(default_factory=list)
    # Every row of the frame this video keeps putting text into — the captions
    # across the middle, the agency credit in the corner — or empty when it
    # puts none anywhere (see subtitle_remover.detect_bands). Worked out once
    # from the extracted sample; WHICH rectangles of a given frame are text is
    # asked again per frame, at render time, because a caption is only on
    # screen part of the time.
    subtitle_bands: list = field(default_factory=list)

    frames: dict[int, FrameRecord] = field(default_factory=dict)
    restored: _LRU = field(default_factory=lambda: _LRU(RESTORED_CACHE_SIZE))
    vary_pools: dict[int, list[str]] = field(default_factory=dict)

    # frame_id -> every candidate in its dense window, scored, in time order.
    # What the extra copies of a multi-figure thumbnail are chosen from, and
    # cached because the choice has to be the SAME every time or a preset
    # switch would silently swap which moments are on screen (see
    # vary._scored_window).
    alternates: dict[int, list] = field(default_factory=dict)

    # (photo path, edit preset) -> the finished cutout PNG for it. Keyed on the
    # PHOTO rather than on the frame, because a frame's extra copies are cut
    # from other photos entirely and two frames can legitimately be asked for
    # the same one.
    cutouts: _LRU = field(default_factory=lambda: _LRU(CUTOUT_CACHE_SIZE))

    # frame_id -> (crop key, weak ref to the base, finished canvas) for the
    # LAST window rendered from each frame. Grading is now applied per crop
    # rather than once over the whole pre-crop frame, which is what keeps the
    # first-enhance cost proportional to what's displayed — but it also means
    # a request for a window that was JUST rendered would redo that work.
    # Toggling a flip or the gradient is exactly that request: the crop
    # doesn't move at all, only the presentation layer on top of it changes.
    # One entry per frame is enough, since these repeats are always
    # immediately consecutive.
    #
    # The base is part of the key, not just the crop geometry — see
    # frame_pipeline.render_window for the preset-switch bug that taught us
    # that. See cached_render for why the reference to it is weak.
    last_render: dict[int, tuple] = field(default_factory=dict)

    def reset(self) -> None:
        self.video_path = None
        self.extract_interval = None
        self.logo_overlays_full = []
        self.logo_overlays_cropped = []
        self.subtitle_bands = []
        self.frames.clear()
        self.restored.clear()
        self.vary_pools.clear()
        self.alternates.clear()
        self.cutouts.clear()
        self.last_render.clear()

    def used_source_paths(self) -> set[str]:
        """
        Every photo the grid is currently showing, named the way the Variation
        pools name their candidates — by the video frame it came from, not by
        the cleaned copy written from it (see FrameRecord.source_raw_path).
        """
        return {f.source_raw_path or f.source_path for f in self.frames.values()}

    def cached_render(self, frame_id: int, key: tuple, base):
        """
        The finished canvas from this frame's last render, if it was the same
        window of the same base — otherwise None.

        The stored reference to the base is WEAK, and that is the whole point
        of this pair of methods. Holding it strongly (as this did) kept every
        base the LRU had already evicted alive anyway, one per frame the user
        had touched: measured at the end of a normal session, RESTORED_CACHE_SIZE
        was capping the cache at 6 bases (729MB) while 20 were still resident
        (1,180MB), because this dict was pinning the other fourteen. The cap
        was doing nothing.

        A weak reference keeps the identity check just as safe as the strong
        one it replaces. The concern that motivated holding the base at all
        was a recycled memory address making `is` return true for a different
        object — but a weakref is cleared the moment its referent is
        collected, so a dead one reads as None and misses, and a live one
        cannot be pointing at anything but the original object.
        """
        entry = self.last_render.get(frame_id)
        if entry is None:
            return None
        cached_key, base_ref, canvas = entry
        if cached_key != key or base_ref() is not base:
            return None
        return canvas

    def remember_render(self, frame_id: int, key: tuple, base, canvas) -> None:
        self.last_render[frame_id] = (key, weakref.ref(base), canvas)

    def bump_generation(self, frame_id: int) -> int:
        record = self.frames[frame_id]
        record.generation += 1
        return record.generation

    def generation_of(self, frame_id: int) -> int:
        record = self.frames.get(frame_id)
        return record.generation if record else 0

    def invalidate(self, frame_id: int) -> None:
        """Drops cached pixels for a slot whose underlying photo or preset changed."""
        self.restored.pop(frame_id, None)
        self.last_render.pop(frame_id, None)
        # The window this frame's extra copies were chosen from was a window
        # around the photo it no longer holds.
        self.alternates.pop(frame_id, None)


session = VideoSession()


def prune_temp(keep_stem: str | None = None) -> int:
    """
    Deletes downloaded videos and extracted frame directories from previous
    runs, keeping only the one identified by `keep_stem` (a video's filename
    without its extension — see video_downloader.expected_stem).

    Nothing used to remove any of it. /cleanup exists but the frontend never
    calls it, so every video ever processed stayed on disk along with its
    hundreds of extracted frames — measured at 4.8GB of accumulated cache on
    a normally-used install. Returns the number of bytes freed.

    Takes the stem rather than a path so it can run BEFORE the download it's
    making room for, which is the only order that actually bounds peak disk:
    called afterwards, the previous video and its ~120MB of extracted frames
    were still on disk alongside the new download for the whole of it.
    """
    if not os.path.isdir(TEMP_DIR):
        return 0

    freed = 0
    for entry in glob.glob(os.path.join(TEMP_DIR, "*")):
        name = os.path.basename(entry)
        stem = os.path.splitext(name)[0]
        if keep_stem and (stem == keep_stem
                          or name == f"frames_{keep_stem}"
                          or name.startswith(f"vary_{keep_stem}_")
                          # A capture slot can end up pointing INTO one of
                          # these: the eye pass swaps a blinking frame for one
                          # out of a window it extracted itself, and that file
                          # then has to outlive selection exactly as the
                          # sample's own frames do (see frame_grab.open_eyes).
                          or name.startswith(f"eyes_{keep_stem}_")
                          # ...and the caption-free copies of this video's
                          # chosen frames, which every slot points AT (see
                          # api._cleaned_source).
                          or name.startswith(f"clean_{keep_stem}_")):
            continue
        try:
            if os.path.isdir(entry):
                freed += sum(
                    os.path.getsize(os.path.join(root, f))
                    for root, _, files in os.walk(entry) for f in files
                )
                shutil.rmtree(entry, ignore_errors=True)
            else:
                freed += os.path.getsize(entry)
                os.remove(entry)
        except OSError:
            continue

    if freed:
        log.info("temp: freed %.1f MB of previous runs' cache", freed / (1024 * 1024))
    return freed
