"""
subtitle_remover.py — Finding burned-in captions and painting them out.

Some of the videos this app is pointed at arrive with the words baked into the
picture. A still cut from one of those is unusable as it stands: whatever the
moment was, it has somebody else's sentence lying across it. So the pipeline
detects that a video is captioned and, when it is, paints the lettering out of
every frame it produces — alongside the restoration and the grading, before
anyone sees a thumbnail.

## Where they are

Not the bottom of the frame. That was the first assumption here and it was
wrong for exactly the footage this is for: a vertical news clip sets its
captions across the MIDDLE of the picture, over the subject, because that is
where a phone viewer is looking. The agency credit, meanwhile, sits in a bottom
corner. Both are lettering burned into fixed rows, both make a still unusable,
and the search covers the whole frame so that it finds either.

## Two questions, and only one of them is hard

WHERE is the text on this frame — text_detector answers that, with a model,
and answers it well: measured on real footage it found every caption and the
small corner credit, in tight boxes, with nothing on the fabric behind them.
The classical version that came before it could not, and the note at the top of
text_detector.py records what was tried and what it cost.

WHICH of that text should be removed is this module's question, and the answer
is not "all of it". A frame can legitimately contain words — a sign, a shirt, a
book — and painting those out damages the picture as surely as leaving a
caption on it does. What separates the two is not how the text looks but WHERE
it keeps appearing: a caption returns to the same rows for the whole running
time, and a sign in the background does not.

So detection runs in two passes:

  the BANDS are a property of the video, worked out once from a sample. Which
  rows does this video keep putting text into, and in enough of it to be its
  captions rather than one shot that happened to have a sign in it?

  the BOXES are a property of one frame, found at render time. Given that this
  video captions those rows, which rectangles of THIS frame are text right now?

That second pass has to be per-frame. Painting the bands out unconditionally
would inpaint a strip across every frame between the lines, where there is
nothing to remove and real picture to lose.

## What it will not tell apart

A caption and any other lettering living in the same place for most of the
video — a permanent ticker, a translated caption, a channel's own recurring
strap, an agency credit. All of them get painted out. That is the intended
reading: a band has to establish itself across the whole video before anything
is touched, and anything that does is furniture the frame is better off
without.

It also does not spare a face. A caption that crosses one is still a caption,
and half a sentence removed is worse than none — so unlike logo_remover, which
rejects any candidate touching face territory outright, this paints where the
text is.

## When the model is missing

Nothing happens. There is no classical fallback, and that is a decision rather
than an omission: one was written, measured, and removed. On six consecutive
frames of real footage it took two captions off cleanly, smeared the picture on
three, and missed one. A frame with a smear across it is worse than a frame
with a caption on it, so with no model there is nothing here worth running.
"""

import logging
from dataclasses import dataclass

import numpy as np

import text_detector
from image_utils import imread
from inpainter import inpaint_boxes

log = logging.getLogger("uvicorn.error")

# How many frames to sample when working out where a video puts its text.
# Each is judged on its own and the answer is how MANY of them agree, so the
# sample size is the resolution of that fraction.
SAMPLE_FRAME_COUNT = 24

# Rows are counted at this height, so one set of fractions means the same thing
# whatever the source was shot at.
ANALYSIS_HEIGHT = 720

# ── What makes a band ─────────────────────────────────────────────────────
# The bands are read off PEAKS in how often each row carries text, not off a
# threshold crossing. A caption occupies the same rows every time it appears,
# so those rows accumulate a count far above their surroundings; incidental
# text — a sign, a label, a shirt — turns up in different rows each time and
# accumulates a low, flat floor. A fixed threshold cannot tell those apart,
# because one low enough to catch a caption that is only on screen half the
# time also catches the floor.
#
# Share of the sampled frames that must carry text in the PEAK row before the
# video is called captioned at all. Well under half on purpose: captions are
# absent between lines, over music, and wherever nobody is speaking.
BAND_PRESENCE_FRAC = 0.30
# How close to the peak a neighbouring row must come to be counted part of the
# same band. This is what sets a band's height.
BAND_PEAK_FRAC = 0.70
# ...with an absolute floor underneath it, so a modest peak does not drag half
# the frame in with it.
ROW_PRESENCE_FRAC = 0.25
# A band taller than this is not lettering. Two or three lines of caption is
# around 12% of frame height; a quarter of the picture is a graphic.
MAX_BAND_H_FRAC = 0.28

# How many separate bands one video may have. Three covers what turns up
# together — captions across the middle, a credit in a corner, a strap along
# the foot — and stops a pathological frame being carved into a dozen.
MAX_BANDS = 3

# How far apart, as a fraction of frame height, two bands must be to count as
# separate things. A single caption produces a cluster of narrow peaks — the
# line, the top of its box, the row where the descenders end — and taking those
# as separate bands returned "three bands" for a video with one caption, whose
# union covered a quarter of the picture. Suppressing by each band's own height
# does not bridge them, because that height is the few percent and not the gap.
BAND_SEPARATION_FRAC = 0.12

# How far the band is grown before the per-frame search is confined to it. The
# band says where text WAS in the sample; a line one pixel taller, or a video
# whose captions shift to clear a lower-third, must not fall outside it.
BAND_MARGIN_FRAC = 0.03

# How far each box is grown before it is painted, as a multiple of the text's
# own height, with a floor as a fraction of frame width.
#
# Proportional to the LINE, because what has to be covered scales with the
# type: a caption is set with an outline, a drop shadow, or a semi-transparent
# box behind it, and all three are sized relative to the lettering. A padding
# fixed to the frame's width left the box standing as a grey rectangle with the
# words neatly removed from inside it, which is worse than not having tried.
BOX_PADDING_LINES = 0.55
BOX_PADDING_FRAC = 0.004
BOX_PADDING_MIN_PX = 6


@dataclass(frozen=True)
class SubtitleBand:
    """
    Rows a video keeps putting text into, as fractions of the FULL frame's
    height.

    Fractions, and of the full frame specifically, because the two images the
    renderer can be working from are not the same rectangle: frame_extractor
    keeps both the uncropped frame and one with the overlay margins trimmed
    off, and a row expressed in one means a different row in the other. One
    statement of where the band is, converted at the point of use (see
    band_rows), is what stops those two answers drifting apart.
    """
    top: float
    bottom: float
    frames_with_text: int
    frames_sampled: int

    @property
    def coverage(self) -> float:
        return self.frames_with_text / max(1, self.frames_sampled)


def _sample(frame_paths: list[str], count: int) -> list[str]:
    """`count` paths spread evenly across the list, or all of them if there are fewer."""
    if len(frame_paths) <= count:
        return list(frame_paths)
    step = len(frame_paths) / count
    return [frame_paths[min(len(frame_paths) - 1, int(i * step))] for i in range(count)]


def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    Overlapping boxes combined into their common bounding box.

    Each box is inpainted independently, so painting two overlapping rectangles
    one after the other means the second reconstructs part of its area from the
    first's output rather than from real picture — which is how a seam appears
    in the middle of a filled region. One box per thing removed avoids it.
    """
    merged: list[list[int]] = []
    for x, y, w, h in sorted(boxes, key=lambda b: (b[1], b[0])):
        x2, y2 = x + w, y + h
        for box in merged:
            if x < box[2] and x2 > box[0] and y < box[3] and y2 > box[1]:
                box[0], box[1] = min(box[0], x), min(box[1], y)
                box[2], box[3] = max(box[2], x2), max(box[3], y2)
                break
        else:
            merged.append([x, y, x2, y2])
    return [(a, b, c - a, d - b) for a, b, c, d in merged]


def _band_around(rows: np.ndarray, peak_row: int, threshold: float) -> tuple[int, int]:
    """The contiguous run of rows around `peak_row` that stays at or above `threshold`."""
    top = peak_row
    while top > 0 and rows[top - 1] >= threshold:
        top -= 1
    bottom = peak_row
    while bottom < len(rows) - 1 and rows[bottom + 1] >= threshold:
        bottom += 1
    return top, bottom + 1


def detect_bands(frame_paths: list[str]) -> list[SubtitleBand]:
    """
    Every row of the frame this video keeps putting text into, as bands.

    Empty when there are none, which is the answer for most videos and costs
    the rest of the pipeline nothing: no bands, no per-frame search, no
    inpainting. Empty too when the text-detection model is unavailable — see
    the module header for why there is nothing to fall back to.
    """
    if not text_detector.is_available():
        return []

    samples = _sample(frame_paths, SAMPLE_FRAME_COUNT)
    if len(samples) < 4:
        return []   # too little of the video to establish anything about it

    rows = np.zeros(ANALYSIS_HEIGHT, dtype=np.int32)
    frames_read = 0
    frames_with_text = 0

    for path in samples:
        image = imread(path)
        if image is None:
            continue
        frames_read += 1
        height = image.shape[0]
        boxes = text_detector.detect(image)
        if not boxes:
            continue
        frames_with_text += 1
        # Reduced to the rows this ONE frame covered before being counted, so
        # a frame carrying three lines of caption votes once per row rather
        # than three times.
        covered = np.zeros(ANALYSIS_HEIGHT, dtype=bool)
        for _, y, _, bh in boxes:
            top = int(y / height * ANALYSIS_HEIGHT)
            bottom = int((y + bh) / height * ANALYSIS_HEIGHT)
            covered[max(0, top):max(0, top) + max(1, bottom - top)] = True
        rows += covered

    if frames_read < 4:
        return []

    remaining = rows.copy()
    bands: list[SubtitleBand] = []

    while len(bands) < MAX_BANDS:
        peak_row = int(np.argmax(remaining))
        peak = int(remaining[peak_row])
        if peak < BAND_PRESENCE_FRAC * frames_read:
            break

        threshold = max(ROW_PRESENCE_FRAC * frames_read, peak * BAND_PEAK_FRAC)
        top, bottom = _band_around(remaining, peak_row, threshold)
        # Out of the running, so the loop always moves on to somewhere else.
        remaining[top:bottom] = 0

        height_frac = (bottom - top) / ANALYSIS_HEIGHT
        if height_frac > MAX_BAND_H_FRAC:
            log.info("captions: ignoring rows %d-%d — they span %.0f%% of the frame, "
                     "which is a graphic and not lettering", top, bottom, height_frac * 100)
            continue

        margin = ANALYSIS_HEIGHT * BAND_MARGIN_FRAC
        bands.append(SubtitleBand(
            top=max(0, int(top - margin)) / ANALYSIS_HEIGHT,
            bottom=min(ANALYSIS_HEIGHT, int(bottom + margin)) / ANALYSIS_HEIGHT,
            frames_with_text=peak,
            frames_sampled=frames_read,
        ))
        # An ACCEPTED band clears its neighbourhood as well as itself (see
        # BAND_SEPARATION_FRAC), so the next band has to be somewhere else in
        # the frame rather than the next ridge of this one. A REJECTED one must
        # not: a too-tall region rejected near the top of the frame would
        # suppress a real caption a third of the way down, and the caption
        # would never be found at all. Measured, in both directions.
        reach = int(ANALYSIS_HEIGHT * BAND_SEPARATION_FRAC)
        remaining[max(0, top - reach):bottom + reach] = 0
        band = bands[-1]
        log.info("captions: text between %.0f%% and %.0f%% of frame height, "
                 "in %d of %d sampled frames at the peak",
                 band.top * 100, band.bottom * 100, peak, frames_read)

    if not bands:
        log.info("captions: none — text in %d of %d sampled frames, but no row carries it "
                 "in more than %d", frames_with_text, frames_read,
                 int(rows.max()) if rows.size else 0)
    return bands


def band_rows(band: SubtitleBand, image_shape: tuple, crop_top: float, crop_bottom: float,
              from_full_frame: bool) -> tuple[int, int]:
    """
    The band as pixel rows of the image actually in hand.

    `from_full_frame` says which of frame_extractor's two outputs this is. The
    cropped one has had `crop_top` of the height taken off the top and
    `crop_bottom` off the bottom, so a row stated as a fraction of the full
    frame lands somewhere else in it — and by an amount large enough to miss
    the text entirely.

    Returns an empty range when the band does not reach this image at all.
    """
    h = image_shape[0]
    if from_full_frame:
        return int(band.top * h), int(band.bottom * h)

    kept = max(1e-6, 1.0 - crop_top - crop_bottom)
    top = (band.top - crop_top) / kept
    bottom = (band.bottom - crop_top) / kept
    return max(0, int(top * h)), min(h, int(bottom * h))


def text_boxes(image: np.ndarray, bands: list[SubtitleBand], crop_top: float,
               crop_bottom: float, from_full_frame: bool) -> list[tuple[int, int, int, int]]:
    """
    The rectangles of THIS frame to paint out, in its own pixel coordinates.

    Empty whenever the frame has nothing showing in any of its bands, which is
    common — between two lines, over a wordless shot, under the closing music.
    That is the reason this runs per frame rather than the bands simply being
    painted out: the difference between a frame with a caption and the next one
    without is picture, and painting the band regardless would take it.

    A box is kept when its MIDDLE falls inside a band. Its middle rather than
    any overlap at all, so a sign that happens to reach into the band from
    outside is left alone, and a caption whose descenders poke past the band's
    edge is still taken whole.
    """
    if not bands:
        return []

    keep = []
    for x, y, w, h in text_detector.detect(image):
        middle = y + h / 2
        for band in bands:
            row_from, row_to = band_rows(band, image.shape, crop_top, crop_bottom, from_full_frame)
            if row_to > row_from and row_from <= middle <= row_to:
                keep.append((x, y, w, h))
                break
    return _merge_boxes(keep)


def remove_subtitles(scaled: np.ndarray, source: np.ndarray, scale: float,
                     bands: list[SubtitleBand] | None, crop_top: float, crop_bottom: float,
                     from_full_frame: bool) -> np.ndarray:
    """
    Paints this frame's captions out of `scaled` — the same frame as `source`,
    multiplied by `scale`.

    Two images for the reason logo_remover.strip_overlays takes two: the text
    is looked for in the frame at its own resolution, and painted out of the
    scaled copy the renderer is actually building from. Detecting on the scaled
    one instead would mean the geometry meant something different at every zoom
    level.
    """
    if not bands:
        return scaled

    boxes = text_boxes(source, bands, crop_top, crop_bottom, from_full_frame)
    if not boxes:
        return scaled

    # Grown per box, since the padding is a multiple of each line's own height
    # (see BOX_PADDING_LINES). inpaint_boxes takes one padding for the lot, so
    # the growing happens here and it is handed rectangles already the right
    # size.
    floor = max(BOX_PADDING_MIN_PX, int(scaled.shape[1] * BOX_PADDING_FRAC))
    grown = []
    for x, y, w, h in boxes:
        pad = max(floor, int(h * scale * BOX_PADDING_LINES))
        grown.append((int(x * scale) - pad, int(y * scale) - pad,
                      int(w * scale) + 2 * pad, int(h * scale) + 2 * pad))
    return inpaint_boxes(scaled, _merge_boxes(grown))
