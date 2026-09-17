"""
logo_remover.py — Detects the static logo overlays remaining after
frame_extractor's destructive top-margin crop, and removes them via inpainting
(LaMa, when available) only on the frames where they actually appear.

CROP_TOP_PCT (frame_extractor.py) can't be made tall enough to always fully
remove a logo without also cutting into headroom the face-size/scale math
needs — some logos poke a few percent below that line. Rather than either
living with the remnant or increasing the crop (which would distort the
face-proportion calculation elsewhere), this detects those surviving strips
once per video and paints over them.

A video can carry MORE THAN ONE overlay. Detection used to compare all sampled
frames at once and return a single box, which silently assumed one logo for the
whole runtime — a video whose channel bug changes partway through (measured on
a real one: a wide banner for the first half, a small badge for the second)
had NEITHER detected, because neither survives a median taken across both
halves: each logo's own pixels disagree with the other half's samples just as
much as moving footage does. So the sample is split into consecutive segments
(see SEGMENT_SAMPLE_COUNT) and each is analysed on its own, then matching
detections are merged across segments — one overlay per distinct logo, however
many the video uses and whenever each is on screen.

Because the boxes now describe overlays that exist only part of the time, each
one carries what it LOOKS like (see Overlay.reference), and is painted over a
given frame only when it's actually there — see present_overlays. Painting a
box unconditionally would smear real content on every frame the other logo's
half of the video is made of.

Never touches a face: a candidate that overlaps where a face was actually
detected across the sampled frames is rejected outright (see
FACE_OVERLAP_REJECT_FRAC), however static/graphic it otherwise looks — a
locked-off talking-head shot can make a subject's own hairline/forehead
score just as "static" as a burned-in logo, and inpainting over that reads
as a seam through the face with no restoration-side check able to catch it
(see face_restorer.py's _seam_* protections, which trust this stage's
output as ground truth).
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass

from image_utils import imread
from inpainter import inpaint, inpaint_boxes, DEFAULT_CONTEXT_MARGIN_PX
from face_detector import detect_faces
from frame_extractor import CROP_TOP_PCT, CROP_BOTTOM_PCT

log = logging.getLogger(__name__)

TOP_BAND_FRAC = 0.30          # only look for a logo remnant within the top 30% of the (already-cropped) frame
STATIC_MAD_THRESHOLD = 6.0    # per-pixel median absolute deviation across sampled frames below this = "doesn't change" = overlay
MIN_OVERLAY_AREA_FRAC = 0.002 # ignore tiny noise blobs — logo must cover at least this fraction of the sampled band
CLOSE_KERNEL_FRAC = 0.004     # morphological-close kernel, as a fraction of frame width (see _segment_overlay)
CLUSTER_GAP_FRAC  = 0.018     # max horizontal gap, as a fraction of frame width, for a neighbouring blob to be
                              # absorbed into the logo cluster; the vertical allowance is half this
MAX_OVERLAY_W_FRAC = 0.30     # reject a static blob wider than this fraction of the frame width — too big to be a logo
MAX_OVERLAY_H_FRAC = 0.60     # reject a static blob taller than this fraction of the scanned band — too big to be a logo
CORNER_FRAC = 0.10            # a logo's near edge must sit within this fraction of the PICTURE's width from its left
                              # or right border. Measured on real channel bugs: their margins ran 0.6-4.8%. The old
                              # 0.40 accepted 80% of the frame as "a corner" and let mid-frame scenery through.
                              # Measured against the picture rather than the video frame because the two are not
                              # always the same rectangle - see _content_bounds.
CONTENT_DETAIL_FRAC = 0.35    # a column counts as picture, not pillarbox, once its detail energy reaches this
                              # fraction of the frame's own 90th-percentile column. Measured on a pillarboxed video:
                              # bar columns averaged 1-3, picture columns 11-130, and the two boundary columns
                              # spiked to ~127 - the hard edge between them. Any cut in that gap separates them.
MIN_CONTENT_W_FRAC = 0.25     # ...but a "picture" narrower than this is not believed, and the whole frame is used
                              # instead. A 9:16 video pillarboxed into 16:9 fills 31.6% of the width, so this sits
                              # below the narrowest real case while still refusing a reading that collapsed to
                              # nothing on unusually flat footage.
MAX_CONTENT_W_FRAC = 0.97     # ...and one this wide means there are no bars at all, which is the ordinary case.
SAMPLE_FRAME_COUNT = 30       # how many frames (evenly spaced) to sample across the video in total
SEGMENT_SAMPLE_COUNT = 6      # ...and how many of those each independently-analysed segment gets. Six is enough for
                              # the median-based statistics below (which need a majority to agree, so at least 3), and
                              # because the sample is spread across the WHOLE video before being split, six
                              # consecutive samples still span a large stretch of runtime — on a 26-minute video,
                              # ~4 minutes and many scene cuts. Smaller segments would localise a logo change more
                              # precisely but would start comparing frames of the same shot, where set dressing
                              # holds as still as an overlay (see MAX_SEGMENT_STATIC_FRAC).
MAX_SEGMENT_STATIC_FRAC = 0.20  # a segment whose scanned band is this static OVERALL didn't cross enough scene change
                              # for "doesn't change" to mean "overlay" — that's a segment sitting inside one
                              # locked-off shot, where a wall or a prop is every bit as motionless as a channel bug.
                              # Skipped rather than trusted. Measured: segments spread across a video ran 2-8% static,
                              # six consecutive frames of a single shot 10-21%.
                              # See _detect_pass: this skip is not the last word, because a video shot ENTIRELY on
                              # one locked-off set trips it in every segment and would otherwise never be scanned.
FALLBACK_MIN_EDGE_ON_STATIC = 45.0  # the edge-energy bar a candidate must clear in the fallback pass that scans the
                              # segments MAX_SEGMENT_STATIC_FRAC skipped (see _detect_pass). Those segments really do
                              # hold motionless set dressing that the normal bar would let through, so the one test
                              # that separates a graphic overlay from a flat surface is tightened rather than the
                              # segment being trusted as-is. Measured on a single-location shoot whose band ran 22-78%
                              # static: its real logo scored 97-216 there, while the set's own static clusters (a wall,
                              # a framed picture, a prop) scored 5-15 — this sits in the middle of that gap, and above
                              # the 20 that the normal, scene-crossing pass can afford.
MIN_STATIC_FILL = 0.25        # fraction of a candidate's box that must be genuinely static. This is what separates an
                              # overlay from scenery that merely holds still: measured on real footage, true logos
                              # filled 31-74% of their box while false positives (a bar's chalkboard menu, a static
                              # mid-frame prop) filled only 5-19%.
MIN_EDGE_ON_STATIC = 20.0     # mean Sobel-gradient magnitude, over just the box's static pixels, that a candidate
                              # must clear. A locked-off camera makes real set dressing (a blank wall, a smooth prop)
                              # every bit as static as a burned-in logo — MIN_STATIC_FILL alone let a plain wall
                              # through. But a wall has no texture where it IS static, while a logo's static pixels
                              # are graphic (bold letters, borders): measured 72-216 for real logos' static pixels,
                              # under 4 for a static wall/prop/poster's.
MERGE_IOU = 0.40              # two segments' detections are the same logo when their boxes overlap by at least this
                              # AND look alike (see MERGE_APPEARANCE_MIN).
                              # The box a logo produces shifts a little between segments (which of its letters and
                              # borders cleared the static test there), so matching has to tolerate that while still
                              # keeping two genuinely different logos — measured at IoU 0.65-0.86 for repeats of one
                              # logo, and ~0 between the two logos of the video that motivated all this.
MERGE_APPEARANCE_MIN = 0.55   # ...and how strongly they must correlate to count as the same logo rather than two
                              # that happen to share a spot. A channel can swap its bug for a different one in the
                              # SAME position partway through the video, and merging those produces a median of two
                              # graphics that matches neither: measured on such a video, the second logo then failed
                              # its own presence test on every frame and survived into the thumbnails. Below the
                              # presence bar on purpose - the same logo seen in two segments sits over different
                              # footage each time, so its correlation with itself is looser than with a frame.
PRESENCE_CORRELATION_MIN = 0.60  # how strongly a frame's pixels must correlate with an overlay's stored appearance for
                              # it to count as on screen there (see present_overlays). Correlation, not a pixel
                              # difference, because a partly translucent bug takes on the tone of whatever is behind
                              # it while keeping its shape. Measured on the two-logo video: 0.95-1.00 wherever a logo
                              # was actually present, -0.39 to 0.16 wherever it wasn't — this sits in the middle of a
                              # gap that wide.
MIN_PRESENCE_MASK_FRAC = 0.05 # if the overlay's own pixels (those static in every segment that found it) come to less
                              # than this share of its box, fall back to correlating the whole box — too few pixels to
                              # judge anything by.
INPAINT_PADDING_FRAC = 0.004  # pad the detected bbox by this fraction of image width — covers the logo's soft
                              # edge/anti-aliasing/drop-shadow. Proportional, since those edges scale with
                              # resolution: a fixed pixel pad tuned at 1080p is a third as wide at 4K.
INPAINT_PADDING_MIN_PX = 6

# A talking-head shot on a locked-off camera makes a subject's own hairline/
# forehead look just as "static" as a burned-in logo (see STATIC_MAD_
# THRESHOLD) — someone who doesn't move much between the sampled frames can
# sit still enough, with hair providing plenty of edge energy to clear
# MIN_EDGE_ON_STATIC, to get misdetected as an overlay near a frame corner
# if their face happens to fall there. Inpainting THAT means literally
# painting over part of a real face — seen on real footage as a seam right
# through an eyebrow/cheek that no amount of restoration-side seam
# protection (see face_restorer.py's _seam_* checks) can catch, since it's
# baked into the frame those checks treat as ground truth. A candidate
# overlapping where a face was actually detected across the same samples is
# rejected outright, no exceptions — a real channel-bug logo never sits on
# top of the subject.
FACE_PRESENCE_FRAC = 0.2      # a pixel counts as "a face was here" if detected in at least this fraction of samples
FACE_OVERLAP_REJECT_FRAC = 0.03  # reject a candidate if more than this fraction of its box falls on face territory
CONTEXT_MARGIN_PX = DEFAULT_CONTEXT_MARGIN_PX  # extra surrounding context given to the inpainter around the (padded)
                               # bbox — see inpainter.inpaint_boxes, which is where the number and its reason live


@dataclass(frozen=True)
class Overlay:
    """
    One burned-in logo: where it sits, and what it looks like there.

    `reference` and `mask` are what let a box detected over part of the video
    be applied only to the frames that actually carry it (see
    present_overlays) — `reference` is the overlay's median grayscale
    appearance over the segments that found it, `mask` marks which pixels of
    the box are the overlay itself rather than the footage around it. Both are
    bbox-sized and stay aligned with it through cropping.
    """
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in the frame space it was detected in
    reference: np.ndarray             # float32, h x w
    mask: np.ndarray                  # bool, h x w


def _content_bounds(sample_paths: list[str], width: int) -> tuple[int, int]:
    """
    The columns the actual picture occupies, as (x0, x1) inclusive.

    A video is not always the shape of the file it arrives in. Vertical
    footage gets published in a 16:9 container with the sides filled by a
    blurred blow-up of the picture itself, and on such a video every
    horizontal measurement taken against the frame is measuring the wrong
    rectangle. That is what hid a real logo: measured on one of these, the
    channel bug sat flush against the picture's right edge - 4px past it,
    even - while being 21% of the FRAME's width away from its border, so the
    corner test rejected it and the logo survived into every thumbnail.

    The bars give themselves away by what they are: a heavy blur of the same
    image. They keep its colours and its broad shapes and lose its detail, so
    a column of bar carries almost no gradient while a column of picture
    carries plenty. Measured on the video above: 1-3 against 11-130, with the
    boundary columns themselves spiking to ~127 because a hard vertical edge
    runs the full height of the frame there.

    Falls back to the whole frame whenever the answer isn't credible - no bars
    found, or a picture too narrow to believe - so a video without bars, which
    is nearly all of them, is treated exactly as it was before this existed.
    """
    energy = None
    read = 0
    for path in sample_paths:
        img = imread(path)
        if img is None:
            continue
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        column = cv2.magnitude(gx, gy).mean(axis=0)
        energy = column if energy is None else energy + column
        read += 1

    if energy is None or read < 3:
        return 0, width - 1
    energy /= read

    threshold = np.percentile(energy, 90) * CONTENT_DETAIL_FRAC
    lit = np.where(energy >= threshold)[0]
    if len(lit) == 0:
        return 0, width - 1

    x0, x1 = int(lit[0]), int(lit[-1])
    span = (x1 - x0 + 1) / width
    if span < MIN_CONTENT_W_FRAC or span > MAX_CONTENT_W_FRAC:
        return 0, width - 1

    log.info("logo: picture occupies columns %d-%d of %d (%.0f%% of the frame) "
             "- corners measured against that", x0, x1, width, span * 100)
    return x0, x1


def _sampled_band(sample_paths: list[str]):
    """
    Reads the top band of each sampled frame, plus a per-pixel tally of where
    faces were detected across them (see FACE_OVERLAP_REJECT_FRAC). Returns
    (stack, face_hits) or None if too few frames could be read for the
    median-based statistics below to mean anything.
    """
    band, face_hits = [], None
    for p in sample_paths:
        img = imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        band_h = int(h * TOP_BAND_FRAC)
        band_img = img[:band_h]
        band.append(cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY))

        if face_hits is None:
            face_hits = np.zeros((band_h, w), dtype=np.int32)
        for (fx, fy, fw, fh) in detect_faces(band_img):
            face_hits[fy:fy + fh, fx:fx + fw] += 1

    if len(band) < 3:
        return None
    return np.stack(band, axis=0).astype(np.float32), face_hits


def _segment_overlay(sample_paths: list[str], static_segments: bool = False,
                     content: tuple[int, int] | None = None):
    """
    Finds a static logo remnant near the top of the frame by comparing this
    segment's sampled frames against each other: real video content changes
    from frame to frame, a burned-in logo doesn't — so most frames agree
    tightly on its pixel values while disagreeing everywhere else. Returns
    (bbox, median_band, static_mask), or None if this segment shows no logo.

    `static_segments` selects the fallback pass (see _detect_pass): it scans
    the segments the normal pass skips for being too static, and pays for the
    lost protection with a higher edge-energy bar on any candidate.

    Uses median absolute deviation, not mean/std-dev: a logo commonly isn't
    on screen for literally 100% of a segment (fades in after an intro, fades
    out before an outro, etc.), so a handful of sampled frames can disagree
    with the rest even where a logo sits throughout most of it. A single such
    outlier is enough to blow up std-dev (e.g. 11 samples agreeing tightly
    plus 1 outlier already pushed a real logo's std past a threshold tuned for
    "always agrees" in testing). Median-based stats keep reporting "this pixel
    doesn't change" as long as a majority of samples agree, tolerating a
    minority of outlier frames.
    """
    sampled = _sampled_band(sample_paths)
    if sampled is None:
        return None
    stack, face_hits = sampled
    n, band_h, band_w = stack.shape

    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)
    static_mask = mad < STATIC_MAD_THRESHOLD

    # Each pass takes the segments the other one leaves: the normal pass skips
    # a segment that never left one shot (nothing there can distinguish an
    # overlay from the set it was filmed on — see MAX_SEGMENT_STATIC_FRAC),
    # and the fallback pass looks at only those.
    too_static = static_mask.mean() > MAX_SEGMENT_STATIC_FRAC
    if too_static != static_segments:
        return None
    min_edge = FALLBACK_MIN_EDGE_ON_STATIC if static_segments else MIN_EDGE_ON_STATIC

    # Gradient energy of the (static) scene itself — see MIN_EDGE_ON_STATIC:
    # distinguishes a graphic overlay's static pixels from a physically
    # static but visually flat surface (a wall, a smooth prop) that a
    # locked-off camera makes look just as unchanging as a real overlay.
    gx = cv2.Sobel(median, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(median, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = cv2.boxFilter(np.sqrt(gx * gx + gy * gy), -1, (9, 9))

    # Seed candidate blobs from pixels that are BOTH static and graphic, not
    # from the static mask alone. A logo commonly sits directly against a
    # physically static, visually flat surface (a wall, a backdrop) with no
    # gap between them in the static mask at all — closing on the static mask
    # alone let a single connected blob swallow logo and wall together, which
    # then got discarded whole for being too large (the logo was never a
    # separate candidate to keep). Seeding on static-and-edge-rich pixels
    # keeps the flat surface out of the blobs from the start; the plain
    # static mask (raw_static) is still consulted afterward as the fill test.
    graphic_mask = (static_mask & (edge_energy > min_edge)).astype(np.uint8) * 255
    raw_static = static_mask  # un-closed — the fill test below must not credit bridged gaps

    # A logo is one (or a couple of nearby) solid blob, not scattered single
    # pixels — morphological close fills small gaps. The kernel is a fraction
    # of frame width, not a fixed 5px: letter spacing scales with resolution,
    # so a fixed kernel that merged a 1080p wordmark left a 4K one shattered
    # into per-letter blobs, none of which cleared the (area-proportional)
    # minimum — a real "B.O.A.T.S." logo went undetected entirely that way.
    k = max(3, int(band_w * CLOSE_KERNEL_FRAC) | 1)
    closed = cv2.morphologyEx(graphic_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Two different scenes can coincidentally agree on a region's brightness
    # (e.g. similar sky, a shared dark corner) without any overlay actually
    # being there — that shows up as a second, unrelated "static" blob
    # elsewhere in the band. Unioning every qualifying blob into one bbox let
    # a single spurious match on the far side of the frame drag the bbox into
    # a strip spanning almost the whole width; keeping only the single
    # largest blob instead went too far the other way, since a wordmark is
    # genuinely several blobs — that lost the left half of a badge logo, and
    # on another video handed the whole detection to a spurious blob that
    # happened to out-area any single letter of the real logo.
    #
    # So: drop anything already too large to plausibly be a logo, then cluster
    # the rest by proximity (within CLUSTER_GAP_FRAC), so adjacent
    # letters/borders/shadows join while an unrelated blob hundreds of pixels
    # away stays separate, and keep the best-scoring cluster. Size limits are
    # re-checked on every merge so a cluster can't creep across the frame.
    blobs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > MAX_OVERLAY_W_FRAC * band_w or h > MAX_OVERLAY_H_FRAC * band_h:
            continue
        blobs.append((x, y, w, h, cv2.contourArea(c)))
    if not blobs:
        return None

    gap_x = band_w * CLUSTER_GAP_FRAC
    gap_y = gap_x * 0.5

    # Merge every blob into proximity clusters — not just one grown from the
    # largest, because the largest cluster is not always the logo. Each is
    # then judged on its own merits below.
    clusters = [[b[0], b[1], b[0] + b[2], b[1] + b[3], b[4]] for b in blobs]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if max(0, max(a[0] - b[2], b[0] - a[2])) > gap_x:
                    continue
                if max(0, max(a[1] - b[3], b[1] - a[3])) > gap_y:
                    continue
                nx0, ny0 = min(a[0], b[0]), min(a[1], b[1])
                nx1, ny1 = max(a[2], b[2]), max(a[3], b[3])
                if (nx1 - nx0) > MAX_OVERLAY_W_FRAC * band_w or (ny1 - ny0) > MAX_OVERLAY_H_FRAC * band_h:
                    continue
                clusters[i] = [nx0, ny0, nx1, ny1, a[4] + b[4]]
                clusters.pop(j)
                merged = True
                break
            if merged:
                break

    min_area = MIN_OVERLAY_AREA_FRAC * band_w * band_h
    candidates = []
    for x0, y0, x1, y1, area in clusters:
        if area < min_area:
            continue
        # A real channel-bug logo hugs a left or right border of the PICTURE,
        # not mid-frame. Which is not the frame's border when the video is
        # pillarboxed - see _content_bounds.
        left_edge, right_edge = content if content else (0, band_w - 1)
        margin = CORNER_FRAC * (right_edge - left_edge + 1)
        if not (x0 < left_edge + margin or x1 > right_edge - margin):
            continue
        # Sanity checks on the cluster's full bbox (which can extend a little
        # past the graphic pixels that seeded it, via padding/merging): it
        # should still be mostly static...
        box_static = raw_static[y0:y1, x0:x1]
        if box_static.mean() < MIN_STATIC_FILL:
            continue
        # ...and where it IS static, that's graphic (bold letters/borders),
        # not a flat surface that a fixed camera merely renders motionless —
        # see MIN_EDGE_ON_STATIC.
        if edge_energy[y0:y1, x0:x1][box_static].mean() < min_edge:
            continue
        # Never a real face — see FACE_OVERLAP_REJECT_FRAC.
        face_territory = face_hits[y0:y1, x0:x1] >= (n * FACE_PRESENCE_FRAC)
        if face_territory.mean() > FACE_OVERLAP_REJECT_FRAC:
            continue
        candidates.append((x0, y0, x1, y1, area))

    if not candidates:
        return None

    x0, y0, x1, y1, _ = max(candidates, key=lambda c: c[4])
    return (x0, y0, x1, y1), median, static_mask


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """
    How alike two same-shaped grayscale patches are, as a correlation in
    [-1, 1]. Returns 0 when either side is featureless, which is the answer
    that makes a caller treat "can't tell" as "not a match".
    """
    if a.size < 16 or a.shape != b.shape:
        return 0.0
    sd_a, sd_b = a.std(), b.std()
    if sd_a < 1e-3 or sd_b < 1e-3:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sd_a * sd_b))


def _same_logo(group: dict, median: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """
    Whether a fresh detection is the SAME logo as one already grouped, rather
    than a different one that happens to occupy the same corner.

    Position alone used to decide this, and position alone is not enough. A
    channel that replaces its bug partway through the video puts the
    replacement exactly where the old one was, so the two boxes overlap almost
    perfectly and were merged — leaving one overlay whose stored appearance
    was the median of two different graphics. That median resembles neither,
    so the presence test then rejected BOTH on every frame and the logo was
    never painted out at all. Measured on a real video: two wordmarks in one
    spot, one of them left burned into every thumbnail.

    Compared over the overlap of the two boxes, since a merge would only ever
    join them there anyway.
    """
    gx0, gy0, gx1, gy1 = group["box"]
    bx0, by0, bx1, by1 = box
    x0, y0 = max(gx0, bx0), max(gy0, by0)
    x1, y1 = min(gx1, bx1), min(gy1, by1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    here = median[y0:y1, x0:x1]
    return any(_correlate(here, m[y0:y1, x0:x1]) >= MERGE_APPEARANCE_MIN
               for m, _ in group["members"])


def _detect_pass(sample_paths: list[str], static_segments: bool,
                 content: tuple[int, int] | None = None) -> list[dict]:
    """
    One scan of the sample's segments, grouping detections that describe the
    same logo. `static_segments` picks which half of the segments this pass
    takes and how strict it is — see _segment_overlay.
    """
    groups: list[dict] = []
    for start in range(0, len(sample_paths), SEGMENT_SAMPLE_COUNT):
        segment = sample_paths[start:start + SEGMENT_SAMPLE_COUNT]
        if len(segment) < 3:
            continue   # trailing remainder too short for the median statistics
        found = _segment_overlay(segment, static_segments=static_segments, content=content)
        if found is None:
            continue
        box, median, static_mask = found
        for group in groups:
            if _iou(group["box"], box) >= MERGE_IOU and _same_logo(group, median, box):
                group["box"] = (min(group["box"][0], box[0]), min(group["box"][1], box[1]),
                                max(group["box"][2], box[2]), max(group["box"][3], box[3]))
                group["members"].append((median, static_mask))
                break
        else:
            groups.append({"box": box, "members": [(median, static_mask)]})
    return groups


def detect_static_overlays(frame_paths: list[str]) -> list[Overlay]:
    """
    Every burned-in overlay this video uses, in the frame coordinates of the
    passed frames. Empty when the video has none.

    Runs once per video (cheap — a few dozen frames, a thin band), not once
    per frame. Each segment of the sample is analysed independently (see the
    module header), and detections that describe the same logo are merged:
    the box is their union, because which parts of a logo clear the static
    test varies between segments and under-covering leaves a visible sliver
    while over-covering only inpaints a little extra corner.

    Two passes, because MAX_SEGMENT_STATIC_FRAC skips any segment too static
    to tell an overlay from the set behind it — which is the right call per
    segment, but on a video shot ENTIRELY on one locked-off set it skips every
    segment and returns nothing for a video that plainly has a logo (measured
    on a single-location shoot: 22-78% static per segment, against the 10-21%
    the threshold was calibrated on). So when the normal pass comes back
    empty, the skipped segments are scanned after all, under the stricter
    FALLBACK_MIN_EDGE_ON_STATIC. Only ever a fallback: a video with even one
    scene-crossing segment is still judged solely by the normal pass, so this
    cannot change what already works.
    """
    if len(frame_paths) < 3:
        return []

    step = max(1, len(frame_paths) // SAMPLE_FRAME_COUNT)
    sample_paths = frame_paths[::step][:SAMPLE_FRAME_COUNT]

    # Each group is one logo: its box, plus the (median band, static mask) of
    # every segment that found it.
    # Where the picture actually is, measured once for the whole video: every
    # segment is the same footage in the same container, so this cannot differ
    # between them.
    probe = imread(sample_paths[0])
    content = _content_bounds(sample_paths, probe.shape[1]) if probe is not None else None

    groups = _detect_pass(sample_paths, static_segments=False, content=content)
    if not groups:
        groups = _detect_pass(sample_paths, static_segments=True, content=content)

    overlays = []
    for group in groups:
        x0, y0, x1, y1 = group["box"]
        # The overlay's appearance, and which pixels are the overlay itself:
        # a pixel counts as the logo only if it held still in EVERY segment
        # that found this logo — the footage behind it did not.
        reference = np.median(np.stack([m[y0:y1, x0:x1] for m, _ in group["members"]]), axis=0)
        mask = np.logical_and.reduce([s[y0:y1, x0:x1] for _, s in group["members"]])
        if mask.mean() < MIN_PRESENCE_MASK_FRAC:
            mask = np.ones_like(mask)
        overlays.append(Overlay(
            bbox=(x0, y0, x1 - x0, y1 - y0),
            reference=reference.astype(np.float32),
            mask=mask,
        ))
    return overlays


def overlays_to_cropped_coords(overlays: list[Overlay], full_height: int) -> list[Overlay]:
    """
    Maps overlays found on the UNCROPPED frames into the cropped frame's
    coordinate space, dropping any that lie entirely above the crop line.

    Detection used to run twice — once over the cropped frames and once over
    the full ones — which meant reading and running face detection over a
    second set of full-resolution frames for a second answer to almost the
    same question. Running it only on the full frames and deriving the cropped
    boxes is both half the work and strictly more informative: the full
    frame's scanned band (its top TOP_BAND_FRAC) fully contains the region the
    cropped frame's own band covers, so nothing the cropped-frame pass could
    have found is missed, while a logo straddling the crop line is now seen
    whole instead of as two unrelated partial detections.
    """
    crop_top = int(full_height * CROP_TOP_PCT)
    crop_bottom = int(full_height * (1 - CROP_BOTTOM_PCT))

    cropped = []
    for overlay in overlays:
        x, y, w, h = overlay.bbox
        y1 = max(y, crop_top)
        y2 = min(y + h, crop_bottom)
        if y2 <= y1:
            continue   # this logo sits entirely in the band the crop removes
        # The reference/mask are sliced the same way, so they stay aligned
        # with the box they describe.
        top, bottom = y1 - y, y2 - y
        cropped.append(Overlay(
            bbox=(x, y1 - crop_top, w, y2 - y1),
            reference=overlay.reference[top:bottom],
            mask=overlay.mask[top:bottom],
        ))
    return cropped


def present_overlays(image: np.ndarray, overlays: list[Overlay]) -> list[Overlay]:
    """
    Which of these overlays are actually on screen in `image` — a frame at the
    native resolution the overlays were detected in.

    A video's overlays don't all run its whole length (that's why they're
    detected per segment in the first place), so the box a logo occupies is
    ordinary footage on the frames where that logo isn't showing. Inpainting
    it there would smear real content for no reason — on the video this was
    built for, the first half's banner covers a quarter of the frame's width
    right where the second half's frames have none.

    Presence is measured as the correlation between the frame's pixels and the
    overlay's stored appearance, over the overlay's own pixels only, so a
    partly translucent bug (whose tone shifts with whatever it sits on, while
    its shape doesn't) still reads as present. See PRESENCE_CORRELATION_MIN.
    """
    if not overlays:
        return []

    ih, iw = image.shape[:2]
    present = []
    for overlay in overlays:
        x, y, w, h = overlay.bbox
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            continue   # a frame of a different size than this was detected on
        region = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        here = region[overlay.mask]
        there = overlay.reference[overlay.mask]
        if here.size < 16:
            continue
        if _correlate(here, there) >= PRESENCE_CORRELATION_MIN:
            present.append(overlay)
    return present


def bbox_in_scaled_coords(bbox: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (int(x * scale), int(y * scale), int(w * scale), int(h * scale))


def remove_logo(image: np.ndarray, scaled_bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """
    Inpaints each given bounding box (in this image's own coordinates —
    normally the full scaled pre-crop image, so there's real surrounding
    context on every side even when the logo sits close to the eventual crop
    edge).

    The painting itself is inpainter.inpaint_boxes, which subtitle_remover
    also calls; what belongs to logos and stays here is how far the box is
    grown before it is painted. LaMa-vs-Telea preference and fallback live in
    inpainter.inpaint.
    """
    pad = max(INPAINT_PADDING_MIN_PX, int(image.shape[1] * INPAINT_PADDING_FRAC))
    return inpaint_boxes(image, scaled_bboxes, pad=pad, context=CONTEXT_MARGIN_PX)
