"""
character_identifier.py — Grouping the candidate pool by who is in the frame.

The selector uses these groups to guarantee coverage, so the failure that
matters here is UNDER-splitting: two people merged into one cluster means the
lower-scoring of them never gets a slot and vanishes from the grid entirely,
however much screen time they had. Over-splitting one person across two
clusters only costs a little variety, because the frames inside each cluster
are already deduplicated. Everything below is biased accordingly.
"""

import cv2
import numpy as np

from image_utils import expand_box

# --- descriptor -------------------------------------------------------------

# Face crop size for the shape (eigenface) half of the descriptor.
FACE_SIZE = 48

# How far past the detected box the descriptor looks. The Haar box is jaw-to-
# brow only; hair volume, hairline and the top of the shoulders are some of
# the most identity-bearing pixels available at this resolution, and they sit
# just outside it.
DESCRIPTOR_BOX_FACTOR = 1.5

# Hue/saturation bins for the color half. Coarse on purpose — this is meant to
# separate skin and hair tone, not to fingerprint a background.
COLOR_BINS = (12, 4)

# Weight of the color half relative to the shape half. Each half is
# L2-normalised on its own first, so this is a straight energy ratio between
# them regardless of their very different dimensionality.
#
# The shape half is histogram-equalised (deliberately — it has to survive a
# lighting change), which throws away exactly the skin/hair tone that tells
# this footage's cast apart most reliably. Carrying tone separately is what
# stops two different people lit the same way from landing closer together
# than one person lit two ways.
COLOR_WEIGHT = 1.0

# --- clustering -------------------------------------------------------------

# PCA components kept before clustering.
N_COMPONENTS = 30

# Roughly how many candidates should fall in a cluster. Only used to avoid
# asking for more clusters than a short pool can meaningfully fill.
FRAMES_PER_CLUSTER = 25

# Hard ceiling on clusters. Also the ceiling on how many "characters" the
# selector will reserve slots for: at MIN_FRAMES_PER_CHARACTER=2 this fits
# inside a 20-slot grid with room left over for the score-ranked fill.
MAX_CLUSTERS = 8


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def _face_vector(image: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray | None:
    """
    One face as a fixed-length vector: an equalised grayscale crop (shape) and
    a coarse hue/saturation histogram (tone), each L2-normalised and then
    concatenated.
    """
    x1, y1, x2, y2 = expand_box(face, DESCRIPTOR_BOX_FACTOR, image.shape)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE))
    gray = cv2.equalizeHist(gray)
    shape_part = _l2(gray.astype(np.float32).flatten())

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(COLOR_BINS), [0, 180, 0, 256])
    tone_part = _l2(hist.flatten().astype(np.float32)) * COLOR_WEIGHT

    return np.concatenate([shape_part, tone_part])


def face_descriptor(image: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray | None:
    """
    The per-face vector cluster_characters projects into eigenspace. Computed
    while the frame is already decoded during scoring (see
    frame_selector._score_one) rather than in a second pass that would have
    to hold every candidate's full-resolution image in memory at once.
    """
    return _face_vector(image, face)


def _cluster(X: np.ndarray, k: int) -> np.ndarray:
    """
    Greedy k-center under cosine distance (X arrives L2-normalised): start
    from the most typical face, then repeatedly take whichever face is
    FARTHEST from every center chosen so far, and finally assign each face to
    its nearest center.

    Not k-means, and not k-means++ either — deliberately, because both spend
    their clusters where the data is dense. On a talking-head video the cast
    is wildly unbalanced: the host is 300 frames and a guest who walks through
    one scene is 4. Lloyd iterations move every center toward the mass it
    already holds, so those 4 frames end up inside the host's cluster no
    matter how many clusters are asked for, and the guest is then outscored
    by the host inside their shared group and never reaches the grid. That was
    measured — over-segmenting alone did not fix it, because k-means gave the
    host six clusters and the guest none.

    Picking centers by distance instead makes cluster count independent of
    screen time: a face that looks unlike anything chosen so far becomes a
    center on its own, whether it occurs four times or four hundred. Centers
    are never re-fitted for the same reason.
    """
    # First center: the face closest to the pool's mean direction, so cluster
    # 0 is the typical face rather than the first outlier the greedy pass
    # would otherwise latch onto.
    centers = [int(np.argmax(np.dot(X, _l2(X.mean(axis=0)))))]
    nearest = np.dot(X, X[centers[0]])
    for _ in range(1, k):
        far = int(np.argmin(nearest))
        centers.append(far)
        nearest = np.maximum(nearest, np.dot(X, X[far]))

    return np.argmax(np.dot(X, X[centers].T), axis=1)


def _cluster_count(n: int) -> int:
    """
    How many clusters to ask for, for a pool of n faces.

    Deliberately NOT an elbow/inertia estimate of the true cast size. That was
    the previous behaviour and it is what emptied the grid: on raw equalised
    eigenfaces the elbow lands at k=2 for almost any real video, so a
    four-person cast collapsed into two groups, and the two people who shared
    a group with someone higher-scoring never surfaced — twenty slots, two
    faces.

    Asking for more clusters than there are people is the cheap direction to
    be wrong in: the extra clusters split one person by pose or lighting, and
    since the selector takes the best frames of each cluster, the result is a
    wider spread of that person rather than a lost one.
    """
    return int(max(2, min(MAX_CLUSTERS, n // FRAMES_PER_CLUSTER or 2, n // 2)))


def cluster_characters(candidates: list[dict]) -> int:
    """
    Eigenface-style clustering over the candidate pool:
      1. Collect every face descriptor (shape + tone, see _face_vector).
      2. Centre the data and take the top N_COMPONENTS principal components.
      3. Project every face into that eigenspace and L2-normalise.
      4. Greedy k-center (see _cluster) to assign character IDs.

    Every resulting cluster is kept. Tiny clusters used to be merged into the
    nearest big one as "noise", which is indistinguishable from a character
    who was only on screen for a couple of seconds — precisely the case the
    selector is supposed to protect. A stray misdetection surviving into its
    own cluster costs one slot; a real person merged away costs them all.

    Returns the number of clusters (see _cluster_count — this is an upper
    bound on the cast size, not a measurement of it).
    """
    valid = [(i, c) for i, c in enumerate(candidates) if c.get("descriptor") is not None]
    for c in candidates:
        if c.get("descriptor") is None:
            c["character_id"] = -1

    if len(valid) < 2:
        for _, c in valid:
            c["character_id"] = 0
        return 1

    valids = [c for _, c in valid]
    X = np.array([c["descriptor"] for c in valids], dtype=np.float32)

    # PCA via compact SVD
    mean = X.mean(axis=0)
    Xc = X - mean
    n_comp = min(N_COMPONENTS, len(X) - 1)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    projections = Xc @ Vt[:n_comp].T

    # L2-normalise projections for cosine distance
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    projections /= np.where(norms > 1e-6, norms, 1.0)

    labels = _cluster(projections, _cluster_count(len(X)))
    for local_i, c in enumerate(valids):
        c["character_id"] = int(labels[local_i])

    # Renumber IDs 0..K-1 so empty clusters leave no gaps.
    id_map: dict[int, int] = {}
    for c in candidates:
        cid = c["character_id"]
        if cid != -1 and cid not in id_map:
            id_map[cid] = len(id_map)
    for c in candidates:
        if c["character_id"] != -1:
            c["character_id"] = id_map[c["character_id"]]

    return len(id_map)
