"""
updater.py — how an installed copy gets new features without being reinstalled.

The install is ~5 GB, nearly all of it CUDA PyTorch and model weights, and
none of that changes when the app does. The app itself is about a megabyte of
Python and static files sitting in one directory:

    <install>/app/          backend/ + frontend/ + requirements.txt

So an update is that directory being replaced, and a patch is a zip of it.
Roughly a megabyte, applied in a couple of seconds, with the 5 GB underneath
untouched. New channels do not even need that much — they are folders of JSON
under content/ (see frontend/content/README.md), which a patch simply carries
along with everything else.

    <install>/version.json      what is installed, and what has been applied
    <install>/updater/          this file, ed25519.py, launcher.py
    <install>/backup/app-X.Y.Z  the previous app/, kept so a bad patch is undoable

Run by the launcher at startup, before the servers come up, which is the one
moment the app's files are reliably not in use. Also usable by hand:

    python updater.py --check
    python updater.py --apply
    python updater.py --apply-file "some-patch.tmpatch"
    python updater.py --rollback
    python updater.py --status

Everything it does is safe to interrupt. The app directory is swapped only
once the replacement is fully staged, and the previous one is kept until the
next update rather than deleted on success.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

APP_NAME = "Thumbnail Maker"

# Where Windows lists the fonts installed for one user. Per-user needs no
# administrator rights and touches nothing outside this user's profile, which
# is what lets an update register a face at all.
FONTS_KEY = r"Software\Microsoft\Windows NT\CurrentVersion\Fonts"

# The one file inside a patch that says what the patch is.
MANIFEST_NAME = "manifest.json"

# What the patch payload is called inside the zip, and where it lands.
PAYLOAD_DIR = "app"

# Other top-level directories a patch is allowed to replace, relative to the
# install root. Just the one: the updater has to be able to fix itself, or a
# bug in this file would be permanent on every machine that already has it.
# Anything outside this list and app/ is refused rather than written.
SIDE_TREES = ("updater",)

# Directories inside app/ that an update carries across rather than replaces.
#
# backend/gfpgan/weights is ~185 MB of face-detection and parsing models that
# facexlib resolves RELATIVE TO THE WORKING DIRECTORY, which for the backend
# is app/backend. They are not part of the app and are not in any patch, so
# replacing app/ wholesale without this would silently throw them away and
# make the next launch re-download them — on a metered connection, quietly,
# with no indication of why the app was suddenly busy for ten minutes.
KEEP_ACROSS_UPDATES = ["backend/gfpgan", "backend/models"]

# Scratch that is deliberately NOT carried across: frames and crops from
# whatever video was open last. It is rebuilt on demand, it can be gigabytes,
# and a patch that changes the pipeline is exactly when a stale cache of its
# output stops being safe to keep.
DISCARD_ON_UPDATE = ["backend/temp"]

# How many previous app/ directories to keep. One is what --rollback needs;
# beyond that they are a megabyte each of nothing anybody will read.
BACKUPS_KEPT = 1

# The port the backend answers on, and the endpoint that proves it is ours
# rather than something else on the port. An update while the app is running
# would replace files out from under a live process.
BACKEND_PORT = 8000
HEALTH_URL = f"http://127.0.0.1:{BACKEND_PORT}/process-video/progress"

# Deliberately short. This runs in front of the launcher, and a feed that is
# unreachable must cost the user a second at most — an app that will not start
# because a server is down is a far worse failure than one that starts a
# version behind.
FEED_TIMEOUT_S = 6
DOWNLOAD_TIMEOUT_S = 300


# --------------------------------------------------------------------------
# what the channel packs bring with them
# --------------------------------------------------------------------------

# A channel is data, not code. It names the type face its titles are set in
# and, if it has one, the texture its highlights are filled with, and the page
# fetches both by URL at run time (frontend/js/channels.js). Nothing fails
# loudly when one of those files is not there: a texture simply does not
# appear, and a missing face is worse than that, because the title is then
# laid out against a fallback's metrics instead of the ones every size and box
# in the design was measured from.
#
# So the files are resolved from the packs themselves rather than listed
# anywhere, and checked: build_installer.py refuses to build a payload with
# one missing, and engine.verify() refuses to finish an install with one
# missing. A channel added next month is covered by both without anyone
# remembering to add it to a list.
#
# This lives here, rather than in engine.py where it is mostly used, for the
# same reason KEEP_ACROSS_UPDATES does: updater.py is the one module that
# exists both in the installer and on an installed machine, so a rule that
# both have to agree on can only have one definition here.

def read_json(path):
    """A dict from a JSON file, or None. A pack that will not parse is not a pack."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _dict(obj, key):
    """obj[key] when it is a dict, {} otherwise — packs are hand-written."""
    value = obj.get(key) if isinstance(obj, dict) else None
    return value if isinstance(value, dict) else {}


def asset_path(rel, frontend, pack_dir):
    """
    The file a path written inside a pack refers to.

    The disk half of assetUrl() in frontend/js/channels.js, and it follows the
    same two rules: a bare path is relative to the frontend root, which is
    where the shipped fonts and textures live, and `./x` is relative to the
    pack's own folder, which is what a self-contained channel carrying its own
    font uses. An absolute URL belongs to someone else and is not ours to
    check, so it is not returned at all.
    """
    if not isinstance(rel, str) or not rel.strip():
        return None
    if re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*:)?//", rel) or rel.startswith("data:"):
        return None
    if rel.startswith("./"):
        return Path(pack_dir) / rel[2:]
    return Path(frontend) / rel


def content_faces(frontend):
    """
    Every type face the packs under `frontend` declare: (label, spec, path).

    Both routes a face can arrive by — its own file under content/faces/, and
    a face object written inline in a channel.json — because both are drawn on
    the canvas, and neither is any more allowed to be missing than the other.
    """
    frontend = Path(frontend)
    content = frontend / "content"
    found = []

    faces_dir = content / "faces"
    if faces_dir.is_dir():
        for path in sorted(faces_dir.glob("*.json")):
            spec = read_json(path)
            if spec:
                found.append((f"faces/{path.name}", spec,
                              asset_path(spec.get("file"), frontend, path.parent)))

    # Both places a channel can write a face inline: the block's, and the one
    # its highlight changes voice to (see `highlight.face` in channels.js). The
    # second is drawn on the same canvas as the first and is no more allowed to
    # be missing, so it is verified and installed by the same route.
    for pack_dir, spec in _packs(content):
        title = _dict(spec, "title")
        for face in (title.get("face"), _dict(title, "highlight").get("face")):
            if isinstance(face, dict):
                found.append((f"channels/{pack_dir.name}/channel.json", face,
                              asset_path(face.get("file"), frontend, pack_dir)))

    return [(label, spec, path) for label, spec, path in found if path]


# Where a channel pack can name a picture, and how to reach it from the
# manifest. A list rather than four hard-coded lookups because the answer to
# "what files does this pack need" has to stay true as packs learn to bring new
# ones: a channel whose backdrop never made it into the payload is a channel
# that installs cleanly and then draws nothing, which is precisely the silent
# failure this whole check exists to prevent.
CONTENT_TEXTURE_KEYS = [
    ("title", "highlight", "texture"),   # the foil a highlight fills its letters with
    ("background", "texture"),           # the backdrop a channel composes on
    ("stamp", "texture"),                # the brand mark it locks to a corner
]


def content_assets(frontend):
    """
    Every file the packs under `frontend` name: (what named it, path).

    The faces, plus every picture a channel brings with it — see
    CONTENT_TEXTURE_KEYS.
    """
    frontend = Path(frontend)
    assets = [(f"{label} -> {spec.get('file')}", path)
              for label, spec, path in content_faces(frontend)]

    for pack_dir, spec in _packs(frontend / "content"):
        for keys in CONTENT_TEXTURE_KEYS:
            node = spec
            for key in keys[:-1]:
                node = _dict(node, key)
            texture = node.get(keys[-1])
            path = asset_path(texture, frontend, pack_dir)
            if path:
                assets.append((f"channels/{pack_dir.name}/channel.json -> {texture}", path))
    return assets


def missing_content_assets(frontend):
    """
    The subset of content_assets() that is absent or empty, described.

    Empty counts as missing: a zero-byte font loads as a broken face, which
    fails exactly the way an absent one does without looking absent.
    """
    problems = []
    for label, path in content_assets(frontend):
        if not path.exists():
            problems.append(f"{label}: missing ({path})")
        elif path.stat().st_size == 0:
            problems.append(f"{label}: empty ({path})")
    return problems


def font_value_name(path, full_name):
    """
    What Windows lists a font under, in its own convention.

    The flavour tag is not decoration: an .otf listed as (TrueType) is a face
    Windows will sometimes decline to load, which produces exactly the silent
    fallback the local() route exists to prevent.
    """
    kind = "TrueType" if Path(path).suffix.lower() in (".ttf", ".ttc") else "OpenType"
    return f"{full_name} ({kind})"


def _packs(content):
    """(folder, manifest) for every channel pack under `content`."""
    folder = Path(content) / "channels"
    if not folder.is_dir():
        return []
    out = []
    for pack_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        spec = read_json(pack_dir / "channel.json")
        if spec:
            out.append((pack_dir, spec))
    return out


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_version(text):
    """
    "1.2.3" -> (1, 2, 3), for comparison. Anything unparsable sorts lowest.

    Tolerant on purpose: the versions here are ours, and a comparison that
    raised would turn a typo in a feed into an update mechanism that cannot
    run at all.
    """
    parts = re.findall(r"\d+", str(text or ""))
    return tuple(int(p) for p in parts[:4]) or (0,)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def app_is_running():
    """True when our own backend is answering — i.e. the app is open right now."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1) as r:
            return r.status == 200 and b"stage" in r.read(400)
    except Exception:
        return False


def open_url(url, timeout):
    """
    Reads an http(s) URL, a file:// URL, or a plain local/UNC path.

    The plain path is not a convenience — it is the cheapest useful way to
    ship patches: a folder on a network share, or a USB stick, needs no
    hosting and no certificate, and it is what makes the whole mechanism
    testable without publishing anything.
    """
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}-updater"})
        return urllib.request.urlopen(req, timeout=timeout)
    return open(url, "rb")


def join_url(base, name):
    """Resolves a patch filename against the feed it was listed in."""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", str(base)):
        return urllib.parse.urljoin(base, name)
    return str(Path(base).parent / name)


class UpdateError(Exception):
    """Something went wrong that the user needs told about, in words they can act on."""


# --------------------------------------------------------------------------
# install state
# --------------------------------------------------------------------------

class State:
    """
    <install>/version.json — what is installed and what has happened to it.

    Separate from app/release.json, which is what a BUILD says about itself.
    This is what a MACHINE says about itself, and it outlives every app
    directory that gets swapped underneath it.
    """

    def __init__(self, install_dir):
        self.dir = Path(install_dir)
        self.path = self.dir / "version.json"
        self.data = self._read()

    def _read(self):
        blank = {
            "schema": 1,
            "app": APP_NAME,
            "app_version": "0.0.0",
            "installed": None,
            "updated": None,
            "feed": None,
            "public_key": None,
            "history": [],
            "extra_pins": [],
            "last_check": None,
            "last_result": None,
        }
        try:
            with open(self.path, encoding="utf-8") as f:
                blank.update(json.load(f))
        except (OSError, ValueError):
            # A missing or unreadable version.json is not fatal: the release
            # file below still says what this build is, and the worst case is
            # an update being offered that turns out to already be applied.
            pass

        release = self.release()
        if blank["app_version"] == "0.0.0" and release.get("version"):
            blank["app_version"] = release["version"]
        # The feed and the key are properties of the RELEASE, so a patch can
        # move them; version.json only overrides when someone has deliberately
        # pointed this machine somewhere else.
        for key in ("feed", "public_key"):
            if blank.get(key) is None:
                blank[key] = release.get(key)
        return blank

    def release(self):
        """app/release.json — the identity of the build currently installed."""
        try:
            with open(self.dir / "app" / "release.json", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)

    @property
    def version(self):
        return self.data.get("app_version") or "0.0.0"

    @property
    def public_key(self):
        raw = self.data.get("public_key")
        if not raw:
            return None
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            raise UpdateError(
                "The update public key in version.json is not valid base64, so no patch "
                "can be checked against it. Fix it or set it to null."
            )


# --------------------------------------------------------------------------
# signatures
# --------------------------------------------------------------------------

def verify_signature(data, signature_b64, public_key, what):
    """
    Checks a patch against the publisher's key, when one is configured.

    An install with no key configured takes the SHA-256 alone, which proves
    the download arrived intact and nothing about who wrote it. An install
    WITH a key refuses anything unsigned — including, deliberately, the case
    where ed25519.py has gone missing: "we could not check" and "it checked
    out" must never come to the same thing.
    """
    if not public_key:
        return

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ed25519
    except ImportError:
        raise UpdateError(
            "This install requires patches to be signed, but ed25519.py is missing from "
            "the updater, so the signature cannot be checked. Reinstall to repair it."
        )

    if not signature_b64:
        raise UpdateError(f"{what} is not signed, and this install only accepts signed patches.")
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except Exception:
        raise UpdateError(f"{what} carries a signature that is not valid base64.")
    if not ed25519.verify(data, signature, public_key):
        raise UpdateError(
            f"{what} is not signed by this install's publisher key. It has been rejected "
            "and nothing has been changed."
        )


# --------------------------------------------------------------------------
# the feed
# --------------------------------------------------------------------------

def fetch_feed(url):
    """Reads updates.json. Returns the parsed document."""
    try:
        with open_url(url, FEED_TIMEOUT_S) as r:
            raw = r.read()
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise UpdateError(f"Could not reach the update server at {url} ({e}).")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise UpdateError(f"The update server returned something that is not an update list ({e}).")


def plan(current_version, feed, feed_url):
    """
    The patches to apply, in order, to get from `current_version` to the newest.

    Patches are whole app directories rather than diffs, so the usual answer
    is one patch however far behind the install is — jumping 1.0 straight to
    1.4 is the same operation as 1.3 to 1.4. `min_app_version` is what makes
    that a claim the publisher gets to make rather than an assumption: a patch
    that needs a step in between (a new pin, a migrated file) says so, and the
    ones before it are applied first.
    """
    entries = []
    for raw in feed.get("patches", []):
        entry = dict(raw)
        entry["_version"] = parse_version(entry.get("version"))
        entry["_min"] = parse_version(entry.get("min_app_version") or "0.0.0")
        if not entry.get("url"):
            if not entry.get("file"):
                continue
            entry["url"] = join_url(feed_url, entry["file"])
        entries.append(entry)
    entries.sort(key=lambda e: e["_version"])

    steps = []
    at = parse_version(current_version)
    while True:
        # The furthest jump this install is allowed to make right now.
        reachable = [e for e in entries if e["_version"] > at and e["_min"] <= at]
        if not reachable:
            break
        chosen = reachable[-1]
        steps.append(chosen)
        at = chosen["_version"]
    return steps


# --------------------------------------------------------------------------
# applying a patch
# --------------------------------------------------------------------------

def download(url, dest, expected_sha, log):
    log(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(url, DOWNLOAD_TIMEOUT_S) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise UpdateError(f"The patch could not be downloaded ({e}).")

    got = sha256_file(dest)
    if expected_sha and got.lower() != str(expected_sha).lower():
        dest.unlink(missing_ok=True)
        raise UpdateError(
            "The downloaded patch does not match the checksum the update list gave for it, "
            "so it is damaged or has been tampered with. Nothing has been changed."
        )
    log(f"Downloaded {dest.stat().st_size / 1024:.0f} KB, checksum verified")
    return dest


def sidecar_signature(patch_path):
    """
    The signature written beside a patch file, for applying one by hand.

    Online, the signature travels in the feed entry — it is a property of the
    listing, so a feed cannot offer a patch it has no signature for. Offline
    there is no listing, so build_patch.py also writes `<patch>.sig` next to
    the patch and the two are copied around together. Returns None when there
    is no sidecar, which an unsigned install accepts and a signed one refuses.
    """
    sig = Path(str(patch_path) + ".sig")
    if not sig.is_file():
        return None
    return sig.read_text(encoding="utf-8").strip()


def read_manifest(patch_path):
    try:
        with zipfile.ZipFile(patch_path) as z:
            with z.open(MANIFEST_NAME) as f:
                return json.loads(f.read().decode("utf-8"))
    except KeyError:
        raise UpdateError(f"{patch_path.name} has no {MANIFEST_NAME} — it is not a patch file.")
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        raise UpdateError(f"{patch_path.name} could not be read ({e}).")


def _force_move(src, dst, attempts=8):
    """
    Moves a directory, waiting out the transient lock.

    os.replace is instant and atomic within a volume, which is what makes the
    swap below a swap rather than a copy the user could catch half-done. What
    it is not is immune to an antivirus scan or a OneDrive sync holding one
    file open for a moment, and that shows up as PermissionError on a
    directory that is otherwise perfectly movable. Retrying briefly turns the
    common case back into a success; the uncommon one still raises.
    """
    last = None
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            last = e
            time.sleep(0.25 * (i + 1))
    raise UpdateError(
        f"Could not move {src} to {dst} ({last}). Something on this machine is holding the "
        f"application's files open — close {APP_NAME} and any window showing that folder, "
        "then try again."
    )


def _carry_over(src_app, dst_app, log):
    """Moves the things an update keeps (see KEEP_ACROSS_UPDATES) into the new tree."""
    for rel in KEEP_ACROSS_UPDATES:
        src = src_app / rel
        if not src.exists():
            continue
        dst = dst_app / rel
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            _force_move(src, dst)
            log(f"kept {rel}")
        except UpdateError:
            # Worth a copy rather than a failed update: this is a cache, and
            # losing it costs a download, not the install.
            shutil.copytree(src, dst, dirs_exist_ok=True)
            log(f"kept {rel} (copied)")


def apply_patch(install_dir, patch_path, log, state=None, source="local file"):
    """
    Replaces <install>/app with the one inside `patch_path`.

    The order is the whole design. Everything that can fail — reading the zip,
    unpacking it, checking that what came out is actually an app — happens
    against a staging directory while the live one is untouched. Only when
    there is a complete, verified replacement on disk does anything move, and
    the move itself is two renames. An interruption before them leaves the
    install exactly as it was; an interruption between them is repaired on the
    next run, because the old app is still sitting in backup/ under its own
    version number.
    """
    install_dir = Path(install_dir)
    state = state or State(install_dir)
    app = install_dir / "app"
    manifest = read_manifest(patch_path)

    version = str(manifest.get("version") or "")
    if not version:
        raise UpdateError(f"{patch_path.name} does not say what version it is.")

    current = state.version
    if parse_version(version) <= parse_version(current):
        raise UpdateError(
            f"{patch_path.name} is version {version}, and {current} is already installed. "
            "Patches only ever move forward; use --rollback to go back one step."
        )
    min_needed = manifest.get("min_app_version")
    if min_needed and parse_version(current) < parse_version(min_needed):
        raise UpdateError(
            f"Version {version} needs {min_needed} or later installed first, and this is "
            f"{current}. Apply the patches in between, or run the full installer."
        )
    if manifest.get("requires_setup"):
        raise UpdateError(
            f"Version {version} changes something a patch cannot ({manifest.get('requires_setup')}). "
            f"Download and run the {APP_NAME} installer instead — it will update this install in "
            "place and keep everything already downloaded."
        )
    if app_is_running():
        raise UpdateError(f"{APP_NAME} is running. Close it and try again.")

    staging = install_dir / "app.new"
    unpacked = install_dir / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(unpacked, ignore_errors=True)

    log(f"Applying {version} (from {current})")
    try:
        with zipfile.ZipFile(patch_path) as z:
            names = z.namelist()
            if not any(n.startswith(PAYLOAD_DIR + "/") for n in names):
                raise UpdateError(f"{patch_path.name} carries no {PAYLOAD_DIR}/ directory.")
            # Every member is checked before a single one is written: a zip
            # entry named ../../something is the oldest trick there is, and an
            # extractall that has already unpacked half the archive is not a
            # thing that can be undone by refusing the rest.
            for m in names:
                parts = Path(m).parts
                if Path(m).is_absolute() or ".." in parts:
                    raise UpdateError(f"{patch_path.name} contains an unsafe path ({m}) and was refused.")
                if parts and parts[0] not in (PAYLOAD_DIR, *SIDE_TREES) and len(parts) > 1:
                    raise UpdateError(
                        f"{patch_path.name} wants to write outside the directories a patch may "
                        f"touch ({m}), and was refused."
                    )
            z.extractall(unpacked)
        _force_move(unpacked / PAYLOAD_DIR, staging)
    except UpdateError:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(unpacked, ignore_errors=True)
        raise
    except (zipfile.BadZipFile, OSError) as e:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(unpacked, ignore_errors=True)
        raise UpdateError(f"{patch_path.name} could not be unpacked ({e}).")

    # The same three files the launcher's preflight refuses to start without.
    # Checked here, against the staged copy, because the moment to find out a
    # patch was built without the frontend is before it replaces the one that
    # works.
    for rel in ("backend/api.py", "frontend/index.html",
                "frontend/fonts/TradeGothicNextLTProHeavyCompressed.otf"):
        if not (staging / rel).exists():
            shutil.rmtree(staging, ignore_errors=True)
            raise UpdateError(
                f"The patch is incomplete — {rel} is missing from it. Nothing has been changed."
            )

    if app.exists():
        _carry_over(app, staging, log)

    backup = install_dir / "backup" / f"app-{current}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(backup, ignore_errors=True)

    if app.exists():
        _force_move(app, backup)
    try:
        _force_move(staging, app)
    except UpdateError:
        # The one window where the install has no app/ at all. Putting the old
        # one straight back is the only correct move, and it is a rename.
        log("Could not put the new version in place — restoring the previous one")
        _force_move(backup, app)
        raise

    for rel in DISCARD_ON_UPDATE:
        shutil.rmtree(app / rel, ignore_errors=True)

    # The updater goes in last, and only once the app is already swapped. It
    # is the thing currently running — from a copy in the temp directory, for
    # exactly this reason (see relocate_and_rerun) — and replacing it before
    # the risky part would mean a failed update leaving behind a mismatched
    # pair: the new updater with the old app.
    for tree in SIDE_TREES:
        incoming = unpacked / tree
        if not incoming.is_dir():
            continue
        target = install_dir / tree
        shutil.rmtree(target, ignore_errors=True)
        try:
            _force_move(incoming, target)
            log(f"replaced {tree}/")
        except UpdateError as e:
            log(f"WARNING: {tree}/ could not be replaced ({e}). The app is updated; "
                f"the {tree} is not.")
    shutil.rmtree(unpacked, ignore_errors=True)

    _prune_backups(install_dir, log)

    # Everything past this point is additive and cannot un-swap the app: a
    # failed pin or a model that would not download leaves a working install
    # of the new version with something missing, which is a state the app
    # already knows how to report. It is logged loudly and not raised.
    problems = []
    try:
        _install_pins(install_dir, manifest, state, log)
    except UpdateError as e:
        problems.append(str(e))
    try:
        _fetch_models(app, manifest, log)
    except UpdateError as e:
        problems.append(str(e))
    try:
        _install_fonts(app, manifest, log)
    except UpdateError as e:
        problems.append(str(e))
    try:
        _rebuild_launcher(install_dir, manifest, log)
    except UpdateError as e:
        problems.append(str(e))

    state.data["app_version"] = version
    state.data["updated"] = now_iso()
    state.data["history"] = (state.data.get("history") or [])[-19:] + [{
        "version": version, "from": current, "applied": now_iso(),
        "source": source, "notes": manifest.get("notes") or [],
        "problems": problems,
    }]
    state.data["last_result"] = "ok" if not problems else "applied with problems"
    state.save()

    log(f"Now on {version}.")
    for p in problems:
        log(f"WARNING: {p}")
    return version, problems


def _prune_backups(install_dir, log):
    folder = install_dir / "backup"
    if not folder.is_dir():
        return
    backups = sorted(
        (p for p in folder.iterdir() if p.is_dir() and p.name.startswith("app-")),
        key=lambda p: parse_version(p.name[4:]),
    )
    for old in backups[:-BACKUPS_KEPT]:
        shutil.rmtree(old, ignore_errors=True)
        log(f"removed old backup {old.name}")


def _venv_python(install_dir):
    exe = Path(install_dir) / "runtime" / "Scripts" / "python.exe"
    if not exe.exists():
        raise UpdateError(f"The application environment is missing from {exe.parent.parent}.")
    return exe


def _run(args, log, what, timeout=3600):
    log("> " + " ".join(str(a) for a in args))
    proc = subprocess.Popen(
        [str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + timeout
    for line in proc.stdout:
        line = line.rstrip()
        if line and not line.startswith("\r"):
            log(line)
        if time.time() > deadline:
            proc.kill()
            raise UpdateError(f"{what} timed out.")
    if proc.wait() != 0:
        raise UpdateError(f"{what} failed — see the update log for what pip reported.")


def _install_pins(install_dir, manifest, state, log):
    """
    Installs whatever new libraries the patch needs.

    Recorded in version.json as well as installed, so the machine's own record
    of what its environment contains stays true — a later repair install reads
    it and does not undo a patch's dependency by rebuilding the venv from the
    shipped requirements alone.

    An entry is a plain pin, or an object with `args` for the cases pip cannot
    be told any other way (a private index, --no-deps for a package that
    downgrades three others if left to itself — see requirements.txt).
    """
    pins = manifest.get("pip") or []
    if not pins:
        return
    python = _venv_python(install_dir)
    recorded = list(state.data.get("extra_pins") or [])

    for pin in pins:
        args = pin["args"] if isinstance(pin, dict) else [pin]
        label = pin.get("what") if isinstance(pin, dict) else pin
        _run([python, "-m", "pip", "install", "--disable-pip-version-check", *args],
             log, f"Installing {label}")
        for a in args:
            if not str(a).startswith("-") and a not in recorded:
                recorded.append(a)

    state.data["extra_pins"] = recorded


def _fetch_models(app, manifest, log):
    """
    Downloads model weights a patch introduces.

    Not carried inside the patch: the point of a patch being a megabyte is
    that it is a megabyte. A new model is a URL and a checksum, fetched once,
    and skipped entirely on a machine that already has the file.
    """
    for model in manifest.get("models") or []:
        rel = model.get("path")
        url = model.get("url")
        if not rel or not url:
            continue
        dest = app / rel
        if dest.exists() and (not model.get("sha256") or sha256_file(dest).lower() == model["sha256"].lower()):
            log(f"model already present: {rel}")
            continue
        log(f"Downloading model {rel} ({model.get('mb', '?')} MB)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with open_url(url, DOWNLOAD_TIMEOUT_S) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        except (urllib.error.URLError, OSError, ValueError) as e:
            tmp.unlink(missing_ok=True)
            raise UpdateError(f"The model {rel} could not be downloaded ({e}).")
        if model.get("sha256") and sha256_file(tmp).lower() != model["sha256"].lower():
            tmp.unlink(missing_ok=True)
            raise UpdateError(f"The model {rel} downloaded damaged and was discarded.")
        os.replace(tmp, dest)


def _install_fonts(app, manifest, log):
    """
    Registers a font a patch brought with it with Windows, for the current user.

    The app serves its own faces and does not need this to render — but the
    stylesheet lists local() sources after the url() precisely so that a
    served file that fails has a second route to THE SAME FACE rather than
    falling through to Impact, and this is what puts it there. See the font
    section of installer/README.md.
    """
    if os.name != "nt":
        return
    import winreg

    user_fonts = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Microsoft" / "Windows" / "Fonts"
    user_fonts.mkdir(parents=True, exist_ok=True)

    def register(src, name):
        if not src.exists():
            log(f"WARNING: {src.name} was named as a font to install but is not in the payload")
            return
        dst = user_fonts / src.name
        try:
            shutil.copy2(src, dst)
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, FONTS_KEY) as key:
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, str(dst))
            log(f"registered font {name}")
        except OSError as e:
            log(f"WARNING: could not register the font {name} ({e}). The app serves its own "
                "copy, so this only costs the fallback route.")

    # Named by the patch. Kept because it is the only way to install a face
    # that is not declared by any pack, and because it is what patch.json has
    # always documented.
    for entry in manifest.get("fonts") or []:
        rel = entry["path"] if isinstance(entry, dict) else entry
        name = (entry.get("name") if isinstance(entry, dict) else None) or Path(rel).stem
        register(app / rel, name)

    # And every face the packs the patch just delivered declare for
    # themselves, on the same terms the installer registers them on: under the
    # first of the `local` names the face says Windows may know it by, and not
    # at all when it declares none. A patch that brings a channel with a new
    # font then puts that font where the local() fallback can find it without
    # anyone having had to remember to list it in patch.json.
    #
    # See the note above content_faces(): this is the same derivation the
    # installer uses, from the same packs, which is why it lives in this file.
    for label, spec, path in content_faces(app / "frontend"):
        names = [n for n in (spec.get("local") or []) if isinstance(n, str) and n.strip()]
        if names and path.exists():
            register(path, font_value_name(path, names[0]))
        elif not names:
            log(f"{label}: served by the app only — it declares no name Windows could "
                "resolve it by")


def _rebuild_launcher(install_dir, manifest, log):
    """
    Rebuilds Thumbnail Maker.exe, for the rare patch that changes the launcher.

    The exe is compiled on the machine at install time, from updater/launcher.py
    — which a patch can replace like anything else. It is not rebuilt unless
    the patch asks, because PyInstaller is a two-minute job and almost no
    patch touches the launcher.
    """
    if not manifest.get("rebuild_launcher"):
        return
    src = Path(install_dir) / "updater" / "launcher.py"
    if not src.exists():
        raise UpdateError(f"The patch asks for the launcher to be rebuilt, but {src} is missing.")

    python = _venv_python(install_dir)
    build = Path(install_dir) / "_build"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)

    args = [python, "-m", "PyInstaller", str(src), "--name", APP_NAME,
            "--onefile", "--noconsole", "--clean", "--noconfirm",
            "--distpath", str(build / "dist"), "--workpath", str(build / "work"),
            "--specpath", str(build)]
    icon = Path(install_dir) / "assets" / "fox_blue.ico"
    if icon.exists():
        args += ["--icon", str(icon)]

    _run([python, "-m", "pip", "install", "--disable-pip-version-check", "pyinstaller==6.11.1"],
         log, "Installing the build tool", timeout=1800)
    _run(args, log, "Rebuilding the launcher", timeout=2400)

    built = build / "dist" / f"{APP_NAME}.exe"
    if not built.exists():
        raise UpdateError("PyInstaller did not produce a new launcher; the old one is still in place.")

    final = Path(install_dir) / f"{APP_NAME}.exe"
    # The running launcher is very likely this exe. Windows will not overwrite
    # a running image but will happily rename it, and the stale copy is swept
    # up on the next update.
    if final.exists():
        try:
            final.replace(final.with_suffix(".exe.old"))
        except OSError:
            final.unlink(missing_ok=True)
    shutil.move(str(built), str(final))
    shutil.rmtree(build, ignore_errors=True)
    log(f"rebuilt {final.name}")


# --------------------------------------------------------------------------
# rollback
# --------------------------------------------------------------------------

def rollback(install_dir, log):
    """
    Puts the previous app/ back.

    Only the app directory. A pin a patch installed stays installed and a
    model it downloaded stays downloaded, because both are additive and
    neither breaks the version being restored — where undoing them could
    easily break the version being restored TO, if it turned out to share
    them. Stated plainly rather than quietly assumed.
    """
    install_dir = Path(install_dir)
    state = State(install_dir)
    folder = install_dir / "backup"
    backups = sorted(
        (p for p in folder.iterdir() if p.is_dir() and p.name.startswith("app-")),
        key=lambda p: parse_version(p.name[4:]),
    ) if folder.is_dir() else []
    if not backups:
        raise UpdateError("There is no previous version to go back to.")
    if app_is_running():
        raise UpdateError(f"{APP_NAME} is running. Close it and try again.")

    previous = backups[-1]
    target_version = previous.name[4:]
    app = install_dir / "app"
    log(f"Rolling back from {state.version} to {target_version}")

    if app.exists():
        _carry_over(app, previous, log)
        discarded = install_dir / f"app-failed-{state.version}"
        shutil.rmtree(discarded, ignore_errors=True)
        _force_move(app, discarded)
    _force_move(previous, app)
    shutil.rmtree(install_dir / f"app-failed-{state.version}", ignore_errors=True)

    state.data["app_version"] = target_version
    state.data["updated"] = now_iso()
    state.data["last_result"] = f"rolled back to {target_version}"
    state.save()
    log(f"Back on {target_version}.")
    return target_version


# --------------------------------------------------------------------------
# the two things the launcher calls
# --------------------------------------------------------------------------

def check(install_dir, log=lambda s: None):
    """
    Asks the feed what is available. Returns a dict, and never raises.

    Never raises because of where it is called from: in front of the launcher,
    on every start. A feed that is down, a machine that is offline, a proxy
    that returns a login page — none of those are reasons the app should fail
    to open, so all of them come back as `available: False` with a `problem`
    the UI can show if it wants to and ignore if it does not.
    """
    result = {"current": None, "latest": None, "available": False, "steps": [],
              "notes": [], "problem": None, "feed": None}
    try:
        state = State(install_dir)
        result["current"] = state.version
        feed_url = state.data.get("feed")
        result["feed"] = feed_url
        if not feed_url:
            result["problem"] = "No update server is configured for this install."
            return result

        feed = fetch_feed(feed_url)
        steps = plan(state.version, feed, feed_url)
        state.data["last_check"] = now_iso()
        state.save()

        result["latest"] = feed.get("latest") or (steps[-1]["version"] if steps else state.version)
        result["available"] = bool(steps)
        result["steps"] = [{"version": s.get("version"), "size": s.get("size"),
                            "notes": s.get("notes") or []} for s in steps]
        result["notes"] = [n for s in steps for n in (s.get("notes") or [])]
        result["_entries"] = steps
    except UpdateError as e:
        result["problem"] = str(e)
    except Exception as e:  # a check must not be able to stop the app starting
        result["problem"] = f"The update check failed unexpectedly ({type(e).__name__}: {e})."
    return result


def update(install_dir, log=print):
    """Checks, then applies everything the plan calls for. Returns (applied, problem)."""
    install_dir = Path(install_dir)
    found = check(install_dir, log)
    if found["problem"]:
        return [], found["problem"]
    if not found["available"]:
        return [], None

    state = State(install_dir)
    applied = []
    with tempfile.TemporaryDirectory(prefix="tnmaker-patch-") as tmp:
        for entry in found["_entries"]:
            name = entry.get("file") or f"{entry.get('version')}.tmpatch"
            local = Path(tmp) / Path(name).name
            try:
                download(entry["url"], local, entry.get("sha256"), log)
                verify_signature(local.read_bytes(), entry.get("signature"),
                                 state.public_key, f"Version {entry.get('version')}")
                version, _ = apply_patch(install_dir, local, log, state=state,
                                         source=entry["url"])
                applied.append(version)
            except UpdateError as e:
                state.data["last_result"] = f"failed: {e}"
                state.save()
                return applied, str(e)
    return applied, None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def default_install_dir():
    """
    Where this file is being run from, if that looks like an install.

    <install>/updater/updater.py is the installed location; running it out of
    the repo, there is no install to point at and one has to be named.
    """
    here = Path(__file__).resolve().parent
    if (here.parent / "app").is_dir() or (here.parent / "version.json").exists():
        return here.parent
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Programs" / APP_NAME


def make_logger(install_dir, echo=True):
    """Logs to <install>/logs/update.log as well as the console."""
    logs = Path(install_dir) / "logs"
    try:
        logs.mkdir(parents=True, exist_ok=True)
        handle = open(logs / "update.log", "a", encoding="utf-8", errors="replace", buffering=1)
        handle.write(f"\n===== {now_iso()} =====\n")
    except OSError:
        handle = None

    def log(text):
        if echo:
            print(text, flush=True)
        if handle:
            handle.write(str(text) + "\n")
    return log


def relocate_and_rerun(argv):
    """
    Re-runs this updater from a copy in the temp directory.

    An update can replace the updater itself, and on Windows a script cannot
    be relied upon to survive the directory it is sitting in being renamed
    out from under the interpreter reading it. Copying the whole updater
    folder somewhere neutral first costs a few milliseconds and removes the
    entire question.
    """
    here = Path(__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp(prefix="tnmaker-updater-"))
    shutil.copytree(here, tmp / "updater", dirs_exist_ok=True)
    args = [sys.executable, str(tmp / "updater" / "updater.py"), "--detached", *argv]
    try:
        return subprocess.call(args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description=f"{APP_NAME} updater")
    p.add_argument("--install-dir", default=None, help="the install to act on")
    p.add_argument("--check", action="store_true", help="report what is available and exit")
    p.add_argument("--apply", action="store_true", help="check, then apply what is available")
    p.add_argument("--apply-file", metavar="PATCH", help="apply a .tmpatch already on disk")
    p.add_argument("--rollback", action="store_true", help="put the previous version back")
    p.add_argument("--status", action="store_true", help="print what is installed")
    p.add_argument("--json", action="store_true", help="machine-readable output (implies --quiet)")
    p.add_argument("--quiet", action="store_true", help="log to the file only")
    p.add_argument("--detached", action="store_true",
                   help=argparse.SUPPRESS)  # set by relocate_and_rerun; not for humans
    args = p.parse_args(argv)

    install_dir = Path(args.install_dir) if args.install_dir else default_install_dir()

    # Anything that rewrites the install runs from a copy of itself elsewhere.
    if (args.apply or args.apply_file or args.rollback) and not args.detached:
        try:
            Path(__file__).resolve().relative_to(Path(install_dir).resolve())
            return relocate_and_rerun(argv)
        except ValueError:
            pass  # already outside the install — nothing to get out of the way of

    log = make_logger(install_dir, echo=not (args.quiet or args.json))

    try:
        if args.status or not (args.check or args.apply or args.apply_file or args.rollback):
            state = State(install_dir)
            info = {"install_dir": str(install_dir), "version": state.version,
                    "feed": state.data.get("feed"), "signed": bool(state.data.get("public_key")),
                    "updated": state.data.get("updated"), "history": state.data.get("history")}
            print(json.dumps(info, indent=2) if args.json
                  else "\n".join(f"{k}: {v}" for k, v in info.items() if k != "history"))
            return 0

        if args.check:
            found = check(install_dir, log)
            found.pop("_entries", None)
            if args.json:
                print(json.dumps(found))
            elif found["problem"]:
                print(found["problem"])
            elif found["available"]:
                print(f"{found['current']} -> {found['latest']}")
                for note in found["notes"]:
                    print(f"  - {note}")
            else:
                print(f"{found['current']} is up to date.")
            return 0

        if args.apply_file:
            patch = Path(args.apply_file)
            if not patch.is_file():
                raise UpdateError(f"There is no patch file at {patch}.")
            state = State(install_dir)
            verify_signature(patch.read_bytes(), sidecar_signature(patch),
                             state.public_key, patch.name)
            version, problems = apply_patch(install_dir, patch, log, state=state,
                                            source=str(patch))
            if args.json:
                print(json.dumps({"applied": [version], "problems": problems}))
            return 0

        if args.apply:
            applied, problem = update(install_dir, log)
            if args.json:
                print(json.dumps({"applied": applied, "problem": problem}))
            elif problem:
                print(problem)
            elif applied:
                print("Updated to " + applied[-1])
            else:
                print("Already up to date.")
            return 1 if problem else 0

        if args.rollback:
            version = rollback(install_dir, log)
            if args.json:
                print(json.dumps({"version": version}))
            return 0

    except UpdateError as e:
        log(f"ERROR: {e}")
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
