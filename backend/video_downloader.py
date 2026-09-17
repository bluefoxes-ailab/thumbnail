import functools
import gdown
import hashlib
import os
import re
import shutil
import sysconfig
import time
import urllib.request
import uuid
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import imageio_ffmpeg
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# Private module, so its absence is survivable: with no way to ask about
# runtimes we simply don't, and downloads behave exactly as they did before
# this check existed rather than refusing to start over a failed import.
try:
    from yt_dlp.utils._jsruntime import (
        DenoJsRuntime, NodeJsRuntime, BunJsRuntime, QuickJsRuntime,
    )
    JS_RUNTIMES = (("deno", DenoJsRuntime), ("node", NodeJsRuntime),
                   ("bun", BunJsRuntime), ("quickjs", QuickJsRuntime))
except ImportError:
    JS_RUNTIMES = ()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

YOUTUBE_PATTERN = re.compile(r"(youtube\.com/watch\?v=|youtube\.com/shorts/|youtu\.be/)")
DROPBOX_PATTERN = re.compile(r"dropbox\.com")

YOUTUBE_MAX_RETRIES = 3   # yt-dlp's signed format URLs intermittently 403 without a JS runtime
                          # (deno) to solve YouTube's signature challenge — a fresh extraction
                          # attempt almost always succeeds on retry
YOUTUBE_RETRY_DELAY = 2.0  # seconds between retries


# ── JavaScript runtime ────────────────────────────────────────────────────
#
# YouTube signs its media URLs with JavaScript served by the watch page.
# yt-dlp can extract that code but not execute it, so on a machine with no JS
# runtime every signed URL comes back rejected and the user sees:
#
#     Download failed: unable to download video data: HTTP Error 403: Forbidden
#
# which says nothing about the actual cause. The retries above exist because a
# fresh extraction sometimes gets lucky; with no runtime at all there is no
# luck to be had, so the three attempts only delay the same failure.
#
# The installer puts deno in its venv's Scripts directory — the first place
# yt-dlp looks — which is why the packaged app works while this same code, run
# from the project folder on the system Python, does not. So the places the
# installer might have left one are searched too, and whatever is found is
# handed to yt-dlp explicitly.

def _deno_candidates():
    """Places a deno may be that yt-dlp itself won't look in."""
    yield os.environ.get("THUMBNAIL_MAKER_DENO")
    # Dropped in by hand next to the project, for a checkout run without the
    # installer. Kept out of the repo — it's a ~97MB binary.
    yield os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runtime", "deno.exe")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        # The installed app's own copy, borrowed rather than duplicated.
        yield os.path.join(local_app_data, "Programs", "Thumbnail Maker",
                           "runtime", "Scripts", "deno.exe")


@functools.lru_cache(maxsize=1)
def _js_runtime():
    """
    (name, path) of a JS runtime yt-dlp can use, or None if the machine has none.

    Asked of yt-dlp's own runtime classes rather than by looking for files:
    what matters is not that a binary exists but that yt-dlp finds it AND
    considers the version usable — a deno older than 2.3.0 is found and still
    fails. Cached because it shells out to `--version` and the answer cannot
    change while the app runs.
    """
    for name, runtime_cls in JS_RUNTIMES:
        info = runtime_cls().info
        if info and info.supported:
            return name, info.path

    for candidate in _deno_candidates():
        if not candidate or not os.path.isfile(candidate):
            continue
        info = DenoJsRuntime(os.path.abspath(candidate)).info
        if info and info.supported:
            return "deno", info.path

    return None


def _no_js_runtime_message():
    return (
        "No JavaScript runtime found, so YouTube downloads cannot work: YouTube "
        "signs its video URLs with JavaScript, and without a runtime to run it "
        "every download is rejected with HTTP 403 Forbidden. Install deno "
        "(>= 2.3.0), or copy deno.exe into " + sysconfig.get_path("scripts") +
        ", or set THUMBNAIL_MAKER_DENO to its full path. Google Drive and "
        "Dropbox links are unaffected."
    )


# The video id inside a YouTube URL. yt-dlp's outtmpl below is "%(id)s.%(ext)s"
# and this IS that id, so the downloaded file's stem is knowable from the URL
# alone — no network round trip — which is what lets the temp cache be pruned
# before the download rather than after it (see expected_stem).
YOUTUBE_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?(?:.*&)?v=|youtube\.com/shorts/|youtu\.be/)([A-Za-z0-9_-]+)"
)


def expected_stem(url: str) -> str | None:
    """
    The filename (without extension) `download_video` will write for this URL,
    worked out from the URL alone — or None when it can't be known in advance.

    Exists so the caller can clear out the PREVIOUS video's cache before
    starting this download instead of afterwards, without throwing away a
    copy of this video that's already on disk (all three download paths reuse
    an existing file rather than re-fetching it, and pruning blindly first
    would defeat that). A None answer just means the caller should fall back
    to pruning once the real filename is known.
    """
    if YOUTUBE_PATTERN.search(url):
        match = YOUTUBE_ID_PATTERN.search(url)
        return match.group(1) if match else None
    if DROPBOX_PATTERN.search(url):
        return f"dropbox_{_dropbox_file_id(url)}"
    try:
        return extract_file_id(url)
    except ValueError:
        return None


def extract_file_id(url: str) -> str:
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"^([a-zA-Z0-9_-]{20,})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract file ID from: {url}")


def _download_youtube(url: str) -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    ydl_opts = {
        # Video-only, highest resolution available — on YouTube, resolutions above
        # ~720p usually only exist as separate video/audio streams; restricting to
        # pre-merged formats (the old "best[ext=mp4]/best") silently caps quality at
        # whatever low-res combined format happens to exist. Audio is irrelevant here
        # (only frames get extracted), so there's no need to also fetch/mux it.
        "format": "bestvideo/best",
        "outtmpl": os.path.join(TEMP_DIR, "%(id)s.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "noplaylist": True,
        "quiet": False,
    }

    # Checked before the first attempt, not after three: a missing runtime is
    # a fixed property of the machine, and the message says what to do about
    # it instead of leaving a bare 403 to be interpreted.
    runtime = _js_runtime() if JS_RUNTIMES else None
    if JS_RUNTIMES and not runtime:
        raise RuntimeError(_no_js_runtime_message())
    if runtime:
        # Named explicitly so the one that was actually verified is the one
        # used — yt-dlp enables only deno by default and searches its own
        # locations, which are not necessarily where this was found.
        name, path = runtime
        ydl_opts["js_runtimes"] = {name: {"path": path}}

    last_error = None
    for attempt in range(1, YOUTUBE_MAX_RETRIES + 1):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                output_path = ydl.prepare_filename(info)
            if os.path.exists(output_path):
                return output_path
            last_error = RuntimeError("Download failed")
        except DownloadError as e:
            last_error = e
        if attempt < YOUTUBE_MAX_RETRIES:
            time.sleep(YOUTUBE_RETRY_DELAY)

    raise last_error


def _dropbox_file_id(url: str) -> str:
    m = re.search(r"/(?:s|scl/fi)/([a-zA-Z0-9]+)", urlparse(url).path)
    return m.group(1) if m else str(abs(hash(url)))


def _dropbox_direct_url(url: str) -> str:
    # Dropbox share pages serve an HTML preview by default (dl=0); forcing
    # dl=1 makes it redirect straight to the raw file bytes instead.
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def _download_dropbox(url: str) -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_id = _dropbox_file_id(url)
    ext = os.path.splitext(urlparse(url).path)[1] or ".mp4"
    output_path = os.path.join(TEMP_DIR, f"dropbox_{file_id}{ext}")
    if os.path.exists(output_path):
        return output_path

    request = urllib.request.Request(
        _dropbox_direct_url(url), headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(request) as response, open(output_path, "wb") as f:
            shutil.copyfileobj(response, f)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Download failed")
    return output_path


def download_video(url: str) -> str:
    if YOUTUBE_PATTERN.search(url):
        return _download_youtube(url)
    if DROPBOX_PATTERN.search(url):
        return _download_dropbox(url)

    os.makedirs(TEMP_DIR, exist_ok=True)
    file_id = extract_file_id(url)
    output_path = os.path.join(TEMP_DIR, f"{file_id}.mp4")
    if os.path.exists(output_path):
        return output_path
    gdown.download(id=file_id, output=output_path, quiet=False)
    if not os.path.exists(output_path):
        raise RuntimeError("Download failed")
    return output_path


# ── Local uploads ─────────────────────────────────────────────────────────
#
# A video the user picked off their own machine rather than pasted a link to.
# It lands in the same TEMP_DIR under the same shape of name as a downloaded
# one, so everything downstream — pruning, frame extraction, the frames_<stem>
# directory, the vary pools — treats it exactly as it treats a download and
# needs no idea of where the bytes came from.

UPLOAD_PREFIX = "upload_"

# Containers ffmpeg will actually open. An allowlist, not "whatever the browser
# labelled video/*": this extension names a file that gets handed to a
# subprocess, so it is load-bearing rather than decoration.
UPLOAD_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".m2ts", ".3gp",
}

# What an upload id is allowed to look like — built from the allowlist above
# rather than a loose "some letters" tail, so the only ids that pass are ones
# this module could actually have issued. A request naming one is naming a
# path this process will open, so the shape is checked rather than trusted:
# because the name is matched whole, it cannot carry a separator or a "..",
# and that is what keeps the resolved path inside TEMP_DIR.
UPLOAD_ID_PATTERN = re.compile(
    r"^" + UPLOAD_PREFIX + r"[0-9a-f]{16}("
    + "|".join(re.escape(e) for e in sorted(UPLOAD_EXTENSIONS))
    + r")$"
)


class UploadWriter:
    """
    Streams an uploaded video to disk, naming it by what it contains.

    The name is a hash of the bytes rather than the user's filename or a fresh
    random token, which buys the upload path the caching the link path already
    had: frame_extractor keys its extracted-frames directory on the video's
    stem, so re-uploading a video you already processed reuses those frames
    instead of spending the minutes again. Hashing also means two files that
    differ only in name cannot collide, and two that differ only in content
    cannot be mistaken for each other — neither of which a filename-derived
    name would give.

    The bytes go to a staging name and are renamed only once the whole stream
    has arrived. A half-written file sitting under the final name is
    indistinguishable from a complete one to every later run, and the run that
    would trip over it is the one that reuses the cache.
    """

    def __init__(self, filename: str):
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in UPLOAD_EXTENSIONS:
            raise ValueError(
                f"{ext or 'That file'} is not a video format this app can read"
            )
        os.makedirs(TEMP_DIR, exist_ok=True)
        self.ext = ext
        self.size = 0
        # 8 bytes -> the 16 hex characters UPLOAD_ID_PATTERN expects. blake2b
        # rather than sha1 because this runs over every byte of a file that
        # can be gigabytes, and it is the faster of the two at equal safety
        # for what is being asked here (telling two videos apart).
        self._digest = hashlib.blake2b(digest_size=8)
        self._staging = os.path.join(TEMP_DIR, f".incoming_{uuid.uuid4().hex}{ext}")
        self._file = open(self._staging, "wb")

    def write(self, chunk: bytes) -> None:
        self._digest.update(chunk)
        self._file.write(chunk)
        self.size += len(chunk)

    def finish(self) -> str:
        """Closes the stream and returns the upload id the finished file sits under."""
        self._file.close()
        if not self.size:
            self.abort()
            raise ValueError("The uploaded file was empty")

        upload_id = f"{UPLOAD_PREFIX}{self._digest.hexdigest()}{self.ext}"
        final = os.path.join(TEMP_DIR, upload_id)
        # These exact bytes are already here from an earlier upload: keep that
        # copy, and with it whatever was extracted from it, and drop the new one.
        if os.path.exists(final):
            os.remove(self._staging)
        else:
            os.replace(self._staging, final)
        return upload_id

    def abort(self) -> None:
        """Drops a partial upload — a closed connection, a disk that filled up."""
        try:
            self._file.close()
        except OSError:
            pass
        try:
            if os.path.exists(self._staging):
                os.remove(self._staging)
        except OSError:
            pass


def upload_stem(upload_id: str) -> str:
    """
    The filename stem an upload id names, having first checked that the id is
    one this module could have issued.

    Split out from uploaded_video_path so the SHAPE of an id can be checked
    the moment a request arrives, without touching the disk. Resolving it is
    the last thing a run does before reading the video, by which point the
    session has already been reset and the previous video's cache deleted —
    far too late for a malformed id to be rejected harmlessly.
    """
    if not UPLOAD_ID_PATTERN.match(upload_id or ""):
        raise ValueError("not a valid upload id")
    return os.path.splitext(upload_id)[0]


def uploaded_video_path(upload_id: str) -> str:
    """
    Where an upload id's video is on disk.

    Raises ValueError when the id isn't one this module issued, or when the
    file is gone. Gone is an ordinary outcome rather than an error worth
    panicking about: prune_temp clears previous runs, so an id held over from
    an earlier video is expected to stop resolving, and the message says what
    to do about it.
    """
    upload_stem(upload_id)   # shape first — this id is about to become a path
    path = os.path.join(TEMP_DIR, upload_id)
    if not os.path.isfile(path):
        raise ValueError("that upload is no longer on disk — pick the file again")
    return path
