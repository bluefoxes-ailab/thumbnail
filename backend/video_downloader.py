import functools
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.request
import uuid
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import imageio_ffmpeg
from yt_dlp import YoutubeDL
import cancel
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
# Any Instagram link at all, not just the shapes that name a video — so that a
# profile or a story link gets a sentence about Instagram rather than falling
# through to the Google Drive path and failing there as "Could not extract
# file ID", which would send the user looking in the wrong place entirely.
INSTAGRAM_PATTERN = re.compile(r"(?:^|[/.])(?:instagram\.com|instagr\.am)/", re.IGNORECASE)

YOUTUBE_MAX_RETRIES = 3   # yt-dlp's signed format URLs intermittently 403 without a JS runtime
                          # (deno) to solve YouTube's signature challenge — a fresh extraction
                          # attempt almost always succeeds on retry
YOUTUBE_RETRY_DELAY = 2.0  # seconds between retries

# How often a Drive download checks whether its run has been stopped (see
# _gdown), and the size each Dropbox read takes before checking again.
GDOWN_POLL_SECONDS = 0.5
DOWNLOAD_CHUNK = 1024 * 1024


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


# ── Logging in to YouTube ─────────────────────────────────────────────────
#
# YouTube answers a download from a datacenter address — a cloud VM, which is
# where the shared server runs — with:
#
#     Sign in to confirm you're not a bot. Use --cookies-from-browser or
#     --cookies for the authentication.
#
# It is a judgement about the ADDRESS, not about the video or about yt-dlp: the
# same link downloads fine from a home connection and fails every time from the
# VM. The one remedy that does not involve moving the server is to download as
# a logged-in user, which yt-dlp does from a cookies file in Netscape format
# exported from a browser where somebody is signed in to YouTube.
#
# Where the file is looked for, first match wins:
#
#   THUMBNAIL_MAKER_YT_COOKIES   a full path, for a server that keeps its
#                                secrets somewhere of its own. Set and wrong is
#                                an error rather than a silent fallback — a
#                                typo in it would otherwise look exactly like
#                                YouTube refusing the cookies.
#   backend/secrets/youtube-cookies.txt
#                                the default. backend/secrets/ is in .gitignore,
#                                and has to stay there: this file IS the login.
#
# With neither, downloads run anonymously exactly as they always did, which is
# still what a machine on a home connection wants.
#
# Use a Google account made for this, not anybody's own. YouTube can flag an
# account whose cookies are used from a server, and the export goes stale on
# its own schedule — the error below says so when it does.

YOUTUBE_COOKIES_ENV = "THUMBNAIL_MAKER_YT_COOKIES"
YOUTUBE_COOKIES_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "secrets", "youtube-cookies.txt")

# How yt-dlp words the refusal. Matched loosely because the apostrophe in
# "you're" arrives as a typographic one, and the sentence around it has been
# reworded before.
YOUTUBE_BOT_CHECK = re.compile(r"not a bot|sign in to confirm", re.IGNORECASE)


def _youtube_cookies():
    """The cookies file to download with, or None to download anonymously."""
    configured = (os.environ.get(YOUTUBE_COOKIES_ENV) or "").strip()
    if configured:
        if not os.path.isfile(configured):
            raise RuntimeError(
                f"{YOUTUBE_COOKIES_ENV} is set to {configured}, but there is no "
                "file there. Fix the path, or unset it to use "
                f"{YOUTUBE_COOKIES_DEFAULT}."
            )
        return configured
    return YOUTUBE_COOKIES_DEFAULT if os.path.isfile(YOUTUBE_COOKIES_DEFAULT) else None


def _working_copy(cookies: str) -> str:
    """
    A private copy of the cookies file for one download to read and rewrite.

    yt-dlp writes the cookie jar back to its file when it finishes, which is
    how a session YouTube keeps refreshing stays alive — but the server runs
    several backends side by side, and two of them rewriting one file at once
    can leave it truncated. Each download therefore works on its own copy, in
    the same folder so that handing the refreshed jar back is a single atomic
    rename (see _keep_refreshed_cookies) rather than a write another process
    can catch halfway.

    A folder that cannot be written to — secrets mounted read-only — gets the
    copy in TEMP_DIR instead. Downloads work the same; the refreshed jar just
    cannot be handed back, which is what read-only was asking for.
    """
    name = f".cookies-{uuid.uuid4().hex}.txt"
    try:
        copy = os.path.join(os.path.dirname(cookies), name)
        shutil.copyfile(cookies, copy)
    except OSError:
        copy = os.path.join(TEMP_DIR, name)
        shutil.copyfile(cookies, copy)
    return copy


def _keep_refreshed_cookies(copy: str, cookies: str) -> None:
    """
    Puts a successful download's refreshed jar back as THE cookies file.

    Only after a success: a refused download can come back with the session
    cookies cleared, and writing that over the file would turn one bad attempt
    into every later one. Best effort — a secrets folder mounted read-only, or
    a copy that had to be made in TEMP_DIR on another filesystem, simply keeps
    the file it was given.
    """
    try:
        os.replace(copy, cookies)
    except OSError:
        _discard(copy)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _bot_check_message(cookies: str | None) -> str:
    if cookies:
        return (
            "YouTube refused the login in " + cookies + " and asked this server "
            "to prove it is not a bot. The cookies have most likely expired or "
            "been signed out: export them again from a browser signed in to "
            "YouTube and replace that file. Google Drive and Dropbox links, and "
            "uploaded videos, are unaffected."
        )
    return (
        "YouTube is refusing this server without a login (\"confirm you're not "
        "a bot\"), which it does to cloud addresses. Export YouTube cookies from "
        "a browser signed in to a Google account kept for this, in Netscape "
        "format, and save them as " + YOUTUBE_COOKIES_DEFAULT + " (or set " +
        YOUTUBE_COOKIES_ENV + " to their path). Until then, upload the video or "
        "use a Google Drive or Dropbox link."
    )


# The video id inside a YouTube URL. yt-dlp's outtmpl below is "%(id)s.%(ext)s"
# and this IS that id, so the downloaded file's stem is knowable from the URL
# alone — no network round trip — which is what lets the temp cache be pruned
# before the download rather than after it (see expected_stem).
YOUTUBE_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?(?:.*&)?v=|youtube\.com/shorts/|youtu\.be/)([A-Za-z0-9_-]+)"
)


# The shortcode of an Instagram post or reel — the part of the link that names
# the media itself. The optional segment before the type is the account name,
# which Instagram now puts in links shared from a profile
# (instagram.com/<user>/reel/<code>/) as well as the bare form.
INSTAGRAM_SHORTCODE_PATTERN = re.compile(
    r"(?:instagram\.com|instagr\.am)/(?:[A-Za-z0-9_.]+/)?(?:p|reels?|tv)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def _instagram_shortcode(url: str) -> str | None:
    match = INSTAGRAM_SHORTCODE_PATTERN.search(url)
    return match.group(1) if match else None


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
    if INSTAGRAM_PATTERN.search(url):
        shortcode = _instagram_shortcode(url)
        return f"instagram_{shortcode}" if shortcode else None
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


def _stop_if_cancelled(_progress: dict) -> None:
    """
    yt-dlp calls this for every chunk it writes, which makes it the download's
    checkpoint: a run whose user has left stops mid-file rather than fetching
    the rest of a video nobody will see (see cancel.py). The exception is a
    BaseException, so it passes through yt-dlp's own error handling and this
    module's retries instead of being taken for a failed download.
    """
    cancel.check()


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
        "progress_hooks": [_stop_if_cancelled],
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "web_embedded", "android"]
            }
        },
        "cookiefile": os.path.join(os.path.dirname(__file__), "cookies.txt") if os.path.exists(os.path.join(os.path.dirname(__file__), "cookies.txt")) else None,
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

    # Resolved once per download, before the first attempt, for the reason the
    # runtime is: a path set wrong is a fixed fact and gets its own message.
    cookies = _youtube_cookies()

    last_error = None
    for attempt in range(1, YOUTUBE_MAX_RETRIES + 1):
        cancel.check()
        # A fresh copy per attempt, so an attempt that failed never hands its
        # jar to the next one (see _keep_refreshed_cookies). Removed however
        # the attempt ends — a success has already renamed it away, so the
        # removal is a no-op there.
        jar = _working_copy(cookies) if cookies else None
        if jar:
            ydl_opts["cookiefile"] = jar
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                output_path = ydl.prepare_filename(info)
            if os.path.exists(output_path):
                if jar:
                    _keep_refreshed_cookies(jar, cookies)
                return output_path
            last_error = RuntimeError("Download failed")
        except DownloadError as e:
            last_error = e
            # The bot check is a verdict on the server's address (or on the
            # cookies), and asking again two seconds later gets the same
            # verdict. Stopping here saves the user the retries' wait and
            # replaces yt-dlp's advice about command-line flags — which nobody
            # using this app can act on — with what to actually do.
            if YOUTUBE_BOT_CHECK.search(str(e)):
                raise RuntimeError(_bot_check_message(cookies)) from e
        finally:
            if jar:
                _discard(jar)
        if attempt < YOUTUBE_MAX_RETRIES:
            cancel.wait(YOUTUBE_RETRY_DELAY)   # woken early by a stop, and the loop checks

    raise last_error


# ── Instagram ─────────────────────────────────────────────────────────────
#
# Through yt-dlp, like YouTube, but with none of YouTube's machinery: Instagram
# does not sign its media URLs with JavaScript, so no runtime is needed, and
# the failures it does have are not the intermittent kind a retry fixes.
#
# What it has instead is a login wall. Public reels usually download
# anonymously, but Instagram decides per request — a private account always
# refuses, and a busy IP gets refused for a while even on public posts. A
# retry from the same machine a second later gets the same answer, so there
# is one attempt and, when it is refused, a message that says what would help.
#
# The help is the same cookies.txt the YouTube path already reads. A Netscape
# cookie file holds cookies for any number of sites and yt-dlp only sends each
# site its own, so instagram.com cookies exported into that file sit beside the
# YouTube ones without either path needing to know about the other.

COOKIE_FILE = os.path.join(os.path.dirname(__file__), "cookies.txt")

# Phrases yt-dlp's Instagram extractor uses when the refusal is about being
# signed in rather than about the link. Matched loosely and in lowercase, since
# the exact wording moves between yt-dlp releases — "--cookies" is the most
# stable of them, because every raise_login_required ends with yt-dlp's own
# hint to pass one.
INSTAGRAM_LOGIN_HINTS = ("login", "log in", "rate-limit", "registered users", "--cookies")


def _instagram_login_message(error: Exception) -> str:
    return (
        "Instagram refused to hand over this video without a signed-in "
        "session. Either the account is private, or Instagram is limiting "
        "anonymous requests from this machine. Sign in to instagram.com in a "
        "browser, export its cookies in Netscape format, and add them to "
        f"{COOKIE_FILE}, then try the link again. (Instagram said: {error})"
    )


def _download_instagram(url: str) -> str:
    shortcode = _instagram_shortcode(url)
    if not shortcode:
        raise ValueError(
            "That Instagram link does not point at a post or a reel. Open the "
            "video itself and copy its link (it contains /reel/ or /p/)."
        )

    os.makedirs(TEMP_DIR, exist_ok=True)
    ydl_opts = {
        # Same reasoning as the YouTube path: only frames are extracted, so
        # the best video stream on its own beats a lower-resolution file that
        # happens to have audio muxed in.
        "format": "bestvideo/best",
        # Named from the shortcode in the URL rather than from yt-dlp's own
        # id, so the stem is known before anything is fetched — that is what
        # expected_stem relies on. yt-dlp's id for a reel is the same
        # shortcode today, but a carousel's entries carry numeric media ids,
        # and nothing here should depend on which one it chooses.
        "outtmpl": os.path.join(TEMP_DIR, f"instagram_{shortcode}.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        # A carousel post is a playlist of its slides, and a photo slide
        # comes back with no formats at all — which yt-dlp would otherwise
        # treat as the whole post failing. Tolerated here so the slides can
        # be looked at and the video among them picked out below.
        "ignore_no_formats_error": True,
        "quiet": False,
        "progress_hooks": [_stop_if_cancelled],
        "cookiefile": COOKIE_FILE if os.path.exists(COOKIE_FILE) else None,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # Looked at first and fetched second, because what to fetch
            # depends on what the post turns out to be. Of a carousel, only
            # the first slide that is a video: every slide would be written
            # to the same name, one over the other, and the run reads one
            # video anyway.
            info = ydl.extract_info(url, download=False)
            if info and info.get("_type") == "playlist":
                info = next((e for e in info.get("entries") or [] if e and e.get("formats")), None)
            if not info or not info.get("formats"):
                raise RuntimeError("That Instagram post has no video in it, only photos")
            info = ydl.process_ie_result(info, download=True)
            output_path = ydl.prepare_filename(info)
    except DownloadError as e:
        if any(hint in str(e).lower() for hint in INSTAGRAM_LOGIN_HINTS):
            raise RuntimeError(_instagram_login_message(e)) from e
        raise

    if not os.path.exists(output_path):
        raise RuntimeError("Download failed")
    return output_path


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
    # Chunk by chunk rather than shutil.copyfileobj, so a stop is noticed
    # between chunks (see cancel.py). BaseException in the cleanup, not
    # Exception: a stop leaves a half-written file under the final name just
    # as a failure does, and the early return above would hand it to the next
    # run of the same link as if it were the whole video.
    try:
        with urllib.request.urlopen(request) as response, open(output_path, "wb") as f:
            while chunk := response.read(DOWNLOAD_CHUNK):
                cancel.check()
                f.write(chunk)
    except BaseException:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Download failed")
    return output_path


def download_video(url: str) -> str:
    if YOUTUBE_PATTERN.search(url):
        return _download_youtube(url)
    if INSTAGRAM_PATTERN.search(url):
        return _download_instagram(url)
    if DROPBOX_PATTERN.search(url):
        return _download_dropbox(url)

    os.makedirs(TEMP_DIR, exist_ok=True)
    file_id = extract_file_id(url)
    output_path = os.path.join(TEMP_DIR, f"{file_id}.mp4")
    if os.path.exists(output_path):
        return output_path
    _gdown(file_id, output_path)
    if not os.path.exists(output_path):
        raise RuntimeError("Download failed")
    return output_path


def _remove_quietly(path: str) -> None:
    # A leftover that cannot be removed - still held open by the process just
    # ended, on Windows - is left for prune_temp rather than failing the run.
    try:
        os.remove(path)
    except OSError:
        pass


def _gdown(file_id: str, output_path: str) -> None:
    """
    Fetches a Drive file through gdown, in a process of its own.

    gdown used to be called in-process, and nothing in that call can be told
    to stop: it has no progress hook and reads the whole file before it
    returns. A run whose user has left (see cancel.py) would download every
    remaining byte for nobody, holding the backend the next user was given. As
    a separate process it can simply be ended. Same package, same version,
    same options as the in-process call: only where it runs has changed.

    gdown writes to `<output><random>.part` and renames at the very end, so a
    process ended mid-download leaves no file under the final name — only the
    part file, which is removed here.
    """
    # A file rather than a pipe for gdown's output: its progress bar writes
    # continuously, and a pipe nobody reads until the end fills up and stalls
    # the download.
    log_path = output_path + ".log"
    try:
        with open(log_path, "w+", encoding="utf-8", errors="replace") as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", "gdown", file_id, "-O", output_path],
                stdout=log_file, stderr=subprocess.STDOUT,
            )
            try:
                while proc.poll() is None:
                    if cancel.wait(GDOWN_POLL_SECONDS):
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        for part in glob.glob(glob.escape(output_path) + "*.part"):
                            _remove_quietly(part)
                        raise cancel.RunCancelled()
            finally:
                # Echoed into this process's own output, where gdown's
                # progress and errors always went when it ran in-process.
                log_file.seek(0)
                output = log_file.read()
                sys.stderr.write(output)
    finally:
        _remove_quietly(log_path)

    if proc.returncode != 0:
        # gdown's own explanation — usually that the file is not shared with
        # "anyone with the link" — which is what the in-process call raised as
        # its message, rather than a bare exit code. Without the progress bars
        # printed before it.
        lines = [l.strip() for l in output.replace("\r", "\n").splitlines()
                 if l.strip() and "%|" not in l]
        raise RuntimeError(" ".join(lines)[:500] or f"gdown exited with {proc.returncode}")


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
