import subprocess
import os
import glob
import shutil
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

# Crop percentages to remove overlays (subtitles at bottom, logos at top)
CROP_TOP_PCT = 0.08
CROP_BOTTOM_PCT = 0.24

# Whole-video sampling density (see extract_frames). MAX_INTERVAL is the one
# that decides whether a brief appearance survives at all: at 1.0s, a
# three-second shot is sampled ~3 times, which is roughly the point where at
# least one frame gets through the quality filters.
SAMPLE_TARGET = 200
MIN_INTERVAL = 0.3
MAX_INTERVAL = 1.0

# Written into a frame directory once extraction has fully succeeded. Without
# it, a run interrupted partway (closed terminal, crash, killed ffmpeg) left a
# directory holding some frames, and the "reuse if it already has frames"
# check below would happily serve that partial set as if it were a complete
# extraction of the video — silently analysing the first N seconds only.
#
# It holds the sampling interval the directory was extracted at, because that
# number is the only thing tying a frame's position in the sorted list back to
# the moment it came from. A second run over the same video at a different
# density reuses this directory, and reusing it while REPORTING the interval
# that was asked for would put every timestamp out by the ratio between the
# two — which is invisible until something reads the timestamps, as the
# Snapchat capture does.
DONE_MARKER = ".extraction-complete"


# ffmpeg is a console program; the backend that spawns it is not. The launcher
# starts uvicorn with CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP (see
# installer/launcher.py spawn), so every child inherits that hidden console and
# that process group. A console control event delivered to the group — a stray
# Ctrl+C, a Ctrl+Break, a console being torn down — is IGNORED by the backend,
# because CREATE_NEW_PROCESS_GROUP disables Ctrl+C for the process it creates,
# but it is not ignored by that process's children. So ffmpeg dies with exit
# status 0xC000013A (STATUS_CONTROL_C_EXIT) while the backend carries on and
# reports the extraction as having failed, which is not what happened: it was
# killed. Giving ffmpeg its own hidden console cuts the link — it is no longer
# attached to the console the event arrives on.
#
# On anything other than Windows the flag does not exist and neither does the
# problem, hence the getattr.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# Windows reports a process that died from something other than its own exit()
# as a raw NTSTATUS, which reaches the user as a ten-digit number saying
# nothing. These are the ones an ffmpeg run realistically ends on; anything
# else still gets reported, just without a translation.
_NTSTATUS = {
    0xC000013A: ("ffmpeg was stopped by a console Ctrl+C/Ctrl+Break rather than "
                 "failing on its own — something outside the app killed it "
                 "(a security product, a console closing, a shutdown)."),
    0xC0000005: "ffmpeg crashed (access violation).",
    0xC0000017: "ffmpeg ran out of memory.",
    0xC00000FD: "ffmpeg crashed (stack overflow).",
    0xC0000135: "ffmpeg could not start — a DLL it needs is missing.",
    0xC0000142: "ffmpeg could not start — a DLL failed to initialise.",
}


class FfmpegError(RuntimeError):
    """A failed ffmpeg run, carrying what ffmpeg said about it."""


def _ffmpeg_failure(returncode: int, stderr: str | None) -> str:
    # Python surfaces an NTSTATUS as a signed int on some paths and an unsigned
    # one on others; normalise before looking it up so both find the entry.
    status = returncode & 0xFFFFFFFF
    reason = _NTSTATUS.get(status)

    # Only when ffmpeg said nothing itself is the number worth printing: for an
    # ordinary failure it is one of ffmpeg's own AVERROR codes, which reads as a
    # ten-digit number that means nothing next to the line above it saying
    # "moov atom not found".
    if reason is None:
        reason = "" if stderr and stderr.strip() else f"ffmpeg failed (status {status:#010x})."
    # The tail only: ffmpeg's last few lines are the complaint, everything
    # above them is stream metadata nobody reading an error message wants.
    said = "\n".join((stderr or "").strip().splitlines()[-8:])
    return f"{reason}\n{said}".strip() if said else reason


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    Runs ffmpeg and, when it fails, raises something a person can act on.

    The point of this over `subprocess.run(..., check=True)` is the message.
    CalledProcessError formats itself as the whole argv followed by the exit
    status and nothing else — so a capture_output run threw away the one thing
    that explains the failure, ffmpeg's own stderr, which it was holding all
    along, and the user was shown a screenful of quoted Windows paths ending in
    a bare ten-digit number. Here the stderr is the message and the argv is not.
    """
    result = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        creationflags=_NO_WINDOW,
    )
    if result.returncode != 0:
        raise FfmpegError(_ffmpeg_failure(result.returncode, result.stderr))
    return result


def get_video_duration(video_path: str) -> float | None:
    """Parse 'Duration: HH:MM:SS.xx' from ffmpeg's stderr output (no ffprobe needed)."""
    # Decoded as UTF-8 explicitly, never via text=True: that decodes with the
    # locale codec, which on a Windows machine outside en-US is a legacy
    # codepage (cp1252 here). ffmpeg's banner and the container's metadata are
    # UTF-8, so one accented title or a stray 0x81 byte raised UnicodeDecodeError
    # — and it raised inside subprocess's stderr reader thread, where nothing
    # propagates it: `result.stderr` was simply left as None and the run died
    # further down as "'NoneType' object has no attribute 'splitlines'".
    result = subprocess.run(
        [FFMPEG, "-i", video_path], capture_output=True,
        encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
    )
    # Duration is an optimisation (it tunes the sampling interval) and the
    # caller already falls back to MAX_INTERVAL, so an unreadable banner should
    # cost sampling density, not the whole extraction.
    for line in (result.stderr or "").splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            ts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def full_frame_path(frame_path: str) -> str:
    """Sibling path where extract_frames keeps the uncropped original of a (cropped) frame."""
    d, name = os.path.split(frame_path)
    return os.path.join(d, name.replace("frame_", "full_", 1))


def _is_complete(frames_dir: str) -> bool:
    return os.path.exists(os.path.join(frames_dir, DONE_MARKER))


def _recorded_interval(frames_dir: str) -> float | None:
    """
    The interval a completed directory was extracted at, or None when it was
    written by a version that did not record one (the marker used to say "ok").
    """
    try:
        with open(os.path.join(frames_dir, DONE_MARKER)) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _mark_complete(frames_dir: str, interval: float | None = None) -> None:
    with open(os.path.join(frames_dir, DONE_MARKER), "w") as f:
        f.write("ok" if interval is None else f"{interval:.6f}")


# ffmpeg crop expressions reproducing exactly what the previous Python
# per-frame crop computed (int(h * TOP) as the offset, int(h * (1 - BOTTOM))
# as the exclusive end row), so the cropped frames every downstream threshold
# was tuned against are byte-identical to before.
_CROP_EXPR = (
    f"crop=x=0:y=trunc(ih*{CROP_TOP_PCT}):w=iw:"
    f"h=trunc(ih*{1 - CROP_BOTTOM_PCT})-trunc(ih*{CROP_TOP_PCT})"
)


def _extract_cmd(video_path: str, vf_head: str, out_dir: str, extra_input: list[str] | None = None) -> list[str]:
    """
    One ffmpeg invocation producing both outputs from a single decode of the
    video:

      full_%04d.jpg   uncropped original — only the manual drag-to-reframe
                      endpoints read these (see api's manual reframe view)
      frame_%04d.jpg  the overlay-cropped frame — the canonical source every
                      automatic result is rendered from, and the only thing
                      face detection and scoring ever see

    Previously ffmpeg wrote only the full frames and Python then re-read,
    cropped and re-encoded every single one — a full JPEG decode plus encode
    per frame, for hundreds of frames, on top of the decode ffmpeg had
    already done. Splitting inside the filter graph does all of it in that
    one pass.
    """
    filter_complex = (
        f"[0:v]{vf_head},split=2[fullout][tocrop];"
        f"[tocrop]{_CROP_EXPR}[cropout]"
    )
    return [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        *(extra_input or []),
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[fullout]", "-q:v", "2", os.path.join(out_dir, "full_%04d.jpg"),
        "-map", "[cropout]", "-q:v", "2", os.path.join(out_dir, "frame_%04d.jpg"),
    ]


def extract_frames(video_path: str, interval: float | None = None) -> tuple[list[str], float]:
    """Returns (frame_paths, interval) — the interval actually used, so callers can map a frame's
    position in the sorted list back to an approximate timestamp (position * interval).

    An `interval` given here is a REQUIREMENT, not a hint: a cached directory
    extracted at some other density is discarded and re-extracted rather than
    served under the requested number. Passing None keeps the adaptive
    behaviour, and then a cached directory is served under whatever interval it
    recorded for itself."""
    frames_dir = os.path.join(TEMP_DIR, "frames_" + os.path.splitext(os.path.basename(video_path))[0])
    requested = interval

    if interval is None:
        # Adaptive sampling: aim for ~SAMPLE_TARGET frames, and never sample
        # further apart than MAX_INTERVAL seconds however long the video is.
        #
        # The old numbers (~80 frames, up to 1.5s apart) were set by what the
        # selector needs to FILL twenty slots, which a dominant character
        # supplies on its own. Coverage needs something else: a character on
        # screen for three seconds only exists in two frames at 1.5s spacing,
        # and the quality filters (sharpness floor, both-eyes-open, profile
        # rejection) reject most single frames — so that character's expected
        # contribution to the pool was well under one usable candidate. No
        # clustering or selection change can recover someone who was never
        # sampled; this is where their frames have to come from.
        duration = get_video_duration(video_path)
        if duration and duration > 0:
            interval = max(MIN_INTERVAL, min(MAX_INTERVAL, duration / SAMPLE_TARGET))
        else:
            interval = MAX_INTERVAL

    if _is_complete(frames_dir):
        existing = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        recorded = _recorded_interval(frames_dir)
        # A directory whose density matches (or one nobody asked a density of)
        # is reused; anything else is re-extracted below. `recorded or interval`
        # covers the pre-marker directories, which is the old behaviour exactly.
        if existing and (requested is None or (recorded is not None and abs(recorded - requested) < 1e-6)):
            return existing, (recorded or interval)

    # Not complete (never run, or interrupted): start clean rather than mixing
    # a partial previous attempt's frames with this one's.
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)

    _run_ffmpeg(_extract_cmd(video_path, f"fps=1/{interval}", frames_dir))

    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if frames:
        _mark_complete(frames_dir, interval)
    return frames, interval


def extract_window(video_path: str, center_time: float, radius: float, interval: float, out_dir: str) -> list[str]:
    """
    Extracts frames from a NARROW, densely-sampled window of the source video
    — [center_time - radius, center_time + radius] at `interval` seconds
    apart — cropped the same way extract_frames crops its whole-video sample,
    and with the same outputs.

    extract_frames' own sample is deliberately sparse (0.3-1.5s apart,
    spread across the whole video) — fine for picking a handful of good
    thumbnail candidates, but too sparse for /vary-frame's "small variation
    of the same shot" search: a fixed number of steps through that sparse
    sample can span many seconds, occasionally crossing a scene cut. This
    samples tightly around ONE timestamp instead, so every frame it returns
    is genuinely close in time to where it's centered.
    """
    if _is_complete(out_dir):
        existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
        if existing:
            return existing

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    start = max(0.0, center_time - radius)
    _run_ffmpeg(_extract_cmd(
        video_path, f"fps=1/{interval}", out_dir,
        extra_input=["-ss", f"{start:.3f}", "-t", f"{2 * radius:.3f}"],
    ))

    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    if frames:
        _mark_complete(out_dir)
    return frames
