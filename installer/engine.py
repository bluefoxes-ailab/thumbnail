"""
The machinery behind the Thumbnail Maker installer: what it looks for on the
machine, and what it does about what it doesn't find.

Kept apart from setup_gui.py so the whole install can be driven from a
terminal (`python -m installer.engine --scan`) when something needs
diagnosing without a window in the way.

The shape of the problem: the app itself is about a megabyte of Python and
static files, but it stands on ~5 GB of CUDA-enabled PyTorch and face/inpaint
model weights. Shipping that would mean a 5 GB download for every user
regardless of what they already have, and would still be wrong for half of
them — a machine with no NVIDIA GPU needs entirely different torch wheels. So
the installer ships light and resolves all of it here, against the machine it
is actually running on.
"""

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import winreg
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Same directory, and the same module the install lays down at
# <install>/updater/updater.py. Imported rather than duplicated so that the
# list of things an update preserves has exactly one definition - a second
# copy here would disagree with it the first time either changed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import updater

APP_NAME = "Thumbnail Maker"
PUBLISHER = "Blue Foxes"


def _release():
    """
    release.json — the version being built, and where its updates come from.

    Read from a file rather than written here because three separate things
    need to agree on it: this installer, the patch builder, and the app itself
    (which reports it, and which the updater compares against the feed). A
    constant in this file would be a fourth place to forget.

    Frozen, the file rides in the payload; from source it is in the repo root.
    """
    for candidate in (
        Path(getattr(sys, "_MEIPASS", "")) / "app" / "release.json" if getattr(sys, "frozen", False) else None,
        Path(__file__).resolve().parent.parent / "release.json",
    ):
        if candidate and candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass
    return {"version": "0.0.0", "feed": None, "public_key": None}


RELEASE = _release()
VERSION = RELEASE.get("version") or "0.0.0"

DEFAULT_INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Programs" / APP_NAME

# Interpreters this app is known to run on. The lower bound is where the
# codebase's syntax starts working; the upper bound is the newest CPython the
# pinned torch 2.6.0 wheels are built for.
MIN_PY = (3, 10)
MAX_PY = (3, 12)

# Fetched only when the machine has no usable interpreter of its own.
BUNDLED_PY = "3.11.9"

# A CPython that is a FOLDER rather than an installation, and the reason this
# app stopped reaching for the python.org installer first.
#
# The python.org bundle is a Windows Installer product. Windows keeps a record
# of it, that record outlives the files, and once it exists the bundle refuses
# to do a fresh install ever again: it switches to maintenance mode, ignores
# the directory it was given, installs nothing, and exits 0. Deleting
# <install> is enough to get there — which an uninstall does, and which a
# half-finished install cleaned up by hand does. One user's machine reached
# that state and could not be talked out of it: install, repair, and a clean
# install after removing the registration all left the target empty.
#
# This has none of that surface. It is a tar.gz that is extracted into a
# folder: no Windows Installer, no registration, no maintenance mode, nothing
# to conflict with and nothing left behind. Uninstalling this app becomes
# deleting a directory again, exactly as its uninstaller always assumed.
#
# It is CPython, built by the python-build-standalone project — the same
# builds `uv python install` puts on millions of machines — and it is complete
# in the two ways this app needs and the embeddable zip is not: tkinter and
# Tcl/Tk are in it (the launcher is a Tk window, frozen out of this
# interpreter), and so are venv, ensurepip and pip.
#
# Not signed by the PSF, so it is not checked the way the python.org bundle
# is. It is pinned instead: an exact URL and the SHA-256 of the exact bytes
# that URL served when this line was written, which is a narrower promise than
# a publisher signature rather than a weaker one. Both are checked before
# anything is unpacked. To move the pin, download the file, hash it, and
# change all three of these together.
PORTABLE_PY = "3.11.9"
PORTABLE_PY_BUILD = "20240814"
PORTABLE_PY_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PORTABLE_PY_BUILD}/cpython-{PORTABLE_PY}+{PORTABLE_PY_BUILD}"
    "-x86_64-pc-windows-msvc-install_only.tar.gz"
)
PORTABLE_PY_SHA256 = "4c71d25731214b8a960d1d87510f24179d819249c5b434aaf7135818421b6215"

# The fallback, for a machine that cannot reach the download above. Kept
# because it is signed by the Python Software Foundation and because it was
# the only route for every install before this one — see ensure_python.
PY_INSTALLER_URL = f"https://www.python.org/ftp/python/{BUNDLED_PY}/python-{BUNDLED_PY}-amd64.exe"

TORCH_PINS = ["torch==2.6.0", "torchvision==0.21.0"]
TORCH_INDEX_CUDA = "https://download.pytorch.org/whl/cu124"
TORCH_INDEX_CPU = "https://download.pytorch.org/whl/cpu"

# The non-torch pins, in the order requirements.txt documents. Torch and the
# GAN stack are deliberately absent — see install_packages() for why the order
# between those two is not negotiable.
BASE_PINS = [
    "fastapi==0.139.0",
    "uvicorn==0.50.2",
    "pydantic==2.11.7",
    "opencv-python==4.13.0.92",
    "numpy==2.2.6",
    "imageio-ffmpeg==0.6.0",
    "gdown==6.1.0",
    # Floating, unlike every other pin here — see the note in
    # requirements.txt: a pinned yt-dlp is one YouTube change away from
    # 403-ing every download, which reaches the user as a broken app.
    "yt-dlp>=2026.8.19",
    "scipy==1.15.3",
    # The background-removal runtime. Ordinary — it moves none of the pins
    # above (see requirements.txt), which is why it is here rather than among
    # the two special cases in install_gan.
    "onnxruntime==1.20.1",
]
GAN_PINS = ["gfpgan==1.3.8", "basicsr==1.4.2", "facexlib==0.3.0"]
LAMA_PIN = "simple-lama-inpainting==0.1.2"

# Rough, and only used to tell the user what they are in for before they
# commit. The CUDA wheels dominate everything else by an order of magnitude.
# How long a step may print nothing before the installer says it is still
# alive, and how much of one line of a child's output is worth keeping. Both
# exist for the same reason and are explained in sh().
HEARTBEAT_S = 60
MAX_LOG_LINE = 1000

# How often the install checks that its own files are still there — see
# watch_files() in Installer.run.
SENTINEL_POLL_S = 15

DOWNLOAD_MB = {
    "python": 40,   # the portable tar.gz; the python.org fallback is 25
    "torch_cuda": 2900,
    "torch_cpu": 220,
    "base": 180,
    "gan": 120,
    "weights": 620,
    # The U2-Net cutout weights, fetched by the app itself on first load — see
    # backend/background_remover.py.
    "cutout": 188,   # the U2-Net matte, plus the person-segmentation prior beside it
    "pyinstaller": 15,
    "deno": 41,
}

# YouTube serves its media from signed URLs, and the signature is produced by
# JavaScript the page runs. yt-dlp cannot execute that on its own, so without a
# JS runtime on the machine it hands back URLs that answer "403 Forbidden" —
# surfacing in the app as "Download failed: unable to download video data".
# yt-dlp accepts deno, bun, node >= 22 or quickjs; deno is the one it treats as
# the primary provider, and it is a single self-contained executable.
#
# It is installed into the venv's Scripts directory because that is the first
# place yt-dlp looks (yt_dlp/utils/_jsruntime.py:_find_exe), which means no
# PATH changes, no configuration, and nothing for the app to know about.
DENO_URL = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip"
DENO_MIN_VERSION = (2, 3, 0)   # yt_dlp.utils._jsruntime.DenoJsRuntime.MIN_SUPPORTED_VERSION

FONT_FILE = "TradeGothicNextLTProHeavyCompressed.otf"
# From the file's own name table: nameID 4. Windows keys installed fonts by
# this, not by the alias the stylesheet invents for it.
#
# Named here as well as in the pack because this is the one face the app
# cannot start without — the scan reports on it by name, the launcher's
# preflight refuses to run without the file, and neither can wait for a
# content directory to be read. Every OTHER face comes from the packs, via
# updater.content_faces(); this one is also in
# frontend/content/faces/trade-gothic-heavy-compressed.json, and the two say
# the same thing.
FONT_FULL_NAME = "Trade Gothic Next LT Pro Heavy Compressed"

# Where Windows lists the fonts installed for one user. Taken from updater.py
# rather than written again here for the same reason KEEP_ACROSS_UPDATES is:
# the installer and an installed copy both register faces, and a key they
# disagreed on would be a font one of them could not find to remove.
FONTS_KEY = updater.FONTS_KEY

# Where the weights torch fetches for itself are kept, under the install root.
#
# Left alone, torch downloads into %USERPROFILE%\.cache	orch — shared with
# every other torch program on the machine, and outside everything this
# installer created. That is ~200 MB of LaMa weights that an uninstall would
# have to either abandon forever or delete out from under somebody else's
# program. Inside the install, "remove the folder" is a complete and honest
# uninstall, and it takes nothing with it that was not ours.
#
# The launcher has to agree, or the app re-downloads them on first use — see
# MODELS in launcher.py.
MODELS_DIRNAME = "models"

# What may be taken from the machine's shared torch cache instead of being
# downloaded again, by exact filename. A list rather than "everything in
# there": that cache belongs to every torch program on the machine, and
# copying whatever it happens to hold could mean gigabytes of somebody else's
# models moved into this install for no reason. This is the one file the app
# fetches through torch — simple-lama-inpainting's ~200 MB, and the 12 MB
# person-segmentation weights the cutout uses to tell a performer from the
# stage they are standing on (see backend/background_remover.py).
TORCH_CACHED_WEIGHTS = ["big-lama.pt", "lraspp_mobilenet_v3_large-d234d4ea.pth"]

# Free space demanded before starting. The venv with CUDA torch lands around
# 6 GB, weights add ~1 GB, and pip needs room for its own build/cache traffic
# on top of both.
REQUIRED_FREE_GB = 12

UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ThumbnailMaker"


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------

# What the installer carries: the app itself and the launcher it will compile.
# Around a megabyte in total, which is the point — everything expensive is
# fetched during the install, and only what this machine turns out to need.
PAYLOAD_TREES = [("backend", "app/backend"), ("frontend", "app/frontend")]
PAYLOAD_FILES = [
    ("requirements.txt", "app/requirements.txt"),
    # Rides inside app/ because that is the directory an update replaces: the
    # version an install reports is then always the version of the files it
    # actually has, with no second place to keep in step.
    ("release.json", "app/release.json"),
]

# What lands in <install>/updater — the machinery that lets an installed copy
# take a patch instead of being reinstalled (see updater.py). launcher.py is
# in here rather than only at the payload root so that a patch can replace it
# and ask for the exe to be rebuilt; the root copy is what PyInstaller
# compiles during THIS install, before there is an updater directory yet.
UPDATER_FILES = ["updater.py", "ed25519.py", "launcher.py"]

# Dropped into <install>/content the first time, and never rewritten after:
# the folder is the user's, and an installer that restores its own README over
# an edited one is an installer that argues with the person using it.
USER_CONTENT_README = "Thumbnail Maker - your own channels\n\nAnything you put here is loaded alongside the channels that came with the\napp, and is never touched by an update. Same layout as\napp\\frontend\\content, which is where the shipped ones are - copy one of\nthose folders here, rename it, and edit its channel.json.\n\n  channels\\<your-channel>\\channel.json   one folder per channel\n  faces\\<your-face>.json                 a type face several may share\n  presets\\<name>.json                    a fragment several may share\n\nA channel here with the same folder name as a shipped one replaces it.\n\nReload the app's page after adding one. There is nothing to restart and\nnothing to rebuild.\n"


def replace_tree(src, dst, ignore=None):
    """
    Copies `src` over `dst`, tolerating a `dst` that will not fully delete.

    rmtree(ignore_errors=True) swallows exactly the failure that matters here:
    one locked file — an antivirus scan, a OneDrive sync, a .pyc still mapped
    by a running process — leaves the directory itself behind, and the
    copytree that follows then dies on a directory that "should not exist".
    Clearing what can be cleared and copying over the rest gets the same
    result without depending on the delete having worked.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def stage_payload(repo_root, dest):
    """
    Copies the app out of the repo into `dest`, in the layout Installer expects.

    Used by build_installer.py to assemble what gets frozen into the setup
    exe, and by the source-run path below so that running this file directly
    behaves identically to running the built installer.
    """
    repo_root, dest = Path(repo_root), Path(dest)
    here = Path(__file__).resolve().parent

    # "models" is the U2-Net weights the app downloads for itself, which are
    # excluded for the same reason "gfpgan" is: 176 MB of file that every
    # install fetches once and no payload should ever carry.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "temp", "gfpgan", "models",
                                    ".pytest_cache")
    for src_name, rel in PAYLOAD_TREES:
        src = repo_root / src_name
        if not src.exists():
            raise InstallError(f"Cannot build the payload: {src} is missing.")
        replace_tree(src, dest / rel, ignore)

    for src_name, rel in PAYLOAD_FILES:
        src = repo_root / src_name
        if src.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)

    (dest / "assets").mkdir(parents=True, exist_ok=True)
    for asset in (here / "assets").glob("*"):
        shutil.copy2(asset, dest / "assets" / asset.name)
    shutil.copy2(here / "launcher.py", dest / "launcher.py")

    (dest / "updater").mkdir(parents=True, exist_ok=True)
    for name in UPDATER_FILES:
        src = here / name
        if not src.exists():
            raise InstallError(f"Cannot build the payload: {src} is missing.")
        shutil.copy2(src, dest / "updater" / name)
    return dest


def payload_source():
    """
    Where the payload is right now.

    Frozen, it is the directory PyInstaller unpacked into. Run from the repo,
    there is nothing to unpack, so it is staged into a temp directory on the
    spot — which keeps the "run it from source to see what it does" path
    honest rather than subtly different from the shipped one.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    staged = Path(tempfile.gettempdir()) / "thumbnail-maker-payload"
    staged.mkdir(parents=True, exist_ok=True)
    return stage_payload(Path(__file__).resolve().parent.parent, staged)


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

OK, MISSING, FATAL, INFO = "ok", "missing", "fatal", "info"


@dataclass
class Check:
    """One line on the scan page."""
    name: str
    state: str          # OK | MISSING | FATAL | INFO
    detail: str
    action: str = ""    # what the install will do about it, if anything


@dataclass
class Scan:
    checks: list = field(default_factory=list)
    python_exe: Path = None          # usable interpreter found on the machine
    needs_python: bool = False
    cuda: bool = False               # install the CUDA torch wheels
    gpu_name: str = ""
    install_dir: Path = DEFAULT_INSTALL_DIR
    venv_packages: dict = field(default_factory=dict)
    download_mb: int = 0

    @property
    def blocked(self):
        return [c for c in self.checks if c.state == FATAL]


# Environment variables this installer must never pass to a child process.
#
# The setup window is tkinter, and PyInstaller's tkinter runtime hook points
# TCL_LIBRARY and TK_LIBRARY at directories inside THIS exe's own unpacked
# copy so that the window can find them. Every child process inherits them,
# and for one child that is fatal: the PyInstaller that compiles
# Thumbnail Maker.exe asks Tcl where its libraries live, is told a path inside
# the installer's own temp directory, finds no tk8.6 beside it, and says so —
#
#     WARNING: TclTkInfo: Tk library/data directory
#     '...\_MEI176922	k8.6' does not exist!
#
# — before building a launcher with no Tk data in it. That exe then cannot
# start on any machine, and fails with the rthook's own message:
#
#     Tk data directory "...\_MEI217162\_tk_data" not found.
#
# Worth this many lines because of how it presented: the install ran to the
# end, verified every file, reported success, and produced one broken exe —
# the only file the user ever double-clicks. Nothing in the install log said
# anything was wrong except one WARNING among four thousand lines of pip.
#
# Stripped for every child rather than only for that one. These variables
# describe where THIS program's copy of Tcl/Tk lives; nothing it starts has
# any business being told that.
INHERITED_TK_VARS = ("TCL_LIBRARY", "TK_LIBRARY", "TKPATH")

# The same argument, for the same reason, about a different Python.
#
# PYTHONPATH and PYTHONHOME name where SOME Python keeps its standard library,
# and they are inherited by every child this installer starts — including the
# venv interpreter that pip and PyInstaller run inside. A machine where one of
# them points at another installation's Lib/ and DLLs/ hands that installation's
# extension modules to a build of a different version, and the exe that comes
# out of it dies on its first double-click with
#
#     ImportError: Module use of python310.dll conflicts with this version
#     of Python
#
# — a failure that is invisible on the machine that built it, because nothing
# about the build says anything went wrong.
#
# Nothing this installer starts should be told where a Python other than its
# own keeps its library. The venv knows where its own is.
FOREIGN_PYTHON_VARS = ("PYTHONPATH", "PYTHONHOME")


def child_env(extra=None):
    """A copy of this process's environment, fit to hand to a child."""
    env = os.environ.copy()
    for name in INHERITED_TK_VARS + FOREIGN_PYTHON_VARS:
        env.pop(name, None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _run(args, timeout=60, cwd=None):
    """Runs a command, never raising, never flashing a console window."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=cwd,
            env=child_env(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _interpreter_version(exe):
    """(major, minor, '64bit') for an interpreter, or None if it won't run."""
    code = "import sys,struct;print(sys.version_info[0],sys.version_info[1],struct.calcsize('P')*8)"
    r = _run([str(exe), "-c", code], timeout=20)
    if not r or r.returncode != 0:
        return None
    try:
        major, minor, bits = r.stdout.split()
        return int(major), int(minor), int(bits)
    except ValueError:
        return None


def _reg_str(key, name):
    """One string value out of an already-open registry key, or None."""
    try:
        value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _subkeys(root, path, flags):
    """
    The names of the subkeys under `root\\path`, or nothing if it will not open.

    Enumeration stops at the first error, which is how winreg says "that was
    the last one" — there is no count to ask for.
    """
    try:
        key = winreg.OpenKey(root, path, 0, winreg.KEY_READ | flags)
    except OSError:
        return []
    with key:
        names, i = [], 0
        while True:
            try:
                names.append(winreg.EnumKey(key, i))
            except OSError:
                return names
            i += 1


def registry_pythons():
    """
    Every interpreter Windows holds a record of.

    This is the only complete answer to "what Python is on this machine", and
    it is here because the three cheaper answers each have the same blind
    spot. The py launcher reads this very registry but is itself optional and
    frequently absent — the private copy THIS installer lays down is
    installed with Include_launcher=0, and it is not the only thing that does
    that. PATH holds at most whichever install won it. The directory globs in
    find_python only know the two places the python.org installer defaults to.

    So an all-users install under C:\\Program Files\\Python311, made without
    the launcher and left off PATH, is invisible to all three. It is not
    invisible to the python.org installer, which sees it, treats a request to
    install that same version somewhere new as a repair of the one already
    there, puts nothing at the location it was given, and exits 0. The result
    was an install that failed saying Python "did not install", on a machine
    that had Python all along. That is what this function exists to prevent;
    ensure_python holds the other half of the fix.

    PEP 514 layout: Software\\Python\\<Company>\\<Tag>\\InstallPath, with the
    interpreter named by ExecutablePath, or the key's default value giving
    the directory it sits in on registrations too old to have that value.
    """
    found = []
    for hive, flags in (
        (winreg.HKEY_CURRENT_USER, 0),
        # Both views of HKLM, deliberately. This installer is frozen 64-bit,
        # so it reads the 64-bit view by default — but a per-machine Python
        # registered in the 32-bit view is still a real install that the
        # python.org bundle will find, and reading only our own view would be
        # recreating the blind spot above on purpose. 32-bit interpreters are
        # rejected later, on what they report rather than on where they were
        # registered.
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
    ):
        try:
            root = winreg.OpenKey(hive, r"Software\Python", 0, winreg.KEY_READ | flags)
        except OSError:
            continue
        with root:
            for company in _subkeys(root, "", flags):
                for tag in _subkeys(root, company, flags):
                    try:
                        key = winreg.OpenKey(root, fr"{company}\{tag}\InstallPath",
                                             0, winreg.KEY_READ | flags)
                    except OSError:
                        continue
                    with key:
                        exe = _reg_str(key, "ExecutablePath")
                        if not exe:
                            base = _reg_str(key, "")
                            exe = str(Path(base) / "python.exe") if base else None
                    if exe:
                        found.append(Path(exe))
    return found


def find_python():
    """
    The best interpreter already on this machine, or None.

    Candidates come from the registry first because it is the only source
    that enumerates every registered install rather than just whichever one
    happens to be first on PATH — see registry_pythons for what missing one
    of them costs. Newest acceptable version wins: all of them can run the
    app, and the newer the interpreter the longer the install stays
    supported.
    """
    candidates = list(registry_pythons())

    r = _run(["py", "-0p"], timeout=20)
    if r and r.returncode == 0:
        for line in r.stdout.splitlines():
            m = re.search(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", line, re.I)
            if m:
                candidates.append(Path(m.group(1)))

    for name in ("python.exe", "python3.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    if base.exists():
        candidates += sorted(base.glob("Python3*/python.exe"))
    # Where an all-users install goes — the case the registry above was added
    # for, kept here as a second chance at a copy whose registration was
    # removed, or written under a company key nobody expected.
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        root = os.environ.get(var)
        if root and Path(root).exists():
            candidates += sorted(Path(root).glob("Python3*/python.exe"))
    for minor in range(MAX_PY[1], MIN_PY[1] - 1, -1):
        candidates.append(Path(f"C:/Python3{minor}/python.exe"))

    best, best_ver = None, None
    seen = set()
    for exe in candidates:
        key = str(exe).lower()
        if key in seen or not exe.exists():
            continue
        seen.add(key)
        ver = _interpreter_version(exe)
        # 32-bit is excluded outright: there are no 64-bit-only torch wheels
        # to fall back to, so a 32-bit interpreter cannot run this app at all.
        if not ver or ver[2] != 64:
            continue
        if not (MIN_PY <= (ver[0], ver[1]) <= MAX_PY):
            continue
        if best_ver is None or ver[:2] > best_ver[:2]:
            best, best_ver = exe, ver
    return best, best_ver


def _ps(script, timeout=90):
    """One PowerShell command, its stdout, or '' — never raising, never fatal."""
    r = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
             timeout=timeout)
    return (r.stdout or "").strip() if r else ""


def security_report(install_dir):
    """
    What Windows knows about security software interfering with this install.

    Written because a process that is killed cannot say so. An antivirus does
    not raise an exception, it calls TerminateProcess — no finally block, no
    atexit, no last line in the log. The log simply stops mid-sentence, and
    from the outside that is indistinguishable from a crash, a hang, or a
    user closing the window.

    This installer is, to a heuristic scanner, an almost perfect description
    of a dropper: a PyInstaller onefile exe that downloads an interpreter,
    unpacks thousands of DLLs into AppData\\Local, and then compiles another
    executable. So "did something kill us" is a question worth being able to
    answer, and Windows will answer it — afterwards, to whoever asks.

    Asked after the fact, on the next run, by report_previous_run. Every part
    is optional: a machine with a third-party scanner has no Get-MpThreat*,
    a locked-down one may refuse the event log, and none of that is a reason
    to fail an install.
    """
    lines = []

    products = _ps("Get-CimInstance -Namespace root/SecurityCenter2 "
                   "-ClassName AntiVirusProduct -ErrorAction SilentlyContinue | "
                   "ForEach-Object { $_.displayName }", timeout=60)
    for name in [p.strip() for p in products.splitlines() if p.strip()]:
        lines.append(f"Security software registered with Windows: {name}")

    # The one that names names. Resources are the paths it acted on, which is
    # what turns "something killed it" into "this product quarantined this
    # file of ours at this time".
    threats = _ps(
        "Get-MpThreatDetection -ErrorAction SilentlyContinue | "
        "Sort-Object InitialDetectionTime -Descending | Select-Object -First 20 | "
        "ForEach-Object { $_.InitialDetectionTime.ToString('yyyy-MM-dd HH:mm:ss') + '  ' + "
        "($_.Resources -join '; ') }", timeout=90)
    ours, others = [], 0
    needle = str(install_dir).lower()
    for line in [t.strip() for t in threats.splitlines() if t.strip()]:
        if needle in line.lower() or APP_NAME.lower() in line.lower():
            ours.append(line)
        else:
            others += 1
    if ours:
        lines.append("Windows Defender acted on files belonging to this install:")
        lines += [f"  {line}" for line in ours]
    elif others:
        lines.append(f"Windows Defender has {others} recent detections, none of them ours.")

    # Windows Error Reporting writes here when a process dies badly, and
    # nothing writes here when a process is simply terminated — so an entry is
    # informative and its absence is informative too.
    events = _ps(
        "Get-WinEvent -LogName Application -MaxEvents 400 -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.Message -match '{APP_NAME.split()[0]}' }} | "
        "Select-Object -First 5 | ForEach-Object { "
        "$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') + '  ' + $_.ProviderName + '  ' + "
        "($_.Message -split \"`n\")[0] }", timeout=90)
    for line in [e.strip() for e in events.splitlines() if e.strip()]:
        lines.append(f"Application log: {line}")

    return lines


def kept_log_dir():
    """
    Where the copy of the log that outlives the install goes.

    Outside the install directory on purpose: the first copy lives inside it,
    which is the directory that goes missing — deleted by a user starting
    clean, or by something else. A record of what went wrong cannot be kept
    only inside the thing that went wrong.
    """
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / f"{APP_NAME} Setup" / "logs"


def _session_file():
    return kept_log_dir() / "session.json"


def session_start(install_dir):
    """
    Records that an install is under way, next to the log that survives.

    Half of a dead man's switch. The file is written when an install starts
    and deleted when it ends — successfully OR with a failure it managed to
    report. So a file still sitting there on the next run means the run
    before it did neither: it did not finish and it did not fail. It stopped.

    That is the one thing the dying process cannot tell anyone. A scanner
    that decides against this installer calls TerminateProcess; there is no
    exception, no finally block, no last line. The log just ends. Read from
    the outside, afterwards, the absence of a clean ending IS the evidence.
    """
    data = {
        "version": VERSION,
        "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "install_dir": str(install_dir),
        "step": "starting",
        "step_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        path = _session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def session_step(label):
    """Records which step is running, so a killed run says where it died."""
    try:
        path = _session_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["step"] = label
        data["step_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def session_end():
    """The other half: reaching this line at all is the good news."""
    try:
        _session_file().unlink()
    except OSError:
        pass


def report_previous_run():
    """
    Lines describing a previous run that was killed, or nothing at all.

    Called at the start of an install, so that the thing nobody could see
    last time is the first thing in the log this time.
    """
    try:
        data = json.loads(_session_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    lines = [
        "",
        "=== THE PREVIOUS RUN OF SETUP DID NOT FINISH ===",
        f"Version {data.get('version', '?')} started {data.get('started', '?')}, "
        f"installing to {data.get('install_dir', '?')}",
        f"It stopped during \"{data.get('step', '?')}\" (last seen "
        f"{data.get('step_at', '?')}) without finishing and without reporting a failure.",
        "A run that fails says so and a run that is closed says so. One that does "
        "neither was ended from outside — the process was terminated, which is why "
        "its log stops mid-sentence rather than at an error.",
        "Asking Windows what it knows about that:",
    ]
    found = security_report(data.get("install_dir", ""))
    lines += [f"  {line}" for line in found] or [
        "  Nothing. No security product named files of ours, and the Application "
        "log has no record of this program. That points at the process being "
        "ended without a trace — Task Manager, a shutdown, or a scanner that "
        "does not report to Windows."]
    lines.append("")
    return lines


def detect_gpu():
    """(has_cuda_gpu, description) via nvidia-smi, the only thing that knows."""
    r = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=25)
    if not r or r.returncode != 0 or not r.stdout.strip():
        return False, ""
    first = r.stdout.strip().splitlines()[0]
    return True, first.strip()


def display_adapters():
    """
    Every display adapter Windows knows about, by name.

    Consulted only when nvidia-smi found nothing, to tell two very different
    situations apart: a machine with no NVIDIA card at all, where CPU-only
    torch is simply the right answer, and a machine that has one but no
    working driver — where the CPU wheels are a silent, permanent downgrade
    the user could fix in ten minutes if anyone told them.
    """
    r = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
              "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"], timeout=60)
    if not r or r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def shell_folder(csidl, fallback):
    """
    Asks Windows where a shell folder actually is.

    Not the same as guessing %USERPROFILE%\\Desktop: OneDrive redirects the
    Desktop into the synced profile, and on a non-English Windows it is not
    even called "Desktop" — this machine's is "OneDrive\\Área de Trabalho". A
    shortcut written to the guessed path lands in a folder the user never
    sees, and the install looks like it silently did nothing.
    """
    try:
        buf = ctypes.create_unicode_buffer(1024)
        # SHGetFolderPathW rather than the newer SHGetKnownFolderPath: it
        # takes a plain integer instead of a GUID struct, needs no COM
        # allocation, and is redirection-aware, which is the part that matters.
        if ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buf) == 0 and buf.value:
            return Path(buf.value)
    except (OSError, AttributeError):
        pass
    return Path(fallback)


CSIDL_DESKTOPDIRECTORY = 0x0010
CSIDL_PROGRAMS = 0x0002


def desktop_dir():
    return shell_folder(CSIDL_DESKTOPDIRECTORY,
                        Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop")


def start_menu_dir():
    return shell_folder(CSIDL_PROGRAMS,
                        Path(os.environ.get("APPDATA", Path.home()))
                        / "Microsoft" / "Windows" / "Start Menu" / "Programs")


def windows_name():
    """
    "Windows 11 (build 26200)" rather than platform.release()'s "10".

    Windows 11 never changed its major version, so release() still answers
    "10" on it — which reads, on a scan page, as though the check got it
    wrong. The build number is the only thing that tells them apart.
    """
    try:
        build = int(platform.version().split(".")[-1])
    except (ValueError, IndexError):
        return f"Windows {platform.release()}"
    return f"Windows {'11' if build >= 22000 else '10'} (build {build})"


def has_internet():
    for host in ("pypi.org", "download.pytorch.org"):
        try:
            with socket.create_connection((host, 443), timeout=6):
                return True
        except OSError:
            continue
    return False


def font_installed():
    """True when the title face is already registered with Windows."""
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, FONTS_KEY) as key:
                for i in range(winreg.QueryInfoKey(key)[1]):
                    name, value, _ = winreg.EnumValue(key, i)
                    if FONT_FULL_NAME.lower() in str(name).lower():
                        return True
                    if isinstance(value, str) and FONT_FILE.lower() in value.lower():
                        return True
        except OSError:
            continue
    return False


def has_msvc_runtime():
    """
    opencv and torch both link the MSVC runtime and fail at import without it.

    Present on essentially every Windows 11 machine, but the failure it causes
    is a bare "DLL load failed" from deep inside a third-party import, so it
    is worth naming here rather than debugging there.
    """
    system32 = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"
    return (system32 / "vcruntime140.dll").exists() and (system32 / "msvcp140_1.dll").exists()


def deno_exe(install_dir):
    return Path(install_dir) / "runtime" / "Scripts" / "deno.exe"


def deno_version(install_dir):
    """(major, minor, patch) of the installed deno, or None if it isn't there."""
    exe = deno_exe(install_dir)
    if not exe.exists():
        return None
    r = _run([str(exe), "--version"], timeout=60)
    if not r or r.returncode != 0:
        return None
    m = re.search(r"deno\s+(\d+)\.(\d+)\.(\d+)", r.stdout)
    return tuple(int(g) for g in m.groups()) if m else None


def venv_packages(install_dir):
    """{name: version} already installed in the target venv, or {} if none."""
    python = Path(install_dir) / "runtime" / "Scripts" / "python.exe"
    if not python.exists():
        return {}
    r = _run([str(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"], timeout=120)
    if not r or r.returncode != 0:
        return {}
    try:
        return {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(r.stdout)}
    except (ValueError, KeyError, TypeError):
        return {}


def missing_pins(installed, pins):
    """The subset of `pins` the venv does not already satisfy exactly."""
    out = []
    for pin in pins:
        name, _, want = pin.partition("==")
        have = installed.get(name.lower().replace("_", "-"))
        # Local-version suffixes (2.6.0+cu124) satisfy their own base pin —
        # that is precisely the build we asked for, not a mismatch.
        if have is None or not (have == want or have.startswith(want + "+")):
            out.append(pin)
    return out


def scan_system(install_dir=DEFAULT_INSTALL_DIR):
    """Everything the installer wants to know before it touches the machine."""
    install_dir = Path(install_dir)
    scan = Scan(install_dir=install_dir)
    add = scan.checks.append

    # --- the machine itself ------------------------------------------------
    is_windows = platform.system() == "Windows"
    is_64 = platform.machine().endswith("64")
    if is_windows and is_64:
        add(Check("Operating system", OK, f"{windows_name()}, 64-bit"))
    else:
        add(Check("Operating system", FATAL,
                  f"{platform.system()} {platform.machine()} — 64-bit Windows is required"))

    # --- disk --------------------------------------------------------------
    anchor = install_dir
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    try:
        free_gb = shutil.disk_usage(anchor).free / (1024 ** 3)
        if free_gb >= REQUIRED_FREE_GB:
            add(Check("Disk space", OK, f"{free_gb:.1f} GB free on {anchor.drive or anchor}"))
        else:
            add(Check("Disk space", FATAL,
                      f"{free_gb:.1f} GB free on {anchor.drive or anchor} — "
                      f"{REQUIRED_FREE_GB} GB required"))
    except OSError as e:
        add(Check("Disk space", FATAL, f"Could not read free space: {e}"))

    # --- network -----------------------------------------------------------
    online = has_internet()
    add(Check("Internet connection", OK if online else FATAL,
              "Reachable" if online else "No connection to pypi.org — required to download components"))

    # --- graphics ----------------------------------------------------------
    scan.cuda, scan.gpu_name = detect_gpu()
    if scan.cuda:
        add(Check("Graphics card", OK, scan.gpu_name,
                  "PyTorch will be installed with CUDA 12.4 support"))
    else:
        adapters = display_adapters()
        scan.gpu_name = ", ".join(adapters)
        if any("nvidia" in a.lower() for a in adapters):
            add(Check("Graphics card", INFO,
                      f"{scan.gpu_name} — found, but its driver is not responding",
                      "PyTorch will be installed in CPU mode. Installing the current "
                      "NVIDIA driver and running Setup again would make face "
                      "restoration many times faster"))
        else:
            add(Check("Graphics card", INFO,
                      f"{scan.gpu_name or 'No NVIDIA GPU detected'} — no NVIDIA GPU",
                      "PyTorch will be installed in CPU mode — face restoration will "
                      "be much slower, but everything works"))

    # --- runtime -----------------------------------------------------------
    scan.python_exe, ver = find_python()
    if scan.python_exe:
        # 3.10 still runs everything here, but yt-dlp has already deprecated it
        # and prints a warning on every download, so it is worth naming rather
        # than leaving the user to wonder about the message.
        aged = (ver[0], ver[1]) == (3, 10)
        add(Check("Python runtime", OK, f"Python {ver[0]}.{ver[1]} 64-bit at {scan.python_exe}",
                  "A private copy of its libraries will be created for this app"
                  + (". Note: yt-dlp has deprecated Python 3.10 — installing "
                     "Python 3.11 or newer and re-running Setup is recommended "
                     "but not required" if aged else "")))
    else:
        scan.needs_python = True
        add(Check("Python runtime", MISSING,
                  f"No Python {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]} (64-bit) found",
                  f"Python {BUNDLED_PY} will be downloaded and installed for this app only"))

    if has_msvc_runtime():
        add(Check("Visual C++ runtime", OK, "Installed"))
    else:
        add(Check("Visual C++ runtime", MISSING, "Not found",
                  "Will be installed — Windows will ask for permission"))

    # --- what the venv already has ----------------------------------------
    scan.venv_packages = venv_packages(install_dir)
    torch_pins_needed = missing_pins(scan.venv_packages, TORCH_PINS)
    base_needed = missing_pins(scan.venv_packages, BASE_PINS)
    gan_needed = missing_pins(scan.venv_packages, GAN_PINS + [LAMA_PIN])

    if not scan.venv_packages:
        add(Check("Python packages", MISSING,
                  f"{len(BASE_PINS + TORCH_PINS + GAN_PINS) + 1} packages required, none installed",
                  "All will be installed into this app's private environment"))
    else:
        outstanding = len(torch_pins_needed) + len(base_needed) + len(gan_needed)
        if outstanding:
            add(Check("Python packages", MISSING,
                      f"{outstanding} of {len(BASE_PINS + TORCH_PINS + GAN_PINS) + 1} "
                      f"packages missing or out of date",
                      "Only the missing ones will be downloaded"))
        else:
            add(Check("Python packages", OK, "All required packages already installed"))

    # --- youtube -----------------------------------------------------------
    have_deno = deno_version(install_dir)
    if have_deno and have_deno >= DENO_MIN_VERSION:
        add(Check("YouTube download engine", OK,
                  "deno {}.{}.{} installed".format(*have_deno)))
    else:
        add(Check("YouTube download engine", MISSING,
                  "No JavaScript runtime found"
                  + (" (deno {}.{}.{} is too old)".format(*have_deno) if have_deno else ""),
                  "deno will be installed (~41 MB). Without it YouTube returns "
                  "\"403 Forbidden\" and videos cannot be downloaded"))

    # --- models ------------------------------------------------------------
    weights = install_dir / "app" / "backend" / "gfpgan" / "weights"
    have_weights = weights.exists() and any(weights.glob("*.pth"))
    if have_weights:
        add(Check("AI model weights", OK, f"Present in {weights}"))
    else:
        add(Check("AI model weights", MISSING, "Face restoration and inpainting models not downloaded",
                  "~620 MB will be downloaded during installation"))

    # --- fonts -------------------------------------------------------------
    # The one dependency whose absence is silent: nothing crashes, titles just
    # come out laid out against Impact's metrics. See install_font().
    if font_installed():
        add(Check("Title font", OK, f"{FONT_FULL_NAME} is installed"))
    else:
        add(Check("Title font", MISSING, f"{FONT_FULL_NAME} is not installed",
                  "Will be installed for the current user, and bundled with the app"))

    # --- existing install --------------------------------------------------
    if (install_dir / "app").exists():
        add(Check("Existing installation", INFO, f"Found at {install_dir}",
                  "It will be updated in place"))

    torch_mb = 0
    if torch_pins_needed:
        torch_mb = DOWNLOAD_MB["torch_cuda"] if scan.cuda else DOWNLOAD_MB["torch_cpu"]

    scan.download_mb = (
        (DOWNLOAD_MB["python"] if scan.needs_python else 0)
        + torch_mb
        + (DOWNLOAD_MB["base"] if base_needed else 0)
        + (DOWNLOAD_MB["gan"] if gan_needed else 0)
        + (0 if have_weights else DOWNLOAD_MB["weights"])
        + (0 if (have_deno and have_deno >= DENO_MIN_VERSION) else DOWNLOAD_MB["deno"])
        + DOWNLOAD_MB["pyinstaller"]
    )
    return scan


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# the uninstaller
# --------------------------------------------------------------------------

# Written into the install directory, verbatim apart from the @@TOKENS@@.
#
# Substituted by hand rather than with str.format because this is PowerShell:
# braces are most of its syntax, and a format string would mean doubling every
# one of them, which is a way to introduce a bug into a script nobody runs
# until the day they are trying to remove the app.

def _fill(template, **values):
    for name, value in values.items():
        template = template.replace(f"@@{name}@@", str(value))
    return template


UNINSTALL_CMD = r"""@echo off
rem Removes @@APP@@. Double-click this file.
rem
rem It exists because Windows opens a .ps1 in an editor when it is
rem double-clicked rather than running it, so the script next to this one
rem cannot be the thing a person clicks.
rem
rem The script is copied to the temp directory and started in its own window,
rem and this file then exits immediately. Both matter: it is about to delete
rem the folder it is sitting in, and Windows will not delete a .cmd that cmd
rem still has open, nor a .ps1 that PowerShell is reading.
setlocal
copy /y "%~dp0uninstall.ps1" "%TEMP%\tm-uninstall.ps1" >nul
if errorlevel 1 (
    echo Could not prepare the uninstaller. Remove "%~dp0" by hand.
    pause
    exit /b 1
)
start "Uninstall @@APP@@" powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\tm-uninstall.ps1"
exit /b 0
"""


UNINSTALL_PS1 = r"""# Removes @@APP@@ and everything it installed.
#
# Everything this app downloaded lives inside the install directory - the
# private Python environment, the libraries, the model weights - so removing
# that directory is nearly all of the job. What is left is the handful of
# things that by definition had to live somewhere else: the shortcuts, the
# per-user font entries, and the Apps & features record.
#
# Nothing else on the machine is touched. In particular no library is removed
# from any Python installation: the app never put one there. It installed its
# own, inside its own folder, which is what is being deleted here.
#
# Run by "Uninstall @@APP@@.cmd", from a copy in the temp directory, because
# this file is inside the folder it deletes.

param([switch]$Yes)
$ErrorActionPreference = 'SilentlyContinue'
$root = '@@ROOT@@'
$fontsKey = 'HKCU:\@@FONTSKEY@@'

Write-Host ''
Write-Host 'Uninstall @@APP@@' -ForegroundColor Cyan
Write-Host ''

if (-not (Test-Path -LiteralPath $root)) {
    Write-Host "Nothing to remove - $root does not exist."
    Start-Sleep -Seconds 3
    exit 0
}

if (-not $Yes) {
    Write-Host "This removes @@APP@@ and everything it installed:"
    Write-Host "  $root"
    Write-Host '  the desktop and Start menu shortcuts'
    Write-Host '  the fonts it registered for your Windows account'
    Write-Host ''
    Write-Host 'Your own channel packs in that folder go with it. Copy them out first'
    Write-Host 'if you want to keep them.'
    Write-Host ''
    $answer = Read-Host 'Remove it? [y/N]'
    if ($answer -notmatch '^(y|Y|yes|YES)$') {
        Write-Host 'Nothing was removed.'
        Start-Sleep -Seconds 2
        exit 0
    }
}

# The app holds its own files open while it is running - the launcher, and the
# two servers it started. Deleting the folder underneath them leaves most of
# it behind, so they are stopped first, by where they are running from rather
# than by name: python.exe is not ours in general, but python.exe inside this
# folder is.
Write-Host 'Closing @@APP@@ if it is running...'
Get-CimInstance Win32_Process |
    Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 2

Write-Host 'Removing shortcuts...'
@@SHORTCUTS@@

Write-Host 'Removing fonts...'
@@FONTS@@

Write-Host 'Removing the Apps and features entry...'
Remove-Item -Path 'HKCU:\@@KEY@@' -Recurse -Force

Write-Host 'Removing files...'
# Retried: a virus scanner or a OneDrive sync can hold one file for a moment,
# and one held file is the difference between a clean removal and a folder the
# user is left to delete by hand.
for ($attempt = 1; $attempt -le 5; $attempt++) {
    Remove-Item -LiteralPath $root -Recurse -Force
    if (-not (Test-Path -LiteralPath $root)) { break }
    Start-Sleep -Seconds 2
}

Write-Host ''
if (Test-Path -LiteralPath $root) {
    Write-Host 'Some files could not be removed - something still has them open.' -ForegroundColor Yellow
    Write-Host "Restart the computer and delete this folder:" -ForegroundColor Yellow
    Write-Host "  $root" -ForegroundColor Yellow
} else {
    Write-Host '@@APP@@ has been removed.' -ForegroundColor Green
}
Write-Host ''
Read-Host 'Press Enter to close'
"""


class InstallError(Exception):
    """Something went wrong that the user has to be told about verbatim."""


def shortcut_paths():
    """Where the app's shortcuts go — and where the uninstaller looks for them."""
    return [desktop_dir() / f"{APP_NAME}.lnk", start_menu_dir() / f"{APP_NAME}.lnk"]


class Installer:
    """
    Runs the install, reporting through two callbacks.

    `log(text)` receives everything, including raw pip output — this is the
    only record of what happened when an install goes wrong on a machine
    nobody can inspect. `progress(fraction, label)` drives the bar.
    """

    def __init__(self, scan, source_dir, log=print, progress=lambda f, s: None):
        self.scan = scan
        self.source = Path(source_dir)      # the unpacked payload
        self.dir = Path(scan.install_dir)
        self.log = log
        self._progress = progress
        self.cancelled = False

        self.app = self.dir / "app"
        self.runtime = self.dir / "runtime"
        self.python = self.runtime / "Scripts" / "python.exe"
        # Where torch is told to keep what it downloads, so that all of it is
        # inside the install — see MODELS_DIRNAME.
        self.models = self.dir / MODELS_DIRNAME
        # (file, registry value) for every face registered with Windows during
        # this run. The uninstaller is written from it.
        self.fonts_installed = []

    # -- plumbing ----------------------------------------------------------

    def steps(self):
        """(label, method) pairs, in the only order that works."""
        return [
            ("Copying application files", self.copy_app),
            ("Setting up the update system", self.install_updater),
            ("Setting up the Python runtime", self.ensure_python),
            ("Creating the application environment", self.create_venv),
            ("Installing core libraries", self.install_base),
            ("Installing PyTorch", self.install_torch),
            ("Installing face restoration libraries", self.install_gan),
            ("Applying compatibility patches", self.patch_basicsr),
            ("Installing the YouTube download engine", self.install_js_runtime),
            ("Installing the title fonts", self.install_fonts),
            ("Downloading AI models", self.download_models),
            ("Building Thumbnail Maker.exe", self.build_launcher),
            ("Creating shortcuts", self.create_shortcuts),
            ("Recording the installed version", self.record_version),
            ("Verifying the installation", self.verify),
        ]

    def run(self):
        steps = self.steps()
        stop = threading.Event()
        sentinel = self.dir / ".setup-running"

        def watch_files():
            """
            Notices our own files being deleted while we are still running.

            The one kind of interference that CAN be caught from the inside.
            Quarantine and termination are separate acts: a scanner often
            removes files first, or removes them and leaves the process
            running, and in that window there is still somebody here to write
            a line about it.

            The sentinel is a file this installer creates and nothing else
            knows about. It does not go missing on its own, no step here
            touches it, and no update or copy replaces the directory it sits
            in. If it existed a moment ago and does not now, something else
            is deleting our files — which is exactly the thing that is
            otherwise invisible until a user reports an empty folder.
            """
            seen = False
            while not stop.wait(SENTINEL_POLL_S):
                if sentinel.exists():
                    seen = True
                elif seen:
                    seen = False
                    self.log(
                        f"WARNING: {sentinel.name} was deleted from {self.dir} while the "
                        "install was running. Something outside this installer is "
                        "removing its files — an antivirus quarantine does exactly this. "
                        "Adding this folder to your security software's exclusions and "
                        "running Setup again is the usual fix.")
                    try:
                        sentinel.write_text("running", encoding="utf-8")
                    except OSError:
                        pass

        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("running", encoding="utf-8")
            threading.Thread(target=watch_files, daemon=True).start()
        except OSError:
            pass

        try:
            for i, (label, fn) in enumerate(steps):
                if self.cancelled:
                    raise InstallError("Installation cancelled.")
                self._progress(i / len(steps), label)
                self.log(f"\n=== {label} ===")
                fn()
            self._progress(1.0, "Done")
        finally:
            stop.set()
            try:
                sentinel.unlink()
            except OSError:
                pass

    def cancel(self):
        self.cancelled = True

    def sh(self, args, cwd=None, what="command", timeout=7200, env_extra=None):
        """
        Runs a subprocess, streaming its output into the log.

        Streaming rather than capturing matters here: a pip install of the
        CUDA wheels is a ~3 GB download, and a progress log that only appears
        at the end is indistinguishable from a hang.

        Everything that has to happen WHILE the child runs — cancellation, the
        timeout, and telling the user it is alive — is done by a second thread
        rather than between lines. See watch() for what that cost before.
        """
        # An inline -c script is echoed as a placeholder rather than verbatim:
        # spilling a dozen lines of Python into the log ahead of the output
        # that actually matters just buries it.
        echo = ["<python script>" if "\n" in str(a) else str(a) for a in args]
        self.log("> " + " ".join(echo))
        proc = subprocess.Popen(
            [str(a) for a in args], cwd=str(cwd) if cwd else None, env=child_env(env_extra),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        started = time.time()
        deadline = started + timeout
        last_output = [started]
        stop = threading.Event()
        ended = []          # "cancelled" or "timeout", written by watch()

        def watch():
            """
            Cancellation, the timeout, and a sign of life.

            All three used to be checked in the reader loop below, which is to
            say: only when the child printed something. That is fine while
            output is flowing and wrong the moment it stops — and it stops for
            a long time in the step this app spends most of its install on.
            pip unpacking torch is roughly a gigabyte of small files with
            nothing to say about any of them; on a laptop with an antivirus
            watching, that is many minutes of complete silence.

            For those minutes the window printed nothing, the bar did not
            move, and **Cancel did nothing at all**, because the loop was
            parked in readline() and never reached the line that reads the
            flag. There is only one way that looks from the outside, and one
            thing left to do about it, and a user did it: the installer had
            hung, so he killed the window.

            None of that was true. It was working. So this thread now does the
            checking, on its own clock, and says so out loud every minute —
            a heartbeat costs one line and is the difference between a step
            that is slow and a program that is dead.
            """
            while not stop.wait(5):
                if self.cancelled:
                    ended.append("cancelled")
                    proc.kill()
                    return
                if time.time() > deadline:
                    ended.append("timeout")
                    proc.kill()
                    return
                if time.time() - last_output[0] >= HEARTBEAT_S:
                    last_output[0] = time.time()
                    self.log(f"  … still working — {what.lower()}, "
                             f"{(time.time() - started) / 60:.0f} min so far. "
                             "This step can be silent for a long time.")

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            for line in proc.stdout:
                last_output[0] = time.time()
                line = line.rstrip()
                # pip's per-chunk download bar is thousands of lines of carriage
                # returns; only the completed lines are worth keeping.
                if line and not line.startswith("\r"):
                    # Truncated because a child that writes progress with
                    # carriage returns and no newline hands this loop one
                    # "line" megabytes long, and a Tk text widget asked to lay
                    # one of those out stops being a window.
                    self.log(line[:MAX_LOG_LINE] + (" …" if len(line) > MAX_LOG_LINE else ""))
        finally:
            stop.set()

        code = proc.wait()
        if "cancelled" in ended:
            raise InstallError("Installation cancelled.")
        if "timeout" in ended:
            raise InstallError(f"{what} timed out after {timeout // 60} minutes.")
        if code != 0:
            raise InstallError(f"{what} failed (exit code {code}). See the log above for details.")
        # Returned, not just checked: a zero here means "the program did not
        # complain", which is not the same as "the program did what we asked".
        # ensure_python is the caller that has to tell those two apart.
        return code

    def pip(self, *args, what="pip install", timeout=7200):
        self.sh([self.python, "-m", "pip", "install", "--disable-pip-version-check", *args],
                what=what, timeout=timeout)

    def download(self, url, dest, what):
        """
        Downloads to `dest`, logging progress in whole percent.

        Two routes, and the second one is not a retry of the first. urllib
        speaks TLS through the OpenSSL that PyInstaller froze into this exe
        from the machine that built it — that library, that version, and
        whatever provider modules it expects to find — and all of it then
        travels to strangers' computers. When it does not load there, the
        failure arrives before a single byte is transferred, with nothing
        wrong with the network:

            Could not download deno: [DSO: LOAD_FAILED] could not load the
            shared library (_ssl.c:4030)

        curl.exe has been in System32 since Windows 10 1803 and speaks TLS
        through Schannel — Windows' own stack, sharing nothing with Python's.
        So the two fail for unrelated reasons, which is the property that
        makes this a fallback worth having rather than the same attempt made
        twice.

        Seen once in the field and not reproducible afterwards, which is its
        own argument: an installer that has to be run twice to find out
        whether today is a good day is one nobody trusts.
        """
        self.log(f"Downloading {what} from {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        last = -1

        def hook(blocks, block_size, total):
            nonlocal last
            if self.cancelled:
                raise InstallError("Installation cancelled.")
            if total > 0:
                pct = min(100, int(blocks * block_size * 100 / total))
                if pct >= last + 10:
                    last = pct
                    self.log(f"  {what}: {pct}% of {total / 1024 / 1024:.0f} MB")

        try:
            urllib.request.urlretrieve(url, dest, hook)
            return dest
        except InstallError:
            # Cancellation, raised out of the hook above. Not something to
            # fall back from.
            raise
        except Exception as e:
            # Deliberately everything else. What is being caught is not one
            # library's exception type but "Python could not fetch this", and
            # the whole point of what follows is that it does not use Python
            # to fetch it.
            first = f"{type(e).__name__}: {e}"
            self.log(f"  {first}")

        curl = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "curl.exe"
        if not curl.exists():
            raise InstallError(f"Could not download {what}: {first}")

        self.log("  Trying again through Windows' own downloader.")
        dest.unlink(missing_ok=True)
        try:
            # Silent, so the log is not a screenful of progress bar, and
            # bounded by curl itself rather than by sh(): a download that
            # prints nothing gives sh nothing to notice a timeout on.
            self.sh([curl, "--location", "--fail", "--silent", "--show-error",
                     "--retry", "2", "--connect-timeout", "30", "--max-time", "3600",
                     "--output", dest, url],
                    what=f"Downloading {what}", timeout=3700)
        except InstallError as e:
            raise InstallError(
                f"Could not download {what}. Python's own downloader failed with "
                f"[{first}], and Windows' failed with [{e}]."
            )
        if not dest.exists() or not dest.stat().st_size:
            raise InstallError(f"Could not download {what}: the file arrived empty.")
        self.log(f"  {what}: {dest.stat().st_size / 1024 / 1024:.0f} MB")
        return dest

    # -- steps -------------------------------------------------------------

    def copy_app(self):
        """
        Lays the payload down in the install directory.

        Copied rather than run from where the installer unpacked it: a frozen
        installer's payload lives in a temp directory that Windows deletes the
        moment the process exits, and the app has to outlive its installer.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)

        # Re-running the installer over an existing copy replaces app/ the
        # same way an update does, and has to keep the same things: the
        # ~185 MB of face weights facexlib resolves relative to app/backend
        # are not in the payload, so a plain overwrite would throw them away
        # and quietly re-download them. See updater.KEEP_ACROSS_UPDATES.
        rescued = self.dir / "_keep"
        shutil.rmtree(rescued, ignore_errors=True)
        for rel in updater.KEEP_ACROSS_UPDATES:
            existing = self.app / rel
            if existing.exists():
                target = rescued / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(existing), str(target))
                self.log(f"holding on to {rel}")

        for name in ("backend", "frontend"):
            src = self.source / "app" / name
            if not src.exists():
                raise InstallError(f"The installer is missing its {name} files ({src}).")
            dst = self.app / name
            replace_tree(src, dst, shutil.ignore_patterns(
                "__pycache__", "*.pyc", "temp", ".pytest_cache"))
            self.log(f"{name}/ -> {dst}")

        for rel in updater.KEEP_ACROSS_UPDATES:
            held = rescued / rel
            if held.exists():
                dst = self.app / rel
                shutil.rmtree(dst, ignore_errors=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(held), str(dst))
                self.log(f"restored {rel}")
        shutil.rmtree(rescued, ignore_errors=True)

        for name in ("requirements.txt", "release.json"):
            src = self.source / "app" / name
            if src.exists():
                shutil.copy2(src, self.app / name)

        assets = self.dir / "assets"
        assets.mkdir(exist_ok=True)
        for asset in (self.source / "assets").glob("*"):
            shutil.copy2(asset, assets / asset.name)

        # The reason the whole font question exists. If this file is not on
        # disk, every title in the app is laid out against a fallback face.
        font = self.app / "frontend" / "fonts" / FONT_FILE
        if not font.exists():
            raise InstallError(
                f"The title font ({FONT_FILE}) is missing from the installer payload. "
                "The app cannot lay out titles correctly without it."
            )
        self.log(f"Title font present: {font} ({font.stat().st_size} bytes)")

    def install_updater(self):
        """
        Lays down the update machinery, and the folder the user's own channels
        live in.

        Both exist so that the next new feature or channel does not mean this
        installer running again. The updater replaces <install>/app from a
        ~1 MB patch; the content folder is read alongside the shipped one and
        is never touched by an update, so a channel dropped in here survives
        every version after it.
        """
        dest = self.dir / "updater"
        dest.mkdir(parents=True, exist_ok=True)
        for name in UPDATER_FILES:
            src = self.source / "updater" / name
            if not src.exists():
                raise InstallError(f"The installer is missing {src}.")
            shutil.copy2(src, dest / name)
            self.log(f"updater/{name}")

        content = self.dir / "content"
        for sub in ("channels", "faces", "presets"):
            (content / sub).mkdir(parents=True, exist_ok=True)
        readme = content / "README.txt"
        if not readme.exists():
            readme.write_text(USER_CONTENT_README, encoding="utf-8")
        self.log(f"Your own channels can go in {content}")

    def record_version(self):
        """
        Writes <install>/version.json - what this machine has, and where it
        looks for what comes next.

        Deliberately not inside app/: that directory is replaced wholesale by
        every update, and the record of what has been applied to a machine
        cannot live in the thing being replaced. app/release.json says what a
        BUILD is; this says what an INSTALL is.
        """
        state = updater.State(self.dir)
        state.data["app_version"] = VERSION
        state.data["installed"] = state.data.get("installed") or updater.now_iso()
        state.data["updated"] = updater.now_iso()
        state.data["feed"] = RELEASE.get("feed")
        state.data["public_key"] = RELEASE.get("public_key")
        state.data["last_result"] = "installed"
        state.save()
        self.log(f"Recorded version {VERSION} in {state.path}")
        if RELEASE.get("feed"):
            self.log(f"Updates will be looked for at {RELEASE['feed']}")
        else:
            self.log("No update server is configured - this copy will not look for patches.")

    def verify_sha256(self, path, expected):
        """
        Refuses a download whose bytes are not the bytes that were pinned.

        The counterpart to verify_signature, for a file that is not signed.
        A publisher signature says "this came from them"; this says "this is
        the exact file that was tested", which for a pinned URL is the
        question that matters.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        self.log(f"SHA-256: {actual}")
        if actual != expected:
            raise InstallError(
                f"The download does not match what was expected ({path.name}).\n"
                f"  expected {expected}\n  got      {actual}\n"
                "It will not be unpacked. This is usually a truncated or "
                "intercepted download; try again."
            )

    def install_portable_python(self, target):
        """
        Unpacks a private Python into `target`. No installer, nothing registered.

        See PORTABLE_PY_URL for why this is the first thing tried rather than
        the last. The whole operation is: download, hash, extract, move into
        place — there is no state anywhere on the machine for it to collide
        with, this time or on any future install.

        Extracted to a staging directory and moved, not extracted over the
        target: a half-written interpreter that only fails later is worse than
        no interpreter, and the check below cannot tell them apart if the
        pieces arrive one at a time.
        """
        with tempfile.TemporaryDirectory() as tmp:
            archive = self.download(PORTABLE_PY_URL, Path(tmp) / "python.tar.gz",
                                    f"Python {PORTABLE_PY}")
            self.verify_sha256(archive, PORTABLE_PY_SHA256)

            staging = Path(tmp) / "unpacked"
            self.log(f"Unpacking Python {PORTABLE_PY}")
            with tarfile.open(archive, "r:gz") as tar:
                try:
                    # Refuses absolute paths, "..", links and device files —
                    # this archive comes off the internet, and an archive that
                    # can write outside the directory it is extracted into is
                    # the oldest trick there is. Added in 3.11.4; the frozen
                    # interpreter here may predate it, hence the fallback.
                    tar.extractall(staging, filter="data")
                except TypeError:
                    tar.extractall(staging)

            root = staging / "python"
            if not (root / "python.exe").exists():
                raise InstallError(
                    f"The Python archive did not contain python.exe where expected "
                    f"({root}). Its layout has changed; the installer needs updating."
                )
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root), str(target))

        exe = target / "python.exe"
        ver = _interpreter_version(exe)
        if not ver:
            raise InstallError(f"The Python unpacked to {target} will not run.")
        self.log(f"Python {ver[0]}.{ver[1]} {ver[2]}-bit unpacked to {target}")

    def python_inventory(self):
        """
        Every Python this machine has a record of, described in one line each.

        Dead registrations are listed too, and are the point. A registry entry
        whose python.exe is gone is invisible to every version check in this
        file — and it is exactly what stops a new install, so a log that omits
        it describes a machine with no Python failing to install Python for no
        reason at all.
        """
        lines = []
        for exe in registry_pythons():
            ver = _interpreter_version(exe)
            if ver:
                lines.append(f"{exe} (Python {ver[0]}.{ver[1]} {ver[2]}-bit)")
            elif exe.exists():
                lines.append(f"{exe} (registered, but will not run)")
            else:
                lines.append(f"{exe} (registered, but the files are gone)")
        return lines

    def _try_sh(self, args, what, timeout=1800):
        """
        sh(), for a step that is allowed to fail.

        The repair and uninstall runs below are attempts, not requirements:
        a bundle asked to repair or remove something that is not there exits
        non-zero, and that is not a reason to end the install — whether an
        interpreter ends up on disk is decided by the step after it, and by
        the check at the end. A cancellation is not one of these; the user
        pressing Cancel means stop, whichever step was running.
        """
        try:
            return self.sh(args, what=what, timeout=timeout)
        except InstallError as exc:
            if self.cancelled:
                raise
            self.log(str(exc))
            return None

    def run_python_installer(self, exe, target, what, extra=(), optional=False):
        """One run of the python.org bundle, with the flags that never vary."""
        # Deliberately inert: no PATH changes, no file associations, no
        # launcher, no Start Menu entries. A tool's installer has no business
        # changing what "python" means everywhere else on the machine.
        args = [
            exe, "/quiet", *extra, "InstallAllUsers=0", f"TargetDir={target}",
            "Include_pip=1", "Include_launcher=0", "PrependPath=0",
            "AssociateFiles=0", "Shortcuts=0", "Include_test=0", "Include_doc=0",
        ]
        code = (self._try_sh(args, what) if optional
                else self.sh(args, what=what, timeout=1800))
        # Said out loud even when it is zero. A non-zero code raises inside
        # sh() and lands in the log by itself; a zero that produced nothing is
        # the interesting case, and a log that never mentions the exit code at
        # all leaves you unable to tell which of the two happened.
        self.log(f"{what} finished with exit code {code}")
        return code

    def ensure_python(self):
        """
        Puts a usable interpreter at self.scan.python_exe, whatever it takes.

        Three sources, in the order of how much of the machine they depend on.

        1. The interpreter the machine already has, if it has a usable one.
        2. A portable CPython, extracted into <install>/python. No installer,
           nothing registered, nothing that can already be in the way.
        3. The python.org installer, which is where this used to start.

        Two is not an optimisation, it is the fix. The python.org bundle only
        honours TargetDir on a FRESH install; where Windows already holds a
        record of that version it switches to maintenance mode, installs
        nothing, ignores the directory it was given, and exits 0. The record
        outlives the files, and this app's private Python was itself a
        per-user MSI install — so deleting <install> was enough to leave
        Windows certain that Python 3.11.9 was present with nothing left of
        it, and every later run of Setup then failed on a machine that had no
        Python at all. A user reached that state and could not be talked out
        of it: install, repair, and a clean install after removing the
        registration all left the target empty.

        Three keeps its own ladder for the machines that cannot reach two,
        and a no-op is not accepted as an answer there either. Repair first,
        since a registration pointing at our own deleted target is repaired
        straight back into place; if that is not it, remove the registration
        and install cleanly. Both are safe by the time they run — see the
        uninstall step for why.
        """
        if not self.scan.needs_python and self.scan.python_exe:
            self.log(f"Using the Python already on this machine: {self.scan.python_exe}")
            return

        target = self.dir / "python"
        candidate = target / "python.exe"
        if candidate.exists() and _interpreter_version(candidate):
            self.log(f"Private Python already installed at {target}")
            self.scan.python_exe = candidate
            return

        try:
            self.install_portable_python(target)
        except InstallError as exc:
            # Not fatal on its own. Anything that stops this — the download
            # blocked, a proxy rewriting the bytes, a disk full — leaves the
            # python.org route below still worth trying, and it is a different
            # server and a different mechanism.
            if self.cancelled:
                raise
            self.log(f"The portable Python could not be installed: {exc}")
            self.log("Falling back to the python.org installer")

        if candidate.exists() and _interpreter_version(candidate):
            self.scan.python_exe = candidate
            self.log(f"Python installed at {candidate}")
            return

        with tempfile.TemporaryDirectory() as tmp:
            exe = self.download(PY_INSTALLER_URL, Path(tmp) / "python-setup.exe",
                                f"Python {BUNDLED_PY}")
            self.verify_signature(exe, "Python Software Foundation")
            self.log("Installing Python (this is private to this app and changes nothing else)")
            # Every attempt is optional, including this one. A bundle that
            # fails outright is no more final than one that quietly does
            # nothing — the repair and the clean install below are worth
            # trying after either — so no single exit code ends the install.
            # What ends it is the check at the bottom: is there an interpreter
            # on disk or not.
            code = self.run_python_installer(exe, target, "The Python installer",
                                             optional=True)

            if not candidate.exists():
                self.log(f"Nothing was installed at {target} despite exit code {code} — "
                         "this machine already has a record of Python "
                         f"{BUNDLED_PY}, so the installer treated that as maintenance")
                for line in self.python_inventory() or ["(the registry lists no Python)"]:
                    self.log(f"  known to Windows: {line}")

                # The gentle one first. If the registration is our own — an
                # <install>/python that was deleted with the rest of a previous
                # install — a repair puts it back exactly where we want it, and
                # a repair cannot damage a Python somewhere else on the machine.
                self.log("Asking the Python installer to repair what Windows thinks is there")
                self.run_python_installer(exe, target, "The Python repair",
                                          extra=("/repair",), optional=True)

            if not candidate.exists():
                # Nothing was repaired into place, so the registration points
                # somewhere else, or at nothing. Remove it and install fresh.
                #
                # Safe by elimination rather than by hope: find_python ran
                # during the scan and runs again below, and both look at the
                # registry. If a working Python 3.11.9 existed anywhere on this
                # machine, one of them would have found it and this method
                # would have returned long before here. So the registration
                # standing between us and a fresh install has no working
                # interpreter behind it — nothing that anybody could be using.
                self.log("Removing the leftover registration, which has no working "
                         "Python behind it, so a clean install can proceed")
                self._try_sh([exe, "/quiet", "/uninstall"],
                             what="Removing the leftover Python registration")
                self.run_python_installer(exe, target, "The Python installer, second attempt",
                                          optional=True)

        if not candidate.exists():
            # Everything above failed, so ask the machine one more time: the
            # repair or the uninstall may itself have exposed an interpreter
            # that was there all along, and using it is a better outcome than
            # stopping.
            self.log(f"Still nothing at {target} — looking again at what this machine has")
            fallback, ver = find_python()
            if fallback:
                self.log(f"Using Python {ver[0]}.{ver[1]} 64-bit at {fallback} instead")
                self.scan.python_exe = fallback
                return

            inventory = self.python_inventory()
            detail = ("\n\nWhat Windows has a record of:\n  " + "\n  ".join(inventory)
                      if inventory else "")
            raise InstallError(
                f"Python could not be installed to {target}. Both routes were tried: "
                "the portable build, which is only unpacked into a folder, and then "
                "the python.org installer three times over — install, repair, and a "
                "clean install after removing the registration Windows already had. "
                "Each left the folder empty." + detail +
                "\n\nSend the log below to support. It says why the portable build was "
                "not used and the exit code of every attempt after it."
            )
        self.scan.python_exe = candidate
        self.log(f"Python installed at {candidate}")

    def verify_signature(self, path, expected_publisher):
        """
        Refuses to run a downloaded installer that isn't signed by its author.

        This is the one place the installer executes something it fetched off
        the internet, so the signature is checked rather than trusted.
        """
        ps = (
            f"$s = Get-AuthenticodeSignature -LiteralPath '{path}'; "
            "Write-Output $s.Status; Write-Output $s.SignerCertificate.Subject"
        )
        r = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=120)
        out = (r.stdout if r else "") or ""
        self.log(f"Signature check: {out.strip()}")
        if "Valid" not in out or expected_publisher.lower() not in out.lower():
            raise InstallError(
                f"The downloaded file is not validly signed by {expected_publisher}, "
                "so it will not be run. Check your internet connection and try again."
            )

    def create_venv(self):
        """
        A private virtualenv, so nothing here can disturb the rest of the machine.

        It matters more than usual for this app: the pins below include a
        specific CUDA build of torch and two libraries that will happily
        downgrade numpy and opencv underneath it. Doing that inside a
        throwaway environment is housekeeping; doing it to a shared system
        Python would break whatever else was using it.
        """
        # Existing is not the same as working, and the difference is not
        # academic here: an install that was interrupted, or one whose base
        # interpreter was uninstalled since, leaves a complete-looking venv
        # whose python.exe answers every command with
        #
        #     No Python at 'C:\\...\\python.exe'
        #
        # and an exit code nobody recognises. Everything downstream — pip, the
        # PyInstaller that compiles the launcher — runs through this
        # interpreter, so a venv that cannot do arithmetic is worth less than
        # no venv at all. build_installer.py asks its own build environment
        # the same question, for the same reason.
        #
        # Rebuilt rather than repaired, and only when it fails to answer:
        # this directory holds several gigabytes of torch by the end, and a
        # venv that WORKS is left alone even if it was made by a different
        # interpreter than the one the scan picked this time. Consistency is
        # what matters, and a working venv is internally consistent by
        # construction.
        if self.python.exists() and not _interpreter_version(self.python):
            self.log(f"The environment at {self.runtime} exists but will not run — "
                     "rebuilding it")
            shutil.rmtree(self.runtime, ignore_errors=True)

        if self.python.exists():
            ver = _interpreter_version(self.python)
            self.log(f"Environment already exists at {self.runtime} "
                     f"(Python {ver[0]}.{ver[1]} {ver[2]}-bit)")
        else:
            self.sh([self.scan.python_exe, "-m", "venv", self.runtime],
                    what="Creating the virtual environment", timeout=600)
        if not self.python.exists():
            raise InstallError(f"The virtual environment was not created at {self.runtime}.")
        # setuptools and wheel are needed in the environment itself, not just
        # in a build sandbox, because basicsr is installed below with build
        # isolation turned off.
        self.pip("--upgrade", "pip", "setuptools", "wheel", what="Updating pip", timeout=900)

    def install_base(self):
        needed = missing_pins(venv_packages(self.dir), BASE_PINS)
        if not needed:
            self.log("Core libraries already present — nothing to download.")
            return
        self.log(f"{len(needed)} of {len(BASE_PINS)} core libraries to install")
        self.pip(*needed, what="Installing core libraries")

    def install_torch(self):
        """
        Torch, from PyTorch's own index rather than PyPI.

        The distinction is the entire point: PyPI's torch is CPU-only, and
        installing it on a machine with an NVIDIA card silently costs an order
        of magnitude in speed on exactly the operation this app is built
        around. The CUDA build only exists on download.pytorch.org.
        """
        index = TORCH_INDEX_CUDA if self.scan.cuda else TORCH_INDEX_CPU
        flavour = "CUDA 12.4" if self.scan.cuda else "CPU-only"
        self.log(f"Installing PyTorch ({flavour}) — the largest download, "
                 f"around {DOWNLOAD_MB['torch_cuda' if self.scan.cuda else 'torch_cpu']} MB")
        self.pip(*TORCH_PINS, "--index-url", index, what="Installing PyTorch")
        self.assert_torch_flavour()

    def install_gan(self):
        """
        The face-restoration stack, and the two ways it sabotages itself.

        gfpgan pulls its own torch on Windows if it can't see one already —
        a CPU-only build that replaces the CUDA one installed above. So torch
        goes first and is re-asserted afterwards, exactly as requirements.txt
        documents.

        simple-lama-inpainting pins old numpy, opencv and pillow in its own
        requirements and downgrades all three on a plain install, breaking the
        versions everything else here is pinned to. It is installed with
        --no-deps; every dependency it actually needs is already satisfied.
        """
        self.pip(*GAN_PINS, "--no-build-isolation", what="Installing face restoration libraries")

        # Re-assert. A no-op when nothing disturbed torch, and the difference
        # between a working GPU install and a mysteriously slow one when
        # something did.
        index = TORCH_INDEX_CUDA if self.scan.cuda else TORCH_INDEX_CPU
        self.pip(*TORCH_PINS, "--index-url", index, what="Re-checking PyTorch")

        self.pip(LAMA_PIN, "--no-deps", what="Installing the inpainting library")

        # And re-assert the pins those two are known to walk over.
        self.pip(*[p for p in BASE_PINS if p.startswith(("numpy", "opencv"))],
                 what="Restoring pinned library versions")
        self.assert_torch_flavour()

    def assert_torch_flavour(self):
        """Confirms the installed torch is the build this machine should have."""
        code = (
            "import torch;"
            "print('torch', torch.__version__, 'cuda_build', torch.version.cuda,"
            " 'cuda_available', torch.cuda.is_available())"
        )
        r = _run([str(self.python), "-c", code], timeout=300)
        out = (r.stdout.strip() if r else "") or (r.stderr.strip() if r else "")
        self.log(out or "torch did not report a version")
        if not r or r.returncode != 0:
            raise InstallError("PyTorch was installed but cannot be imported. See the log above.")
        if self.scan.cuda and "cuda_available True" not in out:
            # Not fatal: the app runs on the CPU path, just slowly. Loud,
            # because silently losing the GPU is the failure that gets
            # reported later as "the app is slow" with no explanation.
            self.log("WARNING: an NVIDIA GPU was detected but PyTorch cannot use it. "
                     "The app will work, but face restoration will run on the CPU. "
                     "Updating your NVIDIA driver usually fixes this.")

    def patch_basicsr(self):
        """
        basicsr 1.4.2 imports a torchvision module that no longer exists.

        torchvision.transforms.functional_tensor was removed in 0.17; basicsr
        has not been updated since. The import is at module scope, so it takes
        down gfpgan — and with it the whole face-restoration path — on first
        use. The function it wants is still there under the public name.
        """
        # Located through sysconfig rather than by importing basicsr, because
        # importing basicsr is exactly what does not work yet: its package
        # __init__ auto-imports every *_dataset module, one of which pulls in
        # degradations, which is the file being fixed. Asking the interpreter
        # to import it first would fail here for the same reason it fails in
        # the app, and this step would never run.
        r = _run([str(self.python), "-c",
                  "import sysconfig;print(sysconfig.get_paths()['purelib'])"], timeout=120)
        if not r or r.returncode != 0:
            raise InstallError("Could not locate the site-packages directory of the app's environment.")
        target = Path(r.stdout.strip()) / "basicsr" / "data" / "degradations.py"
        if not target.exists():
            self.log(f"Nothing to patch: {target} not found")
            return

        source = target.read_text(encoding="utf-8", errors="replace")
        old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new = "from torchvision.transforms.functional import rgb_to_grayscale"
        if old in source:
            target.write_text(source.replace(old, new), encoding="utf-8")
            self.log(f"Patched {target}")
        elif new in source:
            self.log("Already patched.")
        else:
            self.log(f"WARNING: neither import found in {target}; leaving it alone.")

        r = _run([str(self.python), "-c", "import basicsr.data.degradations; print('basicsr ok')"], timeout=300)
        self.log((r.stdout or r.stderr or "").strip() if r else "no output")
        if not r or r.returncode != 0:
            raise InstallError("basicsr still fails to import after patching. See the log above.")

    def install_js_runtime(self):
        """
        Installs deno, without which YouTube downloads fail with 403 Forbidden.

        YouTube's media URLs are signed by JavaScript the watch page executes.
        yt-dlp can extract that code but not run it, so with no JS runtime on
        the machine the URLs it produces are rejected — which reaches the user
        as "Download failed: unable to download video data: HTTP Error 403".
        The app retries three times on the theory that a fresh extraction gets
        lucky; on a machine with no runtime at all, there is no luck to be had.

        It goes into the venv's Scripts directory, the first location yt-dlp
        searches, so nothing needs to be configured or added to PATH.
        """
        have = deno_version(self.dir)
        if have and have >= DENO_MIN_VERSION:
            self.log("deno {}.{}.{} already installed — nothing to do.".format(*have))
            return

        target = deno_exe(self.dir)
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            archive = self.download(DENO_URL, Path(tmp) / "deno.zip", "deno")
            self.log("Extracting deno")
            try:
                with zipfile.ZipFile(archive) as zf:
                    member = next((n for n in zf.namelist()
                                   if n.lower().endswith("deno.exe")), None)
                    if not member:
                        raise InstallError("The deno download did not contain deno.exe.")
                    # Extracted through an explicit open/copy rather than
                    # zf.extract, so the file lands as "deno.exe" regardless of
                    # any directory structure inside the archive.
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            except (zipfile.BadZipFile, OSError) as e:
                raise InstallError(f"Could not unpack deno: {e}")

        version = deno_version(self.dir)
        if not version:
            raise InstallError("deno was installed but will not run.")
        if version < DENO_MIN_VERSION:
            raise InstallError(
                "deno {}.{}.{} is older than the {}.{}.{} yt-dlp requires.".format(
                    *version, *DENO_MIN_VERSION))
        self.log("deno {}.{}.{} installed at {}".format(*version, target))

    def install_fonts(self):
        """
        Installs the packs' type faces into Windows, for the current user.

        The app carries its own copies and serves them over @font-face, which
        is what it normally renders with. This is the backstop for the case
        that produced the broken title overlay in the first place: if a served
        file ever fails to load, the next source in the stylesheet is local()
        — this installed copy — and the layout still comes out right. Without
        it the next source is Impact, ~17% wider at the same size, which turns
        every measured title into a wrong one.

        Which faces there are comes from the packs rather than from a list
        here, so a channel that arrives with its own font is covered by this
        the same way the shipped ones are.

        Only a face that declares `local` names is registered, under the first
        of them, because that list is the pack's own statement of what Windows
        may know the face by. A face that leaves it out is saying the
        opposite, and means it: `scale-condensed-bold` invents its family name
        to pin one instance of a variable font, so anything a machine handed
        back under that name would be some other cut of the family. For those
        the file is the only honest route, and a missing file is reported as
        missing rather than quietly replaced.

        Per-user on purpose: no administrator rights, and nothing outside this
        user's profile is touched.
        """
        fonts_dir = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts"
        fonts_dir.mkdir(parents=True, exist_ok=True)

        for label, spec, source in updater.content_faces(self.app / "frontend"):
            if not source.exists():
                # Not raised here: verify() is what fails the install over a
                # missing pack asset, and it names all of them at once. Doing
                # it here would report a font error for what is a payload
                # problem.
                self.log(f"WARNING: {label} is set in {source.name}, which is not there")
                continue

            names = [n for n in (spec.get("local") or []) if isinstance(n, str) and n.strip()]
            if not names:
                self.log(f"{source.name}: served by the app only — {label} declares no "
                         "name Windows could resolve it by")
                continue
            self.register_font(source, names[0], fonts_dir)

        if self.fonts_installed:
            # Tell every running program the font list changed. Without it the
            # faces are invisible until the user logs out, which includes the
            # browser the app is about to open.
            HWND_BROADCAST, WM_FONTCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001D, 0x0002
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_FONTCHANGE, 0, 0, SMTO_ABORTIFHUNG, 1000, None)

    def register_font(self, source, full_name, fonts_dir):
        """
        Copies one face into the user's font folder and registers it there.

        Recorded in self.fonts_installed as it goes, because the uninstaller
        is written later in the same run and has to remove exactly what was
        added — deriving the list a second time would be a second chance to
        derive it differently.
        """
        installed = fonts_dir / source.name
        value_name = updater.font_value_name(source, full_name)

        try:
            if not installed.exists() or installed.stat().st_size != source.stat().st_size:
                shutil.copy2(source, installed)

            # The registry entry is what makes it a font rather than a file in
            # a folder. Windows keys it by the face's full name; the value may
            # be a bare filename for per-user fonts but is written in full to
            # keep it unambiguous.
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, FONTS_KEY) as key:
                winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, str(installed))

            # Registers it for this session as well, so it works before the
            # next logout.
            ctypes.windll.gdi32.AddFontResourceW(ctypes.c_wchar_p(str(installed)))

            self.fonts_installed.append((installed, value_name))
            self.log(f"Registered '{full_name}' for the current user ({installed.name})")
        except (OSError, KeyError) as e:
            # Non-fatal by design: the bundled copy is the primary source and
            # is already in place. Losing the backstop is worth a warning, not
            # a failed install.
            self.log(f"WARNING: could not install {source.name} system-wide ({e}). "
                     "The app will still use its own bundled copy.")

    def download_models(self):
        """
        Fetches the model weights by asking the app to load its own models.

        Each library resolves its weights to a different directory — gfpgan
        relative to its package, facexlib relative to the working directory,
        simple-lama into torch's hub cache — and hardcoding those paths here
        would mean three guesses that break the day any of them changes. The
        app's own preload functions put every file exactly where the app will
        later look for it, and importing the whole stack proves it works.

        The one of the three that would otherwise land outside the install is
        pointed back inside it with TORCH_HOME; see MODELS_DIRNAME, and note
        that the launcher passes the app the same value.
        """
        self.models.mkdir(parents=True, exist_ok=True)
        self.adopt_torch_cache()

        code = (
            "import sys, face_restorer, inpainter, background_remover, text_detector\n"
            "faces = face_restorer.preload()\n"
            "print('face restoration:', face_restorer.backend_name() if faces else 'UNAVAILABLE')\n"
            "lama = inpainter.preload()\n"
            "print('inpainting:', 'LaMa ready' if lama else 'UNAVAILABLE')\n"
            "cutout = background_remover.preload()\n"
            "print('background removal:', 'U2-Net ready' if cutout else 'UNAVAILABLE')\n"
            "text = text_detector.preload()\n"
            "print('text detection:', 'ready' if text else 'UNAVAILABLE')\n"
            "sys.exit(0 if (faces and lama and cutout and text) else 3)\n"
        )
        self.log("Loading the models once so their weights are downloaded now "
                 f"rather than during the first edit (~{DOWNLOAD_MB['weights'] + DOWNLOAD_MB['cutout']} MB).")
        try:
            self.sh([self.python, "-c", code], cwd=self.app / "backend",
                    what="Downloading the AI models", timeout=5400,
                    env_extra={"TORCH_HOME": self.models})
        except InstallError as e:
            # The app degrades gracefully — it falls back to a classical
            # pipeline without these — so this does not fail the install, but
            # it changes what the user gets and is stated plainly.
            self.log(f"WARNING: {e}")
            self.log("The app will still run, but face restoration, smart inpainting "
                     "and/or background removal will be unavailable until the models "
                     "can be downloaded.")

    def adopt_torch_cache(self):
        """
        Copies weights already in torch's shared cache into the install.

        Only ever saves a download, and never costs one: this machine may
        already have the ~200 MB LaMa file because another torch program
        fetched it, or because this app was run from source before it was
        installed. Copied rather than moved — the shared cache is not ours,
        and something else is entitled to keep using it.
        """
        default = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        target = self.models / "hub" / "checkpoints"
        if not default.is_dir():
            return
        for name in TORCH_CACHED_WEIGHTS:
            src, dst = default / name, target / name
            if not src.exists() or dst.exists():
                continue
            try:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                self.log(f"Reused {src.name} from this machine's torch cache "
                         f"({src.stat().st_size / 1024 / 1024:.0f} MB not downloaded again)")
            except OSError as e:
                # It is a cache. Failing to copy one costs a download, and
                # nothing else.
                self.log(f"(could not reuse {src.name}: {e})")

    def build_launcher(self):
        """
        Compiles launcher.py into "Thumbnail Maker.exe", with the fox icon.

        Built here rather than shipped prebuilt so the exe is produced by the
        same interpreter the app runs on, and so the installer stays small.
        """
        launcher_src = self.source / "launcher.py"
        icon = self.dir / "assets" / "fox_blue.ico"
        if not launcher_src.exists():
            raise InstallError(f"The installer is missing launcher.py ({launcher_src}).")

        self.pip("pyinstaller==6.11.1", what="Installing the build tool", timeout=1800)

        build = self.dir / "_build"
        if build.exists():
            shutil.rmtree(build, ignore_errors=True)
        build.mkdir(parents=True)

        args = [
            self.python, "-m", "PyInstaller", str(launcher_src),
            "--name", APP_NAME,
            "--onefile", "--noconsole", "--clean", "--noconfirm",
            # Where this exe unpacks itself on every launch, instead of the
            # shared %TEMP%. The setup exe and the patch exe were both moved
            # off %TEMP% after unexplained unpack failures on a machine whose
            # %TEMP% held half-finished _MEI directories from other
            # PyInstaller programs — see RUNTIME_TMPDIR in build_installer.py.
            # This is the exe the user actually double-clicks, and it was the
            # one still unpacking into the shared directory.
            "--runtime-tmpdir", r"%LOCALAPPDATA%\Thumbnail Maker Launcher",
            "--distpath", str(build / "dist"),
            "--workpath", str(build / "work"),
            "--specpath", str(build),
        ]
        if icon.exists():
            args += ["--icon", str(icon)]
        self.sh(args, what="Building Thumbnail Maker.exe", timeout=2400)

        built = build / "dist" / f"{APP_NAME}.exe"
        if not built.exists():
            raise InstallError("PyInstaller did not produce Thumbnail Maker.exe.")

        final = self.dir / f"{APP_NAME}.exe"
        if final.exists():
            final.unlink()
        shutil.move(str(built), str(final))
        shutil.rmtree(build, ignore_errors=True)
        self.log(f"Built {final} ({final.stat().st_size / 1024 / 1024:.1f} MB)")
        self.assert_launcher_is_startable(final)

    def assert_launcher_is_startable(self, exe):
        """
        Refuses to accept a launcher that PyInstaller built but Tk cannot open.

        A frozen exe can be produced successfully and still be missing
        something it only needs at startup, and this app has shipped exactly
        that: a launcher with no Tk data in it, from an install that ran to the
        end and verified every file it knew to look for. It died on the user's
        first double-click, days later, with a message about a directory
        nobody had heard of. See INHERITED_TK_VARS for what caused it.

        The cause is fixed. This is here because the cause was invisible: one
        WARNING in four thousand lines of build output, and a build that
        exited 0. Asked of the exe itself, the question has one answer.

        Read rather than run: starting the launcher would start the app's two
        servers, and starting a BROKEN one puts a modal Windows error box on
        screen that waits for a click — which is an installer that hangs.
        """
        probe = (
            "import sys\n"
            "from PyInstaller.archive.readers import CArchiveReader\n"
            "toc = CArchiveReader(sys.argv[1]).toc\n"
            "print(sum(1 for n in toc if n.startswith('_tk_data')))\n"
        )
        r = _run([str(self.python), "-c", probe, str(exe)], timeout=300)
        found = (r.stdout or "").strip() if r else ""
        if not found.isdigit():
            # Not fatal: failing to ask the question is not the same as
            # getting a bad answer, and the exe is otherwise built.
            self.log(f"WARNING: could not check {exe.name} for its Tk data "
                     f"({(r.stderr or '').strip() if r else 'no output'})")
            return
        if int(found) == 0:
            raise InstallError(
                f"{exe.name} was built without the Tk data it needs to start, so it "
                "would fail on the first double-click. This is the environment the "
                "build ran in, not the machine — see the TclTkInfo warning in the log "
                "above."
            )
        self.log(f"OK  {exe.name} carries its Tk data ({found} files)")

        # And then the general question, because the specific one above only
        # catches the failure we already know about. A second user's exe was
        # built with an extension module from a different Python than the one
        # frozen beside it — nothing to do with Tk data, same shape of
        # disaster: an install that reported success and an exe that could not
        # start, discovered days later by the user.
        #
        # Which is now asked of the exe directly. The reservation that kept
        # this to reading rather than running was that starting the launcher
        # starts the app's two servers, and that a broken one puts a modal
        # Windows error box on screen and waits for a click — an installer
        # that hangs. --selftest answers the first (it exits before anything
        # starts) and the timeout answers the second: the box goes when the
        # process it belongs to is killed.
        self.log(f"Checking that {exe.name} can start")
        r = _run([str(exe), "--selftest"], timeout=180)
        if r is None:
            raise InstallError(
                f"{exe.name} did not finish starting within three minutes, which means "
                "it stopped on an error box during startup. It was built on this "
                "machine and cannot run on it, so the install would leave you with an "
                "exe that does nothing when double-clicked."
            )
        if r.returncode != 0:
            # Its stderr is the traceback, and is the whole diagnosis.
            detail = (r.stderr or r.stdout or "").strip()
            raise InstallError(
                f"{exe.name} was built, but fails when started (exit code "
                f"{r.returncode}). The install would otherwise finish and leave you "
                "with an exe that dies on the first double-click."
                + (f"\n\n{detail}" if detail else "")
            )
        self.log(f"OK  {exe.name} starts")

    def create_shortcuts(self):
        """Desktop and Start Menu entries, plus the Apps & features record."""
        target = self.dir / f"{APP_NAME}.exe"
        icon = self.dir / "assets" / "fox_blue.ico"

        for link in shortcut_paths():
            try:
                link.parent.mkdir(parents=True, exist_ok=True)
                ps = (
                    "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
                    "$s.TargetPath='{exe}';"
                    "$s.WorkingDirectory='{cwd}';"
                    "$s.IconLocation='{ico}';"
                    "$s.Description='Thumbnail Maker';"
                    "$s.Save()"
                ).format(lnk=link, exe=target, cwd=self.dir,
                         ico=str(icon) if icon.exists() else str(target))
                r = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=120)
                if r and r.returncode == 0:
                    self.log(f"Shortcut: {link}")
                else:
                    self.log(f"WARNING: could not create {link}")
            except OSError as e:
                self.log(f"WARNING: could not create {link} ({e})")

        self.write_uninstaller()

    def write_uninstaller(self):
        """
        A one-click removal, and the Apps & features entry that points at it.

        Worth the sixty lines: what this installs is several gigabytes in a
        directory most people will never find again on their own.

        Two files rather than one, because a .ps1 is not something Windows
        runs when it is double-clicked — it opens it in an editor. The .cmd
        next to it is what a person clicks; the .ps1 is what does the work,
        and Apps & features goes through the .cmd as well so there is one path
        through this and not two.
        """
        script = self.dir / "uninstall.ps1"
        launcher = self.dir / f"Uninstall {APP_NAME}.cmd"

        shortcuts = "\n".join(
            f"Remove-Item -LiteralPath '{link}' -Force" for link in shortcut_paths()
        )
        # Every face this run registered, and nothing else. A font left behind
        # is a registry entry pointing into a deleted folder, which is how
        # Windows ends up listing fonts it cannot load; a font removed that we
        # did not install would be taking something that was not ours.
        fonts = "\n".join(
            f"Remove-ItemProperty -Path $fontsKey -Name '{value}' -Force\n"
            f"Remove-Item -LiteralPath '{path}' -Force"
            for path, value in self.fonts_installed
        ) or "# no fonts were registered by this installation"

        script.write_text(_fill(UNINSTALL_PS1, APP=APP_NAME, ROOT=str(self.dir),
                                KEY=UNINSTALL_KEY, FONTSKEY=FONTS_KEY,
                                SHORTCUTS=shortcuts, FONTS=fonts),
                          encoding="utf-8")
        launcher.write_text(_fill(UNINSTALL_CMD, APP=APP_NAME), encoding="utf-8")
        self.log(f"Uninstaller: {launcher}")

        # Stats every file in a multi-gigabyte tree, which takes the better
        # part of a minute. Announced first so the step is not silent for that
        # long — an installer that stops printing looks like one that hung.
        self.log("Calculating the installed size (this takes a moment)…")
        try:
            size_kb = sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file()) // 1024
            self.log(f"Installed size: {size_kb / 1024 / 1024:.1f} GB")
        except OSError:
            size_kb = 0

        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
                for name, value in (
                    ("DisplayName", APP_NAME),
                    ("DisplayVersion", VERSION),
                    ("Publisher", PUBLISHER),
                    ("InstallLocation", str(self.dir)),
                    ("DisplayIcon", str(self.dir / f"{APP_NAME}.exe")),
                    # Quoted whole: the path has a space in it, and Windows
                    # hands this string to the shell exactly as written.
                    ("UninstallString", f'"{launcher}"'),
                ):
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
                winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, size_kb)
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            self.log("Registered in Apps & features")
        except OSError as e:
            self.log(f"WARNING: could not register the uninstaller ({e})")

    def verify(self):
        """
        Checks the things that would otherwise fail silently at runtime.

        Every item here has a failure mode that produces no error message in
        the app itself — a missing font that yields a wrong-looking title, a
        missing asset that yields a blank canvas.
        """
        problems = []

        must_exist = [
            self.dir / f"{APP_NAME}.exe",
            self.app / "backend" / "api.py",
            self.app / "frontend" / "index.html",
            self.app / "frontend" / "js" / "main.js",
            self.app / "frontend" / "vendor" / "fabric.min.js",
            self.app / "frontend" / "fonts" / FONT_FILE,
            # Without these, the app runs but can never be updated — and
            # nothing in the app itself would ever say so. An install that
            # cannot take a patch is an install that has to be redone from
            # scratch for every future change, which is the whole thing this
            # arrangement exists to avoid.
            self.app / "release.json",
            self.dir / "version.json",
            self.dir / "updater" / "updater.py",
            self.dir / "updater" / "ed25519.py",
            self.dir / "updater" / "launcher.py",
            # Where the user's own channels go. Created empty; its absence
            # would mean a pack dropped in later is simply never seen.
            self.dir / "content" / "channels",
            # Several gigabytes in a folder most people will never find again
            # on their own. An install with no way out of it is not finished.
            self.dir / f"Uninstall {APP_NAME}.cmd",
            self.dir / "uninstall.ps1",
        ]
        for path in must_exist:
            if path.exists():
                self.log(f"OK  {path}")
            else:
                problems.append(f"missing: {path}")

        packs = self.app / "frontend" / "content" / "channels"
        found = sorted(d.name for d in packs.iterdir() if d.is_dir()) if packs.is_dir() else []
        if found:
            self.log(f"OK  channels: {', '.join(found)}")
        else:
            # Not cosmetic: with no packs the dropdown is empty, every title
            # field stays locked, and the app looks like it has forgotten what
            # it is for.
            problems.append(f"no channel packs were installed ({packs})")

        # Every file the packs name: each channel's type face, and the
        # textures their highlights are filled with. Read from the packs
        # themselves rather than listed here, so a channel added later is
        # covered by this without anyone remembering to come back for it.
        #
        # None of them fails loudly at run time. A missing texture simply does
        # not appear; a face that cannot be reached means every title in that
        # channel is laid out against Impact's metrics instead of the ones the
        # design was measured in. Both are the kind of wrong that surfaces
        # months later as "it looks a bit off", which is why they are checked
        # at the one moment somebody is watching.
        frontend = self.app / "frontend"
        problems += updater.missing_content_assets(frontend)
        for label, path in updater.content_assets(frontend):
            if not path.exists() or not path.stat().st_size:
                continue
            self.log(f"OK  {label} ({path.stat().st_size / 1024:.0f} KB)")
            # And compared against the payload byte for byte: a truncated copy
            # loads as a broken face, which fails the same way a missing one
            # does without looking missing.
            try:
                original = self.source / "app" / path.relative_to(self.app)
            except ValueError:
                continue
            if original.exists() and original.stat().st_size != path.stat().st_size:
                problems.append(f"{label} did not copy intact ({path})")

        # Asked of yt-dlp itself rather than by looking for the file: what
        # matters is not that deno.exe exists but that yt-dlp finds it and
        # considers the version usable, which is a different question and the
        # one that decides whether YouTube downloads work.
        runtime_check = _run([str(self.python), "-c",
                              "from yt_dlp.utils._jsruntime import DenoJsRuntime;"
                              "print(DenoJsRuntime().info)"], timeout=180)
        found = (runtime_check.stdout.strip() if runtime_check else "") or "None"
        self.log(f"JS runtime seen by yt-dlp: {found}")
        if "supported=True" not in found:
            problems.append(
                "yt-dlp cannot find a usable JavaScript runtime — YouTube "
                "downloads would fail with 403 Forbidden")

        code = (
            "import fastapi, uvicorn, cv2, numpy, torch, scipy\n"
            "print('imports ok — torch', torch.__version__,"
            " 'cuda', torch.cuda.is_available())\n"
        )
        r = _run([str(self.python), "-c", code], timeout=600)
        self.log(((r.stdout or "") + (r.stderr or "")).strip() if r else "no output")
        if not r or r.returncode != 0:
            problems.append("the installed Python environment cannot import the app's libraries")

        if problems:
            raise InstallError("Verification failed:\n  - " + "\n  - ".join(problems))
        self.log("Everything verified.")


# --------------------------------------------------------------------------
# terminal entry point, for diagnosing a machine without the GUI in the way
# --------------------------------------------------------------------------

def _print_scan(scan):
    marks = {OK: "[ok]     ", MISSING: "[install]", FATAL: "[BLOCKED]", INFO: "[note]   "}
    print(f"\n{APP_NAME} {VERSION} — system scan\n")
    for c in scan.checks:
        print(f"  {marks[c.state]} {c.name}: {c.detail}")
        if c.action:
            print(f"              -> {c.action}")
    print(f"\n  about {scan.download_mb} MB to download\n")


if __name__ == "__main__":
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_INSTALL_DIR
    result = scan_system(target)
    _print_scan(result)
    if "--install" in sys.argv:
        if result.blocked:
            print("Cannot install:")
            for c in result.blocked:
                print(f"  - {c.name}: {c.detail}")
            sys.exit(1)
        Installer(result, payload_source(), log=print,
                  progress=lambda f, s: print(f"\n[{f * 100:3.0f}%] {s}")).run()
        print("\nDone.")
