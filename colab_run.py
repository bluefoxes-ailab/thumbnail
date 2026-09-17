"""
colab_run.py — setup + launch for Google Colab, in one cell.

Run it from a Colab cell like this (Drive must be mounted by the notebook
itself, so that part stays outside this file):

    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    %run /content/drive/MyDrive/Thumbnail_Maker/colab_run.py

Add --diagnose to check the GAN stack and exit without starting the servers:

    %run /content/drive/MyDrive/Thumbnail_Maker/colab_run.py --diagnose

── What this does differently from the naive Colab script ─────────────────

1. NOTHING IS SILENCED. Every install runs with its output visible and its
   exit status checked. A pip command whose stdout/stderr goes to DEVNULL
   turns "gfpgan failed to build" into "the GAN mysteriously doesn't run",
   which is exactly the failure this file exists to stop happening.

2. Colab's preinstalled torch/torchvision are LEFT ALONE. requirements.txt
   pins torch==2.6.0+cu124 because gfpgan drags in a broken CPU-only wheel on
   Windows; on Colab the preinstalled build is already a matching CUDA one,
   and forcing a downgrade costs a multi-GB download every session and risks
   a driver/runtime mismatch for no gain.

3. The GAN packages install with --no-deps. basicsr 1.4.2 ships as an sdist
   whose dependency list (tb-nightly, an unpinned scikit-image, an old
   opencv) is both fragile to resolve and happy to downgrade numpy/opencv/
   Pillow underneath everything else — the same trap requirements.txt
   documents for simple-lama-inpainting. Its actual import-time needs are
   small and listed explicitly in GAN_SUPPORT below.

4. basicsr's degradations.py is patched for the torchvision API that was
   removed in 0.17 (see requirements.txt), and the patch is VERIFIED rather
   than assumed — a silently-missed patch is a silently-missing GAN.

5. The project is copied to local disk (/content) and only the model weights
   stay on Drive, symlinked back. Drive is a network filesystem: running a
   pipeline that writes hundreds of frames through it is slow enough to look
   like a hang, while the weights are exactly the case it's good for — ~550MB
   downloaded once instead of once per session.

6. Frontend and backend are served from ONE origin (FastAPI serves the static
   frontend itself), so one tunnel is enough. frontend/js/config.js points at
   http://localhost:8000, which through a tunnel resolves to the *viewer's*
   own machine — the backend is simply not reachable that way. The copy under
   /content gets API rewritten to "" (same origin); the Drive original is
   untouched.

7. OpenCV is the ONE preinstalled package that gets replaced, with the pin from
   requirements.txt. Colab now ships OpenCV 5, which removed the cascade
   classifier API the face detector is built on. See OPENCV_SPEC.
"""

import argparse
import html
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request

DRIVE_ROOT = "/content/drive/MyDrive/Thumbnail_Maker"
LOCAL_ROOT = "/content/tnmaker"
PORT = 8000

# Copied from Drive to local disk. Everything else there is either generated
# (temp/, __pycache__/) or Windows-only (libs/ is a local pip target that no
# backend module imports).
COPY_EXCLUDE = ["__pycache__", "temp", ".git", "libs", "*.zip", "*.jpg"]

# Not preinstalled on Colab. Versions match requirements.txt where the app
# actually depends on them.
APP_DEPS = [
    "fastapi==0.139.0",
    "uvicorn==0.50.2",
    "pydantic==2.11.7",
    "imageio-ffmpeg==0.6.0",
    "gdown==6.1.0",
    "yt-dlp==2026.7.4",
]

# What basicsr/facexlib/gfpgan actually import at runtime and Colab doesn't
# already ship. numpy, scipy, opencv, Pillow, tqdm, pyyaml, requests,
# scikit-image, numba and torch are all preinstalled — deliberately not
# reinstalled, since re-resolving them is how a working environment gets
# downgraded into a broken one.
GAN_SUPPORT = ["addict", "future", "lmdb", "yapf", "filterpy"]

GAN_PACKAGES = [
    "basicsr==1.4.2",
    "facexlib==0.3.0",
    "gfpgan==1.3.8",
    "simple-lama-inpainting==0.1.2",
]

# The one preinstalled package this launcher does replace. Colab ships OpenCV
# 5, which dropped cv2.CascadeClassifier from the Python bindings; face_detector
# uses it (and cv2.data.haarcascades, which 5 still ships, so the failure is a
# bare AttributeError at import, not a missing-file error that would point at
# the cause). basicsr and facexlib are 2022 packages written against 4.x too.
# The pin is requirements.txt's, so Colab runs the same OpenCV as the desktop.
OPENCV_SPEC = "opencv-python==4.13.0.92"

# Every OpenCV distribution installs into the SAME cv2/ directory, so leaving a
# second one behind means the newer files win at random and cv2 ends up
# version-mixed the way repair_pillow describes. They all go, then one comes
# back. Colab's exact set varies by image, hence uninstalling names that may
# not be present (pip is fine with that).
OPENCV_DISTRIBUTIONS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
]

# Range rather than a pin: anything in it satisfies torchvision/basicsr/
# facexlib, and pinning an exact version here is what creates the mixed
# install this repairs in the first place. Only used when the runtime's
# Pillow is already broken.
PILLOW_SPEC = "Pillow>=10.0,<12"

# Probed in a subprocess before this process imports any of them, so a broken
# package reports as one clear line instead of a traceback twelve frames deep
# inside whatever happened to import it first. PIL's probe pulls in ImageFont
# specifically: a version-mixed Pillow (see repair_pillow) imports fine at the
# package level and only breaks in the submodules.
CORE_PROBES = {
    "PIL":         "from PIL import Image, ImageFont; print(Image.__version__)",
    "numpy":       "import numpy; print(numpy.__version__)",
    "cv2":         "import cv2; print(cv2.__version__)",
    "scipy":       "import scipy; print(scipy.__version__)",
    "torch":       "import torch; print(torch.__version__)",
    "torchvision": "import torchvision; print(torchvision.__version__)",
}


# How many progress.step() calls a run makes. Split in two because
# --skip-install removes exactly the ones inside install().
SETUP_STEPS = 6
INSTALL_STEPS = 4


class Progress:
    """
    A progress bar for the notebook that does NOT hide the log.

    Point 1 of this file's header is that nothing is silenced, and a progress
    bar that swallows the output would be that same mistake in a friendlier
    costume — the whole reason this launcher exists is that a hidden failure
    reads as a mysterious one. This bar is displayed FIRST and then updated in
    place, so it stays pinned at the top of the cell while every install, probe
    and traceback still streams below it.

    step() renders the label BEFORE incrementing, so the bar shows work
    finished, not work started: while "Installing GFPGAN" is on screen the fill
    has not yet moved for it. Outside IPython the whole thing degrades to a
    printed line, which is what --diagnose from a terminal gets.
    """

    def __init__(self, total: int):
        self.total = max(total, 1)
        self.done = 0
        self.handle = None
        try:
            from IPython.display import display, HTML
        except ImportError:
            return
        self._HTML = HTML
        self.handle = display(self._bar("Starting up"), display_id=True)

    def _bar(self, label: str, color: str = "#3b82f6"):
        pct = int(100 * self.done / self.total)
        return self._HTML(f"""
        <div style="font:13px system-ui,-apple-system,sans-serif;margin:8px 0 12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px">
            <span>{html.escape(label)}</span>
            <span style="opacity:.55">{self.done}/{self.total}</span>
          </div>
          <div style="height:8px;border-radius:4px;background:rgba(128,128,128,.25);
                      overflow:hidden">
            <div style="height:100%;width:{pct}%;background:{color};
                        transition:width .35s ease"></div>
          </div>
        </div>
        """)

    def _render(self, label: str, color: str = "#3b82f6") -> None:
        if self.handle is None:
            print(f"[{self.done}/{self.total}] {label}", flush=True)
        else:
            self.handle.update(self._bar(label, color))

    def step(self, label: str) -> None:
        self._render(label)
        self.done += 1

    def complete(self, label: str) -> None:
        self.done = self.total
        self._render(label, "#22c55e")

    def fail(self, label: str) -> None:
        self._render(label, "#ef4444")


def open_in_new_tab(url: str) -> None:
    """
    Surface the running app as something to click, and try to open it outright.

    The window.open is a best effort and frequently loses: Colab renders cell
    output in a sandboxed iframe, and browsers block popups that no click asked
    for. That is why the link is rendered too and is the thing actually being
    relied on — an auto-open that silently fails while the URL only exists in
    scrolled-away log output is worse than no auto-open at all.
    """
    try:
        from IPython.display import display, HTML
    except ImportError:
        return
    display(HTML(f"""
    <a href="{html.escape(url)}" target="_blank" rel="noopener"
       style="display:inline-block;margin:4px 0 12px;padding:11px 20px;
              background:#22c55e;color:#fff;border-radius:8px;font-weight:600;
              text-decoration:none;font:600 14px system-ui,-apple-system,sans-serif">
      Open Thumbnail Maker &#8599;
    </a>
    <script>
      try {{ window.open({url!r}, "_blank"); }} catch (e) {{}}
    </script>
    """))


def run(cmd: list[str], check: bool = True) -> int:
    """A subprocess call whose output the user can actually see."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    code = subprocess.run(cmd).returncode
    if code != 0:
        print(f"!! command failed with exit code {code}", flush=True)
        if check:
            sys.exit(code)
    return code


def pip(*args: str, check: bool = True) -> int:
    return run([sys.executable, "-m", "pip", "install", "-q", *args], check=check)


# ── Step 1: project files ──────────────────────────────────────────────────

def sync_project() -> None:
    if not os.path.isdir(DRIVE_ROOT):
        sys.exit(
            f"{DRIVE_ROOT} not found. Mount Drive first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive', force_remount=True)"
        )

    excludes = []
    for pattern in COPY_EXCLUDE:
        excludes += ["--exclude", pattern]
    run(["rsync", "-a", "--delete", *excludes, f"{DRIVE_ROOT}/", f"{LOCAL_ROOT}/"])

    # Model weights live on Drive and are symlinked back in, so the ~350MB
    # GFPGANv1.4.pth plus facexlib's detection/parsing weights download once
    # ever instead of once per session. gfpgan resolves that directory
    # relative to the process's cwd ("gfpgan/weights"), which is why the
    # backend must be started from the backend/ directory below.
    drive_weights = f"{DRIVE_ROOT}/backend/gfpgan"
    local_weights = f"{LOCAL_ROOT}/backend/gfpgan"
    os.makedirs(f"{drive_weights}/weights", exist_ok=True)
    if os.path.islink(local_weights) or os.path.isfile(local_weights):
        os.unlink(local_weights)
    elif os.path.isdir(local_weights):
        shutil.rmtree(local_weights)
    os.symlink(drive_weights, local_weights)
    print(f"weights cache: {local_weights} -> {drive_weights}", flush=True)


# Markers proving a file on Drive is the version this launcher expects, as
# (path, marker, what it means). The frontend and backend are copied from
# Drive by hand, one file at a time, so it is entirely possible to update one
# side and not the other — and the two halves of the start/poll/collect flow
# have to match. When they don't, the symptom is a bare JS TypeError with
# nothing pointing at the real cause (an old main.js reads the new endpoint's
# {"status": "started"} as if it were the frame payload and crashes on
# data.frames.map).
VERSION_MARKERS = [
    ("backend/api.py", "/process-video/result",
     "backend/api.py is an old copy — it still answers /process-video with the frames"),
    ("frontend/js/main.js", "waitForResult",
     "frontend/js/main.js is an old copy — it still expects the frames in the POST reply"),
]


def check_project_files() -> None:
    """Fail loudly when the copy on Drive is a mix of versions."""
    stale = []
    for relative, marker, message in VERSION_MARKERS:
        path = f"{LOCAL_ROOT}/{relative}"
        if not os.path.exists(path):
            stale.append(f"{relative} is missing from Drive")
            continue
        with open(path, encoding="utf-8") as f:
            if marker not in f.read():
                stale.append(message)

    if stale:
        sys.exit("\n".join([
            "\nThe project on Drive is out of date:", *(f"  - {s}" for s in stale),
            "\nRe-upload the files above to MyDrive/Thumbnail_Maker/ (keeping the same",
            "subfolders) and run this cell again.",
        ]))
    print("project files: up to date", flush=True)


def patch_frontend_origin() -> None:
    """
    Point the frontend at its own origin instead of http://localhost:8000.

    Only the local copy is rewritten — the Drive/repo original stays as it is,
    since on a normal desktop run the two servers really are on different
    ports and the absolute URL is correct there.
    """
    config = f"{LOCAL_ROOT}/frontend/js/config.js"
    with open(config, encoding="utf-8") as f:
        source = f.read()
    patched = source.replace(
        'export const API = "http://localhost:8000";',
        'export const API = "";  // Colab: frontend is served by the backend itself',
    )
    if patched == source:
        print("!! config.js: API constant not found — check frontend/js/config.js", flush=True)
    with open(config, "w", encoding="utf-8") as f:
        f.write(patched)


def write_colab_app() -> None:
    """
    A FastAPI app that is the real backend plus the frontend as static files.

    The mount goes on last and matches "/" only after every API route has had
    its chance, so /enhance-frame and friends still reach their handlers.
    """
    with open(f"{LOCAL_ROOT}/backend/colab_app.py", "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f'''
            from fastapi.staticfiles import StaticFiles
            from api import app

            app.mount(
                "/",
                StaticFiles(directory="{LOCAL_ROOT}/frontend", html=True),
                name="frontend",
            )
        ''').lstrip())


# ── Step 2: dependencies ───────────────────────────────────────────────────

def install(progress: Progress) -> None:
    progress.step("Installing the app's packages")
    pip(*APP_DEPS)
    progress.step("Installing the GAN's dependencies")
    pip(*GAN_SUPPORT)
    # --no-build-isolation so basicsr's setup.py can see the already-installed
    # torch it imports at module level; --no-deps for the reason in the header.
    progress.step("Installing GFPGAN (this is the slow one)")
    pip("--no-deps", "--no-build-isolation", *GAN_PACKAGES)
    progress.step("Installing the tunnel client")
    run(["npm", "install", "-g", "localtunnel"])


def repair_pillow() -> None:
    """
    Rewrite Pillow from scratch after a version-mixed install.

    Downgrading Colab's preinstalled Pillow (the old launcher pinned 9.5.0)
    can leave a directory holding files from both versions — new `_util.py`
    without the `is_directory` that old `ImageFont.py` imports from it. Every
    package that touches PIL then fails on an ImportError that names neither
    Pillow nor the thing that installed it. --force-reinstall rewrites all of
    them, and --no-cache-dir keeps pip from reusing the wheel that produced
    the mess.
    """
    print("\nPillow is version-mixed — reinstalling it cleanly...", flush=True)
    pip("--force-reinstall", "--no-cache-dir", PILLOW_SPEC)

    result = subprocess.run([sys.executable, "-c", CORE_PROBES["PIL"]],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip(), flush=True)
        sys.exit("Pillow is still broken. Restart the runtime "
                 "(Runtime > Restart session) and run this cell again.")

    print(f"Pillow repaired: {result.stdout.strip()}", flush=True)
    # This process hasn't imported PIL yet, but a partial entry from a failed
    # attempt would shadow the freshly-written files for the rest of the run.
    for module in [m for m in sys.modules if m == "PIL" or m.startswith("PIL.")]:
        del sys.modules[module]


def check_core_packages() -> None:
    """
    Confirm the runtime's foundations import before anything builds on them.

    Colab ships all of these preinstalled and consistent; what breaks them is
    a launcher pinning older versions over the top (Pillow 9.5.0, numpy
    1.26.4), which can leave either a mixed directory or a package compiled
    against the numpy that is no longer there.
    """
    print("\nchecking the runtime's core packages...", flush=True)
    broken = []
    for name, statement in CORE_PROBES.items():
        result = subprocess.run([sys.executable, "-c", statement],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {name:<12} {result.stdout.strip()}", flush=True)
        else:
            error = result.stderr.strip().splitlines()
            print(f"  {name:<12} BROKEN — {error[-1] if error else 'import failed'}", flush=True)
            broken.append(name)

    if not broken:
        return
    if broken == ["PIL"]:
        repair_pillow()
        return
    sys.exit(
        f"\nBroken packages in this runtime: {', '.join(broken)}.\n"
        "These are Colab's own preinstalled packages, so the runtime itself is what's\n"
        "damaged — most likely from an earlier cell pinning older versions over them.\n"
        "Fix: Runtime > Disconnect and delete runtime, then run this cell on a fresh one\n"
        "and do not run the old launcher script in it."
    )


def ensure_opencv() -> None:
    """
    Put the OpenCV the project is written against in place of Colab's.

    Runs on every launch, not just installing ones: --skip-install exists to
    reuse a runtime that already has the packages, and a runtime that already
    has the WRONG OpenCV is precisely the case that needs fixing. The check is
    one subprocess when nothing is wrong, so it costs nothing to always do.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import cv2; print(cv2.__version__)"],
        capture_output=True, text=True,
    )
    version = result.stdout.strip()
    if result.returncode == 0 and version.startswith("4."):
        print(f"opencv: {version} (ok)", flush=True)
        return

    print(f"\nopencv is {version or 'broken'} — the app needs 4.x, reinstalling...", flush=True)
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *OPENCV_DISTRIBUTIONS],
        check=False)
    pip("--no-cache-dir", OPENCV_SPEC)

    # Verified rather than assumed, and by constructing the class that was
    # missing rather than by reading __version__ — the version string coming
    # back right only proves pip wrote a METADATA file.
    result = subprocess.run(
        [sys.executable, "-c",
         "import cv2; cv2.CascadeClassifier("
         "cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'); "
         "print(cv2.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), flush=True)
        sys.exit("OpenCV is still not usable. Restart the runtime "
                 "(Runtime > Restart session) and run this cell again.")
    print(f"opencv repaired: {result.stdout.strip()}", flush=True)


def patch_basicsr() -> bool:
    """
    Repoint basicsr at the torchvision function that replaced the one it
    imports. Returns whether basicsr now imports at all — if this comes back
    False the GAN cannot load, and saying so here beats discovering it as a
    blank traceback later.

    The package is located with find_spec rather than `import basicsr`:
    importing it is precisely what fails until this patch lands, so an import
    used just to find the file on disk would crash before ever writing it.
    find_spec resolves a top-level package's path without executing it.
    """
    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.origin:
        print("!! basicsr is not installed", flush=True)
        return False

    path = os.path.join(os.path.dirname(spec.origin), "data", "degradations.py")
    if not os.path.exists(path):
        print(f"!! basicsr degradations.py not found at {path}", flush=True)
        return False

    with open(path, encoding="utf-8") as f:
        source = f.read()
    patched = source.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale",
    )
    if patched != source:
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"patched {path}", flush=True)

    # Verified by actually importing the GAN stack, in a fresh subprocess so
    # nothing this process may have half-imported earlier can mask the result
    # — a string check on the file only proves the edit landed, not that the
    # replacement API exists in the torchvision this runtime happens to have.
    result = subprocess.run(
        [sys.executable, "-c", "import basicsr, facexlib, gfpgan; print('ok')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), flush=True)
        return False
    print("GAN stack imports: ok", flush=True)
    return True


# ── Step 3: diagnosis ──────────────────────────────────────────────────────

def diagnose() -> None:
    """
    Answers "why is there no face restoration" directly, instead of leaving it
    to be inferred from output that looks slightly flat.

    face_restorer swallows a failed backend load by design (the classical
    pipeline is a legitimate fallback), so the only way to see the cause is to
    provoke the load here with the traceback left intact.
    """
    import traceback

    print("\n" + "=" * 60)
    print("DIAGNOSTIC")
    print("=" * 60)

    try:
        import torch
        print(f"torch          : {torch.__version__}")
        print(f"cuda available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"gpu            : {props.name}, {props.total_memory / 1024 ** 3:.1f}GB")
        else:
            print("!! No GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
            print("   Without one the GAN still runs, but a single frame takes minutes.")
    except Exception:
        traceback.print_exc()
        return

    import torchvision
    print(f"torchvision    : {torchvision.__version__}")

    for module in ("basicsr", "facexlib", "gfpgan"):
        try:
            __import__(module)
            print(f"import {module:<8}: ok")
        except Exception as e:
            print(f"import {module:<8}: FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()
            return

    sys.path.insert(0, f"{LOCAL_ROOT}/backend")
    os.chdir(f"{LOCAL_ROOT}/backend")

    print("\nloading GFPGAN (first run downloads ~350MB of weights to Drive)...")
    try:
        from gfpgan import GFPGANer
        restorer = GFPGANer(
            model_path=(
                "https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.4/GFPGANv1.4.pth"
            ),
            upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as e:
        print(f"!! GFPGANer failed to initialise — {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    print("GFPGANer       : ok")

    import numpy as np
    _, _, out = restorer.enhance(
        np.full((512, 512, 3), 128, dtype=np.uint8),
        has_aligned=False, only_center_face=False, paste_back=True,
    )
    print(f"test inference : ok (returned {None if out is None else out.shape})")

    import face_restorer
    print(f"face_restorer.preload(): {face_restorer.preload()} "
          f"(backend: {face_restorer.backend_name()})")
    print("\nGAN stack is healthy.")


# ── Step 4: launch ─────────────────────────────────────────────────────────

CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-linux-amd64")
CLOUDFLARED_BIN = "/usr/local/bin/cloudflared"


TUNNEL_LOG = f"{DRIVE_ROOT}/tunnel_logs.txt"


def wait_for_url(process: subprocess.Popen, pattern: str, timeout: int = 60) -> str:
    """
    Read the tunnel's announced URL out of its log file.

    The tunnel's output goes to a FILE rather than a pipe, and this polls the
    file. Reading it from a pipe is the obvious approach and it is a trap: the
    reader stops at the line holding the URL and never drains the rest, so the
    tunnel keeps logging into a pipe nobody empties until the ~64KB buffer
    fills, and then blocks forever inside write(). cloudflared logs every edge
    connection it registers and re-registers, so on a long session it gets
    there — and a tunnel frozen mid-write stops serving while its process is
    still alive, which looks exactly like the network being broken.

    Polling the whole file (they are a few KB) rather than accumulating chunks
    also means a URL split across two reads still matches.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(TUNNEL_LOG, encoding="utf-8", errors="replace") as f:
            match = re.search(pattern, f.read())
        if match:
            return match.group(0)
        if process.poll() is not None:
            print(f"!! the tunnel exited with code {process.returncode} "
                  f"— see {TUNNEL_LOG}", flush=True)
            return ""
        time.sleep(0.5)
    print(f"!! the tunnel announced no URL in {timeout}s — see {TUNNEL_LOG}", flush=True)
    return ""


def verify_tunnel(url: str) -> bool:
    """
    Fetch the public URL from inside Colab, and say which side is broken.

    Printing a URL is not evidence that it works, and the two ways it fails
    need opposite fixes: a tunnel that never registered is the notebook's
    problem, while a tunnel that answers here but not in the user's browser is
    a DNS or filtering problem on their machine — several ISPs and endpoint
    security products NXDOMAIN *.trycloudflare.com wholesale, since quick
    tunnels get abused for phishing. Without this check both present as "the
    site can't be reached" and there is nothing to tell them apart.

    An HTTP error status still counts as reachable: it means DNS resolved, the
    edge accepted the connection and something answered.
    """
    import urllib.error

    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                print(f"tunnel check: reachable (HTTP {response.status})", flush=True)
                return True
        except urllib.error.HTTPError as e:
            print(f"tunnel check: reachable (HTTP {e.code})", flush=True)
            return True
        except Exception as e:
            reason = e
            time.sleep(4)
    print(f"!! tunnel check: NOT reachable from Colab either — {reason}", flush=True)
    return False


def start_tunnel(kind: str) -> tuple[subprocess.Popen, str, str]:
    """
    Returns (process, public_url, password) for the chosen tunnel.

    cloudflared is the default because localtunnel drops long-running
    requests. /process-video holds ONE HTTP request open for the entire
    pipeline — download, frame extraction, scoring, clustering, GAN — which
    is minutes on a full video. When loca.lt cuts that connection it answers
    with an HTML error page, the frontend's postJson finds no JSON `detail`
    in it and reports a bare "Error", and the backend never finds out: it
    runs to completion and returns 200 to a client that stopped listening.
    Clicking Process again then starts a SECOND pipeline over the same global
    session while the first is still writing to it.

    cloudflared's quick tunnels hold long requests open and have no password
    interstitial. localtunnel stays available via --tunnel localtunnel.
    """
    log = open(TUNNEL_LOG, "w", encoding="utf-8")

    if kind == "localtunnel":
        process = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(PORT)],
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        url = wait_for_url(process, r"https://[a-z0-9-]+\.loca\.lt")
        try:
            password = urllib.request.urlopen(
                "https://loca.lt/mytunnelpassword").read().decode("utf8").strip()
        except Exception:
            password = "(fetch it manually at https://loca.lt/mytunnelpassword)"
        return process, url, password

    if not os.path.exists(CLOUDFLARED_BIN):
        print("\ndownloading cloudflared...", flush=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)

    process = subprocess.Popen(
        [CLOUDFLARED_BIN, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=log, stderr=subprocess.STDOUT, text=True,
    )
    # cloudflared announces the URL on stderr, inside a drawn box.
    return process, wait_for_url(process, r"https://[a-z0-9-]+\.trycloudflare\.com"), ""


def launch(kind: str, progress: Progress) -> None:
    progress.step("Starting the server and opening the tunnel")
    log_path = f"{DRIVE_ROOT}/backend_logs.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    env = dict(os.environ, PYTHONUNBUFFERED="1")

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "colab_app:app",
         "--host", "0.0.0.0", "--port", str(PORT), "--log-level", "info"],
        stdout=log_file, stderr=subprocess.STDOUT,
        cwd=f"{LOCAL_ROOT}/backend", env=env,
    )

    tunnel, url, password = start_tunnel(kind)

    if url:
        progress.complete("Thumbnail Maker is running")
        open_in_new_tab(url)
    else:
        progress.fail(f"The {kind} tunnel produced no URL")

    print("\n" + "=" * 60)
    if url:
        print("THUMBNAIL MAKER IS RUNNING")
        print("=" * 60)
        print(f"URL      : {url}")
        if password:
            print(f"Password : {password}")
            print("\nOpen the URL, paste the password on the warning page, and the app is ready.")
        else:
            print("\nOpen the URL — no password page with cloudflared.")
        print("Frontend and backend share this one origin — no second tunnel needed.")
        if not verify_tunnel(url):
            print("\nThe tunnel is not answering from Colab itself, so this is not your")
            print("browser. Stop the cell and run it again to get a fresh tunnel.")
        else:
            print("\nIf your browser says DNS_PROBE_FINISHED_NXDOMAIN or 'site can't be")
            print("reached', the URL is fine and your network is blocking trycloudflare.com.")
            print("Try another DNS (1.1.1.1) or a phone hotspot, or restart this cell with:")
            print("    %run .../colab_run.py --tunnel localtunnel")
    else:
        print(f"The {kind} tunnel produced no URL. Stop the cell and run it again.")
    print(f"Backend log: {log_path}")
    print("=" * 60)
    print("\nLive backend log follows (Ctrl-C / stop the cell to shut down):\n", flush=True)

    # Tailing the log into the notebook is the whole point: startup logs
    # whether face restoration loaded and, since face_restorer now warns with
    # a traceback when GFPGAN won't initialise, exactly why it didn't.
    try:
        with open(log_path, encoding="utf-8", errors="replace") as tail:
            while server.poll() is None:
                chunk = tail.read()
                if chunk:
                    print(chunk, end="", flush=True)
                else:
                    time.sleep(0.5)
            # The loop above never runs when the backend dies faster than the
            # first poll — which is exactly the case whose log matters most (an
            # import error at startup). Drain whatever it wrote before exiting.
            log_file.flush()
            print(tail.read(), end="", flush=True)
        progress.fail(f"The backend stopped (exit code {server.returncode}) — see the log below")
        print(f"\n!! Backend exited with code {server.returncode}", flush=True)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        for process in (server, tunnel):
            process.terminate()
        log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true",
                        help="check the GAN stack and exit without serving")
    parser.add_argument("--skip-install", action="store_true",
                        help="reuse the packages already installed in this runtime")
    parser.add_argument("--tunnel", choices=["cloudflared", "localtunnel"],
                        default="cloudflared",
                        help="how to expose the app publicly (see start_tunnel)")
    args, _ = parser.parse_known_args()

    progress = Progress(SETUP_STEPS + (0 if args.skip_install else INSTALL_STEPS))
    # Every failure path in here is a sys.exit with the reason already printed
    # in full above. Catching it only repaints the bar red before letting it
    # through — otherwise a run that died at step 3 leaves a blue bar sitting at
    # 30% forever, which reads as "still working" rather than "stopped".
    try:
        progress.step("Copying the project from Drive")
        sync_project()

        progress.step("Checking the project files")
        check_project_files()
        patch_frontend_origin()
        write_colab_app()

        if not args.skip_install:
            install(progress)

        progress.step("Checking the runtime's packages")
        check_core_packages()

        progress.step("Checking OpenCV")
        ensure_opencv()

        progress.step("Loading the GAN")
        if not patch_basicsr():
            sys.exit("The GAN stack cannot be imported — see the traceback above.")

        if args.diagnose:
            progress.complete("Diagnostics")
            diagnose()
            return
        launch(args.tunnel, progress)
    except SystemExit as e:
        if e.code:
            progress.fail("Setup stopped — see the message above")
        raise


if __name__ == "__main__":
    main()
