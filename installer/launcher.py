"""
Thumbnail Maker — the launcher that becomes "Thumbnail Maker.exe".

The app is a local FastAPI backend plus a static frontend the browser talks
to. Started by hand that means two console windows and a URL to remember;
this replaces all of it with one double-click: start both servers hidden,
wait until they actually answer, open the browser, and stay resident as the
window that shuts them both down again.

Updates normally arrive as a patch the user runs themselves — one file, one
double-click, see installer/patch_gui.py. This is the other route: if an
update server has been configured, the check runs here, in front of
everything else, because starting the app is the one moment its files are
reliably not in use. With no server configured, which is the default, nothing
here runs and nothing is contacted. See updater.py.

It is built by the installer, from the installed runtime, with the fox icon —
see build_launcher() in engine.py. It always runs from the install root:

    <install>/Thumbnail Maker.exe   <- this
    <install>/app/                  <- backend/ + frontend/ + release.json
    <install>/runtime/              <- the virtualenv everything runs in
    <install>/content/              <- channels this machine added
    <install>/updater/              <- updater.py, and what it needs
    <install>/logs/                 <- backend.log, frontend.log, update.log
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox

APP_NAME = "Thumbnail Maker"

# Both are fixed rather than negotiated: the frontend reaches the backend at a
# hardcoded http://localhost:8000 (frontend/js/config.js), so a backend that
# moved to a free port would simply be unreachable. A port already in use is
# therefore reported, not worked around.
BACKEND_PORT = 8000
FRONTEND_PORT = 3000

# Bound to the loopback interface only. The dev scripts bind every interface,
# which puts a video-processing API on the LAN and makes Windows raise a
# firewall prompt on first run — neither is wanted for a desktop tool.
HOST = "127.0.0.1"

# Cold start is dominated by importing torch, so this is generous. Model
# weights load in a background thread after the port opens, so answering at
# all is enough to open the browser on.
STARTUP_TIMEOUT_S = 180

BG = "#080b1e"
BLUE = "#1b2fea"
YELLOW = "#fffa00"
RED = "#FF070C"
FG = "#ffffff"


def install_root() -> Path:
    """The directory the app was installed into."""
    # sys.frozen is the packaged case: sys.executable is the .exe itself,
    # sitting in the install root. Running the .py directly (useful when
    # diagnosing a broken install) resolves the same root from the source
    # layout instead.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = install_root()
APP_DIR = ROOT / "app"
RUNTIME = ROOT / "runtime"
LOGS = ROOT / "logs"

# pythonw, not python: these are long-lived background servers and the plain
# interpreter would put a console window on screen for each one.
PYTHONW = RUNTIME / "Scripts" / "pythonw.exe"
PYTHON = RUNTIME / "Scripts" / "python.exe"

CONTENT = ROOT / "content"
UPDATER = ROOT / "updater" / "updater.py"

# Where torch keeps the weights it fetches for itself — the ~200 MB the
# inpainting model needs.
#
# Left alone it uses %USERPROFILE%\.cache	orch, which is shared with every
# other torch program on the machine and outside everything the installer
# created: weights an uninstall would have to either abandon or delete out
# from under somebody else. The installer downloads them here instead, and
# this has to name the same place or the app quietly downloads its own second
# copy on first use. See MODELS_DIRNAME in installer/engine.py.
MODELS = ROOT / "models"

TITLE_FONT = APP_DIR / "frontend" / "fonts" / "TradeGothicNextLTProHeavyCompressed.otf"

# How long the update check may take before it is abandoned and the app starts
# anyway. Short on purpose: being a version behind is a small problem, and an
# app that will not open because an update server is slow is a large one.
UPDATE_CHECK_TIMEOUT_S = 25

# How long applying one may take. Generous, because a patch that brings a new
# library spends most of it inside pip.
UPDATE_APPLY_TIMEOUT_S = 3600


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def app_version():
    """What is installed, for the window to say. Unknown is not worth failing over."""
    state = _json(ROOT / "version.json")
    return state.get("app_version") or _json(APP_DIR / "release.json").get("version") or "?"


def updates_configured():
    """
    Whether anywhere has been named to look for updates.

    Read straight off disk rather than by asking the updater, because the
    common answer is no: updates are hand-delivered files the user runs
    themselves, and spawning a Python interpreter on every single launch to be
    told there is no server would be a quarter of a second of nothing, forever.

    version.json is the machine's own answer and wins; release.json is what
    the installed version was built with, which is what a machine that has
    never been told otherwise falls back to.
    """
    state = _json(ROOT / "version.json")
    if state.get("feed") is not None:
        return bool(state["feed"])
    return bool(_json(APP_DIR / "release.json").get("feed"))


def preflight():
    """
    Returns a human-readable reason the app cannot start, or None.

    The font is checked alongside the interpreter and the backend because a
    missing .otf is not a cosmetic problem here: the title overlay's entire
    layout is measured from that face, and without it every title is composed
    against a fallback's metrics — the exact failure the installer exists to
    prevent, so it is caught before the browser ever opens.
    """
    if not PYTHONW.exists() and not PYTHON.exists():
        return "The Python runtime is missing from {}.\n\nReinstall {}.".format(RUNTIME, APP_NAME)
    if not (APP_DIR / "backend" / "api.py").exists():
        return "The application files are missing from {}.\n\nReinstall {}.".format(APP_DIR, APP_NAME)
    if not (APP_DIR / "frontend" / "index.html").exists():
        return "The interface files are missing from {}.\n\nReinstall {}.".format(APP_DIR, APP_NAME)
    if not TITLE_FONT.exists():
        return (
            "The title font is missing:\n\n{}\n\n"
            "Titles would be laid out with a fallback font and come out wrong, "
            "so {} will not start. Reinstall it to restore the font."
        ).format(TITLE_FONT, APP_NAME)
    return None


def port_is_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        return s.connect_ex((HOST, port)) == 0


def backend_answers():
    """
    True when our own backend is on the port — not merely something.

    Distinguishing the two matters: finding port 8000 occupied by an unrelated
    program is a fatal, explainable condition, while finding our own backend
    there just means the app is already running and this launch should reuse
    it instead of starting a second copy.
    """
    try:
        url = "http://{}:{}/process-video/progress".format(HOST, BACKEND_PORT)
        with urllib.request.urlopen(url, timeout=1) as r:
            return r.status == 200 and b"stage" in r.read(400)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def spawn(args, cwd, log_name, env_extra=None):
    """Starts a server with no console window, its output tee'd to a log file."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / log_name, "w", encoding="utf-8", errors="replace", buffering=1)

    env = os.environ.copy()
    # Unbuffered, so a crash traceback reaches the log file before the process
    # dies rather than sitting in a pipe buffer that is never flushed.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)

    return subprocess.Popen(
        args, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


class Launcher:
    def __init__(self):
        self.procs = []

        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        icon = ROOT / "assets" / "fox_blue.ico"
        if icon.exists():
            try:
                self.root.iconbitmap(str(icon))
            except tk.TclError:
                pass

        tk.Label(self.root, text=APP_NAME, bg=BG, fg=YELLOW,
                 font=("Segoe UI", 20, "bold")).pack(padx=48, pady=(28, 2))
        # Named here rather than buried in a log: "which version am I on" is
        # the first question of every conversation about an update that did or
        # did not arrive.
        self.version_label = tk.Label(self.root, text=f"version {app_version()}", bg=BG,
                                      fg="#7d86c9", font=("Segoe UI", 8))
        self.version_label.pack(pady=(0, 10))
        self.msg = tk.Label(self.root, text="Starting…", bg=BG, fg=FG,
                            font=("Segoe UI", 10), wraplength=420, justify="center")
        self.msg.pack(padx=48)

        self.button = tk.Button(self.root, text="Quit", command=self.quit, bg=BLUE, fg=FG,
                                activebackground=YELLOW, activeforeground=BLUE,
                                relief="flat", font=("Segoe UI", 10, "bold"),
                                width=18, cursor="hand2")
        self.button.pack(pady=(18, 28))

        self.center()

    def center(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 3
        self.root.geometry("+{}+{}".format(x, y))

    def say(self, text, colour=FG):
        self.root.after(0, lambda: self.msg.configure(text=text, fg=colour))

    def ask(self, title, message):
        """
        A yes/no dialog, asked from the worker thread.

        Tk is not thread-safe and a messagebox raised off the main thread
        either hangs or takes the window with it, so the question is posted to
        the main loop and the worker waits on the answer.
        """
        answer = {}
        done = threading.Event()

        def show():
            try:
                answer["yes"] = messagebox.askyesno(title, message)
            finally:
                done.set()

        self.root.after(0, show)
        done.wait()
        return bool(answer.get("yes"))

    def start(self):
        threading.Thread(target=self._start, daemon=True).start()
        self.root.mainloop()

    def run_updater(self, args, timeout):
        """
        Runs the updater and returns what it reported, or None.

        --json is asked for so this reads one line of structured output rather
        than parsing prose meant for a person; the human-readable version of
        the same run goes to logs/update.log either way.
        """
        python = str(PYTHON if PYTHON.exists() else PYTHONW)
        try:
            done = subprocess.run(
                [python, str(UPDATER), "--install-dir", str(ROOT), *args, "--json"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        for line in reversed((done.stdout or "").strip().splitlines()):
            try:
                return json.loads(line)
            except ValueError:
                continue
        return None

    def check_for_update(self):
        """
        Looks for a patch, offers it, and applies it — before anything starts.

        Every branch here ends with the app starting. An update is worth a
        prompt and worth a wait, and it is never worth a machine that cannot
        open the app because a server was down, a patch was damaged, or pip
        failed halfway. Whatever goes wrong is said once, on screen, and then
        the version already installed is launched.
        """
        if not UPDATER.exists() or not updates_configured():
            return

        self.say("Checking for updates…")
        found = self.run_updater(["--check"], UPDATE_CHECK_TIMEOUT_S)
        if not found or not found.get("available"):
            return

        version = found.get("latest") or "a new version"
        notes = "\n".join(f"  •  {n}" for n in (found.get("notes") or [])[:8])
        if not self.ask(
            APP_NAME,
            f"Version {version} is available (you have {found.get('current')}).\n\n"
            + (notes + "\n\n" if notes else "")
            + "It installs in a few seconds and keeps everything already downloaded.\n\n"
              "Update now?"
        ):
            return

        self.say(f"Updating to {version}… (this window will continue on its own)")
        result = self.run_updater(["--apply"], UPDATE_APPLY_TIMEOUT_S)
        if result is None:
            self.say("The update did not finish. Starting the version you have.", YELLOW)
            return
        if result.get("problem"):
            # Not fatal, and not hidden either: the app on disk is either the
            # old version untouched or the new one with something missing, and
            # both start. Which of the two it is, is in update.log.
            self.say(f"Update problem: {result['problem']}\n\nStarting anyway.", YELLOW)
            return
        if result.get("applied"):
            applied = result["applied"][-1]
            self.root.after(0, lambda: self.version_label.configure(text=f"version {applied}"))
            self.say(f"Updated to {applied}.")

    def _start(self):
        problem = preflight()
        if problem:
            self.fatal(problem)
            return

        if backend_answers():
            # Already running: hand the user the window they already have.
            webbrowser.open("http://localhost:{}".format(FRONTEND_PORT))
            self.say("{} was already running — opened in your browser.".format(APP_NAME))
            self.root.after(0, lambda: self.button.configure(text="Close"))
            return

        # After the "already running" check and before anything is spawned:
        # this is the only point at which the app's files are known to be
        # both present and unused.
        self.check_for_update()

        # The preflight ran against the files as they were. An update has just
        # replaced them, so it is worth asking again — a patch that arrived
        # incomplete should be reported here rather than as a backend that
        # dies eight seconds later with a traceback in a log file.
        problem = preflight()
        if problem:
            self.fatal(problem)
            return

        for port, what in ((BACKEND_PORT, "backend"), (FRONTEND_PORT, "interface")):
            if port_is_open(port):
                self.fatal(
                    "Port {} is already in use by another program, and {} needs it "
                    "for its {}.\n\nClose that program and try again.".format(port, APP_NAME, what)
                )
                return

        python = str(PYTHONW if PYTHONW.exists() else PYTHON)

        self.say("Starting the engine… (the first run takes a moment)")
        # Equivalent to run_backend.py — chdir into backend/, serve api:app —
        # but bound to loopback and with the port under this file's control.
        self.procs.append(spawn(
            [python, "-m", "uvicorn", "api:app", "--host", HOST, "--port", str(BACKEND_PORT)],
            cwd=APP_DIR / "backend", log_name="backend.log",
            env_extra={"TORCH_HOME": str(MODELS)},
        ))
        self.procs.append(spawn(
            [python, "serve.py"],
            cwd=APP_DIR / "frontend", log_name="frontend.log",
            # Where this machine's own channel packs live. Outside app/, so an
            # update replacing app/ cannot take them with it — see serve.py
            # and content/README.md.
            env_extra={"PORT": str(FRONTEND_PORT), "HOST": HOST,
                       "TNMAKER_CONTENT_DIR": str(CONTENT)},
        ))

        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if any(p.poll() is not None for p in self.procs):
                self.fatal(
                    "{} could not start — one of its servers stopped unexpectedly."
                    "\n\nSee {} for details.".format(APP_NAME, LOGS)
                )
                return
            if backend_answers() and port_is_open(FRONTEND_PORT):
                break
            time.sleep(0.4)
        else:
            self.fatal(
                "{} did not finish starting within {} seconds.\n\nSee {} for details."
                .format(APP_NAME, STARTUP_TIMEOUT_S, LOGS)
            )
            return

        webbrowser.open("http://localhost:{}".format(FRONTEND_PORT))
        self.say(
            "{} is running at http://localhost:{}\n\n"
            "Leave this window open while you work — closing it shuts the app down."
            .format(APP_NAME, FRONTEND_PORT)
        )

    def fatal(self, message):
        self.say(message, RED)
        self.root.after(0, lambda: self.button.configure(text="Close"))
        self.root.after(0, lambda: messagebox.showerror(APP_NAME, message))

    def quit(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        # Give them a moment to go quietly; uvicorn releases the port on
        # terminate, and a port left behind would block the next launch.
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.root.destroy()


if __name__ == "__main__":
    # --selftest: reach this line and exit, without starting anything.
    #
    # It exists for the installer, which compiles this file into
    # Thumbnail Maker.exe and then has to find out whether the exe it just
    # produced can start — a frozen build can succeed and still be missing or
    # mismatched in something it only needs at run time. One user's exe was
    # built with a _tkinter.pyd from a different Python than the one frozen
    # beside it, and said so on his first double-click, days later:
    #
    #     ImportError: Module use of python310.dll conflicts with this
    #     version of Python
    #
    # Everything that can go wrong that way has already gone wrong by the
    # time this line runs, because it all lives in the imports at the top of
    # this file. So reaching here is the answer, and the exit code carries
    # it: nothing is printed, because a --noconsole build has nowhere to
    # print to. See assert_launcher_is_startable in engine.py.
    if "--selftest" in sys.argv:
        sys.exit(0)
    Launcher().start()
