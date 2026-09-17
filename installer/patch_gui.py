"""
The window inside a patch — what a user sees when they double-click one.

A patch is delivered by hand: built here, sent over whatever gets a file from
one person to another, and run by the person who receives it. So it has to be
one file that needs nothing explained. It carries its own payload, finds the
install by itself, says what it is about to do, does it, and says whether it
worked.

    Thumbnail Maker Patch 1.1.0.exe

Everything it needs is inside it: the app files, the manifest describing the
release, and the updater that applies them. Nothing is downloaded and nothing
is checked against a server — the file in the user's hands IS the release, and
whoever sent it is the only authority involved.

The heavy lifting is all in updater.py, which is the same code an install uses
for every other route in. This is the window in front of it.
"""

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent))

import updater
from updater import UpdateError

APP_NAME = "Thumbnail Maker"

# Same palette as the launcher, deliberately: this window shows up in the
# middle of using the app and should read as part of it rather than as a
# separate program that has arrived from somewhere.
BG = "#080b1e"
PANEL = "#0e1330"
BLUE = "#1b2fea"
YELLOW = "#fffa00"
GREEN = "#38d17a"
RED = "#FF070C"
FG = "#ffffff"
DIM = "#8b93cc"

# Where the payload sits inside the built exe, and where it sits when this
# file is run straight out of the repo for testing.
PATCH_NAME = "payload.tmpatch"


class _ScrolledText(tk.Text):
    """
    tkinter.scrolledtext.ScrolledText, reimplemented in the few lines of it
    this window actually uses.

    Not a preference. The pinned CPython that build_installer.ensure_base_python
    downloads — the one every exe here is frozen with and the one the app is
    installed on — ships a PRUNED tkinter. commondialog, constants, dialog,
    filedialog, font, messagebox, simpledialog and ttk are all present;
    scrolledtext is not, anywhere in the distribution. So the import froze into
    the patch exe without a word of complaint from PyInstaller and died on its
    own first line, in front of the user, on the one file the whole release
    process exists to deliver:

        ImportError: cannot import name 'scrolledtext' from 'tkinter'

    A --hidden-import cannot fix that, which is worth saying because it is the
    obvious thing to reach for: there is no file on disk for PyInstaller to
    collect. Either the module is vendored in, or it is not used. It is not
    used.

    The original is a Text living in a Frame beside a Scrollbar, forwarding the
    Frame's geometry methods so that callers can treat the pair as one widget.
    Only pack and pack_forget are ever called on it here, so only those are
    forwarded. winfo_ismapped needs none: a Text inside an unpacked Frame is
    not mapped either, which is exactly what toggle_log is asking about.
    """

    def __init__(self, master=None, **kw):
        self.frame = tk.Frame(master)
        self.vbar = tk.Scrollbar(self.frame)
        self.vbar.pack(side="right", fill="y")
        super().__init__(self.frame, yscrollcommand=self.vbar.set, **kw)
        super().pack(side="left", fill="both", expand=True)
        self.vbar.configure(command=self.yview)

    def pack(self, **kw):
        self.frame.pack(**kw)

    def pack_forget(self):
        self.frame.pack_forget()


def bundled_patch():
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / PATCH_NAME


def find_install():
    """
    Where the app is on this machine.

    The registry first, because that is what the installer wrote and it
    survives the user having chosen a different folder. The default location
    second. Neither is guessed at silently — if both miss, the window asks.
    """
    try:
        import winreg
        key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ThumbnailMaker"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
            location = Path(winreg.QueryValueEx(k, "InstallLocation")[0])
            if (location / "app").is_dir():
                return location
    except (ImportError, OSError, ValueError, IndexError):
        pass

    default = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Programs" / APP_NAME
    if (default / "app").is_dir():
        return default
    return None


class PatchWindow:
    def __init__(self, patch_path):
        self.patch = Path(patch_path)
        self.install = find_install()
        self.manifest = None
        self.busy = False
        self.done = False

        # Everything the worker thread wants to put on screen goes through
        # here, and only the main thread ever touches a widget. Tk is not
        # thread-safe and `after()` called from another thread is not a way
        # around that — it registers a command on the interpreter, which is
        # the very thing that must not happen off the main thread.
        self.events = queue.Queue()

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} — update")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        icon = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "fox_blue.ico"
        if icon.exists():
            try:
                self.root.iconbitmap(str(icon))
            except tk.TclError:
                pass

        self.build()
        self.describe()
        self.center()

    # -- layout ------------------------------------------------------------

    def build(self):
        pad = tk.Frame(self.root, bg=BG)
        pad.pack(fill="both", expand=True, padx=36, pady=(28, 24))

        tk.Label(pad, text=APP_NAME, bg=BG, fg=YELLOW,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w")
        self.headline = tk.Label(pad, text="", bg=BG, fg=FG, font=("Segoe UI", 11),
                                 justify="left", wraplength=460)
        self.headline.pack(anchor="w", pady=(4, 0))

        self.where = tk.Label(pad, text="", bg=BG, fg=DIM, font=("Segoe UI", 8),
                              justify="left", wraplength=460)
        self.where.pack(anchor="w", pady=(6, 0))

        # What is in this release, in the words the person who built it wrote.
        # An update the user is asked to run is an update they are entitled to
        # know the contents of.
        self.notes = tk.Label(pad, text="", bg=BG, fg=FG, font=("Segoe UI", 9),
                              justify="left", wraplength=460)
        self.notes.pack(anchor="w", pady=(14, 0))

        self.status = tk.Label(pad, text="", bg=BG, fg=FG, font=("Segoe UI", 9, "bold"),
                               justify="left", wraplength=460)
        self.status.pack(anchor="w", pady=(14, 0))

        # Hidden until something goes wrong or the user asks. A wall of pip
        # output is the only thing that answers "why did it fail", and the
        # only thing nobody wants to see when it didn't.
        self.log_box = _ScrolledText(
            pad, height=10, width=64, bg=PANEL, fg=DIM, insertbackground=FG,
            font=("Consolas", 8), relief="flat", wrap="word")

        buttons = tk.Frame(pad, bg=BG)
        buttons.pack(anchor="w", pady=(18, 0), fill="x")

        self.action = tk.Button(buttons, text="Install update", command=self.start,
                                bg=BLUE, fg=FG, activebackground=YELLOW, activeforeground=BLUE,
                                relief="flat", font=("Segoe UI", 10, "bold"),
                                width=18, cursor="hand2")
        self.action.pack(side="left")

        self.secondary = tk.Button(buttons, text="Close", command=self.close,
                                   bg=PANEL, fg=DIM, activebackground=BG, activeforeground=FG,
                                   relief="flat", font=("Segoe UI", 10),
                                   width=14, cursor="hand2")
        self.secondary.pack(side="left", padx=(10, 0))

        self.details = tk.Button(buttons, text="Details", command=self.toggle_log,
                                 bg=BG, fg=DIM, activebackground=BG, activeforeground=FG,
                                 relief="flat", font=("Segoe UI", 9), cursor="hand2")
        self.details.pack(side="right")

    def center(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 3
        self.root.geometry(f"+{x}+{y}")

    def toggle_log(self):
        if self.log_box.winfo_ismapped():
            self.log_box.pack_forget()
        else:
            self.log_box.pack(anchor="w", pady=(12, 0), fill="x")
        self.center()

    # -- what this patch is ------------------------------------------------

    def describe(self):
        """Fills the window in, and refuses early if this patch cannot be applied here."""
        try:
            self.manifest = updater.read_manifest(self.patch)
        except UpdateError as e:
            self.fatal(str(e))
            return

        version = self.manifest.get("version", "?")
        self.headline.configure(text=f"Update to version {version}")

        notes = self.manifest.get("notes") or []
        self.notes.configure(
            text="\n".join(f"•  {n}" for n in notes) if notes
            else "No release notes came with this update.")

        if not self.install:
            self.headline.configure(text=f"{APP_NAME} was not found on this computer")
            self.where.configure(
                text="An update can only be applied to an existing installation. If it is "
                     "somewhere unusual, point this at it.")
            self.action.configure(text="Find the folder…", command=self.browse)
            return

        state = updater.State(self.install)
        current = state.version
        self.where.configure(text=f"{self.install}    ·    currently on {current}")

        if updater.parse_version(version) <= updater.parse_version(current):
            self.headline.configure(text=f"Version {current} is already installed")
            self.notes.configure(
                text=f"This update is version {version}, which is not newer than what is "
                     "already here. There is nothing to do.")
            self.ready_to_close()
            return

        required = self.manifest.get("min_app_version")
        if required and updater.parse_version(current) < updater.parse_version(required):
            self.headline.configure(text="An earlier update is needed first")
            self.notes.configure(
                text=f"Version {version} needs {required} or later installed before it can be "
                     f"applied, and this computer is on {current}. Apply the update in between "
                     "first, or reinstall.")
            self.ready_to_close()
            return

        if self.manifest.get("requires_setup"):
            self.headline.configure(text="This one needs the full installer")
            self.notes.configure(
                text=f"{self.manifest['requires_setup']}\n\nRun the {APP_NAME} installer "
                     "instead. It updates an existing install in place and keeps everything "
                     "already downloaded.")
            self.ready_to_close()
            return

    def browse(self):
        chosen = filedialog.askdirectory(title=f"Where is {APP_NAME} installed?")
        if not chosen:
            return
        if not (Path(chosen) / "app").is_dir():
            messagebox.showerror(
                APP_NAME,
                f"{chosen}\n\nThat does not look like a {APP_NAME} installation — there is no "
                "app folder in it.")
            return
        self.install = Path(chosen)
        self.action.configure(text="Install update", command=self.start)
        self.describe()

    # -- applying ----------------------------------------------------------

    # -- talking to the window ---------------------------------------------

    def say(self, text, colour=FG):
        """Safe from either thread: the main one applies it, everyone else queues it."""
        if threading.current_thread() is threading.main_thread():
            self.status.configure(text=text, fg=colour)
        else:
            self.events.put(("status", text, colour))

    def log(self, text):
        if threading.current_thread() is threading.main_thread():
            self._append_log(text)
        else:
            self.events.put(("log", str(text), None))

    def _append_log(self, text):
        self.log_box.insert("end", str(text) + "\n")
        self.log_box.see("end")

    def drain(self):
        """
        Applies whatever the worker has queued, then schedules itself again.

        Polling rather than being pushed at, because the alternative is the
        worker reaching into Tk — see the note on `self.events`. 80ms is below
        what reads as lag on a progress line and is nothing at all next to
        what it is reporting on.
        """
        try:
            while True:
                kind, payload, extra = self.events.get_nowait()
                if kind == "status":
                    self.status.configure(text=payload, fg=extra or FG)
                elif kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self.succeeded(payload, extra)
                elif kind == "failed":
                    self.failed(payload)
        except queue.Empty:
            pass
        self.root.after(80, self.drain)

    def start(self):
        if self.busy or not self.install:
            return

        # Checked here rather than left to the updater so the answer is a
        # button the user can press again, not an error they have to
        # understand and a program they have to run twice.
        if updater.app_is_running():
            self.say(f"{APP_NAME} is open. Close it, then press Install update again.", YELLOW)
            return

        self.busy = True
        self.action.configure(state="disabled", text="Installing…")
        self.secondary.configure(state="disabled")
        self.say("Applying the update…")
        threading.Thread(target=self._apply, daemon=True).start()

    def _apply(self):
        file_log = updater.make_logger(self.install, echo=False)

        def log(text):
            file_log(text)
            self.log(text)

        try:
            version, problems = updater.apply_patch(
                self.install, self.patch, log, source="patch file")
        except UpdateError as e:
            log(f"ERROR: {e}")
            self.events.put(("failed", str(e), None))
            return
        except Exception as e:                      # noqa: BLE001 - last line of defence
            log(f"ERROR: {type(e).__name__}: {e}")
            self.events.put(("failed",
                             f"Something unexpected went wrong ({type(e).__name__}: {e}). The "
                             "previous version has been left in place — press Details to see "
                             "what happened.", None))
            return

        self.events.put(("done", version, problems))

    def succeeded(self, version, problems):
        self.busy = False
        self.done = True
        self.headline.configure(text=f"Updated to version {version}", fg=GREEN)
        self.notes.configure(text="Open Thumbnail Maker as usual — the next time it starts it "
                                  "will be the new version.")
        if problems:
            # The app directory is on the new version and something alongside
            # it is not — a library that would not install, a model that would
            # not download. It starts, and what is missing is stated rather
            # than discovered later as a feature that quietly does nothing.
            self.say("Installed, but with problems:\n" + "\n".join(f"•  {p}" for p in problems),
                     YELLOW)
            if not self.log_box.winfo_ismapped():
                self.toggle_log()
        else:
            self.say("Done.", GREEN)
        self.action.configure(state="normal", text="Close", command=self.close)
        self.secondary.pack_forget()
        self.center()

    def failed(self, message):
        self.busy = False
        self.say(message, RED)
        self.action.configure(state="normal", text="Try again", command=self.start)
        self.secondary.configure(state="normal")
        if not self.log_box.winfo_ismapped():
            self.toggle_log()
        self.center()

    def ready_to_close(self):
        """Nothing to do, and nothing wrong — one button, and it closes."""
        self.action.configure(text="Close", command=self.close)
        self.secondary.pack_forget()

    def fatal(self, message):
        self.headline.configure(text="This update cannot be used", fg=RED)
        self.notes.configure(text=message)
        self.ready_to_close()

    def close(self):
        if self.busy:
            # Half an update is the one state this has to avoid, and the swap
            # itself is two renames — waiting it out is always the right call.
            if not messagebox.askyesno(
                    APP_NAME,
                    "The update is still being applied. Closing now could leave the "
                    "installation half-updated.\n\nClose anyway?"):
                return
        self.root.destroy()

    def run(self):
        self.drain()
        self.root.mainloop()


def main():
    # An explicit path is the repo-testing route: run this file with a
    # .tmpatch and it behaves exactly as the built exe does.
    patch = Path(sys.argv[1]) if len(sys.argv) > 1 else bundled_patch()
    if not patch.is_file():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            APP_NAME,
            f"This update file is damaged — its contents are missing ({patch.name}).\n\n"
            "Ask for a fresh copy.")
        return 1
    PatchWindow(patch).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
