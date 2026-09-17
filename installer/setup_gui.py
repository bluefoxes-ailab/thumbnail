"""
Thumbnail Maker Setup — the window the user sees.

Three pages, in the order the work happens: what is about to be installed and
where, what this particular machine turns out to be missing, and the install
itself with its log on screen. All engine work happens on a worker thread;
Tk is only ever touched from the main thread, via after().

The text is English throughout, deliberately — the app's own interface is
English, and an installer that speaks a different language than the thing it
installs reads as a different product.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import engine
from engine import APP_NAME, FATAL, INFO, MISSING, OK, VERSION

# The app's own palette, so setup and app look like one product.
BG = "#080b1e"
PANEL = "#0e1330"
BLUE = "#1b2fea"
BORDER = "#026dff"
YELLOW = "#fffa00"
FG = "#ffffff"
MUTED = "#9aa4d6"
GREEN = "#3ddc84"
RED = "#FF070C"

# How many lines of the install log stay on screen — see append_log.
LOG_MAX_LINES = 3000
LOG_TRIM_BLOCK = 500

STATE_STYLE = {
    OK:      ("✓", GREEN,  "Ready"),
    MISSING: ("↓", YELLOW, "Will be installed"),
    FATAL:   ("✗", RED,    "Blocked"),
    INFO:    ("•", MUTED,  "Note"),
}

WINDOW_W, WINDOW_H = 780, 620


class Setup(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} Setup")
        self.configure(bg=BG)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(WINDOW_W, WINDOW_H)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.assets = Path(__file__).resolve().parent / "assets"
        if getattr(sys, "frozen", False):
            self.assets = Path(getattr(sys, "_MEIPASS", ".")) / "assets"
        icon = self.assets / "fox_blue.ico"
        if icon.exists():
            try:
                self.iconbitmap(str(icon))
            except tk.TclError:
                pass

        self.scan = None
        self.installer = None
        self.install_dir = tk.StringVar(value=str(engine.DEFAULT_INSTALL_DIR))
        self.launch_after = tk.BooleanVar(value=True)
        self.busy = False
        self.finished = False

        # Worker threads never touch Tk. They push here; the main thread
        # drains it on a timer. Doing it the other way round is how Tk
        # installers end up deadlocking halfway through a pip install.
        self.events = queue.Queue()

        self._build_chrome()
        self.show_welcome()
        self.after(60, self._drain)
        self.center()

    # -- chrome ------------------------------------------------------------

    def center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() - WINDOW_W) // 2
        y = max(0, (self.winfo_screenheight() - WINDOW_H) // 2 - 30)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")

    def _build_chrome(self):
        header = tk.Frame(self, bg=BLUE)
        header.pack(fill="x")

        self.logo = None
        logo_path = self.assets / "fox_blue.png"
        if logo_path.exists():
            try:
                # Tk reads PNG natively from 8.6 on, so the icon needs no
                # imaging library — worth the subsample dance to avoid
                # shipping Pillow inside the installer for one picture.
                image = tk.PhotoImage(file=str(logo_path))
                factor = max(1, image.width() // 56)
                self.logo = image.subsample(factor, factor)
                tk.Label(header, image=self.logo, bg=BLUE).pack(side="left", padx=(20, 14), pady=14)
            except tk.TclError:
                self.logo = None

        titles = tk.Frame(header, bg=BLUE)
        titles.pack(side="left", pady=14)
        tk.Label(titles, text=APP_NAME, bg=BLUE, fg=YELLOW,
                 font=("Segoe UI", 19, "bold")).pack(anchor="w")
        tk.Label(titles, text=f"Setup — version {VERSION}", bg=BLUE, fg=FG,
                 font=("Segoe UI", 9)).pack(anchor="w")

        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True)

        footer = tk.Frame(self, bg=PANEL)
        footer.pack(fill="x", side="bottom")
        self.hint = tk.Label(footer, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
        self.hint.pack(side="left", padx=20, pady=14)
        self.next_btn = self._button(footer, "Continue", self.noop, primary=True)
        self.next_btn.pack(side="right", padx=(8, 20), pady=12)
        self.back_btn = self._button(footer, "Cancel", self.on_close)
        self.back_btn.pack(side="right", pady=12)
        # Made here, shown only by show_failed. See open_log for why a failed
        # install needs a button rather than a sentence naming a path.
        self.log_btn = self._button(footer, "Open log", self.open_log)

    def _button(self, parent, text, command, primary=False):
        return tk.Button(
            parent, text=text, command=command,
            bg=BLUE if primary else PANEL, fg=YELLOW if primary else FG,
            activebackground=YELLOW, activeforeground=BLUE,
            disabledforeground=MUTED, relief="flat", bd=0,
            font=("Segoe UI", 10, "bold" if primary else "normal"),
            padx=22, pady=8, cursor="hand2",
        )

    def clear(self):
        for child in self.body.winfo_children():
            child.destroy()

    def h1(self, parent, text):
        tk.Label(parent, text=text, bg=BG, fg=FG,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 6))

    def p(self, parent, text, colour=MUTED, pady=(0, 14)):
        tk.Label(parent, text=text, bg=BG, fg=colour, font=("Segoe UI", 10),
                 justify="left", wraplength=WINDOW_W - 90).pack(anchor="w", pady=pady)

    def noop(self):
        pass

    # -- page 1: welcome ---------------------------------------------------

    def show_welcome(self):
        self.clear()
        page = tk.Frame(self.body, bg=BG)
        page.pack(fill="both", expand=True, padx=34, pady=28)

        self.h1(page, f"Install {APP_NAME}")
        self.p(page,
               "This installer checks what this computer already has, then downloads and "
               "installs only what is missing — the Python runtime, the imaging and AI "
               "libraries, the face-restoration models and the title font.\n\n"
               "When it has finished it builds Thumbnail Maker.exe and puts a shortcut on "
               "your desktop.")

        box = tk.Frame(page, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        box.pack(fill="x", pady=(6, 0))
        tk.Label(box, text="Install location", bg=PANEL, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=16, pady=(14, 6))

        row = tk.Frame(box, bg=PANEL)
        row.pack(fill="x", padx=16, pady=(0, 6))
        entry = tk.Entry(row, textvariable=self.install_dir, bg=BG, fg=FG,
                         insertbackground=YELLOW, relief="flat",
                         font=("Segoe UI", 10), highlightbackground=BORDER,
                         highlightthickness=1)
        entry.pack(side="left", fill="x", expand=True, ipady=6)
        self._button(row, "Browse…", self.choose_dir).pack(side="left", padx=(10, 0))

        tk.Label(box,
                 text="Around 8 GB once everything is installed. A folder outside "
                      "OneDrive is recommended, so several gigabytes of libraries are "
                      "not uploaded to the cloud.",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9), justify="left",
                 wraplength=WINDOW_W - 120).pack(anchor="w", padx=16, pady=(0, 16))

        self.hint.configure(text="Nothing is changed on your computer until you choose Install.")
        self.back_btn.configure(text="Cancel", command=self.on_close, state="normal")
        self.next_btn.configure(text="Scan this PC", command=self.start_scan, state="normal")

    def choose_dir(self):
        chosen = filedialog.askdirectory(title=f"Where should {APP_NAME} be installed?",
                                         initialdir=self.install_dir.get())
        if chosen:
            path = Path(chosen)
            # Picking an existing "Thumbnail Maker" folder should mean "here",
            # not "one more level down" — the browse dialog is used both ways.
            if path.name.lower() != APP_NAME.lower():
                path = path / APP_NAME
            self.install_dir.set(str(path))

    # -- page 2: scan ------------------------------------------------------

    def start_scan(self):
        self.clear()
        page = tk.Frame(self.body, bg=BG)
        page.pack(fill="both", expand=True, padx=34, pady=28)
        self.h1(page, "Checking this computer")
        self.p(page, "Looking for the graphics card, the Python runtime, the libraries, "
                     "the models and the font…")
        bar = ttk.Progressbar(page, mode="indeterminate")
        bar.pack(fill="x", pady=10)
        bar.start(12)

        self.back_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled", text="Continue")
        self.hint.configure(text="This takes a few seconds.")

        target = self.install_dir.get()
        threading.Thread(target=self._scan_worker, args=(target,), daemon=True).start()

    def _scan_worker(self, target):
        try:
            result = engine.scan_system(Path(target))
            self.events.put(("scan", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def show_scan(self, scan):
        self.scan = scan
        scan.install_dir = Path(self.install_dir.get())
        self.clear()

        page = tk.Frame(self.body, bg=BG)
        page.pack(fill="both", expand=True, padx=34, pady=(24, 10))

        pending = [c for c in scan.checks if c.state == MISSING]
        blocked = scan.blocked

        self.h1(page, "System scan")
        if blocked:
            self.p(page, "This computer cannot run Thumbnail Maker yet:", RED, (0, 10))
        elif pending:
            self.p(page,
                   f"{len(pending)} of {len(scan.checks)} things still need installing "
                   f"— about {scan.download_mb} MB to download.", FG, (0, 10))
        else:
            self.p(page, "Everything is already in place. The installer will verify it "
                         "and rebuild the app.", GREEN, (0, 10))

        # Scrolled, because the list grows with every check and a fixed
        # layout would quietly hide the last few on a small screen.
        canvas = tk.Canvas(page, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        rows = tk.Frame(canvas, bg=BG)
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        for check in scan.checks:
            self._scan_row(rows, check)

        self.back_btn.configure(text="Back", command=self.show_welcome, state="normal")
        if blocked:
            self.next_btn.configure(text="Install", state="disabled")
            self.hint.configure(text="Resolve the blocked items above, then run Setup again.")
        else:
            self.next_btn.configure(text="Install", command=self.start_install, state="normal")
            self.hint.configure(
                text=f"Installing to {scan.install_dir}"
                     + ("  •  CUDA build" if scan.cuda else "  •  CPU build"))

    def _scan_row(self, parent, check):
        mark, colour, label = STATE_STYLE[check.state]
        row = tk.Frame(parent, bg=PANEL, highlightbackground="#1a2350", highlightthickness=1)
        row.pack(fill="x", pady=3, padx=(0, 8))

        tk.Label(row, text=mark, bg=PANEL, fg=colour, font=("Segoe UI", 13, "bold"),
                 width=3).pack(side="left", padx=(10, 0), pady=8)

        text = tk.Frame(row, bg=PANEL)
        text.pack(side="left", fill="x", expand=True, pady=8)
        tk.Label(text, text=check.name, bg=PANEL, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(text, text=check.detail, bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                 justify="left", wraplength=WINDOW_W - 260).pack(anchor="w")
        if check.action:
            tk.Label(text, text="→ " + check.action, bg=PANEL, fg=colour,
                     font=("Segoe UI", 9), justify="left",
                     wraplength=WINDOW_W - 260).pack(anchor="w", pady=(2, 0))

        tk.Label(row, text=label, bg=PANEL, fg=colour,
                 font=("Segoe UI", 9, "bold")).pack(side="right", padx=14)

    # -- page 3: install ---------------------------------------------------

    def start_install(self):
        self.clear()
        self.busy = True
        page = tk.Frame(self.body, bg=BG)
        page.pack(fill="both", expand=True, padx=34, pady=(24, 10))

        self.h1(page, "Installing")
        self.step_label = tk.Label(page, text="Starting…", bg=BG, fg=YELLOW,
                                   font=("Segoe UI", 10, "bold"))
        self.step_label.pack(anchor="w")

        self.bar = ttk.Progressbar(page, mode="determinate", maximum=1000)
        self.bar.pack(fill="x", pady=(8, 4))
        tk.Label(page,
                 text="The largest step is PyTorch — several gigabytes, and it can sit at "
                      "the same percentage for a while. That is normal.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 10))

        wrap = tk.Frame(page, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        wrap.pack(fill="both", expand=True)
        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=8)
        self.log_box = tk.Text(wrap, bg="#05070f", fg="#c8d2ff", relief="flat",
                               font=mono, wrap="none", height=10, padx=10, pady=8,
                               insertbackground=YELLOW)
        log_scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=log_scroll.set, state="disabled")
        log_scroll.pack(side="right", fill="y")
        self.log_box.pack(side="left", fill="both", expand=True)
        self.log_box.tag_configure("warn", foreground=YELLOW)
        self.log_box.tag_configure("head", foreground=GREEN)
        self.log_lines = 0

        self.back_btn.configure(text="Cancel", command=self.cancel_install, state="normal")
        self.next_btn.configure(text="Installing…", state="disabled")
        self.hint.configure(text="Please leave this window open.")

        # Two copies, and the second one is the point.
        #
        # The log used to live only inside the install directory, which is the
        # directory that goes missing. A user whose install failed deleted the
        # folder to start clean — an entirely reasonable thing to do, and what
        # most people do — and then went looking for the log to send, which he
        # had thrown away half an hour earlier without knowing it. The record
        # of what went wrong cannot be stored inside the thing that went
        # wrong.
        #
        # So the surviving copy sits beside the setup exe's own scratch
        # directory, outside the install entirely. It is the one Open log
        # reaches for when the other is gone, and it is the reason there is
        # still something to read after a user cleans up.
        self.log_path = Path(self.install_dir.get()) / "logs" / "install.log"
        self.log_path_kept = engine.kept_log_dir() / "install.log"
        self.log_files = []
        for path in (self.log_path, self.log_path_kept):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.log_files.append(open(path, "a", encoding="utf-8", buffering=1))
            except OSError:
                pass
        for handle in self.log_files:
            handle.write(f"\n=== {APP_NAME} Setup {VERSION} — {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"— installing to {self.install_dir.get()} ===\n")

        threading.Thread(target=self._install_worker, daemon=True).start()

    def _install_worker(self):
        try:
            # First, because it is about the run BEFORE this one and belongs at
            # the top of this log rather than buried in it. It costs a second
            # and says nothing at all when the last run ended properly.
            for line in engine.report_previous_run():
                self.events.put(("log", line))
            engine.session_start(self.install_dir.get())

            source = engine.payload_source()
            self.installer = engine.Installer(
                self.scan, source,
                log=lambda line: self.events.put(("log", line)),
                progress=lambda frac, label: self.events.put(("step", (frac, label))),
            )
            self.installer.run()
            engine.session_end()
            self.events.put(("done", None))
        except engine.InstallError as e:
            engine.session_end()
            self.events.put(("failed", str(e)))
        except Exception:
            engine.session_end()
            self.events.put(("failed", traceback.format_exc()))

    def cancel_install(self):
        if not self.busy:
            return
        if messagebox.askyesno(f"{APP_NAME} Setup",
                               "Stop the installation?\n\nPartly installed files will be "
                               "left behind; running Setup again will resume from where "
                               "it stopped."):
            if self.installer:
                self.installer.cancel()
            self.back_btn.configure(state="disabled", text="Stopping…")

    def append_log(self, line):
        tag = ()
        if line.startswith("==="):
            tag = ("head",)
        elif "WARNING" in line or "ERROR" in line:
            tag = ("warn",)
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n", tag)
        # Bounded, because an install is an hour long and pip is not quiet:
        # this widget kept every line of it, with its tags, in a process that
        # also has to survive to the end. The complete record is the file
        # below; what is on screen only has to be the part somebody could
        # still scroll back to. Trimmed in blocks because deleting from a Text
        # widget is the expensive half, not inserting.
        self.log_lines += 1
        if self.log_lines > LOG_MAX_LINES + LOG_TRIM_BLOCK:
            self.log_box.delete("1.0", f"{LOG_TRIM_BLOCK + 1}.0")
            self.log_lines -= LOG_TRIM_BLOCK
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        for handle in getattr(self, "log_files", ()):
            try:
                handle.write(line + "\n")
            except OSError:
                pass

    # -- page 4: finish ----------------------------------------------------

    def show_done(self):
        self.busy = False
        self.finished = True
        self.clear()
        page = tk.Frame(self.body, bg=BG)
        page.pack(fill="both", expand=True, padx=34, pady=34)

        tk.Label(page, text="✓", bg=BG, fg=GREEN, font=("Segoe UI", 40)).pack(anchor="w")
        self.h1(page, f"{APP_NAME} is installed")
        self.p(page,
               f"Thumbnail Maker.exe has been created in {self.scan.install_dir}, with a "
               "shortcut on your desktop and in the Start menu.\n\n"
               "The first launch takes a little longer than the rest — the AI models are "
               "loaded into memory before the browser window opens.", FG)

        gpu = ("Face restoration will use your GPU." if self.scan.cuda
               else "No NVIDIA GPU was found, so face restoration runs on the CPU and "
                    "will be slower.")
        self.p(page, gpu)

        # Said here because it changes what a user does with the next version:
        # they wait for it rather than going looking for an installer. The
        # several gigabytes just downloaded are the reason it matters.
        self.p(page,
               "New versions arrive as updates. Thumbnail Maker checks when it starts and "
               "asks before installing one — a few seconds, and nothing you have just "
               "downloaded is fetched again.\n\n"
               f"Your own channels go in {self.scan.install_dir / 'content'}, and updates "
               "never touch them.")

        tk.Checkbutton(page, text=f"Launch {APP_NAME} now", variable=self.launch_after,
                       bg=BG, fg=FG, selectcolor=BLUE, activebackground=BG,
                       activeforeground=YELLOW, font=("Segoe UI", 10),
                       highlightthickness=0).pack(anchor="w", pady=(6, 0))

        self.back_btn.configure(text="Open install folder", state="normal",
                                command=lambda: webbrowser.open(str(self.scan.install_dir)))
        self.next_btn.configure(text="Finish", state="normal", command=self.finish)
        self.hint.configure(text=f"Log saved to {getattr(self, 'log_path', '')}")

    def open_log(self):
        """
        Puts the log on the user's Desktop and shows it to them.

        Three earlier attempts at this failed for the same reason, which is
        worth stating once: the log lived somewhere the person who needs it
        cannot go. The path runs through AppData, which Explorer hides, so a
        sentence naming it asks the user to navigate to a folder they cannot
        see. One went looking and could not find it. Told to paste the path
        into Run instead, that failed too — his Desktop is redirected into
        OneDrive, so %USERPROFILE%\Desktop does not exist on his machine.

        So this does not send anybody anywhere. It copies the log to the
        Desktop — the real one, asked of Windows rather than assembled from
        environment variables, which is what desktop_dir() is for — under a
        .txt name that opens in Notepad on a double-click and drags into a
        chat window. Then it selects it, so it is on screen and not merely
        somewhere.

        Nothing here may raise. This runs from a Tk callback on a window that
        has already failed once, and an exception escaping it is the last
        thing that window should do.
        """
        source = None
        for candidate in (getattr(self, "log_path", None), getattr(self, "log_path_kept", None)):
            if candidate and Path(candidate).exists():
                source = Path(candidate)
                break

        if source is None:
            self.append_log("There is no log file to open yet.")
            return

        target = source
        try:
            target = engine.desktop_dir() / f"{APP_NAME} install log.txt"
            shutil.copy2(source, target)
            self.append_log(f"The log has been copied to your Desktop: {target}")
        except Exception as exc:
            # Copying is the convenience, not the point. A read-only Desktop,
            # a sync client holding the file, a redirected folder that is
            # offline — none of those should cost the user the log itself, so
            # fall back to showing where it already is.
            target = source
            self.append_log(f"Could not copy the log to the Desktop ({exc}). "
                            f"It is at {source}")

        try:
            # One string, not a list: explorer's /select takes the path as part
            # of the same token and does not survive being re-quoted.
            subprocess.Popen(f'explorer /select,"{target}"')
        except Exception:
            try:
                os.startfile(target.parent)
            except Exception:
                self.append_log(f"Could not open Explorer. The log is at {target}")

    def show_failed(self, message):
        self.busy = False
        self.append_log("\n=== INSTALLATION FAILED ===")
        for line in message.splitlines():
            self.append_log(line)
        self.step_label.configure(text="Installation failed", fg=RED)
        self.back_btn.configure(text="Close", command=self.on_close, state="normal")
        self.next_btn.configure(text="Try again", state="normal", command=self.start_scan)
        self.log_btn.pack(side="right", padx=(0, 8), pady=12)
        self.hint.configure(text="Open log has the whole story — send that file if you "
                                 "are asking for help.")
        messagebox.showerror(f"{APP_NAME} Setup",
                             message.strip().splitlines()[-1] if message.strip() else
                             "The installation failed. See the log for details.")

    def finish(self):
        if self.launch_after.get():
            exe = self.scan.install_dir / f"{APP_NAME}.exe"
            if exe.exists():
                subprocess.Popen([str(exe)], cwd=str(self.scan.install_dir))
        self.destroy()

    # -- event pump --------------------------------------------------------

    def _drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "step":
                    frac, label = payload
                    self.bar["value"] = frac * 1000
                    self.step_label.configure(text=label)
                    engine.session_step(label)
                elif kind == "scan":
                    self.show_scan(payload)
                elif kind == "done":
                    self.show_done()
                elif kind == "failed":
                    self.show_failed(payload)
                elif kind == "error":
                    messagebox.showerror(f"{APP_NAME} Setup", payload)
                    self.show_welcome()
        except queue.Empty:
            pass
        self.after(60, self._drain)

    def on_close(self):
        if self.busy:
            self.cancel_install()
            return
        self.destroy()


if __name__ == "__main__":
    Setup().mainloop()
