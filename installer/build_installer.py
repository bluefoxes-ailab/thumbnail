"""
Builds the thing you hand to a user: dist/Thumbnail Maker Setup <version>.exe.

Run it after changing anything under backend/ or frontend/ — the app is
frozen into the setup exe, so an unbuilt change ships nothing.

    python installer/build_installer.py

The result is one self-contained file of roughly 12 MB. That is the whole
point of the split: the ~5 GB of CUDA PyTorch and model weights the app
actually needs is not in here, because what a given machine needs is not
knowable until the installer is looking at it.

A build for one customer ships only that customer's channels:

    python installer/build_installer.py --group "QDS Karma"

Repeat --group to include more than one. Without it, every channel in the
repo ships — see keep_only_groups for why this is a build-time choice rather
than something the repo holds.

A private build environment is created outside the project on first run, so
building the installer never depends on what happens to be installed in the
system Python — and so that a copy of the project carried to another machine
does not carry a venv pointing at the first machine's interpreter.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Scratch space lives outside the project on purpose. This repo sits in a
# OneDrive-synced folder, and OneDrive keeps handles open on files it is
# syncing — which makes deleting a build directory fail with "access denied"
# on folders that are already empty, at random. PyInstaller rewrites its work
# directory constantly, so building in there is a coin flip. Only the finished
# installer is written back into the project.
BUILD = Path(tempfile.gettempdir()) / "thumbnail-maker-build"
DIST = HERE / "dist"

# The build venv lives out here for the same reason the work directory does,
# and it was moved here after hitting exactly that: a venv inside the project
# could not be replaced, because "access denied" on a directory OneDrive was
# holding made `python -m venv --clear` fail on a tree it had already emptied.
#
# A venv also records the absolute path of the interpreter that built it, so
# one that travels with the project — copied to another machine, restored from
# a backup, synced down — is a working directory full of files whose
# python.exe points at somebody else's disk. Outside the project it is local
# to the machine by construction, which is the property that matters.
VENV = BUILD / "venv"
VENV_PY = VENV / "Scripts" / "python.exe"

# The version goes in the filename, and it is not decoration. Every build is
# one file mailed to somebody, and two builds of the same version — a rebuilt
# 1.1.0 carrying a fix — are two different files with one name, sitting in two
# people's downloads folders. A user reporting a problem can then be looking
# at either of them, and the screenshot does not say which. The full name is
# assembled in main(), where the version has been read.
SETUP_NAME = "Thumbnail Maker Setup"
PYINSTALLER = "pyinstaller==6.11.1"

# Where the onefile bootloader unpacks itself before running.
#
# It defaults to %TEMP%, and a onefile exe is not a small unpack: this one is
# 1067 files, 917 of them the Tcl/Tk data the setup window needs, written out
# on every launch and deleted on exit. %TEMP% is shared with everything else
# on the machine, and on the machine this was first tested on it contained
# half-finished _MEI directories left by other PyInstaller programs — one of
# them empty, from months earlier.
#
# That machine also produced two failures that look like exactly this: a DLL
# that would not load, and, on the next build, a _tk_data directory that never
# appeared at all — 893 of 1067 files unpacked, the archive itself verified
# complete both times. Neither was reproducible afterwards.
#
# So the unpack goes somewhere that is ours: nothing else writes there, and a
# directory left behind by a crash is ours to find. It is not a proven fix for
# something never reproduced — it is removing a shared resource from the one
# step that has to work before any of our code runs.
#
# Windows expands the environment variable; POSIX would not, which is why the
# PyInstaller documentation warns about it. This installer is Windows-only.
RUNTIME_TMPDIR = r"%LOCALAPPDATA%\Thumbnail Maker Setup"


def run(args, **kwargs):
    print("> " + " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def venv_is_usable():
    """
    Whether .build-venv can actually run anything.

    Existing is not the same as working. A venv records the absolute path of
    the interpreter that created it, so one that arrived with the project —
    copied from another machine, restored from a backup, synced out of
    OneDrive — is a complete directory whose python.exe reports

        No Python at 'C:\\Users\\someone-else\\...\\python.exe'

    and fails every command with an exit code nobody recognises. Asking it to
    do arithmetic is a cheaper question than reading pyvenv.cfg and comparing
    paths, and it is the question that actually matters.
    """
    if not VENV_PY.exists():
        return False
    probe = subprocess.run([str(VENV_PY), "-c", "print(1)"], capture_output=True, text=True)
    return probe.returncode == 0 and probe.stdout.strip() == "1"


def ensure_base_python(engine):
    """
    The interpreter this installer is frozen with: the same one it installs.

    It used to be `sys.executable` — whatever Python happened to run this
    script. On the machine this is built on that is a Python 3.10, so the
    setup exe went out carrying a 3.10 `_tkinter.pyd` and `python310.dll` in
    its bundle, while the app it installs runs on 3.11.

    Two Pythons in one process tree is a loaded gun, and it went off. The
    setup exe compiles Thumbnail Maker.exe on the user's machine, with
    PyInstaller, out of the 3.11 runtime it just built — and on one user's
    machine that build picked up the 3.10 tkinter extension out of the
    installer's own unpacked bundle instead of the 3.11 one. The launcher was
    produced without complaint and died on the first line that needed Tk:

        ImportError: Module use of python310.dll conflicts with this version
        of Python

    Rather than hunt for which of the several paths between those two
    processes carried it, the two Pythons are made one. This downloads the
    same pinned CPython build that the installer installs — same URL, same
    SHA-256, same bytes — so there is no second version anywhere in the
    picture for anything to pick the wrong one of.

    Cached in the build directory, so this costs 40 MB once.
    """
    base = BUILD / "python"
    exe = base / "python.exe"
    if exe.exists():
        probe = subprocess.run([str(exe), "-c", "import sys;print(sys.version.split()[0])"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            print(f"Base interpreter: Python {probe.stdout.strip()} at {base}")
            return exe
        shutil.rmtree(base, ignore_errors=True)

    archive = BUILD / "python.tar.gz"
    if not archive.exists():
        print(f"Fetching the base interpreter from {engine.PORTABLE_PY_URL}")
        BUILD.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(engine.PORTABLE_PY_URL, archive)

    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != engine.PORTABLE_PY_SHA256:
        archive.unlink(missing_ok=True)
        raise SystemExit(f"FATAL: the base interpreter download does not match its pin.\n"
                         f"  expected {engine.PORTABLE_PY_SHA256}\n"
                         f"  got      {digest.hexdigest()}")

    staging = BUILD / "python-unpacked"
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(staging, filter="data")
        except TypeError:
            tar.extractall(staging)
    shutil.move(str(staging / "python"), str(base))
    print(f"Base interpreter: Python {engine.PORTABLE_PY} unpacked to {base}")
    return exe


def ensure_build_env(engine):
    """A venv with PyInstaller in it, created once and reused after that."""
    base_python = ensure_base_python(engine)

    # A venv built on a different base than the one we now want is exactly the
    # mismatch this is here to prevent, so it is checked rather than assumed.
    if VENV_PY.exists():
        cfg = (VENV / "pyvenv.cfg").read_text(encoding="utf-8") if (VENV / "pyvenv.cfg").exists() else ""
        if str(base_python.parent) not in cfg or not venv_is_usable():
            print(f"The build environment in {VENV} is not built on {base_python.parent} "
                  "— rebuilding it")
            shutil.rmtree(VENV, ignore_errors=True)

    if not VENV_PY.exists():
        print(f"Creating the build environment in {VENV}")
        run([base_python, "-m", "venv", VENV])
        run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"])

    have = subprocess.run(
        [str(VENV_PY), "-c", "import PyInstaller; print(PyInstaller.__version__)"],
        capture_output=True, text=True,
    )
    if have.returncode != 0:
        run([VENV_PY, "-m", "pip", "install", PYINSTALLER, "--disable-pip-version-check"])
    else:
        print(f"PyInstaller {have.stdout.strip()} already present")


def keep_only_groups(payload, groups):
    """
    Removes every channel pack outside `groups` from the staged payload.

    Which channels ship is a per-customer question, not a property of the
    repo. All the packs are developed here, and a build handed to one client
    has no business carrying another client's brands — but neither should
    that mean moving folders out of frontend/content/channels before a build
    and remembering to put them back, which is exactly the kind of manual
    step that eventually ships the wrong thing. So it happens to the staged
    copy: the repo is untouched, and the next build without --group is
    everything again.

    Matching is on the pack's own `group`, case-insensitively. That key
    already exists and already means this — it is how the packs say who they
    belong to, and how the app groups its channel picker.
    """
    root = payload / "app" / "frontend" / "content" / "channels"
    wanted = {g.strip().casefold() for g in groups}
    kept, dropped, present = [], [], set()

    for pack in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = pack / "channel.json"
        try:
            group = json.loads(manifest.read_text(encoding="utf-8")).get("group") or ""
        except (OSError, ValueError) as exc:
            raise SystemExit(f"FATAL: cannot read {manifest}: {exc}")
        present.add(group)
        if group.casefold() in wanted:
            kept.append(pack.name)
        else:
            shutil.rmtree(pack, ignore_errors=True)
            dropped.append(f"{pack.name} ({group or 'no group'})")

    # A misspelled group is otherwise a silent build of nothing at all: no
    # pack matches, every pack is deleted, and the installer ships with an
    # empty channel picker that nobody looks at until a user opens it.
    unknown = wanted - {g.casefold() for g in present}
    if unknown:
        known = ", ".join(sorted(g for g in present if g)) or "(none)"
        raise SystemExit(f"FATAL: no channel belongs to the group(s) "
                         f"{', '.join(sorted(unknown))}.\nGroups in this repo: {known}")

    print(f"channel groups: {', '.join(sorted(groups))}")
    for name in dropped:
        print(f"  excluded: {name}")
    return kept


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Builds dist/Thumbnail Maker Setup.exe.")
    parser.add_argument(
        "--group", action="append", metavar="NAME", default=[],
        help="Ship only the channels in this group, e.g. --group \"QDS Karma\". "
             "Repeatable. Omit to ship every channel in the repo.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE))
    import check_language
    import engine
    import make_icon
    import updater

    # First, and fatal. The app is English throughout, and the cheapest moment
    # to enforce that is before two minutes of PyInstaller rather than after a
    # user has read a sentence in the wrong language.
    print("=== language ===")
    check_language.assert_english()

    print("\n=== icon ===")
    make_icon.main()

    print("\n=== payload ===")
    payload = BUILD / "payload"
    # Not cleared with a plain rmtree first: a directory left over from the
    # last build is routinely still held for a moment by OneDrive or an
    # antivirus scan, and deleting it outright fails with "access denied" on a
    # folder that is already empty. stage_payload replaces each tree
    # individually and tolerates exactly that.
    payload.mkdir(parents=True, exist_ok=True)
    engine.stage_payload(ROOT, payload)

    # Before every check below it, so that the asset check asks its question
    # about the channels this build actually ships and not about the ones it
    # just removed.
    if args.group:
        keep_only_groups(payload, args.group)

    size = sum(f.stat().st_size for f in payload.rglob("*") if f.is_file())
    print(f"payload: {size / 1024:.0f} KB in {payload}")

    # Not a nicety: the whole reason the installer exists in this shape is
    # that a missing font file produces a silently wrong text overlay rather
    # than an error. Better to fail the build than ship that.
    font = payload / "app" / "frontend" / "fonts" / engine.FONT_FILE
    if not font.exists() or font.stat().st_size < 10_000:
        raise SystemExit(f"FATAL: the title font is missing or truncated at {font}")
    print(f"title font: {font.name} ({font.stat().st_size} bytes)")

    # The same question asked of every channel, rather than only of the face
    # the app cannot start without: each pack names a type face and possibly a
    # texture, and a payload missing either produces an install that works
    # everywhere except in one channel, silently. Read from the packs, so this
    # covers a channel added after this line was written.
    frontend = payload / "app" / "frontend"
    broken = updater.missing_content_assets(frontend)
    if broken:
        raise SystemExit("FATAL: the channel packs name files the payload does not have:\n  - "
                         + "\n  - ".join(broken))
    assets = updater.content_assets(frontend)
    print(f"channel assets: {len(assets)} files, all present")
    for label, path in assets:
        print(f"  {path.stat().st_size / 1024:8.0f} KB  {label}")

    packs = sorted(d.name for d in (frontend / "content" / "channels").iterdir() if d.is_dir())
    print(f"channels: {len(packs)} — {', '.join(packs)}")

    release = payload / "app" / "release.json"
    if not release.exists():
        raise SystemExit(f"FATAL: release.json is missing from the payload at {release}")
    info = json.loads(release.read_text(encoding="utf-8"))
    print(f"version: {info.get('version')}")
    if info.get("feed"):
        print(f"updates: {info['feed']}")
    else:
        # Not fatal — a build for a machine that will never be patched is a
        # perfectly reasonable thing to make. Said out loud because the
        # alternative is discovering it months later, when a patch is ready
        # and nothing is looking for it.
        print("updates: DISABLED — release.json has no feed, so copies installed from this "
              "exe will never look for a patch. Set it before shipping to anyone.")

    print("\n=== build environment ===")
    ensure_build_env(engine)

    print("\n=== freezing ===")
    # ignore_errors for the same OneDrive reason; --noconfirm overwrites
    # whatever survives.
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir(parents=True, exist_ok=True)

    sep = os.pathsep  # ';' on Windows, and PyInstaller wants the native one
    setup_name = f"{SETUP_NAME} {info.get('version')}"
    args = [
        VENV_PY, "-m", "PyInstaller", str(HERE / "setup_gui.py"),
        "--name", setup_name,
        "--onefile",
        "--noconsole",
        "--clean", "--noconfirm",
        "--icon", str(HERE / "assets" / "fox_blue.ico"),
        "--paths", str(HERE),
        "--runtime-tmpdir", RUNTIME_TMPDIR,
        # The payload rides inside the exe and is unpacked to a temp
        # directory at run time; Installer.copy_app moves it somewhere
        # permanent before that directory evaporates.
        "--add-data", f"{payload / 'app'}{sep}app",
        "--add-data", f"{payload / 'assets'}{sep}assets",
        "--add-data", f"{payload / 'launcher.py'}{sep}.",
        # The update machinery, copied to <install>/updater during the
        # install. It is what lets the next release be a ~1 MB patch instead
        # of this exe again — see installer/build_patch.py.
        "--add-data", f"{payload / 'updater'}{sep}updater",
        # engine is imported by name from a --paths directory, which the
        # analysis follows, but naming it costs nothing and makes the
        # dependency explicit if setup_gui's imports are ever reshuffled.
        "--hidden-import", "engine",
        # engine imports it for the one list they have to agree on (what an
        # update preserves), and it is a data file as well as a module.
        "--hidden-import", "updater",
        "--distpath", str(DIST),
        "--workpath", str(BUILD / "work"),
        "--specpath", str(BUILD),
    ]
    run(args)

    exe = DIST / f"{setup_name}.exe"
    if not exe.exists():
        raise SystemExit("FATAL: PyInstaller did not produce the setup exe")

    print(f"\nBuilt: {exe}")
    print(f"Size:  {exe.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nHand that single file to a user. Everything else is downloaded on demand.")


if __name__ == "__main__":
    main()
