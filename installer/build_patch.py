"""
Builds the thing you hand to a user who already has the app: an update.

    python installer/build_patch.py

That is the whole release process for a change to `backend/`, `frontend/`, or
a channel under `frontend/content/`. It produces one file:

    installer/patches/Thumbnail Maker Patch 1.1.0.exe

Send it however you send a file. The person who gets it double-clicks it. It
finds their installation, tells them what is in the release, applies it, and
says whether it worked. There is no command to type and nothing to explain.

The setup exe is still how a NEW machine gets the app, and `build_installer.py`
is still what builds it. This is for the machines that already have it: they
are carrying ~5 GB of PyTorch and model weights that a new feature does not
change, and asking them to uninstall and reinstall to get one is asking them
to download all of it again.

## What goes in an update, and what cannot

Everything under `app/` — the backend, the frontend, the content packs, the
requirements file — plus the updater itself. That covers new features, fixed
bugs, and new channels.

Three things ride alongside it as instructions rather than as payload, because
they are large or live outside `app/`:

    pip     new libraries, installed into the existing runtime
    models  new weights, downloaded once from their own URL
    fonts   a new face registered with Windows as well as served

They go in `patch.json` (below). What an update cannot do is change the Python
version, move the install, or rebuild the environment from scratch: set
`requires_setup` in `patch.json` and the patch will say so plainly and send the
user to the installer instead of half-applying something.

## patch.json

Optional, in the repo root, describing the release being built. Empty its
`notes`/`pip`/`models` between releases — it describes THIS update, not the
app.

    {
      "notes": ["Two new channels", "Faster reframing"],
      "min_app_version": "1.0.0",
      "pip": ["some-new-lib==1.2.3"],
      "models": [{"path": "backend/weights/x.pth", "url": "https://…",
                  "sha256": "…", "mb": 40}],
      "fonts": [{"path": "frontend/fonts/New.otf", "name": "New Face"}],
      "rebuild_launcher": false,
      "requires_setup": null
    }

`notes` is the one that matters most: it is what the user reads in the window
before they press the button.

## The other two outputs

Alongside the exe, the same payload is written as a bare `.tmpatch`. It is the
exe's own contents, and it exists for two situations the exe does not cover:

    --apply-file        applying an update from a script, or on a machine
                        where running an exe from e-mail is not allowed
    updates.json        a feed, if this ever grows past hand-delivery

The feed is only written when `release.json` has a `feed` URL set. With five
machines and a file you send yourself, it stays off and nothing looks for it.

## Signing

    python installer/build_patch.py --keygen

Only worth it once updates are published somewhere rather than handed over.
A file you sent someone yourself is already as authenticated as it is going to
get; a file sitting on a server is not. The public half goes in `release.json`
and ships inside every installer; the private half stays outside the repo, at
`%LOCALAPPDATA%\\ThumbnailMaker\\signing.key`, and losing it means no existing
install will ever accept a signed patch again.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE))
import ed25519  # noqa: E402  (path has to be set first — this file is not a package)

# Not under dist/ — build_installer.py clears that directory on every run, and
# a build of the setup exe silently deleting every patch you have published
# would be a very quiet way to break every install pointing at them.
PATCH_DIR = HERE / "patches"
FEED_NAME = "updates.json"

# Where a signing key is looked for when --key is not given. Outside the repo
# on purpose: a private key in a project folder is a private key in a backup,
# in a sync client, and eventually in a zip somebody shares.
DEFAULT_KEY = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ThumbnailMaker" / "signing.key"

APP_NAME = "Thumbnail Maker"


def load_release():
    with open(ROOT / "release.json", encoding="utf-8") as f:
        release = json.load(f)
    if not release.get("version"):
        raise SystemExit("FATAL: release.json has no version.")
    return release


def load_patch_spec():
    """The optional description of this release. Absent means "just the files"."""
    path = ROOT / "patch.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return {k: v for k, v in spec.items() if not k.startswith("_")}


def stage(version, release):
    """
    Assembles what the patch contains, in the layout the updater unpacks.

        app/        exactly what an install has at <install>/app
        updater/    so a patch can fix the thing that applies patches

    The app tree is staged by the installer's own stage_payload, so a patch
    and a fresh install are built from one definition of what the app is —
    two lists of files to copy would disagree the first time either changed.
    """
    import engine

    staged = Path(tempfile.mkdtemp(prefix="tnmaker-patch-build-"))
    engine.stage_payload(ROOT, staged)

    # stage_payload also lays down assets/ and launcher.py for the installer's
    # benefit. A patch wants the launcher — it is what a rebuild_launcher
    # patch replaces — but under updater/, where the install keeps it.
    updater = staged / "updater"
    updater.mkdir(exist_ok=True)
    for name in ("updater.py", "ed25519.py"):
        shutil.copy2(HERE / name, updater / name)
    shutil.move(str(staged / "launcher.py"), str(updater / "launcher.py"))
    shutil.rmtree(staged / "assets", ignore_errors=True)

    # The build's own identity, read back by the app and by the next update.
    with open(staged / "app" / "release.json", "w", encoding="utf-8") as f:
        json.dump({**release, "version": version}, f, indent=4)
        f.write("\n")

    return staged


def manifest_for(version, release, spec, staged):
    packs = staged / "app" / "frontend" / "content" / "channels"
    channels = sorted(d.name for d in packs.iterdir() if d.is_dir()) if packs.is_dir() else []
    return {
        "schema": 1,
        "app": APP_NAME,
        "version": version,
        "built": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "min_app_version": spec.get("min_app_version") or "0.0.0",
        "notes": spec.get("notes") or [],
        "pip": spec.get("pip") or [],
        "models": spec.get("models") or [],
        "fonts": spec.get("fonts") or [],
        "rebuild_launcher": bool(spec.get("rebuild_launcher")),
        "requires_setup": spec.get("requires_setup") or None,
        # Not read by the updater — it is here so that opening a patch and
        # reading one file tells you what channels that version shipped with,
        # which is the question anyone debugging "where did my channel go"
        # asks first.
        "channels": channels,
    }


def write_patch(version, staged, manifest):
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    out = PATCH_DIR / f"thumbnail-maker-{version}.tmpatch"
    if out.exists():
        out.unlink()

    # Deflate, sorted, and with a fixed timestamp on every entry: two builds
    # of unchanged sources then produce byte-identical patches, so "did
    # anything actually change" is answerable by comparing checksums instead
    # of by trusting the version number.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, json.dumps(manifest, indent=2))
        for path in sorted(staged.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(staged).as_posix()
            entry = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            z.writestr(entry, path.read_bytes())
    return out


def build_exe(version, patch):
    """
    Freezes patch_gui.py with the payload inside it, into one double-clickable file.

    Built with the same private build environment as the setup exe, so making
    an update never depends on what happens to be installed in the system
    Python. It costs a minute or two and about 10 MB of PyInstaller runtime on
    top of the payload — which is the price of the user not having to type
    anything, and is still a rounding error against reinstalling.
    """
    # engine is what names the pinned CPython the build environment has to be
    # built on. build_installer takes it as an argument rather than importing
    # it itself, so the two builds cannot end up on two interpreters — the
    # exact failure ensure_base_python exists to prevent. Passing it was
    # missed when that argument was added, and this is the only caller
    # outside build_installer.py.
    import build_installer
    import engine

    build_installer.ensure_build_env(engine)

    work = Path(tempfile.mkdtemp(prefix="tnmaker-patch-exe-"))
    try:
        # The GUI looks for its payload under a fixed name, so the version
        # number in the filename cannot become something it has to parse.
        staged = work / "payload.tmpatch"
        shutil.copy2(patch, staged)

        name = f"{APP_NAME} Patch {version}"
        icon = HERE / "assets" / "fox_blue.ico"
        sep = os.pathsep

        args = [
            build_installer.VENV_PY, "-m", "PyInstaller", str(HERE / "patch_gui.py"),
            "--name", name,
            "--onefile", "--noconsole", "--clean", "--noconfirm",
            "--paths", str(HERE),
            # Same window, same bootloader, same reason — see RUNTIME_TMPDIR
            # in build_installer.py. A patch exe is the other file a user
            # double-clicks, so it unpacks in the same place rather than into
            # whatever %TEMP% happens to be doing.
            "--runtime-tmpdir", build_installer.RUNTIME_TMPDIR,
            "--add-data", f"{staged}{sep}.",
            # updater does the work and ed25519 is imported by name from it;
            # neither is reachable by static analysis from patch_gui alone.
            "--hidden-import", "updater",
            "--hidden-import", "ed25519",
            "--distpath", str(PATCH_DIR),
            "--workpath", str(work / "work"),
            "--specpath", str(work),
        ]
        if icon.exists():
            args += ["--icon", str(icon), "--add-data", f"{icon}{sep}."]

        print(f"\n=== freezing {name}.exe ===")
        subprocess.run([str(a) for a in args], check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    exe = PATCH_DIR / f"{APP_NAME} Patch {version}.exe"
    if not exe.exists():
        raise SystemExit("FATAL: PyInstaller did not produce the patch exe")
    return exe


def sign_patch(patch, key_path):
    """Signs the patch bytes, writes the sidecar, and returns the base64 signature."""
    priv = base64.b64decode(key_path.read_text(encoding="utf-8").strip(), validate=True)
    if len(priv) != 32:
        raise SystemExit(f"FATAL: {key_path} does not hold a 32-byte Ed25519 private key.")
    signature = base64.b64encode(ed25519.sign(patch.read_bytes(), priv)).decode()
    # Written beside the patch as well as into the feed, so a patch copied
    # onto a stick and applied with --apply-file can still be checked.
    Path(str(patch) + ".sig").write_text(signature + "\n", encoding="utf-8")
    return signature


def rebuild_feed(release):
    """
    Rewrites updates.json from every patch sitting in the folder.

    Generated rather than appended to, so the file always describes what is
    actually there: a patch deleted from the folder disappears from the list,
    and one copied in by hand appears in it. The alternative — a feed edited
    alongside the folder — is a feed that eventually offers a download that
    404s, and the user sees the update fail rather than never being offered.
    """
    patches = []
    for path in sorted(PATCH_DIR.glob("*.tmpatch")):
        try:
            with zipfile.ZipFile(path) as z:
                manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        except (zipfile.BadZipFile, KeyError, ValueError) as e:
            print(f"  skipping {path.name}: {e}")
            continue
        sig = Path(str(path) + ".sig")
        entry = {
            "version": manifest["version"],
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "min_app_version": manifest.get("min_app_version") or "0.0.0",
            "notes": manifest.get("notes") or [],
            "built": manifest.get("built"),
        }
        if sig.exists():
            entry["signature"] = sig.read_text(encoding="utf-8").strip()
        patches.append(entry)

    patches.sort(key=lambda e: [int(n) for n in e["version"].replace("-", ".").split(".") if n.isdigit()])
    feed = {
        "schema": 1,
        "app": APP_NAME,
        "generated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "latest": patches[-1]["version"] if patches else release.get("version"),
        "patches": patches,
    }
    out = PATCH_DIR / FEED_NAME
    with open(out, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)
        f.write("\n")
    return out, feed


def keygen(where):
    priv, pub = ed25519.generate()
    where.parent.mkdir(parents=True, exist_ok=True)
    if where.exists():
        raise SystemExit(
            f"FATAL: {where} already exists. Every install carrying the old public key would "
            "reject everything signed with a new one — move it aside deliberately if that is "
            "really what you want."
        )
    where.write_text(base64.b64encode(priv).decode() + "\n", encoding="utf-8")
    try:
        os.chmod(where, 0o600)
    except OSError:
        pass

    print(f"Private key written to {where}")
    print("  Keep it. Back it up somewhere that is not this project folder.")
    print("  Losing it means no existing install will ever accept another patch.\n")
    print("Public key — put this in release.json as \"public_key\":\n")
    print(f'    "public_key": "{base64.b64encode(pub).decode()}"\n')


def main():
    p = argparse.ArgumentParser(description="Build an update patch for an installed copy.")
    p.add_argument("--version", help="override the version in release.json")
    p.add_argument("--key", type=Path, default=None,
                   help=f"Ed25519 private key to sign with (default: {DEFAULT_KEY})")
    p.add_argument("--unsigned", action="store_true",
                   help="build without signing, even though a key is available")
    p.add_argument("--keygen", action="store_true", help="create a signing key pair and exit")
    p.add_argument("--group", action="append", metavar="NAME",
                   help="ship only the channels in this group, matched on the pack's "
                        "own `group` key, case-insensitively; repeat for more than "
                        "one. It must name what the TARGET MACHINE already has.")
    p.add_argument("--no-exe", action="store_true",
                   help="build only the .tmpatch, skipping the double-clickable exe")
    p.add_argument("--feed-only", action="store_true",
                   help="rebuild updates.json from the patches already in installer/patches")
    args = p.parse_args()

    if args.keygen:
        keygen(args.key or DEFAULT_KEY)
        return

    release = load_release()

    if args.feed_only:
        out, feed = rebuild_feed(release)
        print(f"{out}  ({len(feed['patches'])} patches, latest {feed['latest']})")
        return

    version = args.version or release["version"]
    spec = load_patch_spec()

    # Before anything is staged. Every update carries the whole frontend, so a
    # Portuguese string anywhere in it would reach every machine at once.
    import check_language
    print("=== language ===")
    check_language.assert_english()

    print(f"\n=== staging {APP_NAME} {version} ===")
    staged = stage(version, release)
    try:
        # Filtered to one customer's channels, on the staged copy and by the
        # installer's own function, so a patch and a setup exe cannot disagree
        # about what "the QDS Karma build" contains.
        #
        # This matters more for a patch than it does for an installer, because
        # an update replaces <install>/app WHOLE. An unfiltered patch does not
        # merely fail to respect a filtered install — it ADDS every other
        # customer's brands to it. And a patch filtered to the wrong group
        # REMOVES the packs that were there. So the flag names what the machine
        # being patched already has, not what this release happens to be about.
        if args.group:
            import build_installer
            build_installer.keep_only_groups(staged, args.group)

        size = sum(f.stat().st_size for f in staged.rglob("*") if f.is_file())
        print(f"payload: {size / 1024:.0f} KB")

        # The same check build_installer.py makes, for the same reason: a
        # missing title font does not produce an error, it produces titles laid
        # out against a font nobody can see. Better to fail the build.
        font = staged / "app" / "frontend" / "fonts" / "TradeGothicNextLTProHeavyCompressed.otf"
        if not font.exists() or font.stat().st_size < 10_000:
            raise SystemExit(f"FATAL: the title font is missing or truncated at {font}")

        # And the same question asked of every channel the patch carries. A
        # patch is how a new channel reaches a machine, so this is the build
        # that would ship one whose font was never copied — an install that
        # works everywhere except in the channel the update was for.
        import updater
        broken = updater.missing_content_assets(staged / "app" / "frontend")
        if broken:
            raise SystemExit("FATAL: the channel packs name files the patch does not carry:"
                             + "".join(f"\n  - {b}" for b in broken))

        manifest = manifest_for(version, release, spec, staged)

        # patch.json is scanned like everything else, but the notes deserve
        # naming separately: they are the sentences the user actually reads in
        # the update window, and they are written fresh for every release —
        # which is exactly when a language slips.
        for note in manifest["notes"]:
            bad = check_language.offending_chars(note) or check_language.offending_words(note)
            if bad:
                raise SystemExit(
                    f"FATAL: the release note {note!r} is not English. It is what the user "
                    "reads in the update window.")

        print(f"channels: {', '.join(manifest['channels']) or 'none'}")
        for note in manifest["notes"]:
            print(f"  note: {note}")
        for pin in manifest["pip"]:
            print(f"  pip:  {pin}")

        patch = write_patch(version, staged, manifest)
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    print(f"\nPayload: {patch}  ({patch.stat().st_size / 1024:.0f} KB)")

    key = args.key or DEFAULT_KEY
    if args.unsigned:
        pass
    elif key.exists():
        sign_patch(patch, key)
        print(f"Signed with {key}")
    elif release.get("public_key"):
        Path(str(patch) + ".sig").unlink(missing_ok=True)
        print(f"WARNING: release.json carries a public key but there is no signing key at "
              f"{key}, so every install built from it will REFUSE this patch. Sign it or "
              "clear the key.")
    else:
        Path(str(patch) + ".sig").unlink(missing_ok=True)

    exe = None
    if not args.no_exe:
        exe = build_exe(version, patch)

    # Only when updates are actually published somewhere. Hand-delivered
    # updates have no list to be on, and writing one nothing reads would only
    # ever be something to wonder about later.
    if release.get("feed"):
        out, feed = rebuild_feed(release)
        print(f"\nFeed: {out}  (latest {feed['latest']}, {len(feed['patches'])} patches)")
        print(f"Installs look for updates at {release['feed']} — publish {PATCH_DIR} there.")

    print()
    if exe:
        print(f"=== Send this file ===\n  {exe}")
        print(f"  {exe.stat().st_size / 1024 / 1024:.1f} MB")
        print("\nThe person who receives it double-clicks it. It finds their installation,")
        print("shows them what is in the update, and applies it. Nothing to type, and")
        print("nothing already downloaded is fetched again.")
        if manifest["notes"]:
            print("\nThey will see:")
            for note in manifest["notes"]:
                print(f"  •  {note}")
        else:
            print("\nNOTE: this update has no release notes, so the window will say so.")
            print('      Put them in patch.json under "notes".')
    else:
        print(f"Built the payload only: {patch}")
        print("Apply it with:  updater.py --apply-file <path>")


if __name__ == "__main__":
    main()
