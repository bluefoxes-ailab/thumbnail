# Thumbnail Maker — installer

Builds `dist/Thumbnail Maker Setup.exe`: one self-contained file of ~9.5 MB
that installs the app on a machine that has nothing on it.

```bash
python installer/build_installer.py
```

**This is for a machine that does not have the app yet.** Everything after the
first install ships as an update exe the user double-clicks — see
[RELEASING.md](../RELEASING.md).
The app is a megabyte; the install is five gigabytes of PyTorch and model
weights that a new feature does not change, so a reinstall to get one is five
gigabytes of nothing.

Rebuild the setup exe when the feed URL or the signing key in `release.json`
changes, or when a new machine needs a starting point that is not several
patches behind. Both paths stage the app from the same `stage_payload`, so
they can never carry different files.

## Why the installer is small and the install is not

The app is about a megabyte of Python and static files. It stands on ~5 GB of
CUDA-enabled PyTorch plus the weights for face restoration, inpainting and
background removal.

Shipping that would mean a 5 GB download for everyone, and it would still be
wrong for half of them: a machine with no NVIDIA GPU needs entirely different
torch wheels. So the setup exe carries only the app, scans the machine it is
run on, and downloads exactly what that machine turns out to need.

## What the scan looks for

| Check | What it decides |
|---|---|
| 64-bit Windows, ≥12 GB free, internet | whether the install can proceed at all |
| NVIDIA GPU (`nvidia-smi`) | CUDA 12.4 torch wheels, or the CPU ones |
| Python 3.10–3.12, 64-bit | reuse the machine's own, or install a private 3.11 |
| Visual C++ runtime | torch and opencv fail to import without it |
| Packages already in the app's venv | only the missing ones are downloaded |
| JS runtime (deno) | see below — without it YouTube 403s |
| Model weights | ~800 MB, or nothing if already there |
| Title font | see below |
| Channel assets | every face and texture the packs name — see below |

An adapter that is an NVIDIA card with no working driver is reported as such
rather than quietly downgraded to the CPU wheels — it is a ten-minute fix, and
otherwise shows up months later as "the app is slow".

## What the install does

1. Copies `backend/` and `frontend/` into `<install>/app`
2. Installs a private Python if the machine has none in range (downloaded from
   python.org, **Authenticode signature verified** before it is run)
3. Creates a venv at `<install>/runtime` — nothing outside it is touched
4. Installs the pins in the one order that works (below)
5. Patches `basicsr` for the torchvision API it still imports
6. Installs deno, so YouTube downloads don't 403
7. Installs the channels' type faces for the current user
8. Downloads the model weights by loading the app's own models once
9. Builds `Thumbnail Maker.exe` with PyInstaller and the fox icon
10. Creates desktop and Start-menu shortcuts, the uninstaller, and an Apps &
    features entry
11. Verifies all of it, and fails loudly if anything is missing

Re-running is safe: every step is idempotent, and pip skips what is already
satisfied.

## Step 2: the private Python is a folder, not an installation

It used to be an installation, and that is the single thing in this installer
that has cost the most.

The python.org installer only honours `TargetDir` on a **fresh** install.
Where Windows already holds a record of that exact version, the bundle
switches to maintenance mode: it installs nothing, ignores the directory it
was given, and **exits 0**.

And the record outlives the files. The private Python was a real per-user MSI
install, so deleting `<install>` — an uninstall, a half-finished install
tidied up by hand — left Windows certain that Python 3.11.9 was present while
nothing of it remained. Every later run of Setup then downloaded the bundle,
watched it do nothing, and failed:

```
The Python installer finished with exit code 0
Python did not install to C:\Users\...\Thumbnail Maker\python.
```

— on a machine with no Python at all, which could then never install this app
again. One user's machine reached that state and stayed there through
install, repair, and a clean install after removing the registration.

**So the first choice is no longer an installer.** `install_portable_python`
downloads a CPython built by the python-build-standalone project — the builds
`uv python install` uses — checks it against a pinned SHA-256, and extracts it
into `<install>/python`. Download, hash, extract, move. There is no Windows
Installer anywhere in that sentence, so there is no record to outlive
anything, nothing that can already be in the way, and nothing left behind
when the uninstaller deletes the folder.

It is complete in the two ways this app needs and the embeddable zip is not:
tkinter and Tcl/Tk are in it (the launcher is a Tk window frozen out of this
interpreter), and so are `venv`, `ensurepip` and `pip`.

Not signed by the PSF, so it is not checked the way the python.org bundle is —
it is **pinned** instead, URL and digest together, which is a narrower promise
rather than a weaker one. Moving the pin means changing `PORTABLE_PY_URL` and
`PORTABLE_PY_SHA256` together.

### The python.org route is still there, behind it

For a machine that cannot reach that download. It keeps the ladder that was
built when it was the only route, because a no-op must not be accepted as an
answer:

1. Install. If `<install>/python` now exists, done.
2. **Repair.** A registration pointing at our own deleted `<install>/python`
   is repaired straight back into place, and a repair cannot damage a Python
   anywhere else.
3. **Remove the registration and install cleanly.** Safe by elimination, not
   by hope: `find_python` has already searched the registry twice by this
   point, so if any working 3.11.9 existed on the machine we would be using
   it and would never have got here. What is left has no interpreter behind
   it and belongs to nobody.

Every attempt is optional and every exit code is logged, including the zeros.
What ends the install is the question at the bottom — is there an interpreter
on disk — not any single step's return value.

`find_python` reads the registry first for the same reason all of this
exists: PEP 514's `Software\Python\<Company>\<Tag>\InstallPath`, under HKCU
and both views of HKLM, is the only complete list. The `py` launcher is
optional and often absent, `PATH` holds at most one, and the directory globs
only know the python.org defaults. `python_inventory` lists dead
registrations too — the ones no version check can see, and the ones that
caused this.

## The exe the user double-clicks is tested before the install is called done

`Thumbnail Maker.exe` is compiled on the user's machine at step 9, and a
frozen build can succeed and still be unable to start. It has happened twice,
both times the same shape: an install that ran to the end, reported success,
verified every file — and one broken exe, found by the user days later.

| | |
|---|---|
| Tk data missing from the bundle | `INHERITED_TK_VARS` — the installer's own `TCL_LIBRARY` reaching the child build |
| `Module use of python310.dll conflicts with this version of Python` | `FOREIGN_PYTHON_VARS` — a `PYTHONPATH`/`PYTHONHOME` on the machine pointing the build at another Python's `Lib` and `DLLs` |

Both causes are fixed at the source, by stripping those variables from every
child process. But both were invisible at build time, so the build is no
longer taken at its word: `assert_launcher_is_startable` reads the archive for
its Tk data, and then **runs the exe** with `--selftest`, a flag that reaches
the far side of `launcher.py`'s imports and exits without starting the app's
servers. Everything of this kind has already gone wrong by the time that line
runs.

A broken exe would stop on a modal Windows error box instead of exiting; the
180-second timeout kills the process, which takes the box with it, and the
install fails there — on the machine, in front of the person who can still do
something about it.

## How an install that was killed says so

A process that is terminated writes nothing. No exception, no `finally`, no
`atexit` — an antivirus does not raise an error, it calls `TerminateProcess`,
and the log simply stops mid-sentence. From the outside that is
indistinguishable from a crash, a hang, or somebody closing the window.

That matters here because of what this installer looks like to a heuristic
scanner: a PyInstaller onefile exe that downloads an interpreter, unpacks
thousands of DLLs into `AppData\Local`, and then compiles another executable.
That is also the description of a dropper.

So the question is answered three ways, none of which require the dying
process to do anything:

| | |
|---|---|
| **A second log** | `%LOCALAPPDATA%\Thumbnail Maker Setup\logs\install.log`, outside the install directory, because the first copy is inside the folder that goes missing |
| **A dead man's switch** | `session.json` is written when an install starts and deleted when it ends — succeeding *or* failing. Still there on the next run means the run before it did neither. `report_previous_run` puts that at the top of the new log, with the step it died on |
| **A sentinel** | a file only this installer knows about, checked every 15 seconds. Quarantine and termination are separate acts, so when files go first there is still somebody here to write a line about it |

After finding a killed run, `security_report` asks Windows what it knows:
which security products are registered, what Defender quarantined recently
(**with the paths**, so ours are recognisable), and whether the Application
log has anything. All of it is optional and none of it can fail an install —
a machine with a third-party scanner has no `Get-MpThreat*` at all.

## The install order is not negotiable

Straight from the comments in `requirements.txt`, learned the hard way:

- **torch before gfpgan.** gfpgan pulls its own CPU-only torch on Windows if
  it cannot see one, replacing the CUDA build. Torch is installed first and
  re-asserted afterwards.
- **`--no-build-isolation` for basicsr.** Its `setup.py` imports torch at
  build time, which an isolated build environment does not have.
- **`--no-deps` for simple-lama-inpainting.** It pins old numpy, opencv and
  pillow and downgrades all three on a plain install. Everything it actually
  needs is already satisfied.
- **basicsr needs patching.** It imports
  `torchvision.transforms.functional_tensor`, removed in torchvision 0.17, at
  module scope — which takes gfpgan down with it. The function is still there
  under the public name.

## The JS runtime, and the 403

YouTube signs its media URLs with JavaScript that the watch page executes.
yt-dlp can extract that code but cannot run it, so on a machine with no JS
runtime the URLs it produces are rejected — reaching the user as:

```
Download failed: ERROR: unable to download video data: HTTP Error 403: Forbidden
```

`video_downloader.py` retries three times on the theory that a fresh
extraction gets lucky. On a machine with no runtime at all there is no luck to
be had, and every attempt fails the same way.

So the installer installs **deno** into `runtime/Scripts/` — the first
directory yt-dlp searches (`yt_dlp/utils/_jsruntime.py:_find_exe`), which means
no PATH changes and nothing for the app to configure. Verification asks yt-dlp
whether it can see a usable runtime rather than merely checking that the file
exists, because those are different questions and only the first one decides
whether downloads work.

yt-dlp accepts deno ≥ 2.3.0, bun ≥ 1.2.11, node ≥ 22 or quickjs; if one is
already present the installer leaves it alone.

## Downloads have two routes

Everything the installer fetches goes through `Installer.download`, which
tries Python's own `urllib` and falls back to `curl.exe` from System32.

Not a retry: `urllib` speaks TLS through the OpenSSL that PyInstaller froze
into the setup exe from the machine that built it, and that library then
travels to strangers' computers. When it fails to load there, the download
fails before a byte moves —

    Could not download deno: [DSO: LOAD_FAILED] could not load the shared
    library (_ssl.c:4030)

— with nothing wrong with the network. curl has been in System32 since
Windows 10 1803 and goes through Schannel, Windows' own TLS, sharing nothing
with Python's. The two fail for unrelated reasons, which is what makes the
second one worth trying.

Seen once during testing and not reproducible afterwards, which is its own
argument for the fallback: an installer that has to be run twice to find out
whether today is a good day is one nobody trusts.

## Four models, and where each one's weights land

The app loads four networks, and none of their weights ship in anything the
installer carries — they are downloaded once, by asking the app to load its own
models (`Installer.download_models`), which puts every file exactly where the
app will later look for it.

| | What it does | Where its weights go |
|---|---|---|
| GFPGAN | face restoration | `app/backend/gfpgan/weights` |
| LaMa | inpainting: logo removal, and the autofill past a dragged crop | `models/` (torch's hub cache, pointed here by `TORCH_HOME`) |
| U2-Net | background removal: where exactly the subject's edge is | `app/backend/models/u2net.onnx` |
| LR-ASPP | background removal: which pixels are a person at all | `models/` (torch's hub cache, as LaMa) |

Two of the four live under `app/`, which every update replaces wholesale — so
both are named in `updater.KEEP_ACROSS_UPDATES`, and that is the only thing
standing between a one-megabyte patch and an 800 MB re-download.

Each degrades rather than failing: no GFPGAN means the classical pipeline
alone, no LaMa means OpenCV Telea, no LR-ASPP means a cutout that keeps
whatever scenery the saliency model thought was part of the subject, and no
U2-Net means the classical segmenter in `region_segmenter.py` — a visibly
rougher cutout, but a cutout. Which one actually ran is logged at startup and
never guessed at.

The last two are a pair and split the job between them: LR-ASPP knows what a
person is and could not trace a strand of hair; U2-Net traces the strand and
does not know a person from a spotlight. See `background_remover.py`.

U2-Net is reached through `onnxruntime` and not through `rembg`, which is the
usual way to these weights: rembg pins old numpy, opencv and pillow and
downgrades all three on a plain install, exactly as simple-lama-inpainting
does, and everything it would add beyond the `.onnx` file is a resize and a
normalise. See `backend/background_remover.py`.

## The fonts, and everything else a channel brings with it

A channel is data — a folder of JSON under `content/channels/` (see
[frontend/content/README.md](../frontend/content/README.md)). It names the
type face its titles are set in, and sometimes a texture its highlight boxes
are filled with, and the page fetches both by URL at run time.

Nothing fails loudly when one of those files is absent. A missing texture
simply does not appear. A missing face is worse, and is the subject of the
rest of this section.

So the files are read out of the packs themselves — `updater.content_assets`,
which is where it lives because the installer and an installed copy both need
it, and which covers every picture a pack can name: the highlight foil, the
backdrop a channel composes on, and its brand mark (`CONTENT_TEXTURE_KEYS`).
They are checked in three places:

- `build_installer.py` and `build_patch.py` refuse to build with one missing,
  so a channel whose font never made it into the payload cannot ship.
- `engine.verify()` refuses to finish an install with one missing.
- The installer registers each face with Windows, and so does a patch that
  brings a new one — under the first of the `local` names the face declares,
  and not at all when it declares none. That list is the pack's own statement
  of what Windows may know the face by. `scale-condensed-bold` leaves it out
  on purpose: it invents its family name to pin one instance of a variable
  font, so anything a machine handed back under that name would be a different
  cut. For a face like that the file is the only honest route.

A channel added next month is covered by all three without anyone editing a
list.

## The title font in particular

The title overlay's whole layout is measured from Trade Gothic: sizes,
highlight boxes, line positions. Impact — first in the fallback stack — is
~17% wider at the same size, so a face that fails to load does not produce
"a different font", it produces a title composed against numbers describing a
font nobody can see. Oversized boxes, text sitting high inside them, lines
overflowing the canvas.

Four things now stand between the app and that:

1. The `.otf` is bundled and served by the app's own server, and the build
   fails if it is missing or truncated from the payload.
2. The installer registers the same face with Windows, and the stylesheet
   lists `local()` sources after the `url()` — so if the served file ever
   fails, the next source is *the same face by another route*, not Impact.
3. `frontend/js/fontguard.js` verifies at runtime that the face is real by
   measuring it against a sentinel generic, and shows a red banner naming the
   missing file instead of rendering a wrong-looking title silently.
4. `Thumbnail Maker.exe` refuses to start if the font file is missing.

The face's own family name is `Trade Gothic Next LT Pro`, with `Heavy
Compressed` as its subfamily; the full name the stylesheet uses as a family
only exists because `@font-face` declares it. That is why the Windows registry
entry is keyed on the full name and the `local()` fallbacks list both it and
the PostScript name `TradeGothicNextLTPro-HvCm`.

## Layout after installing

```
%LOCALAPPDATA%\Programs\Thumbnail Maker\
  Thumbnail Maker.exe     built here, from launcher.py, with the fox icon
  app\                    backend\ + frontend\ + release.json
  runtime\                the venv everything runs in
  models\                 what torch downloads for itself (TORCH_HOME)
  content\                channels this machine added
  updater\                updater.py, ed25519.py, launcher.py
  backup\                 the previous app\, for --rollback
  assets\                 fox_blue.ico
  logs\                   install.log, backend.log, frontend.log, update.log
  version.json            what is installed, and what has been applied
  Uninstall Thumbnail Maker.cmd   double-click to remove it
  uninstall.ps1                   what that runs
```

The split across those directories is the update mechanism, and each line of
it is deliberate:

- **`app\` is disposable.** A patch replaces it whole. Nothing that a machine
  would miss may live inside it — which is why `version.json` is outside it,
  and why `updater.KEEP_ACROSS_UPDATES` names the two things that have to be
  carried across anyway: the ~185 MB of face weights facexlib resolves
  relative to `app\backend`, and the 176 MB U2-Net file beside them. Neither
  is in any payload, and without that list every patch would download both
  again.
- **`content\` is the user's.** Channel packs dropped in here are merged with
  the shipped ones by the frontend server and are never touched by an update.
  See `frontend/content/README.md`.
- **`updater\` can replace itself.** A patch may carry a new copy, applied
  last and from a temp copy of itself, so a bug in it is fixable rather than
  permanent.
- **`version.json` is what the machine says about itself**, as opposed to
  `app\release.json`, which is what the build says about itself. The two are
  separate because the second is replaced by every update and the first is
  the record of those updates.
- **`models\` is inside the install because torch would otherwise put it
  outside.** Left alone, torch downloads the ~200 MB inpainting weights into
  `%USERPROFILE%\.cache\torch`, shared with every other torch program on
  the machine. An uninstaller would then have to either abandon them forever
  or delete them out from under something else. `TORCH_HOME` points at this
  directory instead — set by the installer when it downloads them and by the
  launcher when it starts the app, which have to agree or the app quietly
  fetches a second copy on first use.

## Removing it

`Uninstall Thumbnail Maker.cmd`, in the install folder, is what a person
double-clicks; Apps & features runs the same file. It exists because Windows
opens a `.ps1` in an editor rather than running it, so `uninstall.ps1` cannot
itself be the thing anyone clicks.

The script is copied to the temp directory and run from there, because it is
about to delete the folder it lives in — and Windows will not delete a file
that cmd or PowerShell still has open.

What it removes: the install directory, which is nearly everything (the
private Python environment, the libraries and the model weights are all inside
it), the desktop and Start-menu shortcuts, the fonts the install registered —
by exactly the names it registered them under, recorded as it did — and the
Apps & features entry.

What it does not: touch any Python on the machine. The app never installed a
library into one. It also asks before starting, and stops the app if it is
running, because deleting a folder underneath a process that has files open in
it leaves most of the folder behind.

## Files here

| File | |
|---|---|
| `build_installer.py` | builds the setup exe, for a machine with nothing |
| `build_patch.py` | builds an update exe, for a machine that already has it |
| `patch_gui.py` | the window inside that exe |
| `setup_gui.py` | the installer window — English, three pages |
| `engine.py` | the scan and the install steps; runs headless too |
| `updater.py` | checks the feed, applies a patch, rolls one back |
| `ed25519.py` | signature checking, in pure Python |
| `launcher.py` | becomes `Thumbnail Maker.exe`, and runs the update check |
| `make_icon.py` | generates the blue fox icon from `frontend/fox.png` |

`engine.py` runs without the GUI, which is the fastest way to see what a
machine reports:

```bash
python installer/engine.py --scan
```

`updater.py` does the same for the update side, against a real install:

```bash
python installer/updater.py --install-dir "%LOCALAPPDATA%\Programs\Thumbnail Maker" --status
python installer/updater.py --install-dir "%LOCALAPPDATA%\Programs\Thumbnail Maker" --check
```

## Where the version comes from

`release.json`, in the repo root, and nowhere else. `engine.VERSION` reads it,
`build_patch.py` reads it, and it ships inside the payload as
`app/release.json` so the running app can report what it is. It also carries
the update feed URL and the publisher's public key, which is what makes those
properties of a build rather than of a machine — change either one and the
next installer and every patch after it carry the change.
