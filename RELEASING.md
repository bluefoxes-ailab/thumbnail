# Shipping a change

Five people, five local machines, and a file you send them. That is the whole
distribution model, and everything here is built around it.

```bash
python installer/build_patch.py
```

Out comes one file:

```
installer/patches/Thumbnail Maker Patch 1.1.0.exe
```

Send it — WhatsApp, e-mail, a shared folder, a stick. The person double-clicks
it. A window opens, tells them what is in the update, they press **Install
update**, and it is done in a few seconds. Nothing to type, nothing to
uninstall, and none of the ~5 GB already on their machine is downloaded again.

---

## The loop

1. Make the change — anything under `backend/`, `frontend/`, or a channel pack
   in `frontend/content/channels/`.
2. Bump `version` in [release.json](release.json).
3. Write what changed in [patch.json](patch.json) under `notes`, **in
   English**. That is what the user reads in the window before pressing the
   button - the one field worth never leaving empty, and the build refuses a
   note in any other language (see CLAUDE.md).
4. `python installer/build_patch.py`
5. Send the `.exe`.

The first build takes a couple of minutes (it makes a private build
environment); after that it is well under one.

## What the user sees

```
  Thumbnail Maker
  Update to version 1.1.0
  C:\Users\...\Programs\Thumbnail Maker    ·    currently on 1.0.0

  •  Two new channels
  •  Faster reframing

  [ Install update ]   [ Close ]              Details
```

It refuses, in plain words, when it should:

| Situation | What it says |
|---|---|
| The app is open | "Close it, then press Install update again" |
| Already on that version or newer | "Version 1.1.0 is already installed" |
| A step in between is missing | "Version 1.2.0 needs 1.1.0 installed first" |
| The app is not on this machine | offers a folder picker |
| Something goes wrong midway | the previous version is put back, **Details** shows why |

Everything it does is also written to `<install>\logs\update.log`, which is
what to ask for when someone says it did not work.

## What an update can and cannot carry

Everything under `app/` — backend, frontend, channel packs, requirements —
plus the updater itself. That covers new features, fixed bugs, new channels.

Three more things go in `patch.json` as instructions rather than payload,
because they are large or live outside `app/`:

| | |
|---|---|
| `pip` | new libraries, installed into the runtime that is already there |
| `models` | new weights, downloaded once from their own URL |
| `fonts` | a face registered with Windows as well as served, that no pack declares |

Outside that: the Python version, the install location, the virtualenv, the
deno runtime. If a release needs one of those, put a sentence in `patch.json`'s
`requires_setup` saying why. The patch window will show it and send the user to
the installer instead of applying half a release.

Re-running the setup exe over an existing install is the fallback and is safe:
it keeps the venv, keeps the model weights, keeps the user's own channels, and
skips everything already satisfied.

## Adding a channel

A channel is a folder of JSON under `frontend/content/channels/` — no code, no
`@font-face`, no rebuild. See [frontend/content/README.md](frontend/content/README.md).
Ship it like any other change: bump the version, build a patch, send it.

Its font and its texture ride along with it, and are not listed anywhere: both
builds read the files a pack names straight out of the pack and refuse to build
if one of them is not there, and both the installer and the updater register a
face declaring `local` names with Windows on their own. The `fonts` row above
is only for a face that arrives without a pack to declare it.

A channel a *user* adds themselves goes in `<install>\content\channels\`, and
updates never touch that folder.

## Shipping one customer's channels only

Every pack in the repo ships by default. A build for a single customer names
their group instead:

```bash
python installer/build_installer.py --group "QDS Karma"
```

The group is the `group` key each pack already declares, matched
case-insensitively; repeat `--group` for more than one. Nothing moves in the
repo — the packs are dropped from the staged payload — so the next build
without the flag is everything again, and a misspelled group name fails the
build rather than quietly producing an installer with no channels.

`build_patch.py` takes the same flag, and a filtered install needs it:

```bash
python installer/build_patch.py --group "QDS Karma"
```

Without it a patch carries every channel in the repo. That is not a cosmetic
difference, because an update replaces `<install>pp\` **whole**: an
unfiltered patch sent to a filtered install *adds* every other customer's
brands to it, and a patch filtered to the wrong group *removes* the packs that
were there. The flag names what the target machine already has, not what the
release happens to be about.

### `dist/` is scratch, `installer/releases/` is what you keep

Every build starts by deleting `installer/dist/` whole, and the filename it
writes carries the version and nothing else. Two builds of one version for two
customers are therefore the same path, and the second one silently replaces
the first — the group is the one thing that distinguishes them and the one
thing the name does not say.

So a build that is going to be handed to somebody is moved out and renamed as
it leaves:

```bash
installer/releases/Thumbnail Maker Setup 1.3.0 (Snapchat).exe
```

`releases/` is not touched by any build. Record the same file in
[DEPLOYMENTS.md](DEPLOYMENTS.md) when it is actually sent.

To find out what a machine has when nobody wrote it down, the installer they
were given still knows — its staged pack names survive in the exe:

```bash
grep -a -c undercover-ceo "Thumbnail Maker Setup 1.2.5.exe"
```

## When one goes wrong

The previous `app/` is kept at `<install>\backup\app-<version>` until the next
update replaces it, so there is always exactly one step back:

```bash
"%LOCALAPPDATA%\Programs\Thumbnail Maker\updater\updater.py" --rollback
```

Only `app/` is restored. A library or model an update installed stays
installed — both are additive, and removing them could break the version being
restored *to* if it turned out to share them.

---

## Appendix: the parts you are not using

The same machinery supports automatic updates, and it is off. Nothing checks
anything, nothing is contacted, and the launcher does not even start the
updater unless a server has been named. Worth knowing it is there for the day
five machines becomes fifty.

**Automatic updates.** Set `feed` in `release.json` to where you publish the
`installer/patches/` folder — an https URL, a network share, a `file://` path.
Builds then also write an `updates.json` next to the patches, and every install
that carries that URL checks it at startup and offers what it finds. Turning it
on for a machine that already exists does not need a reinstall: `feed` in
`<install>\version.json` wins over the built-in one, and a hand-delivered patch
carries a new `release.json` anyway.

**Signing.** `python installer/build_patch.py --keygen`. Only meaningful once
updates sit on a server rather than being handed over: a file you sent someone
yourself is already as authenticated as it is going to get. The public half
goes in `release.json`; the private half stays outside the repo, and losing it
means no install carrying the public half will accept a patch again.

**The bare `.tmpatch`.** Built alongside the exe. It is the exe's own payload,
for a machine where running an exe from e-mail is not allowed, or for a script:

```bash
"%LOCALAPPDATA%\Programs\Thumbnail Maker\updater\updater.py" --apply-file patch.tmpatch
```

---

## Files

| | |
|---|---|
| `release.json` | the version being built (and, if ever used, the feed and public key) |
| `patch.json` | what is new in the update being built now |
| `installer/build_patch.py` | builds the update exe |
| `installer/patch_gui.py` | the window inside it |
| `installer/updater.py` | applies an update, and rolls one back |
| `installer/ed25519.py` | signature checking, for the feed route |
| `installer/build_installer.py` | still how a new machine gets the app |
| `frontend/content/` | the channel packs, as data |
