# Who has what

One row per machine. It exists because two things a build needs cannot be
recovered from the repo: the version a machine is on, and the `--group` it was
built with.

Both matter for the same reason — an update replaces `<install>/app/` **whole**
(see [RELEASING.md](RELEASING.md)). A patch built with the wrong `--group`
removes the packs that machine had, or adds another customer's brands to it.
The flag names what the target machine already has, not what the release
happens to be about.

| Machine / customer | Group(s) | Version | Last file sent | Notes |
|---|---|---|---|---|
| QDS Karma | `QDS Karma` | 1.3.0 | Setup 1.3.0, 2 Sep | Confirmed working on the user's machine |
| Smart News | `Snapchat` | 1.3.0 | Setup 1.3.0, 2 Sep | Rebuilt 4 Sep as `Setup 1.3.0 (Snapchat).exe` |
| — | `Binge` | — | built 4 Sep, not sent | `laugh-society-1`, `laugh-society-2` |
| — | `France TV` | — | not shipped | `la-parenthese-inattendue` |

Fill a row in when a file is sent, not when it is built. The files themselves
live in `installer/releases/`, with the group in the name — `installer/dist/`
is deleted by every build and holds only the most recent one.

## When a row is missing or in doubt

The installer that machine was given still knows which packs were staged into
it — the names survive in the exe:

```bash
grep -a -c undercover-ceo "Thumbnail Maker Setup 1.3.0.exe"
```

And the machine itself knows its version: `<install>\version.json`, which also
records every update applied to it.

## Groups currently declared by the packs

Read out of the `group` key in each pack under `frontend/content/channels/`:

| Group | Packs |
|---|---|
| QDS Karma | `best-of-supermission`, `karens-unleashed`, `superlove`, `supermission`, `undercover-ceo` |
| Snapchat | `hall-of-femme`, `fitness-story`, `fight-source`, `determined`, `killer-bites` |
| Binge | `laugh-society-1`, `laugh-society-2` |
| France TV | `la-parenthese-inattendue` |

This table is derived, not authoritative — the packs are. Regenerate it rather
than editing it by hand when a pack's group changes.
