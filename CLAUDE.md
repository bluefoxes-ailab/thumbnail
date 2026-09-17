# Conventions

Short list. These are the things that quietly rot if nobody writes them down.

## English, everywhere, always

Every string a person can see is in English. Buttons, labels, placeholders,
status lines, error messages, installer pages, the update window, release
notes in `patch.json`, log lines, and the comments and docstrings around all
of it.

This holds regardless of what language a conversation about the code is
happening in. A request made in Portuguese does not make the answer
Portuguese — the code and the interface stay English.

It is enforced, not trusted:

```bash
python installer/check_language.py
```

It runs first inside `build_installer.py` and `build_patch.py` and **fails the
build**, so nothing in the wrong language can be shipped. It flags accented
letters and a short list of unmistakably Portuguese words.

It is a backstop, not a proof — Portuguese written in plain ASCII using none
of the listed words gets through. Write English; the checker is there for
slips, not instead of the rule.

Two deliberate exceptions:

- **English prose quoting something foreign**, where the foreign thing is the
  subject — a Windows path that really is called "Área de Trabalho", the
  accented capitals a title has to leave room for. Declared in `EXEMPT` in
  `check_language.py`, each with its reason. An exemption that stops matching
  is reported as stale and fails the build, so the list cannot become a
  blanket permission.
- **Channel names**, under `frontend/content/channels/`. A channel's `name` is
  a brand — whatever that channel is actually called. Everything else about a
  pack is still English.

## Comments carry the reasoning

The comments in this codebase say *why*, not *what*, and they are long where
the reason is long. That is the house style: a decision that took an afternoon
to reach gets the two paragraphs that stop someone undoing it. Match the
density of the file being edited rather than trimming it.

JSON has nowhere to put a comment, so the content packs use `_comment` keys.
The loader strips any key starting with `_`.

## Where things are written down

| | |
|---|---|
| [RELEASING.md](RELEASING.md) | how a change reaches a user |
| [DEPLOYMENTS.md](DEPLOYMENTS.md) | which machine is on which version, and which `--group` |
| [installer/README.md](installer/README.md) | what the installer does and why |
| [frontend/content/README.md](frontend/content/README.md) | how to add a channel |

## Two invariants worth knowing before editing

- **`<install>/app/` is disposable.** An update replaces it whole. Nothing a
  machine would miss may live inside it — that is why `version.json`,
  `content/` and `updater/` sit outside, and why
  `updater.KEEP_ACROSS_UPDATES` exists for the model weights that have no
  choice.
- **The title font is not cosmetic.** Every size, box and line position is
  measured from Trade Gothic. A missing face does not produce a different
  look, it produces a layout computed against a font nobody can see. Four
  separate things guard it; see the font section of
  [installer/README.md](installer/README.md) before touching any of them.
