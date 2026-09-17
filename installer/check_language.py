"""
Refuses to ship a build with non-English text in it.

The app is written in English throughout — every label, every button, every
error, and every comment. That is a decision, not an accident, and this is
what keeps it from eroding one string at a time: it runs inside
build_installer.py and build_patch.py, so a Portuguese sentence cannot reach a
user without the build failing first.

    python installer/check_language.py

## What it looks for

Two things, in the files a user's eyes or a maintainer's eyes land on:

1. **Letters carrying diacritics** — `não`, `você`, `configuração`, `é`. This
   is the strong signal, because Portuguese cannot go far without one, and
   English here needs none. Typographic punctuation is a different matter and
   is allowed: the prose in this codebase leans on em dashes, curly quotes and
   bullets, and none of those say anything about language.

2. **A short list of unmistakably Portuguese words** — only ones with no
   English meaning and no accent to catch them by. Deliberately short: `de`,
   `a`, `e`, `no`, `os` and their like are also English words, or Python, or
   part of an identifier, and a checker that cries wolf is a checker somebody
   switches off.

## What it does not look for

This is a backstop, not a proof. A Portuguese sentence written in plain ASCII
using none of the listed words will pass — "Nada para fazer" would. What it
catches is the realistic case: text typed naturally by a Portuguese speaker,
which reaches for an accent within a few words.

## Foreign text that is not a mistake

Some English prose here has to quote something foreign, because the foreign
thing is the subject: a Windows path that really is called "Area de Trabalho",
the accented capitals a Portuguese title puts above the cap line. Those are
declared in EXEMPT below rather than marked in the source, so that every
exception lives in one list somebody can read in ten seconds instead of being
scattered through the files as pragmas nobody ever revisits.

Each exemption names the text it excuses. When that text changes or goes away,
the exemption stops matching and is reported as stale — so the list cannot
quietly grow into a blanket permission for the file it names.

Channel packs under `frontend/content/` are exempt wholesale. A channel's
`name` is a brand — whatever that channel is actually called — and "Casos
Reais" is its name, not an interface string in the wrong language.
"""

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where interface text and the prose around it live. Everything else is either
# third-party (libs/, vendor/), generated, or data.
SCANNED = [
    ("frontend", "index.html"),
    ("frontend", "styles.css"),
    ("frontend", "serve.py"),
    ("frontend/js", "*.js"),
    ("backend", "*.py"),
    ("installer", "*.py"),
    (".", "*.md"),
    (".", "*.bat"),
    (".", "release.json"),
    (".", "patch.json"),
]

SKIP_DIRS = {"libs", "vendor", "node_modules", "__pycache__", ".build-venv",
             "dist", "patches", "build", "temp", "gfpgan", "content"}

# Punctuation and symbols this codebase uses in English prose. None of them
# implies a language, and flagging them would make the check unusable.
ALLOWED_NON_ASCII = set(
    "—–‑…“”‘’·•×÷≥≤≠→←↑↓✓✗°±§¶©®™"
    "−"                # U+2212, the minus in the zoom control's own button
    "─│┌┐└┘├┤┬┴┼"      # the rules that separate sections in comments
    " ﻿"      # non-breaking space, byte-order mark
)

# Unaccented, and Portuguese only. Anything ambiguous with English, with a
# Python keyword, or with a plausible identifier is left out on purpose.
PORTUGUESE_WORDS = {
    "nao", "voce", "usuario", "usuarios", "arquivo", "arquivos", "pasta",
    "senha", "janela", "botao", "mensagem", "imagem", "atualizacao",
    "configuracao", "instalacao", "clique", "aguarde", "carregando",
    "selecione", "escolha", "falhou", "sucesso", "erro", "erros",
    "salvar", "abrir", "fechar", "enviar", "baixar", "instalar",
    "atualizar", "canal", "canais", "quadro", "legenda", "miniatura",
}

# (file, the exact text excused, why). A line is let through when it contains
# the excused text. Every entry has to still match something, or the check
# reports it as stale and fails — an exemption for text that no longer exists
# is an exemption quietly covering whatever replaced it.
EXEMPT = [
    ("frontend/js/text.js", "Á, Ã, É",
     "the accented capitals being measured — the subject of the comment"),
    ("backend/image_utils.py", "Área de",
     "a real Windows path, which is why imread cannot open it"),
    # Matched without the backslash in front of it: that line lives in a
    # docstring, where the separator is written "\\" and an exemption spelling
    # it out would break the day the quoting changed rather than the day the
    # text did.
    ("installer/engine.py", "Área de Trabalho",
     "the same real path, on the machine this was written on"),
    ("installer/check_language.py", None,
     "this file is the dictionary — it has to contain what it looks for"),
    ("CLAUDE.md", "Área de Trabalho",
     "the rule's own worked example of a foreign name that is the subject"),
]


def exemptions_for(rel_path):
    return [text for path, text, _ in EXEMPT if path == rel_path]


WORD_RE = re.compile(r"[A-Za-z]{4,}")


def offending_chars(line):
    """The characters in `line` that are neither ASCII nor allowed punctuation."""
    bad = []
    for ch in line:
        if ord(ch) < 128 or ch in ALLOWED_NON_ASCII:
            continue
        # A letter with an accent on it is the signal; an unfamiliar symbol is
        # not, so they are reported differently.
        decomposed = unicodedata.normalize("NFD", ch)
        if len(decomposed) > 1 and unicodedata.category(ch).startswith("L"):
            bad.append((ch, "accented letter"))
        elif unicodedata.category(ch).startswith("L"):
            bad.append((ch, "non-Latin letter"))
        else:
            bad.append((ch, "unexpected symbol"))
    return bad


def offending_words(line):
    return sorted({w.lower() for w in WORD_RE.findall(line) if w.lower() in PORTUGUESE_WORDS})


def files_to_scan():
    seen = []
    for folder, pattern in SCANNED:
        base = ROOT / folder
        if not base.is_dir():
            continue
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1]):
                continue
            if path not in seen:
                seen.append(path)
    return seen


def check():
    """Returns a list of (path, line number, line, reason) — empty when clean."""
    findings = []
    used = set()

    for path in files_to_scan():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            findings.append((path, 0, "", f"could not be read as UTF-8 ({e})"))
            continue

        excused = exemptions_for(rel)
        if None in excused:                     # the whole file is excused
            used.add((rel, None))
            continue

        for n, line in enumerate(text.splitlines(), 1):
            hit = next((t for t in excused if t and t.strip() in line), None)
            if hit:
                used.add((rel, hit))
                continue
            chars = offending_chars(line)
            if chars:
                what = ", ".join(f"{c!r} ({why})" for c, why in dict(chars).items())
                findings.append((path, n, line.strip(), what))
                continue
            words = offending_words(line)
            if words:
                findings.append((path, n, line.strip(), "Portuguese: " + ", ".join(words)))

    for rel, excused_text, reason in EXEMPT:
        if (rel, excused_text) not in used:
            findings.append((ROOT / rel, 0, "",
                             f"stale exemption — nothing matches {excused_text!r} any more "
                             f"({reason}). Remove it from EXEMPT."))
    return findings


def assert_english(quiet=False):
    """
    Raises SystemExit if anything is not English. Called by both builders.

    Fails the build rather than warning, because a warning printed in the
    middle of a two-minute PyInstaller run is a warning nobody reads, and the
    thing being protected here is what a user sees.
    """
    findings = check()
    if not findings:
        if not quiet:
            print(f"language: {len(files_to_scan())} files checked, all English")
        return

    # Everything below goes through `safe`, because the console this runs in
    # is routinely cp1252 and the entire point of the output is to show
    # characters it cannot encode. A report that dies while reporting is
    # worse than no report at all.
    def safe(text):
        return str(text).encode("ascii", "backslashreplace").decode("ascii")

    print("\nFATAL: non-English text found. The interface is English throughout.\n")
    for path, n, line, reason in findings[:40]:
        print(f"  {path.relative_to(ROOT)}:{n}  {safe(reason)}")
        if line:
            print(f"      {safe(line[:110])}")
    if len(findings) > 40:
        print(f"  ... and {len(findings) - 40} more")
    print("\nFix them, or — if one is a false positive — widen ALLOWED_NON_ASCII "
          "in installer/check_language.py and say why.")
    raise SystemExit(1)


if __name__ == "__main__":
    assert_english()
    sys.exit(0)
