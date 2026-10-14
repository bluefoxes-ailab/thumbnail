"""
reference.py - how a person is named, line after line of one account.

A caption grid reads down a column, and forty lines in a row that all begin
with the same two words read as a database dump rather than as a story:

    Dolly Parton's story was not exaggerated.
    Dolly Parton was born on January 19, 1946, in a one-room cabin.
    Dolly Parton's family was extremely poor.
    Dolly Parton picked up a guitar at the age of seven.
    Dolly Parton had an extraordinary music career.

That is real output, from when every caption was written to stand alone over
its own photograph: the model was told each line had to stand by itself, and
the safest way for a model to make a line stand by itself is to name the
subject in full. Every line.

The account is one text now (see narrator.py), which takes that pressure
away, and the prompt asks for the variation outright. But asking was tried
before, twice, the second time with the rule in capitals, and measured both
times: seven of seventeen lines still opened with the full name, six of them
consecutive. The model follows a concrete instruction and ignores a
discipline held across a list - see the note above narrator.SYSTEM, where the
same thing happened to three other rules. So the variation is held by
machinery, whatever the model does: the full name, then a pronoun, a role,
the surname, the given name.

## What it does

Where a line opens with the same person the line before it opened with, that
opening is swapped for another way of naming them, cycling through whatever
forms are available, in this order:

    full name   Dolly Parton        (always; what the first mention uses)
    pronoun     She                 (only with evidence of gender - see below)
    role        The singer          (only if the text said so - see below)
    surname     Parton
    given name  Dolly

Only ever the SECOND and later of a run. The first mention of a person is left
in full, because that is the one doing the introducing. And never a form that
would not say who: a surname or a given name that two people in the account
share is skipped, since in a story with Dolly and Stella Parton in it,
"Parton said" could be either sister.

The order is the one an account is written in - "Dolly Parton was born...",
"She grew up...", "The singer moved...", "Parton recorded..." - and it is the
order the app was asked to name people in. It used to lead with the surname,
on the argument that a slice of the name already on screen cannot be wrong;
that was right while every caption stood alone over its own photograph, and
the account is read as one text now, where a pronoun after the full name is
simply how prose goes on. The two conditional forms are why that is safe
rather than bold: each is only offered where it is known, and the pronoun only
where it is unambiguous.

## Why it is safe to use a pronoun here

A pronoun is the alternative that could do real damage: "She was born in 1946"
under a photograph of the wrong woman is a caption that lies. The reason it is
safe is the same reason the swap is happening at all - it fires only when the
PREVIOUS line opened with this person - and one more condition on top: that
line named nobody else. "Dolly Parton hugged her sister Stella Parton" followed
by "She said..." could mean either of them, so after a line like that the
rotation skips the pronoun and names the person another way. That is exactly
the condition under which a pronoun is unambiguous, and it is checked rather
than hoped for - as far as names can be found at all, which is names of two
words or more (see _names_in).

## What is deliberately not done

Gender is never guessed from a name. It is read out of the words themselves -
"her mother", "his aunt" - and where the text never says, there is no pronoun
form and the rotation simply skips it. A name tells you nothing reliable about
pronouns, and this module is not going to pretend otherwise.

A role is never invented either. It is lifted verbatim, article and all, from
an appositive the text actually contains ("Dolly Parton, the singer") and used
as it was written. Nothing here composes a noun phrase it did not read.

Possessive forms ("Dolly Parton's family") are handled only where the language
marks possession on the NAME, which in practice means English's "'s". A
Portuguese caption says "a familia de Dolly Parton" and the possession sits on
a preposition halfway down the sentence, nowhere near the opening this module
edits - so the case never arises there rather than being got wrong. The same
caution kills possessive pronouns outside English: Portuguese "sua" and French
"son" agree with the thing possessed rather than the owner, so no table of
them could be applied without parsing the noun that follows.

## Never fatal

Every step answers "leave it alone" when it is unsure: no person found, no
repeat, no alternative available, a language this file has no table for. The
lines come back exactly as they arrived, which is a working account.
"""

import re

# The handful of non-ASCII characters this file needs, written as code points.
#
# CLAUDE.md requires every source file here to be English and ASCII, and
# installer/check_language.py fails the build over an accented letter — which
# it did over this file, correctly. These are not prose in another language,
# they are language DATA: a Spanish pronoun, the capital letters a European
# name can start with, and the curly apostrophe a transcript can carry. Naming
# them here keeps the rule intact and says what each one is, which a bare
# escape in the middle of a table would not.
E_ACUTE = chr(0xE9)          # e with an acute accent, as in the Spanish "el"
A_GRAVE = chr(0xC0)          # A with a grave accent - the first accented capital
U_DIAERESIS = chr(0xDC)      # U with a diaeresis - the last one worth covering
RIGHT_QUOTE = chr(0x2019)    # the curly apostrophe, in "Parton" + this + "s"

# Subject and possessive pronouns, by language and by gender.
#
# The possessive slot is None wherever the language declines it against the
# thing possessed rather than the owner - see the module header. That is most
# of them, and it is why English is the only entry with two forms.
PRONOUNS = {
    "en": {"f": ("she", "her"), "m": ("he", "his")},
    "pt": {"f": ("ela", None), "m": ("ele", None)},
    "es": {"f": ("ella", None), "m": (E_ACUTE + "l", None)},
    "fr": {"f": ("elle", None), "m": ("il", None)},
    "it": {"f": ("lei", None), "m": ("lui", None)},
    "de": {"f": ("sie", None), "m": ("er", None)},
}

# What the text has to say for this module to believe it knows someone's
# gender. Read out of the captions, never inferred from the name.
#
# Deliberately not the subject pronouns alone: "her mother" and "his aunt" are
# the forms that actually appear near a person being written about, and a
# subject pronoun is often the very thing that has been left out.
GENDER_MARKERS = {
    "en": {"f": ("she", "her", "hers"), "m": ("he", "him", "his")},
    "pt": {"f": ("ela", "dela"), "m": ("ele", "dele")},
    "es": {"f": ("ella",), "m": (E_ACUTE + "l",)},
    "fr": {"f": ("elle",), "m": ("il", "lui")},
    "it": {"f": ("lei",), "m": ("lui",)},
    "de": {"f": ("sie", "ihre", "ihr"), "m": ("er", "seine", "sein")},
}

# The articles a role has to arrive with to be taken. Requiring one is what
# keeps this from lifting a bare noun and having to invent the article itself
# - "singer announced" is not a sentence in any of these languages.
ARTICLES = {
    "en": ("the", "a", "an"),
    "pt": ("o", "a", "os", "as", "um", "uma"),
    "es": ("el", "la", "los", "las", "un", "una"),
    "fr": ("le", "la", "les", "un", "une"),
    "it": ("il", "lo", "la", "i", "gli", "le", "un", "una"),
    "de": ("der", "die", "das", "ein", "eine"),
}

# How many words after the article a role may run to.
#
# Two, which takes "the singer" and "the country star" and stops before it
# takes half a clause. A caption whose subject is six words long has not been
# shortened.
ROLE_MAX_WORDS = 2

# A word that starts with a capital. The range covers the accented capitals a
# Latin-script name can begin with; anything outside it (a name in a script
# with no case at all) simply never matches, and this module then finds no
# people and changes nothing.
CAPITALISED = re.compile("[A-Z" + A_GRAVE + "-" + U_DIAERESIS + r"][^\W\d_]*$",
                         re.UNICODE)

# The possessive marker English puts on the name itself, straight or curly.
POSSESSIVE = ("'s", RIGHT_QUOTE + "s")


class Person:
    """One person as the captions refer to them, and every way they may be named."""

    def __init__(self, full):
        self.full = full
        # Every spelling of this person the captions actually use, which is
        # not one spelling. A transcript says "Dolly Parton" in most lines and
        # "Dolly Rebecca Parton" in the one that gives her full name, and both
        # have to be recognised at the start of a caption. Keeping only the
        # longest was the first version, and it silently switched the whole
        # pass off on real data: every caption opening "Dolly Parton" stopped
        # matching the person now recorded as "Dolly Rebecca Parton", so
        # nothing was ever a repeat and nothing was ever varied.
        self.spellings = {full}
        parts = full.split()
        self.given = parts[0] if len(parts) > 1 else None
        self.surname = parts[-1] if len(parts) > 1 else None
        self.role = None      # filled in from an appositive, if the text has one
        self.gender = None    # filled in from the words around them, if the text says

    @property
    def key(self):
        """
        What makes two spellings the same person: the first name and the last.

        Both, and not the surname alone, which is what this used to be. A story
        about a family is full of people who share one - "Dolly Parton" and
        "Stella Parton" are sisters - and keyed by surname they were ONE
        person, so a line opening with Stella straight after a line opening
        with Dolly was a "repeat", and was varied into "She" or "Dolly": a
        caption putting one sister's words in the other's mouth. The first and
        last word together still merge the spellings that really are one
        person, "Dolly Parton" and "Dolly Rebecca Parton".
        """
        return " ".join(filter(None, (self.given, self.surname))).lower() or self.full.lower()


def _tokens(text):
    return text.split()


def _names_in(text):
    """
    Every run of two or more capitalised words in `text`.

    Two is the floor and it is what keeps sentence-initial words out: "The
    world found out about..." opens with a capital, and "world" does not
    follow it in capitals, so the run is one word long and is not a name.

    A name of a single word is therefore missed. That is the right trade at
    this end - the cost of missing one is a caption that keeps repeating a
    name, and the cost of a false positive is a caption whose first word gets
    replaced by a pronoun for no reason.
    """
    found = []
    run = []
    for token in _tokens(text):
        bare = token.strip(",.;:!?\"'()[]")
        for marker in POSSESSIVE:
            if bare.endswith(marker):
                bare = bare[:-len(marker)]
                break
        if bare and CAPITALISED.match(bare):
            run.append(bare)
            continue
        if len(run) > 1:
            found.append(" ".join(run))
        run = []
    if len(run) > 1:
        found.append(" ".join(run))
    return found


def _people(texts):
    """
    The people the lines talk about, keyed by first and last name (see
    Person.key).

    Spellings are merged rather than counted separately: "Dolly Parton" and
    "Dolly Rebecca Parton" are one person, and the longest spelling seen is
    kept as the full name because that is the one worth using on first
    mention.
    """
    people = {}
    for text in texts:
        for name in _names_in(text):
            person = Person(name)
            existing = people.get(person.key)
            if existing is None:
                people[person.key] = person
                continue
            existing.spellings.add(name)
            if len(name.split()) > len(existing.full.split()):
                # The longest spelling is what a first mention is worth
                # writing out as; every spelling stays in `spellings` so that
                # a caption opening with a shorter one is still recognised.
                existing.full = name
    return people


def _find_role(person, texts, articles):
    """
    A role for `person`, lifted from an appositive the text actually contains.

    Looks for the name, a comma, an article, and one or two lower-case words:
    "Dolly Parton, the singer, said..." gives "the singer". The article has to
    be there — see ARTICLES — and the words after it have to be lower case, so
    "Brian Siever, Dolly Parton's nephew" does not come back as a role made
    out of somebody else's name.
    """
    if not articles:
        return None
    pattern = re.compile(
        re.escape(person.full) + r"\s*,\s*(" + "|".join(articles) + r")\s+"
        + r"((?:[^\W\d_]+)(?:\s+[^\W\d_]+){0,%d})" % (ROLE_MAX_WORDS - 1),
        re.IGNORECASE | re.UNICODE)
    for text in texts:
        match = pattern.search(text)
        if not match:
            continue
        words = match.group(2).split()
        if any(CAPITALISED.match(w) for w in words):
            continue
        return "%s %s" % (match.group(1).lower(), " ".join(words))
    return None


def _find_gender(person, texts, markers, by_surname=True):
    """
    Which pronouns the text uses about `person`, or None if it never says.

    Counted over the lines that mention them, which is a blunt instrument and
    knowingly so: a line naming two people of different genders votes for
    both. The tie it produces answers None, and None means no pronoun form -
    the rotation just has one fewer option.

    A line mentions them if it carries one of their spellings, or their
    surname alone when `by_surname` - which the caller turns off when somebody
    else in the account shares it, since then "Parton said" is not evidence
    about either of them.
    """
    if not markers:
        return None
    names = [spelling.lower() for spelling in person.spellings]
    if by_surname and person.surname:
        names.append(person.surname.lower())
    score = {"f": 0, "m": 0}
    for text in texts:
        lowered = text.lower()
        if not any(name in lowered for name in names):
            continue
        words = {w.lower().strip(",.;:!?\"'()") for w in _tokens(text)}
        for gender, forms in markers.items():
            score[gender] += len(words & set(forms))
    if score["f"] == score["m"]:
        return None
    return "f" if score["f"] > score["m"] else "m"


def _opening(text, people):
    """
    The person a caption opens by naming, the exact text of that opening, and
    whether it was possessive - or None if it does not open with a name.

    Longest match wins, so "Dolly Rebecca Parton" is not read as "Dolly
    Rebecca" plus a stray word.
    """
    stripped = text.lstrip()
    best = None
    for person in people.values():
        for spelling in person.spellings:
            if not stripped.lower().startswith(spelling.lower()):
                continue
            rest = stripped[len(spelling):]
            possessive = any(rest.startswith(m) for m in POSSESSIVE)
            length = len(spelling) + (2 if possessive else 0)
            # It has to be the whole name, not a prefix of a longer word.
            if not possessive and rest[:1] and (rest[:1].isalnum() or rest[:1] == "-"):
                continue
            if best is None or length > best[2]:
                best = (person, possessive, length)
    return best


def _forms(person, possessive, pronouns, pronoun_ok=True, surname_ok=True, given_ok=True):
    """
    The ways of naming `person` that this text supports, in rotation order.

    The order an account is written in, and the one the module header gives:
    a pronoun, the role, then the two slices of the name. Every one of them
    may be absent, and each for the same reason - it would not say WHO:

      * the pronoun, with no evidence of gender, or when `pronoun_ok` is False
        because the line before named somebody else (see vary);
      * the role, with no appositive to lift one from;
      * the surname or the given name, when somebody else in the account has
        the same one. "Parton said" after a line about Dolly, in a story that
        also has her sister Stella Parton in it, could be either of them.
    """
    forms = []
    pronoun = (pronouns or {}).get(person.gender) if pronoun_ok else None
    if pronoun:
        subject, owned = pronoun
        word = owned if possessive else subject
        if word:
            # A pronoun replaces the possessive marker rather than taking one:
            # "her family", never "her's family".
            forms.append((word, True))
    if person.role:
        forms.append(person.role)
    if person.surname and person.given:
        if surname_ok:
            forms.append(person.surname)
        if given_ok:
            forms.append(person.given)
    return forms


def vary(lines, language=None):
    """
    The lines of the account with repeated openings varied, in order.

    A new list, never the one handed in edited in place: the caller keeps its
    own, and narrator compares the two to know which lines this changed.

    Plain text in and plain text out. This module decides how a person is
    named and nothing else - it never sees where a line will be set or when
    its words were spoken, and is built so that it cannot.
    """
    if not lines:
        return list(lines)

    texts = list(lines)
    people = _people(texts)
    if not people:
        return texts

    # How many people in the account answer to each surname and each given
    # name - a form two of them share names neither (see _forms).
    surnames, givens = {}, {}
    for person in people.values():
        for table, part in ((surnames, person.surname), (givens, person.given)):
            if part:
                table[part.lower()] = table.get(part.lower(), 0) + 1

    def unique(table, part):
        return bool(part) and table.get(part.lower()) == 1

    pronouns = PRONOUNS.get(language)
    for person in people.values():
        person.role = _find_role(person, texts, ARTICLES.get(language))
        person.gender = _find_gender(person, texts, GENDER_MARKERS.get(language),
                                     by_surname=unique(surnames, person.surname))

    out = []
    previous_key = None
    previous_line = ""
    turn = 0
    for line in texts:
        found = _opening(line, people)
        before, previous_line = previous_line, line
        if found is None:
            # Somebody else, or nobody, is the subject: the next mention of
            # the person before them starts the count again, because a
            # pronoun is only unambiguous while nobody has intervened.
            previous_key = None
            out.append(line)
            continue

        person, possessive, length = found
        if person.key != previous_key:
            previous_key, turn = person.key, 0
            out.append(line)
            continue

        # The line before opened with this person; if it ALSO named somebody
        # else, "she" could be either of them, and the pronoun sits this one
        # out (see the module header). Any other name counts, a place as much
        # as a person: the cost of being wrong that way is a surname where a
        # pronoun would have done.
        opening = before.lstrip()
        others = [name for name in _names_in(before) if not opening.startswith(name)]
        forms = _forms(person, possessive, pronouns, pronoun_ok=not others,
                       surname_ok=unique(surnames, person.surname),
                       given_ok=unique(givens, person.given))
        if not forms:
            out.append(line)
            continue

        form = forms[turn % len(forms)]
        turn += 1
        replacement, drop_possessive = form if isinstance(form, tuple) else (form, False)
        if possessive and not drop_possessive:
            replacement += "'s"

        stripped = line.lstrip()
        out.append((replacement[0].upper() + replacement[1:]) + stripped[length:])

    return out
