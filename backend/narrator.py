"""
narrator.py - the transcript, rewritten as reported information.

transcriber.py ends with an admission and an invitation. It says its sentence
assembly decides whether a line is COMPLETE, not whether it is INFORMATIVE,
that judging the second thing is a language model's job, and that the seam a
language model would be dropped into is the list of Sentence it hands back.
This is that model, dropped into that seam.

What it is for is one specific failure of raw transcription as caption. These
videos are cut from news packages, and a news package is half narration and
half interview. The narration already reads as information. The interview does
not: it is somebody talking about themselves, in the first person, to a
reporter who is not on screen. Laid on a still as a caption it reads as a
caption of nobody -

    "I am Brian, Dolly's son."
    "...and that is how the world found out she had passed."

- two lines a viewer scrolling past has no way to place. The same two facts
written as report are one line that stands on its own:

    "The world learned of her death through a message from her son Brian."

So this module rewrites first-person speech into third-person report, and it
does it with the whole transcript in view, because that is the only way the
second line above can know who "I" was. The identity is established in one
sentence and needed in another.

## What it may and may not do

It rewrites. It does not research, and it does not embellish. Every name,
number, relationship and event in the output has to be somewhere in the input,
because the alternative is a caption asserting something nobody said over a
photograph of a real person. The prompt says so and the validation below
cannot check it - that is worth stating plainly rather than implying the guard
is stronger than it is.

## Why the timestamps cannot drift

A caption is only right because it is the words that were being said when that
frame was on screen (see transcriber.captions_for). A rewrite that loses the
timing does not produce a worse caption, it produces a caption of a different
moment - which is invisible in review, because every line still reads well.

One thing downstream of here does loosen that, and it is worth knowing about
before reading the contract below as stronger than it is. A sentence that
several frames land on is SPLIT between them rather than repeated on all of
them (transcriber.captions_for), so a frame in the middle of one carries the
middle of it. The sentence still belongs to the moment; what is no longer true
is that the frame carries the whole of what was said at it. That is a decision
about repetition on the grid, taken where the grid is known — and it is
exactly why it is not taken here: this module cannot see how many frames will
land on a line it is writing.

So the model is never asked for timings. It is asked for TEXT KEYED TO THE IDS
it was given, and the spans come from the sentences that were already there.
The contract it works under is deliberately narrow:

  * it may rewrite the text at an id.
  * it may not reorder, invent an id, drop one, or overlap two runs.

Nor is it asked WHERE a caption should begin. The transcript is gathered into
thoughts before the model sees it — an interview turn and the line that
completes it become one item, one caption — so each id it is handed is
already a whole unit and its only job is to say that unit as report. That was
not the first design: the model was offered a syntax for joining ids itself,
and used it zero times in sixty-three sentences, then zero times again when
told joining was the expected move. See the grouping block below.

Anything it returns that breaks that is not repaired and not retried: the ids
involved keep their ORIGINAL sentence. So the worst case of a bad generation is
the transcript as spoken on those frames, which is a caption that is merely
plain rather than one that is wrong. There is no failure mode in here that
produces a confident line under the wrong picture.

## It is written for a grid of a known size

How many captions the account comes out in is not the transcript's decision,
it is the run's: the rewrite is told how many stills are about to be fetched
and writes that many lines (see `_group`). Forty stills of a twenty-minute
video is a caption per fifteen seconds of speech; two hundred of the same
video is one per three, out of the same words.

Which makes the frame-count slider a control over the WRITING as much as over
the pictures, and that is worth saying out loud because nothing in the
interface says it. Left to the transcript, a long video makes far more
thoughts than a short run has frames, and the ones that miss out are not
reported anywhere - they simply never reach a still, and the story on the grid
has holes in it that the output cannot show.

## Never fatal, for the reason transcriber.py is never fatal

A frame fetch that produced forty good stills has done its job. A missing
model, a machine that has never downloaded the weights, a generation that
comes back as prose instead of the format asked for - each of those returns
the sentences it was given, unchanged, and the run carries on as the channel
that does not ask for a rewrite does.

## The language is still the video's

Same rule as the transcript it rewrites, and for the same reason: these are
the video's own words, reported. A Portuguese interview reported in English
would be a caption of a translation nobody asked for. The model is told to
answer in the language it was given. Every string in THIS file, and every
label the user reads around the caption, is English.
"""

import logging
import re
import threading
from pathlib import Path

import reference
from transcriber import (CAPTION_CHARS, CAPTION_CHARS_MAX, DANGLING, MAX_CHARS,
                         MIN_WORDS, TERMINAL, Sentence, split_caption)

# ── Grouping ──────────────────────────────────────────────────────────────
#
# What the model is handed is not one transcript sentence per caption. It is
# one THOUGHT per caption, and the difference is the whole reason a caption
# can read as information.
#
# The case this exists for, from a real transcript:
#
#     "...My name is Brian Siever, son of Cassie Parton"
#     "and I'm representing the Parton and Owen's family today"
#     "announcing the passing of my Aunt Dolly, Rebecca Parton Dean."
#
# Three captions, none of which is news. One caption, and it is the story.
#
# Asking the model to make that join does not work. It is offered the syntax
# and, measured on this transcript, used it zero times in sixty-three
# sentences; told in a rewritten prompt that joining was the normal move, with
# a worked example, it used it zero times again. So the join is taken away
# from it and done here, where it is arithmetic.
#
# The rule is not new. It is transcriber._merge_continuations, run a second
# time with the brake off. That pass already wanted to join the first two
# lines above — the pause between them is 0.2s — and refused because the
# result was 21 words against a MAX_WORDS of 18. The brake is right there and
# wrong here: what has to fit in four lines of Arial Black is the model's
# ANSWER, not the material it reads. Grouping is what condensing needs.

# How far apart two sentences may be and still be one thought.
#
# Larger than transcriber.PAUSE_HARD_S (1.1s), which separates a hesitation
# inside a sentence from a boundary between two. This is a boundary between
# two by assumption, so the question is different: is the speaker still on the
# same thought. Past two seconds the video has almost certainly cut, and one
# caption stretched across a cut sits over pictures from both sides of it.
GROUP_MAX_GAP_S = 2.0

# ...and how much may end up in one group.
#
# Four sentences, because five is a paragraph and no single line can honestly
# report a paragraph. The character cap is three times a caption's own budget:
# the output condenses, so the input is allowed to be longer than it, but a
# group the model would have to compress five-to-one comes back having dropped
# whatever it decided mattered least, and that decision is not one this can
# check.
GROUP_MAX_SENTENCES = 4
GROUP_MAX_CHARS = MAX_CHARS * 3

# ...and how much may end up in one group when the GRID is what decides how
# many groups there are (see the second pass in _group).
#
# Both caps above are quality guards: they say how much material one caption
# may honestly report. This is not one. When a run asks for forty stills of a
# twenty-minute video it has asked for forty captions, and forty captions of
# twenty minutes IS five-to-one compression — the decision was made at the
# frame-count slider, and refusing it here would not undo it, it would leave
# the story with holes in it instead. The extra sentences would simply never
# reach a frame.
#
# So this bounds the PROMPT rather than the reporting: twelve captions' worth
# of speech is far past the point where condensing to one line is useful, and
# it is there to stop a pathological target (one frame of a feature film)
# building a single item nothing can generate from. A run that cannot reach
# its target inside it says so in the log and comes back with more captions
# than frames, which costs the last of them their place on the grid.
GROUP_TARGET_MAX_CHARS = CAPTION_CHARS * 12

# ...and the least a piece of a divided sentence may be, in words and in
# characters, for the division to be worth making.
#
# Well above transcriber.MIN_WORDS, which is the floor for a run to be a
# SENTENCE. This is a higher bar because it answers a harder question: not
# "can this stand as a line" but "is there a second thought in here worth its
# own photograph". Measured at MIN_WORDS it was not: "My name is Brian
# Siever, son of Cassie Parton" divided into two halves of five words, and the
# model, handed half a self-introduction with nothing in it to report, echoed
# it back — putting a first-person fragment on a still, which is the one thing
# this module exists to prevent.
#
# At eight words and forty characters a half is a clause with a fact in it,
# which is something a model can report. Both numbers were measured against
# the two cases that matter: "My name is Brian Siever, son of Cassie Parton"
# stays whole (neither half reaches forty characters), and "...songs in her
# lifetime, won eleven Grammy awards, and sold over one hundred million
# records" divides at its first comma into two halves that each say something.
#
# The cost is that fewer sentences divide, so a short video leaves more of a
# long grid blank — and a blank still the user types into is a better outcome
# than a caption in the wrong voice.
SPLIT_MIN_WORDS = 8
SPLIT_MIN_CHARS = 40

# Where a spoken sentence may be divided: the speaker's own internal marks.
# The terminal ones are not here, because a run that still contains one was
# never a single sentence to begin with.
SPLIT_BREAK = re.compile(r"[,;:\u2014\u2013](?=\s)")

# What to call the language in the prompt, given the ISO code Whisper answers.
#
# The NAME and not the code, because "Write your answer in Portuguese" is an
# instruction a model has seen a hundred thousand times and "Write your answer
# in pt" is not. An unlisted code falls back to the code itself, which is
# weaker but still better than the nothing this used to say.
#
# The list is short on purpose: these are the languages this app's channels
# are actually cut in, plus the neighbours a European news package tends to
# carry an interview in. Adding one is a line.
LANGUAGE_NAMES = {
    "en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French",
    "it": "Italian", "de": "German", "nl": "Dutch", "ca": "Catalan",
    "gl": "Galician", "ro": "Romanian", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "tr": "Turkish",
    "ru": "Russian", "uk": "Ukrainian", "ar": "Arabic", "he": "Hebrew",
    "hi": "Hindi", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
}

# Words that give away a line which has drifted into English.
#
# Function words only, and only ones that are not also words of the languages
# this is guarding — no Romance language spells anything "the", "and", "of" or
# "was". A content word would be no good here: a Portuguese caption may well
# quote an English song title, and "I Will Always Love You" must not be read
# as a translation.
ENGLISH_MARKERS = frozenset("""
    the and of was were with from that this these those been being are is
    his her their they them there which while after before through into
""".split())

# How many DISTINCT markers make a line English rather than a line with an
# English name in it.
#
# Two, and the asymmetry is deliberate. A false positive costs that frame the
# spoken sentence instead of the rewritten one — the wrong language is the one
# failure this guard exists to prevent, and it cannot be the price of avoiding
# it. A false negative is an English caption on a Portuguese video, which is
# the thing itself. So the guard leans towards rejecting.
ENGLISH_MARKERS_MIN = 2

# How much of a caption's wording may already have been in the caption before
# it before the two are saying the same thing.
#
# Three quarters, measured against the shorter of the pair — see _repeats. Set
# lower, two captions about the same person doing different things start
# colliding on the name alone; set higher, a line that only swapped its last
# three words for three others gets through, and that is exactly the padding
# this is here to stop.
REPEAT_MAX_OVERLAP = 0.75

# ...and the length below which two captions are not compared at all.
#
# At four distinct words "The funeral is today" and "The funeral is tomorrow"
# are three-quarters alike and say opposite things, so the overlap measure
# stops meaning anything before it stops being computable. Five is where a
# caption has enough words for the ones it shares to be evidence.
REPEAT_MIN_WORDS = 5

# Splits a caption into lowercase words, apostrophes kept inside them so
# "l'accident" does not become a bare "l" and an "accident".
WORDS = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)

# Where a caption may be cut short — see _fit.
#
# The trailing lookahead is the whole of it, and it was put there by a failing
# test rather than by foresight. Without it, the last punctuation mark inside
# the budget of "...more than 3,000 songs written" is the thousands separator,
# and the caption comes out ending "...more than 3". A clause boundary is a
# mark with a SPACE after it; a comma inside a number, a decimal point and the
# full stop in "9 to 5." are not places a sentence comes apart.
CLAUSE_END = re.compile(r"[,;:.!?—–](?=\s|$)")

log = logging.getLogger("uvicorn.error")  # see the note in face_restorer.py

# The model, by the name huggingface_hub resolves, already converted to the
# CTranslate2 format and quantised to int8.
#
# Llama rather than the obvious current answer, and the reason is a pin two
# modules away. Qwen2 support landed in CTranslate2 4.6.0; this project pins
# ctranslate2==4.5.0 and the comment in requirements.txt explains why in
# detail - the resolver's own choice crashes the machine with an access
# violation rather than an exception. Llama has been supported since 4.0, so
# it is the family that runs on the engine this app already ships. Do not
# raise the pin to reach a newer model without reading that comment first.
#
# 3B rather than 1B because the job is fidelity, not fluency: the 1B writes
# perfectly readable sentences and loses a name out of the middle of one. 3B
# at int8 is about 3.2 GB on disk. Dropping to
# "jncraton/Llama-3.2-1B-Instruct-ct2-int8" is a one-line change here and
# nothing else in this file cares, if that trade is ever worth making.
#
# Llama 3.2 lists Portuguese among the languages it officially supports, which
# is not incidental - the transcripts this rewrites are routinely Portuguese,
# and a model that merely tolerates a language will quietly translate it.
MODEL_REPO = "jncraton/Llama-3.2-3B-Instruct-ct2-int8"

# Beside the app's own code, alongside Whisper and U2-Net and for the reason
# background_remover.MODEL_DIR gives: an uninstall that deletes the install
# folder should take everything it downloaded with it and nothing else.
# `backend/models` is named in updater.KEEP_ACROSS_UPDATES, so a patch that
# replaces app/ wholesale does not throw 3 GB away and make the user fetch it
# a second time.
MODEL_DIR = Path(__file__).resolve().parent / "models" / "narrator"

# CPU and int8, and this is not a fallback either - see the same decision in
# transcriber.py, which it inherits wholesale. The card is already
# oversubscribed between GFPGAN, LaMa and U2-Net, vram.py exists because they
# were exhausting it between them, and a capture run is where all three are
# used on every frame in the grid. This runs ONCE per video. It has no
# business in that queue.
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# How many sentences are rewritten in one generation.
#
# Not a context-window limit - Llama 3.2 has far more than this needs. It is a
# limit on how long one CPU generation runs before anything at all comes back,
# and on how much the model is holding in its head when it answers. Forty
# sentences is roughly four minutes of speech, which is long enough for a
# narrative to be consistent across it.
CHUNK_SENTENCES = 40

# ...and how many already-rewritten sentences are shown ahead of a chunk as
# read-only context.
#
# Without this, sentence 41 has no idea who has been speaking, and the chunk
# boundary is visible in the output as the exact point where a name turns back
# into a pronoun. They are shown as context and are NOT reproduced in the
# answer - see the prompt, and the parser, which ignores an id outside the
# chunk it asked about.
CONTEXT_SENTENCES = 8

# What a rewritten line may weigh before it stops being settable.
#
# transcriber.CAPTION_CHARS_MAX, and the number is not this module's to pick:
# it is a fact about the GRID. A caption inside CAPTION_CHARS is set on one
# still and a longer one is split across two consecutive stills, so the
# ceiling here is exactly "two frames' worth" and a line past it is a line
# nothing downstream has anywhere to put (see transcriber.captions_for).
#
# The model is told CAPTION_CHARS — the length to write TO — rather than this
# one. Told the ceiling, it writes to the ceiling: every line came back near
# 160 and every one of them then took two frames, which turns the two-frame
# case from the exception it is meant to be into what the run does. Told 80,
# it writes 80 and the long ones are the thoughts that genuinely need it.
#
# The parser enforces the ceiling, and a line over it is CUT rather than
# dropped (see _fit). A line that cannot be cut keeps the original, which was
# inside transcriber.MAX_CHARS by construction.
MAX_CHARS_REWRITTEN = CAPTION_CHARS_MAX

# Tokens one generation may produce. A chunk of 40 sentences answers in
# roughly 40 short lines, and a line is around 30 tokens with its id - so this
# is about twice what a well-behaved answer needs, which is the room a model
# needs to finish a sentence rather than the room to write an essay.
MAX_GENERATION_TOKENS = 2048

# Greedy, stated rather than inherited.
#
# These two are also CTranslate2's own defaults, which is precisely why they
# are written down: the task has one right answer and one wrong one, sampling
# only moves probability towards paraphrase, and a caption that comes out
# differently on two runs of the same video is a caption nobody can review. A
# reader changing that should have to mean it.
#
# Temperature is deliberately NOT among them, and must not be added. With
# sampling_topk at 1 there is a single candidate to pick from, so a
# temperature reshapes a distribution of one - a no-op at best, and 0.0 is a
# division waiting to happen inside the kernel rather than the "be
# deterministic" it looks like.
BEAM_SIZE = 1
SAMPLING_TOPK = 1

# One line of the model's answer: an id, or a run of consecutive ids, then the
# rewritten sentence. "7> ..." and "7-9> ..." are the only two shapes accepted.
#
# A bare number and an angle bracket rather than JSON, and that is a decision
# about failure rather than about taste. A model that loses its way inside
# JSON produces a document that will not parse at all, and the whole chunk is
# lost with it. A model that loses its way here produces some lines that match
# this and some that do not, and every line that matches is still usable - the
# ones that do not simply keep their original sentence.
ANSWER_LINE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*>\s*(.+?)\s*$")

# Leading list furniture a model adds when it is being helpful. Stripped
# rather than rejected, because the sentence after it is usually right.
BULLET = re.compile("^[-*•]\\s+")

# The chat format Llama 3.1 and 3.2 are trained on, written out here rather
# than applied by transformers.
#
# `transformers` is not a dependency of this project, and adding it to reach
# one string-formatting function would pull torch's whole model zoo in behind
# it - see requirements.txt, which is unusually careful about what a new
# package drags along. `tokenizers` alone is already installed (faster-whisper
# brings it) and it reads the tokenizer.json sitting in the same repo as the
# weights, so the only thing missing is the template, and the template is
# this.
PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Where the model is told to stop. Llama 3.2 ends a turn with this; without it
# the generator runs to MAX_GENERATION_TOKENS every time and the answer has a
# second, imagined conversation stapled to the end of it.
END_TOKEN = "<|eot_id|>"

# What this prompt can and cannot get out of a 3B, measured rather than
# assumed. Worth reading before rewording any of it, because the pattern is
# consistent and it is not the one prompt-writing intuition predicts.
#
# It obeys a concrete, local instruction. "Write in Portuguese" turned a
# transcript that came back 100% translated into one that came back 100% in
# its own language, on the first try.
#
# It ignores stylistic discipline held across a list. Three separate rules
# have been written and measured here, and none moved its number:
#
#   * joining lines that belong together - offered the syntax, used it 0 times
#     in 63 sentences; promoted to rule 3 with a worked example, 0 again. Now
#     done in _group before the model sees anything.
#   * not repeating itself - still produced three consecutive captions that
#     were one sentence with its last three words swapped. Now caught by
#     _repeats in _apply.
#   * not opening every line with the same name - rule 5 below, which is the
#     second wording tried and the first in capitals. Still 7 of 17 captions
#     opening with "Dolly Parton", six of them consecutive.
#
# So a rule here is worth writing when it is a fact the model needs (the
# language, the budget, the answer format) and worth being sceptical of when
# it asks for restraint across many lines. That kind wants machinery, or a
# bigger model. Rule 5 is kept because it costs nothing and is the honest
# statement of the intent; it is not what enforces anything.
SYSTEM = (
    "You rewrite the transcript of a news video into caption text.\n"
    "\n"
    "The transcript mixes a narrator with interviews. The interviews are in "
    "the first person: people talking about themselves. Your job is to turn "
    "all of it into third-person reporting, so that every line reads as "
    "information about what happened rather than as somebody speaking.\n"
    "\n"
    "Rules, in order of importance:\n"
    "1. WRITE IN {language}. The transcript is in {language} and every line "
    "you write must be in {language} too. These captions are laid back over "
    "the video the words were spoken in, so a caption in another language is "
    "captioning something nobody said - it is wrong no matter how well it "
    "reads. You are not translating. These instructions are in English; your "
    "answer is not.\n"
    "2. Never add a fact. Every name, number, relationship, place and event in "
    "your answer must already be somewhere in the transcript you were given. "
    "If a speaker is never named, describe them by their role instead of "
    "inventing a name.\n"
    "3. Use the whole transcript to work out who is speaking. A person who "
    "introduces themselves early is still that person later, and a line that "
    "says 'I' should name them if the transcript ever named them.\n"
    "4. Each numbered item may be several sentences of speech that belong "
    "together. Answer it with ONE sentence reporting what the whole item "
    "says. One line out per item in, in the same order.\n"
    "4b. THE ITEMS ARE CONSECUTIVE MOMENTS OF ONE VIDEO, in the order they "
    "happen, and your lines are read in that order - one per photograph, "
    "down a grid. So write them as ONE ACCOUNT: line by line the story moves "
    "forward, nothing in the middle is missing, and nothing is said twice. "
    "Before writing a line, read the item AFTER it - what this line has to do "
    "is get the reader from where they are to what comes next.\n"
    "4c. AND EVERY LINE MUST STILL NAME WHO IT IS ABOUT. The reader sees one "
    "photograph and one line at a time and cannot look back at the line "
    "before, so never open with 'he', 'she', 'they', 'this' or 'that' "
    "unless the same line has already said who it means, and name the person "
    "the first time each line mentions them.\n"
    "4d. Those two are not in conflict, and the difference is the kind of "
    "word you open with. A word that carries the story FORWARD is what you "
    "want - 'Within hours', 'By 1992', 'The response was', 'The family "
    "then' - because it moves the reader on and still says what it is about. "
    "A word that points BACKWARD is what you must not use, because it asks "
    "the reader for a line they cannot see. Reach for the first kind often: "
    "it is what makes twelve separate lines read as one story rather than as "
    "twelve facts in a row.\n"
    "5. NEVER BEGIN A LINE WITH A NAME THAT ALREADY OPENED THE LINE BEFORE "
    "IT. The context above these items shows what you have already written; "
    "anyone named there is known, and naming them again in full is repetition "
    "a reader sees immediately. Use the surname, or what the person is (the "
    "singer, her son), or start the sentence somewhere else entirely - with "
    "the year, the place, the thing that happened. Vary how lines open.\n"
    "6. Never write two lines that say the same thing. If an item adds "
    "nothing the line before it did not already say, report the part that IS "
    "new, and keep it short.\n"
    "7. WRITE IN BLOCKS OF {budget} CHARACTERS. A line is not a sentence you "
    "then measure - it is one screen of caption, and {budget} characters is "
    "what a screen holds. So choose how much of the item goes on this line by "
    "what will FIT on it, and finish the thought inside that. Shorter is "
    "better; an unfinished thought is not.\n"
    "7b. A thought that genuinely will not go into {budget} may run to "
    "{ceiling} and no further - and when it does, WRITE IT AS TWO COMPLETE "
    "SENTENCES, each one under {budget} characters and each ending in a full "
    "stop. A line that long is shown across two screens, split at the full "
    "stop between them, and the second screen is seen on its own: it must "
    "name who it is about rather than starting with 'and', 'who' or 'he'. "
    "One long sentence with a comma in the middle is the thing to avoid - "
    "cut in half it leaves a fragment on one still and a verb with no "
    "subject on the next.\n"
    "8. Report plainly. No commentary, no drama, and no hedging words unless "
    "the transcript hedged first.\n"
    "\n"
    "Answer format. One line per item, and nothing else - no preamble, no "
    "numbering of your own, no blank lines:\n"
    "  7> The rewritten sentence.\n"
    "Answer for every id you are given, once each, in order.\n"
    "\n"
    "Write every line in {language}."
)

# What the model is asked when a finished caption came back too long.
#
# A second pass, and it exists because the first one cannot be made to obey a
# length. Rule 7 states the budget plainly, in capitals, with the reason
# attached, and measured on a real video 22 of 37 captions still came back
# over it - several at half as long again. That is the same shape of failure
# as the three rules the note above SYSTEM describes: the model follows a
# concrete instruction about THIS line and ignores a discipline held across a
# list. So the discipline is taken away from it and asked as a concrete
# instruction instead - here is one line, it is too long, say it shorter -
# which is a question it answers well.
#
# Shortening is also a genuinely easier job than reporting. The first pass has
# to decide what a piece of speech MEANS and who it is about; this one is
# handed a finished sentence and asked only to say it with fewer words, so a
# 3B model does it reliably where it drifts on the harder question.
#
# What it must not do is trim a sentence into a fragment - that is the failure
# `_fit` used to produce by cutting at a clause boundary, and it is worse than
# a long line, so it is rule 4 here and is checked on the way back out.
CONDENSE_SYSTEM = (
    "You shorten caption lines that came out too long to fit on screen.\n"
    "\n"
    "Rules, in order of importance:\n"
    "1. WRITE IN {language}. Every line you write must be in {language}, the "
    "same language as the line you were given. You are not translating.\n"
    "2. Say the same thing in under {budget} characters. That is the whole "
    "job: each numbered line below is a finished caption that does not fit, "
    "and your answer is that caption, shorter.\n"
    "3. Keep every name, number, date and place. What you drop is the words "
    "that are not carrying one - qualifiers, scene-setting, anything the "
    "sentence still says without them. Never add a fact that is not already "
    "in the line you were given.\n"
    "4. ANSWER WITH A COMPLETE SENTENCE. A line cut off in the middle of a "
    "thought is worse than a line that is too long, because it is shown on "
    "its own over a photograph and there is nothing after it. If you cannot "
    "say it in {budget} characters and still finish it, get as close as you "
    "can and finish it.\n"
    "5. Report plainly, in the third person, as the line already does. No "
    "commentary and no drama.\n"
    "\n"
    "Answer format. One line per item, and nothing else - no preamble, no "
    "numbering of your own, no blank lines:\n"
    "  7> The shorter sentence.\n"
    "Answer for every id you are given, once each, in order.\n"
    "\n"
    "Write every line in {language}."
)

# The same instruction again, last thing before the transcript itself.
#
# Not redundancy for its own sake. The rules above are English prose and they
# are long; by the time a 3B model reaches the words it has to rewrite, the
# language rule is a hundred tokens behind it and the thing nearest to hand is
# an English sentence. Measured on real Portuguese and French transcripts,
# every line came back translated with the rule stated once. Restating it
# where the model is about to start writing is what stops that.
LANGUAGE_REMINDER = "Answer in {language}. Do not translate anything.\n\n"


_generator = None
_tokenizer = None
_load_failed = False
# Serialises the LOAD, not the generation, for the reason transcriber's lock
# gives: on an install that has yet to fetch the weights, two runs arriving
# together would otherwise start the same multi-gigabyte download into the
# same directory. Generation itself needs no lock - one video is processed at
# a time (see the process lock in api.py).
_load_lock = threading.Lock()


def _download() -> str | None:
    """
    The weights on this machine, fetching them once if they are not here yet.

    Through huggingface_hub rather than a plain URL, unlike U2-Net and the
    text detector, because this is a repository of eight files rather than one
    .onnx, and the hub client already handles the resume, the checksum and the
    half-written file those two each hand-roll. It is installed either way -
    faster-whisper brings it, and Whisper's own weights arrive through exactly
    this call.
    """
    from huggingface_hub import snapshot_download

    log.info("narration: downloading %s (~3 GB, once)", MODEL_REPO)
    return snapshot_download(MODEL_REPO, cache_dir=str(MODEL_DIR))


def _ensure_loaded(download: bool = True) -> bool:
    """
    Builds the generator once, downloading the weights if this machine has not
    got them yet. False means rewriting is unavailable and captions will be
    the transcript as spoken - which is a working channel, not a broken one.
    """
    global _generator, _tokenizer, _load_failed
    if _generator is not None:
        return True
    if _load_failed:
        return False

    with _load_lock:
        if _generator is not None:
            return True
        if _load_failed:
            return False
        try:
            import ctranslate2
            from tokenizers import Tokenizer
        except Exception as e:
            # Neither of these is a package this feature adds: ctranslate2 is
            # pinned in requirements.txt for Whisper, and tokenizers comes
            # with faster-whisper. Missing, this is an install that predates
            # the transcription work entirely, and naming the failed import is
            # the whole of the fix.
            log.warning("narration: %s - captions will not be rewritten", e)
            _load_failed = True
            return False

        # Nothing on disk and nothing allowed to arrive: report unavailable
        # rather than letting the hub client reach for the network on a caller
        # that asked it not to.
        local = list(MODEL_DIR.rglob("model.bin"))
        if not local and not download:
            return False

        try:
            path = str(local[0].parent) if local else _download()
            if path is None:
                _load_failed = True
                return False
            log.info("narration: loading %s (%s/%s)", MODEL_REPO, DEVICE, COMPUTE_TYPE)
            _generator = ctranslate2.Generator(path, device=DEVICE, compute_type=COMPUTE_TYPE)
            _tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
            return True
        except Exception as e:
            log.warning("narration: %s could not be loaded (%s)", MODEL_REPO, e)
            _generator = None
            _load_failed = True
            return False


def preload(download: bool = True) -> bool:
    """
    Build the model now rather than inside the run that first needs it. Called
    at server startup alongside the others; returns whether rewriting is
    available at all.
    """
    return _ensure_loaded(download)


def is_available() -> bool:
    """Whether a run can expect rewritten captions. False means the transcript as spoken."""
    return _generator is not None or not _load_failed


def loaded() -> bool:
    """
    Whether the model is in memory RIGHT NOW — which, after a rewrite has been
    attempted, is the same question as "was anything actually rewritten".

    `is_available` is the optimistic form and is the right one to ask before a
    run: it answers True while the weights have yet to be tried, because they
    usually work. This is the pessimistic form and is the right one to ask
    after, when a caller wants to report what happened rather than what was
    hoped for.

    The distinction earned its place. Every way this module declines to work -
    no weights on the machine, a load that threw, a generation that came back
    as prose - returns the sentences it was given, unchanged and in order,
    which is indistinguishable at the call site from a rewrite that ran and
    found nothing to change. A run that quietly captioned every frame with
    the transcript as spoken therefore looked exactly like a run that worked,
    and the only trace was one warning in a log nobody reads while a video is
    processing.
    """
    return _generator is not None


def _continues(previous: Sentence, following: Sentence) -> bool:
    """
    Whether `following` carries on the thought `previous` started.

    Two signals, and both are about SHAPE rather than meaning — the same
    discipline transcriber's own assembly holds itself to.

      * `previous` never terminated. No full stop, no question mark: the
        speaker stopped mid-thought and what follows finishes it.
      * `following` starts in lower case. Whisper capitalises what it hears as
        the start of a sentence, so a run that begins "and I'm representing"
        or "announcing the passing" is the model itself saying this was not a
        new sentence.

    The second test is deliberately about case and not about a list of
    conjunctions. A word list is a language list, and this has to work on
    every language a video might be shot in; capitalisation carries the same
    information in all of the cased ones. In a script with no case at all —
    Japanese, Chinese, Arabic — `islower()` is false for every character, the
    test simply never fires, and grouping falls back to the punctuation signal
    alone. That is a weaker pass, not a broken one.
    """
    if not TERMINAL.search(previous.text):
        return True
    head = following.text.lstrip()[:1]
    return bool(head) and head.islower()


def _group(sentences: list[Sentence], target: int | None = None) -> list[list[Sentence]]:
    """
    The sentences gathered into the thoughts they belong to, in order — as
    many of them as the grid has room for.

    Every sentence appears exactly once and none moves, so the groups tile the
    timeline the transcript already had. A sentence that continues nothing and
    is continued by nothing comes back as a group of one, which is what most
    narration is.

    `target` is how many captions the run can actually show: one per still it
    is about to fetch. It is not a nicety. The captions are matched to frames
    by TIME (see transcriber.captions_for), so a transcript that makes ninety
    thoughts on a run of forty stills does not produce ninety captions and
    forty frames — it produces forty frames carrying whichever forty thoughts
    happened to fall under them, and fifty that are never seen. The story on
    the grid then has holes in it that nothing in the output shows. Written to
    the grid instead, the same account comes out in forty lines that tile it.

    Which means the frame-count slider is a decision about the WRITING and not
    only about the pictures, and that is worth knowing before moving it: forty
    stills of a long video is a caption per fifteen seconds of speech, and two
    hundred is one per three.

    Two passes, and they answer different questions.

      * The first is about MEANING, and runs whatever the target is: a
        sentence that finishes the one before it belongs with it (see
        `_continues`). This is the pass that turns three lines of somebody
        introducing themselves into one piece of news.
      * The second is about the GRID, and runs only while there are still more
        groups than frames to put them on. It joins whichever neighbouring
        pair has the least silence between them, over and over, because the
        longest silences are where the video most likely cut and a caption
        stretched across a cut sits over pictures from both sides of it. The
        caps the first pass respects do not apply — see
        GROUP_TARGET_MAX_CHARS for why, and for the one bound that does.

    A target LARGER than the number of sentences runs a third pass, which is
    the second one in reverse: the transcript is coarser than the grid, so the
    longest sentences are DIVIDED at their own internal punctuation until
    there is a thought per still. That is still only the video's own words cut
    finer — nothing is invented — and it stops as soon as no sentence can be
    divided without leaving a fragment (see SPLIT_MIN_WORDS). A video with
    less to say than the run has frames for still comes back with blank stills
    at the end, and that is the honest answer: eight thoughts do not become
    forty captions.
    """
    if target is not None and target >= len(sentences):
        return _divide([[sentence] for sentence in sentences], target)

    groups: list[list[Sentence]] = []
    for sentence in sentences:
        current = groups[-1] if groups else None
        if (current
                and len(current) < GROUP_MAX_SENTENCES
                and sentence.start - current[-1].end < GROUP_MAX_GAP_S
                and len(" ".join(s.text for s in current)) + len(sentence.text) + 1 <= GROUP_MAX_CHARS
                and _continues(current[-1], sentence)):
            current.append(sentence)
        else:
            groups.append([sentence])

    if target is None or len(groups) <= target:
        return groups

    # The silence between each neighbouring pair, and the pair with the least
    # of it joined first. Recomputed rather than kept in a heap: this runs
    # once per video, over a few hundred groups at most, and a heap of stale
    # gaps around a list that is being mutated is the kind of code that is
    # wrong for a year before anybody notices.
    def joinable():
        best, least = None, float("inf")
        for i in range(len(groups) - 1):
            if (len(" ".join(s.text for s in groups[i]))
                    + len(" ".join(s.text for s in groups[i + 1])) + 1) > GROUP_TARGET_MAX_CHARS:
                continue
            gap = groups[i + 1][0].start - groups[i][-1].end
            if gap < least:
                best, least = i, gap
        return best

    while len(groups) > target:
        at = joinable()
        if at is None:
            log.info("narration: %d thoughts will not condense to %d frames — "
                     "the last of them have no still to appear on", len(groups), target)
            break
        groups[at] = groups[at] + groups[at + 1]
        del groups[at + 1]

    return groups


def _halve(sentence: Sentence) -> list[Sentence] | None:
    """
    One spoken sentence cut in two at its own punctuation, or None when it has
    nowhere to come apart.

    The cut goes at the internal mark nearest the middle - a comma, a
    semicolon, a dash - because that is where the speaker themselves paused,
    and both halves have to be at least SPLIT_MIN_WORDS long or the cut is
    refused. A sentence with no internal mark is not divided at all: guessing
    a break inside a clause is how a caption ends up opening with "and".

    The span is divided in proportion to the characters on each side. That is
    an estimate and it is allowed to be: what it decides is which still a
    caption lands NEAR, and both halves stay inside the seconds the whole
    sentence covered, so neither can drift onto a picture the sentence never
    described.
    """
    text = sentence.text
    middle = len(text) / 2
    for cut in sorted((m.end() for m in SPLIT_BREAK.finditer(text)),
                      key=lambda c: abs(c - middle)):
        head, tail = text[:cut].strip(), text[cut:].strip()
        if any(len(half) < SPLIT_MIN_CHARS
               or len(re.findall(r"\S+", half)) < SPLIT_MIN_WORDS
               for half in (head, tail)):
            continue
        span = sentence.end - sentence.start
        at = sentence.start + span * (len(head) / max(1, len(text)))
        return [Sentence(text=DANGLING.sub("", head), start=sentence.start, end=at),
                Sentence(text=tail, start=at, end=sentence.end)]
    return None


def _divide(groups: list[list[Sentence]], target: int) -> list[list[Sentence]]:
    """
    Groups cut finer until there is one per still, for a video whose
    transcript made fewer thoughts than the run has frames.

    The longest group goes first, every time, because the longest is both the
    one most likely to be holding two thoughts and the one whose halves are
    most likely to stand on their own. A group that will not divide is set
    aside rather than retried, so the loop ends when every remaining group has
    refused - which is what leaves a short video's grid with blank stills at
    the end rather than with forty fragments.

    Only ever divides a group of ONE. A group of several was assembled by the
    meaning pass, which had a reason for every sentence it joined, and taking
    that apart here to reach a number would be this function arguing with that
    one. It never comes up in practice either: a target above the sentence
    count means nothing was ever joined.
    """
    refused: set[int] = set()
    while len(groups) < target:
        widest, at = 0, None
        for i, group in enumerate(groups):
            if len(group) != 1 or id(group[0]) in refused:
                continue
            if len(group[0].text) > widest:
                widest, at = len(group[0].text), i
        if at is None:
            break
        halves = _halve(groups[at][0])
        if halves is None:
            refused.add(id(groups[at][0]))
            continue
        groups[at:at + 1] = [[halves[0]], [halves[1]]]
    return groups


def _block(group: list[Sentence]) -> Sentence:
    """
    A group written out as the single unit the model is asked about.

    A Sentence again, and that is what keeps everything downstream unchanged:
    the chunking, the prompt, the parser and the span arithmetic all go on
    working on a list of Sentence and none of them has to learn what a group
    is. The span is the group's own — from where the first started to where
    the last ended.
    """
    return Sentence(text=" ".join(s.text for s in group),
                    start=group[0].start, end=group[-1].end)


def _listing(sentences: list[Sentence], first: int, last: int) -> str:
    """The sentences from `first` to `last` as the numbered lines the model reads."""
    return "\n".join(f"{i}> {sentences[i].text}" for i in range(first, last))


def _generate(system: str, user: str) -> str:
    """
    One turn, as text. Empty for anything that did not work - the caller reads
    that as "keep the originals".

    CTranslate2 speaks in token STRINGS rather than ids, which is why the
    round trip goes through the tokenizer twice: encode to strings on the way
    in, look the strings back up to decode on the way out.
    """
    prompt = PROMPT.format(system=system, user=user)
    tokens = _tokenizer.encode(prompt, add_special_tokens=False).tokens
    try:
        result = _generator.generate_batch(
            [tokens],
            max_length=MAX_GENERATION_TOKENS,
            beam_size=BEAM_SIZE,
            sampling_topk=SAMPLING_TOPK,
            include_prompt_in_result=False,
            end_token=END_TOKEN,
        )
    except Exception as e:
        log.warning("narration: generation failed (%s)", e)
        return ""

    if not result or not result[0].sequences:
        return ""
    ids = [_tokenizer.token_to_id(t) for t in result[0].sequences[0]]
    return _tokenizer.decode([i for i in ids if i is not None])


def _drifted_to_english(text: str) -> bool:
    """
    Whether a line has been translated into English rather than rewritten.

    The one failure this cannot be allowed to ship. A caption is the video's
    own words put back on the video's own pictures; in another language it is
    a caption of something nobody said, and unlike every other way a rewrite
    can go wrong it does not LOOK wrong — it reads perfectly, and a reviewer
    skimming forty frames has no reason to stop on it.

    The prompt is what should prevent it, and mostly does. This is here
    because "mostly" is not a property worth relying on when the check is a
    set lookup: a 3B model handed a long English instruction and a Portuguese
    sentence answers in English far more readily than its own rules suggest,
    and that was measured rather than assumed.

    Only called when the video was NOT in English, so an English video's
    captions never come near it.
    """
    words = {w.lower() for w in WORDS.findall(text)}
    return len(words & ENGLISH_MARKERS) >= ENGLISH_MARKERS_MIN


def _fit(text: str) -> str:
    """
    A line brought inside the caption budget, or "" if it cannot be.

    Two budgets, because a caption may take two screens (CAPTION_CHARS_MAX)
    but only if it actually comes apart into two. A line over one screen is
    put through the same splitter the grid will use, and kept at that length
    only when it survives it — see transcriber.split_caption for what "comes
    apart" means and for the two frames of nonsense that taught it. A line
    that does not survive is treated exactly as an over-long one: cut back at
    the last clause boundary that fits ONE screen, since the second screen was
    never going to be usable.

    Rejecting an over-long answer outright was the first behaviour and it was
    measured to be the wrong one: every rejection in a real run was a length
    rejection, and the lines being thrown away were good. One of them was
    136 characters against a budget of 132 — a finished, accurate caption
    discarded over four characters, and the frame fell back to the first
    person sentence this whole module exists to get rid of.

    So an over-long line is CUT rather than dropped, at the last clause
    boundary that fits. That is transcriber._split_overlong's rule, and for
    its reason: the speaker's own punctuation is the only marked-up place a
    sentence comes apart without a reader noticing the seam. Cutting at a
    character count instead would end a caption in the middle of a name.

    What is lost is the tail, and that is the right thing to lose. A caption
    is a headline, not a summary: "...earning 11 Grammys and selling over 100
    million records" is the line, and the clause about song-writing that
    followed it is what a 540-pixel frame did not have room for.

    Only a line past CAPTION_CHARS_MAX with nowhere to cut comes back empty,
    and its group then falls back to the sentences as spoken. That last resort
    is deliberately hard to reach, and it was once far too easy: for a while
    any line over one screen that would not divide cleanly was rejected
    outright, which handed the frame back to the first-person speech this
    whole module exists to get rid of. A caption ten characters too long is a
    caption. "My name is Brian Siever, son of Cassie Parton" is not one, and
    no length rule is worth paying that for.
    """
    if len(text) <= CAPTION_CHARS:
        return text

    # Over one screen, and welcome to stay that way if the grid can actually
    # show it: two pieces out of the splitter means two stills.
    if len(text) <= CAPTION_CHARS_MAX and len(split_caption(text, CAPTION_CHARS)) == 2:
        return text

    # Between one screen and two, it is KEPT. It will either be split across
    # two stills or wrap on one, and both are better than what used to happen
    # here: the line was cut back at the last clause boundary under
    # CAPTION_CHARS, which sounds careful and in practice decapitates the
    # sentence. A clause boundary is not the end of a thought, it is the
    # middle of one, so what came back was
    #
    #     "Brian Siever, son of Cassie Parton and representing the Parton
    #      and Owen's"
    #     "The response to Dolly Parton's passing fell short of properly"
    #
    # — two captions with the predicate cut off, on two photographs, with no
    # way for a reader to finish either. Truncation earned its place when the
    # budget was a whole paragraph and a line reached it once in a while; at
    # eighty characters it fires on most lines the model writes, and the thing
    # it removes is the half that says what happened.
    if len(text) <= CAPTION_CHARS_MAX:
        return text

    # Past two screens it cannot be shown whole however forgiving the setting,
    # so the tail goes after all — at the last clause boundary that fits, and
    # only if enough of the sentence survives to still say something.
    #
    # DANGLING is transcriber's: a comma says "this goes on", which is true of
    # the sentence and false of the caption, since nothing follows it on that
    # frame. A full stop is left alone — that one ends it properly.
    cut = max((m.start() for m in CLAUSE_END.finditer(text)
               if m.start() < CAPTION_CHARS_MAX), default=-1)
    if cut < CAPTION_CHARS:
        return ""
    return DANGLING.sub("", text[:cut + 1]).strip()


def _repeats(text: str, previous: str) -> bool:
    """
    Whether a line says what the line before it already said.

    The failure it catches, from a real run, three consecutive captions:

        "...version of I Will Always Love You was one of the best-selling
         singles of all time."
        "...version of I Will Always Love You was a huge commercial success."
        "...version of I Will Always Love You was also a hit."

    A model with a numbered list to fill in and nothing new to say will
    paraphrase rather than leave a line out, and the shorter the new thought
    the more of the old line it reaches for to pad it. The prompt asks for
    something else and this is what happens anyway, which is the argument for
    checking rather than asking.

    Measured against the SHORTER of the two, not against the pair. A short
    line whose every word appears in a long one adds nothing to it — that is
    the third caption above — while comparing both ways would score that pair
    as only half alike and let it through.

    Very short lines are left alone entirely — see REPEAT_MIN_WORDS. "The
    funeral is today" and "The funeral is tomorrow" are three-quarters the
    same words and opposite in meaning, and no overlap threshold can tell
    those apart.
    """
    a = {w.lower() for w in WORDS.findall(text)}
    b = {w.lower() for w in WORDS.findall(previous)}
    shorter = a if len(a) <= len(b) else b
    if len(shorter) < REPEAT_MIN_WORDS:
        return False
    return len(a & b) / len(shorter) >= REPEAT_MAX_OVERLAP


def _parse(answer: str, first: int, last: int,
           guard_english: bool = False) -> dict[int, tuple[int, str]]:
    """
    The model's answer as a map from the FIRST id of each run to (last id,
    text) - and the whole of the contract's enforcement.

    Everything in here is a reason to discard a line rather than to repair
    one. A line that does not match the format, names an id outside the chunk,
    runs backwards, overlaps something already claimed, or comes out longer
    than a caption can be, is dropped; its ids then find no entry in this map
    and keep the sentence they already had. That is the design stated in the
    header: a bad line costs the plain original, never the wrong moment.

    `claimed` is what makes the no-overlap rule hold across the whole answer
    rather than line by line. A model that answers "7-9>" and then "8>" has
    contradicted itself about what sentence 8 is, and the second statement is
    not more true than the first - the first is kept because it came first,
    and the second is dropped.
    """
    runs: dict[int, tuple[int, str]] = {}
    claimed: set[int] = set()

    for line in answer.splitlines():
        match = ANSWER_LINE.match(BULLET.sub("", line.strip()))
        if not match:
            continue

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        text = match.group(3).strip()

        # In range, the right way round, and not something an earlier line has
        # already spoken for.
        if not (first <= start <= end < last) or any(i in claimed for i in range(start, end + 1)):
            continue
        text = _fit(text)
        if not text:
            continue
        # ...and the line that reads well and says the wrong thing. Dropped
        # like any other rejection, so those ids keep the sentence as spoken —
        # which is, at minimum, in the right language.
        if guard_english and _drifted_to_english(text):
            continue

        runs[start] = (end, text)
        claimed.update(range(start, end + 1))

    return runs


def _apply(groups: list[list[Sentence]], blocks: list[Sentence],
           runs: dict[int, tuple[int, str]], first: int, last: int,
           previous: str = "") -> list[Sentence]:
    """
    The chunk rebuilt from whatever the model got right, walked in id order so
    the output is in the video's order whatever order the answer arrived in.

    The span arithmetic is the whole reason the timestamps survive: a run's
    Sentence starts when its FIRST block started and ends when its LAST one
    ended, both read off the originals. Nothing in here reads a number the
    model produced.

    What a REJECTED group falls back to is the one thing grouping changed
    here, and it matters. Not the joined block — that was assembled to be read
    by a model and can be three sentences long, which is not a caption
    anybody can set. It falls back to the group's own sentences, put back
    individually, exactly as they came from the transcript. So a group whose
    rewrite is thrown away costs those frames the transcript as spoken, one
    caption per sentence and each inside the length budget by construction. The promise the module header makes survives grouping.

    The repetition check lives here rather than in _parse, and it has to. The
    question is whether a caption says what the caption BEFORE IT ON SCREEN
    said, and only this function knows what that was: _parse sees the model's
    answer, not the sentences a rejected group falls back to, so a rewritten
    line that echoed the fallback ahead of it went straight through. It did,
    on a real run — two neighbouring captions with identical wording. Here,
    `last_text` is whatever was actually emitted, rewritten or fallen back.
    """
    out: list[Sentence] = []
    last_text = previous
    i = first
    while i < last:
        run = runs.get(i)
        if run is not None:
            end, text = run
            if not (last_text and _repeats(text, last_text)):
                out.append(Sentence(text=text, start=blocks[i].start, end=blocks[end].end))
                last_text = text
                i = end + 1
                continue
        # Rejected by the parser, or a repeat of the line before it: the
        # group's own sentences, as spoken.
        out.extend(groups[i])
        last_text = groups[i][-1].text
        i += 1
    return out


def _needs_condensing(text: str) -> bool:
    """
    Whether a finished caption is too long for the grid to show.

    Over one screen is not automatically too long: a caption up to
    CAPTION_CHARS_MAX is allowed to take TWO stills, and one that divides
    cleanly into two readable halves is the sanctioned exception rather than a
    fault (see transcriber.split_caption). What this catches is the line that
    is over a screen and will not divide - the one that lands whole on a
    single card and wraps to eight lines of type inside a box built for four.
    """
    return (len(text) > CAPTION_CHARS
            and len(split_caption(text, CAPTION_CHARS)) == 1)


def _condense(texts: dict[int, str], language: str | None) -> dict[int, str]:
    """
    Over-long captions said again, shorter, keyed by the index they came from.

    Asked in one turn per chunk rather than one per line, for the reason the
    first pass is chunked: a generation costs about the same whether it
    answers one item or forty, and forty separate turns is forty times the
    wait for the same words.

    An answer is taken only if it is SHORTER than what it replaces and still
    inside the budget, and only if it did not drift into English on a video
    that was not (the same guard the first pass uses). Anything else leaves
    the long line alone: too long is a caption that overflows its card, and
    every way this could go wrong instead - a fragment, a translation, an
    invented fact - is a caption that is wrong.
    """
    if not texts:
        return {}

    named = LANGUAGE_NAMES.get(language, language) if language else None
    system = CONDENSE_SYSTEM.format(
        budget=CAPTION_CHARS,
        language=named or "the same language as the line you were given")
    reminder = LANGUAGE_REMINDER.format(language=named) if named else ""
    guard_english = bool(language) and language != "en"

    order = sorted(texts)
    shorter: dict[int, str] = {}
    for at in range(0, len(order), CHUNK_SENTENCES):
        batch = order[at:at + CHUNK_SENTENCES]
        user = (reminder
                + "Shorten these. Answer for every id:\n"
                + "\n".join("%d> %s" % (i, texts[i]) for i in batch))
        for line in _generate(system, user).splitlines():
            match = ANSWER_LINE.match(BULLET.sub("", line.strip()))
            if not match:
                continue
            i, text = int(match.group(1)), match.group(3).strip()
            if i not in texts or not text:
                continue
            if len(text) >= len(texts[i]) or len(text) > CAPTION_CHARS:
                continue
            if guard_english and _drifted_to_english(text):
                continue
            shorter[i] = text
    return shorter


def rewrite(sentences: list[Sentence], language: str | None = None,
            frames: int | None = None) -> list[Sentence]:
    """
    The transcript as third-person report, on the same timeline, in the same
    language, and in as many captions as the run has stills to put them on.

    `frames` is how many stills this run is about to deliver, and it decides
    how much of the transcript goes into one caption: forty of them is an
    account in forty lines, two hundred is the same account in two hundred.
    See `_group`, which is where it is spent, for why the frame-count slider
    is a decision about the writing and not only about the pictures.

    None means "as many captions as the transcript makes", which is what this
    did before the grid was known here and is still the right answer for a
    caller that is not building one.

    `language` is the ISO code Whisper detected while transcribing this same
    audio (transcriber.detected_language). It is not optional in spirit: with
    nothing to name, the prompt can only ask for "the same language as the
    transcript", and that was measured to fail on every line of a Portuguese
    transcript and every line of a French one. Passing None is the honest
    behaviour for a caller that genuinely does not know, and it leaves the
    English guard off, since guarding against English is meaningless when the
    video might be English.

    Returns the sentences it was given, unchanged, for anything that did not
    work - no model, no weights on this machine, a generation that came back
    as prose. See the module header: a frame fetch that produced good stills
    is not allowed to fail over the words on them, and neither is a rewrite.

    Chunked, and each chunk is shown the tail of the PREVIOUS chunk's own
    output as context. That is what carries a name across a boundary: a model
    reading "her son Brian" in the lines above it goes on writing "Brian",
    where a model reading the first-person original would reintroduce him. The
    context lines are labelled as context and their ids are outside the range
    the parser accepts, so a model that rewrites them anyway is ignored rather
    than allowed to overwrite work already done.
    """
    if not sentences:
        return sentences
    if not _ensure_loaded():
        return sentences

    # The name if it is one this file knows, the bare code if not, and the
    # vaguest possible wording if the caller had nothing at all.
    named = LANGUAGE_NAMES.get(language, language) if language else None
    system = SYSTEM.format(budget=CAPTION_CHARS, ceiling=CAPTION_CHARS_MAX,
                           language=named or "the same language as the transcript")
    reminder = LANGUAGE_REMINDER.format(language=named) if named else ""
    guard_english = bool(language) and language != "en"

    # The sentences gathered into thoughts before the model sees any of them,
    # so that what it is asked for is one caption per thought and it is never
    # asked to decide where a caption should begin. See the grouping block at
    # the top of this file for why that decision was taken away from it.
    groups = _group(sentences, frames)
    blocks = [_block(g) for g in groups]

    done: list[Sentence] = []
    rewritten = 0

    for first in range(0, len(blocks), CHUNK_SENTENCES):
        last = min(first + CHUNK_SENTENCES, len(blocks))

        context = ""
        if done:
            tail = done[-CONTEXT_SENTENCES:]
            context = ("Context - the lines just before these, already "
                       "rewritten. Do not answer for them:\n"
                       + "\n".join(s.text for s in tail) + "\n\n")

        user = (context
                + reminder
                + "Rewrite these. Answer for every id from "
                + f"{first} to {last - 1}:\n"
                + _listing(blocks, first, last))

        runs = _parse(_generate(system, user), first, last, guard_english)
        rewritten += sum(1 + end - start for start, (end, _) in runs.items())
        done.extend(_apply(groups, blocks, runs, first, last,
                           previous=done[-1].text if done else ""))

    # The lines that came back too long to show, asked again and shorter (see
    # CONDENSE_SYSTEM). Between the two passes rather than after the naming
    # one, so that what reference.vary looks at is the wording that will
    # actually be set — an opening it varied on a line that is about to be
    # replaced is work thrown away, and worse, a variation it would have made
    # on the replacement is one it never sees.
    long_lines = {i: s.text for i, s in enumerate(done) if _needs_condensing(s.text)}
    shortened = _condense(long_lines, language)

    # ...and a shortened line is only taken if it does not collide with the one
    # before it. Two captions can say different things at full length and the
    # same thing once both are cut to a sentence each: measured on a real
    # video, "The idea was simple: every child from birth to age five would
    # receive one free book every month, no application, no means test" and
    # the line after it came back as "Every child received a free book every
    # month through the Imagination Library" and "Children received a free
    # book every month through the Imagination Library" — one thought, twice,
    # on two consecutive stills.
    #
    # _repeats already guards against that, but it runs inside _apply, which
    # is over by the time this pass writes anything. So it is asked again
    # here, against whatever is actually going to be set, and a collision
    # keeps the LONG line: a caption that overflows its card is a caption, and
    # the same caption twice is not.
    taken = 0
    for i in sorted(shortened):
        previous = done[i - 1].text if i else ""
        if previous and _repeats(shortened[i], previous):
            continue
        done[i] = Sentence(text=shortened[i], start=done[i].start, end=done[i].end)
        taken += 1

    # ...and last, the one thing the prompt could not be made to do: stop
    # opening line after line with the same full name. Rule 5 above asks for
    # it and was measured twice not getting it; reference.py takes the
    # decision away in the same spirit _group did. It touches the wording of
    # a caption's opening and nothing else — never the timeline.
    varied = reference.vary(done, language)

    log.info("narration: %s, %d sentences grouped into %d, %d rewritten, "
             "%d captions on the timeline, %d shortened of %d over budget, "
             "%d openings varied",
             named or "language unknown", len(sentences), len(blocks), rewritten,
             len(varied), taken, len(long_lines),
             sum(1 for a, b in zip(done, varied) if a.text != b.text))
    return varied
