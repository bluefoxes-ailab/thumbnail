"""
narrator.py - the transcript, rewritten as one reported account and dealt out
over the frames of a run.

transcriber.py ends with an admission and an invitation. It says its sentence
assembly decides whether a line is COMPLETE, not whether it is INFORMATIVE,
that judging the second thing is a language model's job, and that the seam a
language model would be dropped into is the list of Sentence it hands back.
This is that model, dropped into that seam.

What it is for is one specific failure of raw transcription as caption. These
videos are cut from news packages, and a news package is half narration and
half interview. The narration already reads as information. The interview does
not: it is somebody talking about themselves, in the first person, to a
reporter who is not on screen. Laid over the stills it reads as nobody's
words -

    "I am Brian, Dolly's son."
    "...and that is how the world found out she had passed."

- two lines a viewer scrolling past has no way to place. The same two facts
written as report are news:

    "The world learned of her death through a message from her son Brian."

So this module turns direct speech into described speech - "I am Dolly
Parton's grandson" becomes "Brian said he was Dolly Parton's grandson" - and
it does it with the whole transcript in view, because that is the only way the
second line above can know who "I" was. The identity is established in one
sentence and needed in another.

## The dynamics, in order

  1. The transcript arrives as TEXT. The seconds each sentence was spoken at
     stay behind in transcriber.py: nothing here maps a caption to a moment
     and nothing here keeps a timestamp. See "Why there is no timeline".
  2. The frame count is confirmed. It is the number of stills the run
     actually DELIVERS - api.py asks for the rewrite once the frames are
     prepared - so a video that yielded thirty-one usable stills out of the
     forty asked for is written for thirty-one, and no line is left over.
  3. The budget follows from it: CAPTION_CHARS per frame. Forty frames is an
     account of at most 3,200 characters; two hundred, of at most 16,000.
  4. The transcript is planned into ITEMS - runs of consecutive sentences,
     each to be reported by one line of the account - and every item is given
     one frame or two, the sizes adding up to the frame count (see _plan).
  5. The model writes one line per item, in order, as ONE account: a single
     text read from the first frame to the last, not forty captions that each
     have to stand alone.
  6. The account is held to the budget. No sentence runs past two frames'
     worth - CAPTION_CHARS_MAX, each frame within CAPTION_CHARS: one that does
     is cut there and carries on in the next (see _bounded). Lines that came
     back longer than their frames are joined with a short neighbour where the
     two fit, shortened by a second pass where they do not (see
     CONDENSE_SYSTEM), and only as the very last resort set longer than a
     frame holds (see _deal).
  7. Names are varied. reference.py takes the second and later of a run of
     lines that open with the same person and names them another way - a
     pronoun, a role, the surname, the given name.
  8. The account is dealt out over the frames, in order: the first piece to
     the first still, the next to the next. A line of up to CAPTION_CHARS
     takes one still and a longer one takes two. EVERY frame gets text, and
     no text is left without a frame.

## What it may and may not do

It rewrites. It does not research, and it does not embellish. Every name,
number, relationship and event in the output has to be somewhere in the input,
because the alternative is a caption asserting something nobody said over a
photograph of a real person. The prompt says so and the validation below
cannot check it - that is worth stating plainly rather than implying the guard
is stronger than it is.

The same rule decides what happens to a short video on a long grid. Eight
sentences of speech do not become forty sentences of account: the lines are as
long as the transcript can honestly fill, and the dealing spreads them over
more frames each, so every frame still carries a piece of the text - a shorter
one. The budget is a ceiling, not a quota.

## Why there is no timeline

This module used to keep one. Every caption carried the seconds of the
sentences it reported, and was matched to the frames by them, and it cost
more than it gave: frames are chosen on picture quality, so two of them fell
inside one sentence while the next sentence had none, and a run came back with
blanks in the middle of a story that had exactly enough words to fill it.

Order is the only thing kept now, and it is enough. The frames come in the
order they occur in the video and so does the transcript, so line ten of forty
lands on still ten of forty and both are about a quarter of the way through.
What is given up is the claim that a caption sits over the instant its words
were spoken - which a REPORT of what was said was never really making.

## Never fatal, for the reason transcriber.py is never fatal

A frame fetch that produced forty good stills has done its job. A missing
model, a machine that has never downloaded the weights, a generation that
comes back as prose instead of the format asked for - each of those returns
None, and the run captions its frames the way a channel that does not ask for
a rewrite does: with the transcript as spoken.

Short of that, a line the model got wrong costs its own item and not the run.
An item with no usable answer keeps its words as spoken, cut to the frames it
was planned for; a line that only repeats the one before it is dropped, and
the frames it would have had go to its neighbours. There is no failure mode in
here that puts a line on the wrong frame or out of order.

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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import reference
from transcriber import CAPTION_CHARS, CAPTION_CHARS_MAX, MAX_CHARS, TERMINAL

# ── Planning ──────────────────────────────────────────────────────────────
#
# What the model is handed is not one transcript sentence per line. It is one
# THOUGHT per line, and the difference is the whole reason a line can read as
# information.
#
# The case this exists for, from a real transcript:
#
#     "...My name is Brian Siever, son of Cassie Parton"
#     "and I'm representing the Parton and Owen's family today"
#     "announcing the passing of my Aunt Dolly, Rebecca Parton Dean."
#
# Three lines, none of which is news. One line, and it is the story.
#
# Asking the model to make that join does not work. It is offered the syntax
# and, measured on this transcript, used it zero times in sixty-three
# sentences; told in a rewritten prompt that joining was the normal move, with
# a worked example, it used it zero times again. So the join is taken away
# from it and done here, where it is arithmetic.
#
# The rule is not new. It is transcriber._merge_continuations, run a second
# time with the brake off. That pass already wanted to join the first two
# lines above and refused because the result was 21 words against a MAX_WORDS
# of 18. The brake is right there and wrong here: what has to fit on the
# frames is the model's ANSWER, not the material it reads.

# How much one thought may gather.
#
# Four sentences, because five is a paragraph and no single line can honestly
# report a paragraph. The character cap is three times a spoken sentence's own
# ceiling: the output condenses, so the input is allowed to be longer than it,
# but a thought the model would have to compress five-to-one comes back having
# dropped whatever it decided mattered least, and that decision is not one
# this can check.
#
# Both are about MEANING. How many items the account is written in is decided
# separately, by the frame count - see _plan, which goes on joining thoughts
# past these caps when the grid has fewer frames than the transcript has
# thoughts, because then the compression is what the run asked for.
GROUP_MAX_SENTENCES = 4
GROUP_MAX_CHARS = MAX_CHARS * 3

# What one turn of the model is shown at most, in characters of transcript -
# and therefore also the most one ITEM may gather.
#
# The caps above are quality guards. This is not one. When a run asks for
# forty stills of a twenty-minute video it has asked for an account of forty
# lines, and forty lines of twenty minutes IS five-to-one compression - the
# decision was made at the frame-count slider, and refusing it here would not
# undo it. So this bounds the PROMPT rather than the reporting: roughly six
# thousand tokens of transcript in one generation, which a CPU gets through
# in a minute or two.
#
# A run that cannot reach its frame count inside it - one frame of an
# hour-long video - plans more items than it has frames, says so in the log,
# and is squeezed onto the grid by _deal.
CHUNK_MAX_CHARS = 20000
ITEM_MAX_CHARS = CHUNK_MAX_CHARS

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
# Two, and the asymmetry is deliberate. A false positive costs that item its
# rewrite - it keeps the words as spoken instead. The wrong language is the one
# failure this guard exists to prevent, and it cannot be the price of avoiding
# it. A false negative is an English caption on a Portuguese video, which is
# the thing itself. So the guard leans towards rejecting.
ENGLISH_MARKERS_MIN = 2

# How much of a line's wording may already have been in the line before it
# before the two are saying the same thing.
#
# Three quarters, measured against the shorter of the pair — see _repeats. Set
# lower, two lines about the same person doing different things start
# colliding on the name alone; set higher, a line that only swapped its last
# three words for three others gets through, and that is exactly the padding
# this is here to stop.
REPEAT_MAX_OVERLAP = 0.75

# ...and the length below which two lines are not compared at all.
#
# At four distinct words "The funeral is today" and "The funeral is tomorrow"
# are three-quarters alike and say opposite things, so the overlap measure
# stops meaning anything before it stops being computable. Five is where a
# line has enough words for the ones it shares to be evidence.
REPEAT_MIN_WORDS = 5

# Splits a line into lowercase words, apostrophes kept inside them so
# "l'accident" does not become a bare "l" and an "accident".
WORDS = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)

# ── Where the account may be broken between two stills ───────────────────
#
# Every space in the account is a place it COULD be broken, and they are not
# equal. After a full stop the reader has finished something; after a comma
# they are in the middle of it but at a place the writer chose; anywhere else
# they are mid-phrase. So each break is ranked by the mark in front of it, and
# every cut in this file - the CAPTION_CHARS_MAX rule, the split of a line
# over its two stills, the spreading of a short account over a long grid -
# takes the best-ranked break it can before the nearest one.
#
# A mark only counts with a SPACE after it, which is what keeps "3,000", "9.5"
# and "l'accident" from ever being taken for places a sentence comes apart.
SENTENCE_MARKS = ".!?…"
CLAUSE_MARKS = ",;:—–"
# ...and what may stand between a mark and the space without hiding it, so
# that '..."' and '...)' end a sentence as firmly as '...' does.
CLOSERS = "\"'”’)]"

# The fewest words a still may be left holding when a short account is spread
# over a long grid (see _deal).
#
# Three, so that spreading never puts "Yes," on a frame of its own while there
# is still a longer line it could have divided instead. It is relaxed to one
# word only when nothing at all can be divided under it, which is the run whose
# transcript barely has a word per frame - and there, a word on every frame is
# what "every frame gets text" honestly comes to.
SPREAD_MIN_WORDS = 3

log = logging.getLogger("uvicorn.error")  # see the note in face_restorer.py

# ── The model ─────────────────────────────────────────────────────────────

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

# How many items are written in one generation - with CHUNK_MAX_CHARS, which
# bounds the same turn by how much transcript it shows.
#
# Not a context-window limit - Llama 3.2 has far more than this needs. It is a
# limit on how long one CPU generation runs before anything at all comes back,
# and on how much the model is holding in its head when it answers. Forty
# items is long enough for a narrative to be consistent across it, and it is
# the whole of the default run.
CHUNK_ITEMS = 40

# ...and how many lines of the account already written are shown ahead of a
# chunk as read-only context.
#
# Without this, the first line of the second chunk has no idea who has been
# speaking, and the account reads as two texts stapled together: the chunk
# boundary is visible as the exact point where a name turns back into a
# pronoun, or a person is introduced a second time. They are shown as context
# and are NOT reproduced in the answer - see the prompt, and the parser, which
# ignores an id outside the chunk it asked about.
CONTEXT_LINES = 8

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
# rewritten line. "7> ..." and "7-9> ..." are the two shapes accepted, and the
# item's size may be echoed between them ("7 [80]> ..."), since that is how
# the item was printed in the question.
#
# A bare number and an angle bracket rather than JSON, and that is a decision
# about failure rather than about taste. A model that loses its way inside
# JSON produces a document that will not parse at all, and the whole chunk is
# lost with it. A model that loses its way here produces some lines that match
# this and some that do not, and every line that matches is still usable - the
# ones that do not simply cost their own item.
ANSWER_LINE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*(?:\[\s*\d+\s*\])?\s*>\s*(.+?)\s*$")

# Leading list furniture a model adds when it is being helpful. Stripped
# rather than rejected, because the sentence after it is usually right.
BULLET = re.compile("^[-*•]\\s+")

# ...and an item's size copied into the line itself ("7> [80] The family..."),
# which a model echoing the question's format does as readily as it echoes it
# before the bracket. Stripped for the same reason: what follows it is the
# line, and a "[80]" set on a frame is not.
SIZE_TAG = re.compile(r"^\[\s*\d+\s*\]\s*")

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
# the generator runs to its token ceiling every time and the answer has a
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
#     in 63 sentences; promoted to a rule with a worked example, 0 again. Now
#     done in _plan before the model sees anything.
#   * not repeating itself - still produced three consecutive lines that were
#     one sentence with its last three words swapped. Now caught by _repeats
#     as the account is assembled.
#   * not opening every line with the same name - asked twice, the second time
#     in capitals, and still 7 of 17 lines opening with "Dolly Parton", six of
#     them consecutive. Now done by reference.vary.
#
# That last one was measured while every line was ALSO told it had to stand
# alone over its own photograph - and the safest way to make a line stand
# alone is to name its subject in full. The account is now one text, which
# takes that pressure away and is why rule 7 below asks for the variation
# outright; reference.vary stays as the machinery that holds it whether or
# not the model does.
#
# So a rule here is worth writing when it is a fact the model needs (the
# language, the size of the line it is writing, the answer format) and worth
# being sceptical of when it asks for restraint across many lines. That kind
# wants machinery, or a bigger model.
#
# Which is why the budget is printed BESIDE EACH ITEM ("7 [80]> ...") rather
# than stated once. "This line goes on one photograph, 80 characters" is a
# fact about the line the model is writing at that moment; "3,200 characters
# in all" is a discipline across forty lines, and is in the prompt only so the
# sizes make sense. Nothing downstream relies on the model hitting either:
# _bounded, the condensing pass and _deal hold the budget whatever comes back.
SYSTEM = (
    "You rewrite the transcript of a news video as the text of a photo story.\n"
    "\n"
    "The transcript mixes a narrator with interviews. The interviews are in "
    "the first person: people talking about themselves. Your job is to turn "
    "all of it into third-person reporting, so that every line reads as "
    "information about what happened rather than as somebody speaking. Direct "
    "speech becomes described speech: \"I grew up on that farm\" becomes "
    "\"Her brother said he grew up on the farm\".\n"
    "\n"
    "The story is shown over {frames} photographs, a piece of it on each, in "
    "order, and it is read from the first photograph to the last as ONE text - "
    "a single short article, not a list of separate captions. It has room for "
    "{budget} characters in all: {size} per photograph.\n"
    "\n"
    "Rules, in order of importance:\n"
    "1. WRITE IN {language}. The transcript is in {language} and every line "
    "you write must be in {language} too. These lines are laid back over the "
    "video the words were spoken in, so a line in another language is "
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
    "4. One line out per numbered item in, in the same order. Each item shows "
    "its size in brackets. [{size}] means its line goes on one photograph and "
    "must fit in {size} characters. [{ceiling}] means its line goes on two "
    "photographs and may use up to {ceiling} characters - best written as two "
    "sentences of up to {size} each. Never go over an item's size. Shorter is "
    "fine; an unfinished thought is not.\n"
    "5. NO SENTENCE LONGER THAN {ceiling} CHARACTERS. A thought that needs "
    "more ends its sentence there and carries on in the next one.\n"
    "6. THE ITEMS ARE CONSECUTIVE MOMENTS OF ONE VIDEO, in the order they "
    "happen, so write them as ONE ACCOUNT: line by line the story moves "
    "forward, nothing in the middle is missing, and nothing is said twice. "
    "Before writing a line, read the item AFTER it - what this line has to do "
    "is get the reader from where they are to what comes next. Words that "
    "carry the story forward - 'Within hours', 'By 1992', 'The family then' - "
    "are what make the lines read as one story rather than as facts in a "
    "row.\n"
    "7. VARY HOW PEOPLE ARE NAMED. Name each person in full the first time. "
    "After that, refer to them in different ways: a pronoun, the surname, the "
    "first name, or what the transcript says they are - the singer, the "
    "musician. Never open two lines in a row with the same name.\n"
    "8. Report plainly. No commentary, no drama, and no hedging words unless "
    "the transcript hedged first.\n"
    "\n"
    "Answer format. One line per item, and nothing else - no preamble, no "
    "numbering of your own, no blank lines:\n"
    "  7> The rewritten line.\n"
    "Answer for every id you are given, once each, in order.\n"
    "\n"
    "Write every line in {language}."
)

# What the model is asked when the account came back longer than the frames
# can hold.
#
# A second pass, and it exists because the first one cannot be made to obey a
# length. A budget stated plainly, in capitals, with the reason attached, was
# measured on a real video: 22 of 37 lines still came back over it - several
# at half as long again. That is the same shape of failure as the three rules
# the note above SYSTEM describes: the model follows a concrete instruction
# about THIS line and ignores a discipline held across a list. So the
# discipline is taken away from it and asked as a concrete instruction
# instead - here is one line, it is too long, say it shorter - which is a
# question it answers well.
#
# Shortening is also a genuinely easier job than reporting. The first pass has
# to decide what a piece of speech MEANS and who it is about; this one is
# handed a finished sentence and asked only to say it with fewer words, so a
# 3B model does it reliably where it drifts on the harder question.
#
# What it must not do is trim a sentence into a fragment - that is worse than a
# long line, so it is rule 4 here and is checked on the way back out.
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
    "thought is worse than a line that is too long. If you cannot say it in "
    "{budget} characters and still finish it, get as close as you can and "
    "finish it.\n"
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
    Whether the model is in memory RIGHT NOW.

    `is_available` is the optimistic form and is the right one to ask before a
    run: it answers True while the weights have yet to be tried, because they
    usually work. This is the pessimistic form, and the right one to ask after
    a rewrite came back None, when a caller wants to say WHY: a model that was
    never there is a machine to fix, and a model that ran and produced nothing
    usable is a video to look at.
    """
    return _generator is not None


# ── Planning the items ────────────────────────────────────────────────────


@dataclass
class _Item:
    """A run of consecutive transcript sentences, and the stills its one line is written for."""
    source: str
    stills: int


def _continues(previous: str, following: str) -> bool:
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
    if not TERMINAL.search(previous):
        return True
    head = following.lstrip()[:1]
    return bool(head) and head.islower()


def _thoughts(sentences: list[str]) -> list[list[str]]:
    """
    The sentences gathered into the thoughts they belong to, in order.

    Every sentence appears exactly once and none moves. A sentence that
    continues nothing and is continued by nothing comes back as a thought of
    one, which is what most narration is.
    """
    groups: list[list[str]] = []
    for sentence in sentences:
        current = groups[-1] if groups else None
        if (current
                and len(current) < GROUP_MAX_SENTENCES
                and len(" ".join(current)) + 1 + len(sentence) <= GROUP_MAX_CHARS
                and _continues(current[-1], sentence)):
            current.append(sentence)
        else:
            groups.append([sentence])
    return groups


def _plan(sentences: list[str], frames: int) -> list[_Item]:
    """
    The transcript as the items the account is written in, each with the
    number of stills its line is for - one or two, adding up to `frames`.

    Two questions, answered in this order.

    HOW MANY ITEMS. The transcript's own thoughts first (see _thoughts), and
    then the grid. A transcript with more thoughts than the run has frames is
    joined down until there is one item per frame, the neighbouring pair that
    is shortest together first - so the items come out even, and no line is
    asked to report a minute of speech while the next reports a sentence. The joins ignore the caps that formed the thoughts, because past
    this point the compression is what the run asked for (see
    CHUNK_MAX_CHARS for the one bound that does still hold).

    HOW MANY STILLS EACH. One, unless the transcript has fewer thoughts than
    the run has frames. Then the spare frames go to the LONGEST items, one
    each, as items written for two stills - up to CAPTION_CHARS_MAX
    characters, set across a pair of consecutive frames. That is the
    allowance for a thought that needs more than one frame holds, spent where
    there is the most to say.

    The longest rather than an even spread, and rather than cutting the
    transcript's sentences finer to make more items: an item is a whole
    thought, and a half of one handed to the model on its own came back as an
    echo of half a self-introduction - a first-person fragment, the one thing
    this module exists to prevent.

    A transcript with fewer thoughts than HALF the frames cannot fill the grid
    this way - every item is already written for two. What it can say is all
    there is, and _deal spreads it thinner over the frames that are left.
    """
    groups = [" ".join(g) for g in _thoughts(sentences)]

    while len(groups) > frames:
        best, least = None, None
        for i in range(len(groups) - 1):
            size = len(groups[i]) + 1 + len(groups[i + 1])
            if size > ITEM_MAX_CHARS:
                continue
            if least is None or size < least:
                best, least = i, size
        if best is None:
            log.info("narration: %d items will not join down to %d frames inside one "
                     "turn of the model - the account will be squeezed to fit",
                     len(groups), frames)
            break
        groups[best:best + 2] = [groups[best] + " " + groups[best + 1]]

    stills = [1] * len(groups)
    spare = frames - len(groups)
    if spare > 0:
        longest = sorted(range(len(groups)), key=lambda i: len(groups[i]), reverse=True)
        for i in longest[:spare]:
            stills[i] = 2

    return [_Item(source=g, stills=s) for g, s in zip(groups, stills)]


def _chunks(items: list[_Item]):
    """
    The items as consecutive (first, last) runs, one generation each.

    At most CHUNK_ITEMS of them and CHUNK_MAX_CHARS of transcript, and never
    fewer than one item, so an item larger than the cap still gets a turn of
    its own rather than none.
    """
    first = 0
    while first < len(items):
        last, size = first + 1, len(items[first].source)
        while (last < len(items) and last - first < CHUNK_ITEMS
               and size + len(items[last].source) <= CHUNK_MAX_CHARS):
            size += len(items[last].source)
            last += 1
        yield first, last
        first = last


# ── Writing ───────────────────────────────────────────────────────────────


def _listing(items: list[_Item], first: int, last: int) -> str:
    """The items from `first` to `last` as the numbered lines the model reads, each with its size."""
    return "\n".join(f"{i} [{items[i].stills * CAPTION_CHARS}]> {items[i].source}"
                     for i in range(first, last))


def _answer_tokens(stills: list[int]) -> int:
    """
    How many tokens one generation may produce, for lines written to `stills`.

    About twice what a well-behaved answer needs: each line its size in
    characters at three and a half characters a token, plus a few for its id.
    That is the room a model needs to finish a sentence rather than the room to
    write an essay - and it has to scale with what was asked, because an item
    written for two stills answers twice as long as one written for one, and a
    fixed ceiling sized for the first cuts a chunk of the second off halfway.
    """
    chars = sum(stills) * CAPTION_CHARS
    return int(2 * (chars / 3.5 + 4 * len(stills)))


def _generate(system: str, user: str, max_tokens: int) -> str:
    """
    One turn, as text. Empty for anything that did not work - the caller reads
    that as "this turn produced nothing usable".

    CTranslate2 speaks in token STRINGS rather than ids, which is why the
    round trip goes through the tokenizer twice: encode to strings on the way
    in, look the strings back up to decode on the way out.
    """
    prompt = PROMPT.format(system=system, user=user)
    tokens = _tokenizer.encode(prompt, add_special_tokens=False).tokens
    try:
        result = _generator.generate_batch(
            [tokens],
            max_length=max_tokens,
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


def _repeats(text: str, previous: str) -> bool:
    """
    Whether a line says what the line before it already said.

    The failure it catches, from a real run, three consecutive lines:

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
    the third line above — while comparing both ways would score that pair
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
    runs backwards, overlaps something already claimed, is empty, or drifted
    into English on a video that was not, is dropped; its ids then find no
    entry in this map, and _write gives those items their own words instead.

    Length is NOT a reason, and that is a decision. A line over its size is a
    finished, accurate sentence that came out long, and throwing it away for
    that hands its frame back to the first-person speech this module exists to
    get rid of. Holding the account to the budget is _bounded's job and the
    condensing pass's, which shorten rather than discard.

    `claimed` is what makes the no-overlap rule hold across the whole answer
    rather than line by line. A model that answers "7-9>" and then "8>" has
    contradicted itself about what item 8 says, and the second statement is
    not more true than the first - the first is kept because it came first,
    and the second is dropped.

    A run ("7-9>") is accepted rather than refused. Nothing here ties a line
    to a moment any more, so a model that reports three items in one line has
    simply written a shorter account, and the frames those items were planned
    for go to the lines around it.
    """
    runs: dict[int, tuple[int, str]] = {}
    claimed: set[int] = set()

    for line in answer.splitlines():
        match = ANSWER_LINE.match(BULLET.sub("", line.strip()))
        if not match:
            continue

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        text = SIZE_TAG.sub("", match.group(3)).strip()

        if not (first <= start <= end < last) or any(i in claimed for i in range(start, end + 1)):
            continue
        if not text:
            continue
        if guard_english and _drifted_to_english(text):
            continue

        runs[start] = (end, text)
        claimed.update(range(start, end + 1))

    return runs


def _fallback(item: _Item) -> str:
    """
    An item's own words, as spoken, cut to the stills it was planned for.

    What an item gets when the model gave it nothing usable - no line for its
    id, or a line in the wrong language. Plainer, not wrong: the words are the
    video's and they are in the video's language, and the cut keeps the line
    to its share of the budget so one failed item cannot crowd its neighbours
    off the grid. What is lost is the rest of what that item's speech said,
    which a model that answered would have condensed rather than cut.
    """
    head, _ = _cut(item.source, item.stills * CAPTION_CHARS)
    return head


def _write(items: list[_Item], system: str, reminder: str,
           guard_english: bool) -> tuple[list[str], dict[str, int]]:
    """
    The account: one line per item the model answered, in order, with the
    context that keeps it one text.

    Chunked, and each chunk is shown the tail of the account written so far.
    That is what carries a name across a boundary: a model reading "her son
    Brian" in the lines above it goes on writing "Brian", where a model
    reading the first-person original would reintroduce him. The context lines
    are labelled as context and their ids are not in the range the parser
    accepts, so a model that rewrites them anyway is ignored rather than
    allowed to overwrite work already done.

    The repetition check runs here, against whatever line was actually kept
    before this one - rewritten or kept as spoken - because that is what the
    reader will have just read. A line that repeats it is DROPPED rather than
    replaced: the thought is already on the frame before, and the frames this
    line was planned for are dealt to its neighbours.

    Also returns what happened, for the log: lines `written` by the model,
    lines kept as `spoken`, and `repeats` dropped.
    """
    lines: list[str] = []
    stats = {"written": 0, "spoken": 0, "repeats": 0}

    for first, last in _chunks(items):
        context = ""
        if lines:
            context = ("Context - the account so far ends with these lines, "
                       "already written. Do not answer for them:\n"
                       + "\n".join(lines[-CONTEXT_LINES:]) + "\n\n")

        user = (context
                + reminder
                + "Rewrite these. Answer for every id from "
                + f"{first} to {last - 1}:\n"
                + _listing(items, first, last))

        answer = _generate(system, user, _answer_tokens([it.stills for it in items[first:last]]))
        runs = _parse(answer, first, last, guard_english)

        i = first
        while i < last:
            run = runs.get(i)
            if run is not None:
                end, text = run
                kind = "written"
                i = end + 1
            else:
                text = _fallback(items[i])
                kind = "spoken"
                i += 1
            if lines and _repeats(text, lines[-1]):
                stats["repeats"] += 1
                continue
            lines.append(text)
            stats[kind] += 1

    return lines, stats


# ── Holding the account to the budget ─────────────────────────────────────


def _breaks(text: str) -> list[tuple[int, int, int]]:
    """
    Every place `text` may be broken, as (end of the head, start of the tail,
    rank): rank 0 after a sentence, 1 after a clause, 2 anywhere else. See the
    block above SENTENCE_MARKS.
    """
    out = []
    for gap in re.finditer(r"\s+", text):
        if gap.start() == 0 or gap.end() == len(text):
            continue
        mark = text[:gap.start()].rstrip(CLOSERS)[-1:]
        rank = 0 if mark and mark in SENTENCE_MARKS else 1 if mark and mark in CLAUSE_MARKS else 2
        out.append((gap.start(), gap.end(), rank))
    return out


def _cut(text: str, limit: int, keep=None) -> tuple[str, str]:
    """
    `text` cut so its head is at most `limit` characters, as (head, rest).

    The cut goes at the best-ranked break in the back half of the limit, the
    latest of those: the end of a sentence if there is one there, a clause if
    not, a word if neither. The back half, because a cut at the first full
    stop of a long line would satisfy the rule and leave the head a fraction
    of what it may hold. `keep`, when given, is a further test the head has to
    pass. Only a text with no acceptable break at all - a single word longer
    than a frame, which is a URL, not a sentence - is returned whole, with
    nothing left over.
    """
    if len(text) <= limit and (keep is None or keep(text)):
        return text, ""
    options = [b for b in _breaks(text)
               if b[0] <= limit and (keep is None or keep(text[:b[0]]))]
    if not options:
        return text, ""
    late = [b for b in options if b[0] >= limit // 2] or options
    at = min(late, key=lambda b: (b[2], -b[0]))
    return text[:at[0]], text[at[1]:]


# Remembered, because the budget is checked over and over against an account
# that changes a line at a time (see _shorten), and every check re-divides the
# same few hundred lines: on a synthetic transcript of 1,500 sentences dealt
# onto 200 frames, that was four seconds of the run spent working out answers
# it already had, and with these it is half of one. Both are pure functions of
# a string, and bounded, so a server that runs all day does not keep every
# line it ever set.
@lru_cache(maxsize=16384)
def _settable(text: str) -> bool:
    """
    Whether a piece of the account can be set as it is: on one still, or
    divided over two with neither holding more than CAPTION_CHARS.

    The second half of that is not the same thing as "at most
    CAPTION_CHARS_MAX". A piece of a hundred and fifty-nine characters fits
    two frames only if it has a space within a character or two of its middle;
    without one, the most even division it has is eighty-one and seventy-eight,
    and that eighty-first character is a fifth line of type on a card built
    for four. So the rule is held as what it means - two frames' worth, each
    within its frame - rather than as the number it is usually written as.
    """
    if len(text) <= CAPTION_CHARS:
        return True
    if len(text) > CAPTION_CHARS_MAX:
        return False
    return any(s <= CAPTION_CHARS and len(text) - e <= CAPTION_CHARS
               for s, e, _ in _breaks(text))


@lru_cache(maxsize=4096)
def _bounded(line: str) -> tuple[str, ...]:
    """
    A line of the account as the pieces it is set in: each settable on one
    frame or on two (see _settable), which is at most CAPTION_CHARS_MAX.

    Almost every line comes back as itself. One that does not fit two frames
    is cut there and CONTINUED, not shortened and not thrown away: the head
    takes its frames, and what was cut off carries on as the next piece - and
    is cut again, if it too runs past the limit. That is the rule as stated: a
    sentence may run to CAPTION_CHARS_MAX and no further, and a thought that
    needs more goes on in the next one.

    Cut at a sentence end where there is one (see _cut), which is what makes
    "the next one" usually a sentence of its own. Where there is not, the
    continuation begins mid-sentence - and in an account read as one text
    that reads as what it is, the same sentence going on over the next frame.
    """
    pieces, rest = [], line.strip()
    while not _settable(rest):
        head, tail = _cut(rest, CAPTION_CHARS_MAX, _settable)
        if not tail:
            break
        pieces.append(head)
        rest = tail
    if rest:
        pieces.append(rest)
    return tuple(pieces)


def _set(lines: list[str]) -> list[str]:
    """The lines of the account as the pieces they are set in, in order."""
    return [piece for line in lines for piece in _bounded(line)]


def _stills(text: str) -> int:
    """How many stills a piece of the account takes on its own: one if it fits one, two if not."""
    return 1 if len(text) <= CAPTION_CHARS or len(text.split()) < 2 else 2


def _joined(pieces: list[str], frames: int) -> tuple[list[list], int]:
    """
    The pieces as blocks of [text, stills], neighbours joined while that
    saves a frame and loses nothing - and how many stills the blocks need.

    Only runs while the account needs more stills than `frames`. A join is
    taken only where the two pieces fit, together, inside the stills the
    joined block will have: two short pieces on one frame, or a long piece
    and a short one on two (see _settable). Nothing is shortened and nothing
    overflows, so this always goes first, before anything that costs words.

    The smallest pair first, so what ends up sharing a frame is the shortest
    lines - "She was 79." beside the line before it - rather than whichever
    two happened to come first.
    """
    blocks = [[piece, _stills(piece)] for piece in pieces]
    total = sum(stills for _, stills in blocks)
    while total > frames:
        best = None
        for i in range(len(blocks) - 1):
            (a, sa), (b, sb) = blocks[i], blocks[i + 1]
            stills = sa + sb - 1
            size = len(a) + 1 + len(b)
            if stills > 2 or size > stills * CAPTION_CHARS:
                continue
            if stills == 2 and not _settable(a + " " + b):
                continue
            if best is None or size < best[1]:
                best = (i, size, stills)
        if best is None:
            break
        i, _, stills = best
        blocks[i:i + 2] = [[blocks[i][0] + " " + blocks[i + 1][0], stills]]
        total -= 1
    return blocks, total


def _condense(texts: dict[int, str], language: str | None) -> dict[int, str]:
    """
    Over-long lines said again, shorter, keyed by the index they came from.

    Asked in one turn per chunk rather than one per line, for the reason the
    first pass is chunked: a generation costs about the same whether it
    answers one item or forty, and forty separate turns is forty times the
    wait for the same words.

    An answer is taken only if it is SHORTER than what it replaces and fits
    one frame, and only if it did not drift into English on a video that was
    not (the same guard the first pass uses). Anything else leaves the long
    line alone: too long is a line that takes another frame, and every way
    this could go wrong instead - a fragment, a translation, an invented fact -
    is a caption that is wrong.
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
    for at in range(0, len(order), CHUNK_ITEMS):
        batch = order[at:at + CHUNK_ITEMS]
        user = (reminder
                + "Shorten these. Answer for every id:\n"
                + "\n".join("%d> %s" % (i, texts[i]) for i in batch))
        for line in _generate(system, user, _answer_tokens([1] * len(batch))).splitlines():
            match = ANSWER_LINE.match(BULLET.sub("", line.strip()))
            if not match:
                continue
            i, text = int(match.group(1)), SIZE_TAG.sub("", match.group(3)).strip()
            if i not in texts or not text:
                continue
            if len(text) >= len(texts[i]) or len(text) > CAPTION_CHARS:
                continue
            if guard_english and _drifted_to_english(text):
                continue
            shorter[i] = text
    return shorter


def _shorten(lines: list[str], frames: int, language: str | None) -> tuple[list[str], int]:
    """
    The account brought inside `frames` by saying its long lines shorter -
    and how many were.

    Only reached when joining neighbours was not enough. The lines that take
    more than one frame are the candidates, because a line shortened under
    CAPTION_CHARS gives the rest back; the SHORTEST of them first, because
    those lose the least by it - a line of eighty-five characters says nearly
    everything in eighty, and a line of a hundred and fifty does not.

    Whole LINES, as the model wrote them, and never the pieces _bounded cut
    them into: the piece after a cut is half a sentence, and a model asked to
    shorten it hands back a whole one that no longer follows from the piece
    it came after.

    Twice as many are asked as the excess, and a few more, since not every
    answer comes back usable; they are taken in that same order, and only
    until the account fits. A shortened line is only taken if it does not
    collide with the one before it. Two lines can say different things at
    full length and the same thing once both are cut down, and the same
    caption twice is worse than a caption that runs onto a second frame.
    """
    excess = _joined(_set(lines), frames)[1] - frames
    if excess <= 0:
        return lines, 0

    candidates = sorted((i for i, line in enumerate(lines) if len(line) > CAPTION_CHARS),
                        key=lambda i: len(lines[i]))
    asked = candidates[:2 * excess + 4]
    shorter = _condense({i: lines[i] for i in asked}, language)

    lines = list(lines)
    taken = 0
    for i in asked:
        if i not in shorter:
            continue
        if i and _repeats(shorter[i], lines[i - 1]):
            continue
        lines[i] = shorter[i]
        taken += 1
        if _joined(_set(lines), frames)[1] <= frames:
            break
    return lines, taken


def _vary(lines: list[str], language: str | None) -> tuple[list[str], int]:
    """
    The account with repeated openings varied by reference.vary - and how
    many were.

    Whole lines again, for the reason _shorten gives: every line the model
    wrote starts a sentence, and the pieces _bounded cut out of them do not.
    One of those can begin with a name halfway through a sentence - "...the
    library, which" / "Dolly Parton founded in 1995." - and swapping that name
    for "She" would put a capital letter in the middle of a sentence that was
    never this module's to restructure.

    A varied opening that would make its line take more frames keeps the
    original wording instead: "The singer" is a few characters longer than
    "Parton", and the budget is not something a stylistic pass gets to spend.
    """
    varied = reference.vary(lines, language)
    out = []
    for old, new in zip(lines, varied):
        cost = sum(_stills(p) for p in _bounded(old))
        out.append(new if sum(_stills(p) for p in _bounded(new)) <= cost else old)
    return out, sum(1 for old, new in zip(lines, out) if old != new)


# ── Dealing the account out over the frames ──────────────────────────────


def _pieces(text: str, count: int) -> list[str]:
    """
    One block of the account divided into `count` consecutive pieces, one per
    still - its words in order, none dropped and none repeated.

    Divided in two halves at a time, each cut aimed at where an even division
    would fall. Near that point the best-ranked break wins - a sentence end,
    then a clause, then any word (see _breaks) - and a break too far from it
    is only taken when nothing nearer exists. For the common case, a line on
    two stills, there is one more condition and it comes first: both halves
    should fit a frame, so a line of a hundred and fifty characters becomes
    two of about seventy-five rather than one of a hundred and one of fifty.
    A block that cannot fit two frames whichever way it is cut - one squeezed
    onto them by _deal - is cut as evenly as it can be instead, since then the
    rank of the break matters less than not leaving one frame twice as full as
    the other.

    The pieces are the account's own text, punctuation included. A frame that
    ends on a comma is not left looking unfinished by mistake: the account is
    read as one text, the next frame carries it on, and the comma is telling
    the reader exactly that.

    Every piece keeps at least one word for every still still to be cut from
    it, so a caller that checked the block has `count` words always gets
    `count` pieces back.
    """
    if count <= 1:
        return [text]

    left = count // 2
    target = len(text) * left / count
    reach = len(text) / (2 * count)

    options = [b for b in _breaks(text)
               if len(text[:b[0]].split()) >= left and len(text[b[1]:].split()) >= count - left]
    if not options:
        return [text] + [""] * (count - 1)

    if count == 2:
        fitting = [b for b in options
                   if b[0] <= CAPTION_CHARS and len(text) - b[1] <= CAPTION_CHARS]
        if not fitting:
            at = min(options, key=lambda b: (abs(b[0] - target), b[2]))
            return [text[:at[0]], text[at[1]:]]
        options = fitting

    at = min(options, key=lambda b: (b[2] if abs(b[0] - target) <= reach else 3,
                                     abs(b[0] - target)))
    return _pieces(text[:at[0]], left) + _pieces(text[at[1]:], count - left)


def _deal(pieces: list[str], frames: int) -> list[str]:
    """
    The account as exactly `frames` captions, in order: the first piece of it
    on the first still, the next on the next.

    Three stages, and which of the last two runs depends on which way the
    account missed its size - usually by a frame or two, since the items were
    planned to the frame count and the pieces mostly come back near the sizes
    they were asked for.

      * JOIN. Neighbours that fit together are put on one frame (see _joined).
        Loses nothing, so it always goes first.

      * SQUEEZE, when the account still needs more stills than there are. The
        last resort, and the one place the budget gives: a block is set on
        fewer frames than it needs, so a frame carries more than
        CAPTION_CHARS. Whichever move leaves the least on one frame is taken,
        one at a time - a two-frame line set on one, or two neighbours set on
        one frame fewer than they need between them. Nothing is cut: a long
        caption wraps onto more lines of type, and the user can trim it, where
        a cut one is gone. The condensing pass has already been asked to make
        this unnecessary.

      * SPREAD, when the account is shorter than the grid - a short video on a
        long run, or a model that wrote briefer than it was allowed. Blocks
        are given more stills until every frame has text: first a second one
        to the longest of the blocks that have one, so that lines take two
        frames before any takes three, and then a further one to whichever
        holds the most text per frame. No still is left fewer than
        SPREAD_MIN_WORDS words while a longer line could take the cut
        instead.

    A frame comes back "" only when the whole account has fewer words than
    there are frames - when there is not a word left to put on it. That is
    the one case "every frame gets text" cannot honestly meet.
    """
    if frames < 1:
        return []

    blocks, total = _joined(pieces, frames)

    while total > frames:
        # Each option is (characters on the fullest frame it leaves, where,
        # what): 0 sets block `where` on one still, 1 joins it with the next.
        options = [(len(text), i, 0) for i, (text, stills) in enumerate(blocks) if stills == 2]
        options += [((len(blocks[i][0]) + 1 + len(blocks[i + 1][0]))
                     / (blocks[i][1] + blocks[i + 1][1] - 1), i, 1)
                    for i in range(len(blocks) - 1)
                    if blocks[i][1] + blocks[i + 1][1] - 1 <= 2]
        if not options:
            break
        _, i, move = min(options)
        if move == 0:
            blocks[i][1] = 1
        else:
            blocks[i:i + 2] = [[blocks[i][0] + " " + blocks[i + 1][0],
                                blocks[i][1] + blocks[i + 1][1] - 1]]
        total -= 1

    for least, pairs_first in ((SPREAD_MIN_WORDS, True), (1, False)):
        while total < frames:
            options = [i for i, (text, stills) in enumerate(blocks)
                       if len(text.split()) >= (stills + 1) * least]
            if not options:
                break
            i = max(options, key=lambda i: (pairs_first and blocks[i][1] == 1,
                                            len(blocks[i][0]) / blocks[i][1]))
            blocks[i][1] += 1
            total += 1

    captions = [piece for text, stills in blocks for piece in _pieces(text, stills)]
    return (captions + [""] * frames)[:frames]


def rewrite(sentences: list[str], language: str | None = None,
            frames: int = 0) -> list[str] | None:
    """
    The transcript as one third-person account, dealt out as exactly `frames`
    captions in the order the frames come - or None if no part of it could be
    rewritten.

    `sentences` are the transcript's sentences as TEXT, in the order they were
    spoken. Their timestamps are not wanted here (see the module header).

    `frames` is how many stills the run is delivering, confirmed after they
    were prepared, and it is the whole of the budget: CAPTION_CHARS each, so
    forty frames is an account of at most 3,200 characters. Every one of them
    comes back with text, unless the transcript has fewer words than there are
    frames (see _deal).

    `language` is the ISO code Whisper detected while transcribing this same
    audio (transcriber.detected_language). It is not optional in spirit: with
    nothing to name, the prompt can only ask for "the same language as the
    transcript", and that was measured to fail on every line of a Portuguese
    transcript and every line of a French one. Passing None is the honest
    behaviour for a caller that genuinely does not know, and it leaves the
    English guard off, since guarding against English is meaningless when the
    video might be English.

    None means the caller should caption the frames the way a channel that
    asks for no rewrite does: no model, no weights on this machine, or a
    model that did not produce one usable line for the whole video. See the
    module header: a frame fetch that produced good stills is not allowed to
    fail over the words on them, and neither is a rewrite.
    """
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences or frames < 1:
        return None
    if not _ensure_loaded():
        return None

    # The name if it is one this file knows, the bare code if not, and the
    # vaguest possible wording if the caller had nothing at all.
    named = LANGUAGE_NAMES.get(language, language) if language else None
    budget = frames * CAPTION_CHARS
    system = SYSTEM.format(frames=frames, budget=budget, size=CAPTION_CHARS,
                           ceiling=CAPTION_CHARS_MAX,
                           language=named or "the same language as the transcript")
    reminder = LANGUAGE_REMINDER.format(language=named) if named else ""
    guard_english = bool(language) and language != "en"

    items = _plan(sentences, frames)
    lines, stats = _write(items, system, reminder, guard_english)
    if not stats["written"]:
        log.warning("narration: not one line of the rewrite could be used - "
                    "the frames keep the transcript as spoken")
        return None

    # Shortened before the names are varied, not after, so that what
    # reference.vary looks at is the wording that will actually be set - an
    # opening it varied on a line about to be replaced would be work thrown
    # away, and the variation it would have made on the replacement is one it
    # would never see.
    lines, shortened = _shorten(lines, frames, language)
    lines, varied = _vary(lines, language)

    # ...and only then cut into the pieces it is set in, each inside two
    # frames, and dealt out.
    captions = _deal(_set(lines), frames)

    log.info("narration: %s, %d frames (up to %d characters): %d sentences "
             "planned as %d items, %d lines written, %d kept as spoken, %d "
             "repeats dropped, %d shortened, %d openings varied; %d characters "
             "set, %d frames over %d characters, %d left blank",
             named or "language unknown", frames, budget, len(sentences), len(items),
             stats["written"], stats["spoken"], stats["repeats"], shortened, varied,
             sum(len(c) for c in captions),
             sum(1 for c in captions if len(c) > CAPTION_CHARS), CAPTION_CHARS,
             sum(1 for c in captions if not c))
    return captions
