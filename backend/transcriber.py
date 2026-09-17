"""
transcriber.py — what was being SAID when a frame appeared.

A frame fetch already knows WHEN each still it hands back happens: every slot
carries the second it was cut from (see frame_grab.Grab.timestamp, and the
`timestamp` each frame is sent to the frontend with in api.py). That number
was recorded and never used — "for anything that wants to compute with them",
as the comment there put it. This is the thing that wants to.

It supplies the other half of the pair: the words of the voice track, each
with the seconds it was spoken at. Cross the two and a frame can be captioned
with the sentence that was being said over it, which is what the Snapchat pack
turns into the line of type at the bottom of every still it produces.

## Why sentences and not segments

Whisper hands back SEGMENTS, and a segment is a unit of decoding, not a unit
of meaning. It ends where the model's attention window ended, which is
routinely in the middle of a clause: "the council said on Tuesday that the
bridge" / "would stay closed until spring". Either half, dropped on a frame on
its own, is a caption that says nothing — the reader has no previous card to
carry the rest of the thought over from, because there is no previous card.
Each frame is looked at alone.

So the words are re-assembled into sentences here, and everything in the
sentence-assembly block below exists to answer one question: is this run of
words something a person could read by itself. Three rules, and they are
deliberately about SHAPE rather than about meaning:

  * a full stop, a question mark or an exclamation mark ends a sentence. That
    is the speaker's own punctuation as the model heard it, and it is by far
    the strongest signal available.
  * a long enough pause ends one too, but only once enough words have been
    said to be worth ending. Speech has no commas in it; the breath between
    two thoughts is where the punctuation would have gone, and a model that
    missed the full stop rarely misses the silence.
  * anything left too short to stand up is glued back onto the neighbour it
    belongs to — forward when it never terminated (it is the head of what
    comes next), backward when it did (it is the tail of what came before).

That is a heuristic and it is worth being honest about its ceiling: it decides
whether a line is COMPLETE, not whether it is informative. "He said that was
not the case." is a whole sentence and tells a reader very little. Judging the
second thing is a language model's job, and there is not one in this app —
every model here is a convolutional net doing one measurable thing. The
interface this module presents (a list of Sentence, and captions_for) is the
seam a language model would be dropped into if that ever changes; nothing
upstream of it would need to know.

## The language is the video's

The caption comes out in whatever was spoken, because it is the video's own
words being put back on the video's own frames — a Portuguese clip captioned
in English would be captioning something nobody said. That is content and not
interface, so it is not what the English rule in CLAUDE.md is about; every
string in THIS file, and every label the user reads around the caption, is
still English.

## Never fatal

A frame fetch that produced forty good stills has done its job. No failure in
here — no audio track, no model, no disk space for the weights, a video that
is thirty minutes of music — may take that away, so everything public returns
empty rather than raising, and the caller carries on with frames that simply
have no caption on them.
"""

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frame_extractor import FFMPEG, _NO_WINDOW

log = logging.getLogger("uvicorn.error")  # see the note in face_restorer.py

# Which Whisper the app runs, by the name faster-whisper resolves against its
# published conversions.
#
# "small" is the smallest one whose punctuation can be relied on, and the
# punctuation is not a nicety here: it is the first of the three rules the
# sentence assembly runs on. "base" transcribes recognisably but drops full
# stops for whole paragraphs at a time, which leaves every sentence boundary
# to the pause rule alone and produces captions that break mid-clause. Going
# up to "medium" triples the download and roughly triples the time for an
# accuracy gain that does not change where the sentences are.
MODEL_SIZE = "small"

# Beside the app's own code and not in the user's home cache, for the reason
# background_remover.MODEL_DIR gives: an uninstall that deletes the install
# folder should take everything it downloaded with it and nothing else.
# `backend/models` is named in updater.KEEP_ACROSS_UPDATES, so a patch
# replacing app/ wholesale does not throw the weights away.
MODEL_DIR = Path(__file__).resolve().parent / "models" / "whisper"

# CPU, int8, and that is a decision rather than a fallback.
#
# The GPU in this app is already oversubscribed: GFPGAN, LaMa and U2-Net all
# want it, vram.py exists because they were exhausting it between them, and a
# capture run is the one place all three are used on every frame in the grid.
# Whisper on CUDA would join that queue for a job that is not the bottleneck —
# transcription runs ONCE per video while restoration runs once per frame.
#
# int8 on the CPU transcribes several times faster than the video plays, which
# for the clips this channel cuts from is a handful of seconds, and it leaves
# the card entirely to the pipeline that needs it. It also sidesteps
# ctranslate2's own CUDA/cuDNN discovery on Windows, which is a second set of
# native libraries to be missing on a user's machine.
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# What Whisper is trained on. The audio is decoded straight to it rather than
# resampled afterwards, so ffmpeg does the conversion once and there is no
# second opinion about it.
SAMPLE_RATE = 16000

# ── Sentence assembly ─────────────────────────────────────────────────────

# What ends a sentence when the model wrote it down. Closing brackets and
# quotes are allowed to follow, because "...over." and "...over.'" end equally
# firmly.
TERMINAL = re.compile("[.!?…][\"'”’)\\]]*$")

# What marks a boundary INSIDE a sentence rather than the end of one. Only
# _split_overlong reads it, and only when it has run out of every better place
# to break — see the note there.
INTERNAL = re.compile("[,;:—–][\"'”’)\\]]*$")

# ...and the same marks, stripped off the end of a run that a split left
# holding one. A comma says "this goes on", which is true of the sentence and
# false of the caption: what the frame carries has to look finished, because
# nothing follows it on that frame. A full stop is never stripped — that one
# is the speaker's.
DANGLING = re.compile("[,;:—–]+$")

# A silence this long, or longer, is where the sentence the speaker was not
# punctuating actually ended.
#
# 0.65s is above the gap inside fluent speech (a comma's worth of breath
# measures around 0.2-0.4s) and below the pause between two spoken sentences,
# which is rarely under 0.7s in delivered narration of this kind. Set lower,
# every clause becomes its own caption; set much higher, two sentences the
# model failed to punctuate arrive as one over-long line.
PAUSE_SPLIT_S = 0.65

# ...and the pause above which a break is taken to be a real one even though
# nothing was punctuated.
#
# The gap rule above has to be set low, at the length of a hesitation, or it
# misses the sentence boundaries the model failed to write down. Set that low
# it also fires INSIDE sentences, on the breath somebody takes mid-clause —
# which is precisely the cut this module exists to undo. The two cases are not
# distinguishable by punctuation (there is none, by assumption) but they are
# distinguishable by length: a hesitation is under a second, and the silence
# between two delivered sentences is longer than that.
#
# So a run left unterminated is put back together with the one after it when
# the silence between them is shorter than this, and left alone when it is
# longer. See _merge_continuations, which also refuses the join outright when
# the result would be too long to set — an over-long caption is a different
# failure from a truncated one, and not a better one.
PAUSE_HARD_S = 1.1

# Below this many words a run cannot stand on its own whatever punctuation it
# carries, and is merged back into a neighbour. Four, because three-word
# utterances that ARE whole ("He was arrested.") are rare next to three-word
# fragments that are not ("...and the police").
MIN_WORDS = 4

# ...and above these it stops being a caption. Four lines of Arial Black
# across a 540px frame is roughly 110 characters before the type is too small
# to read at Snapchat's size, so a sentence past either bound is split at its
# own widest internal pause rather than handed over to be set at 9 points.
MAX_WORDS = 18
MAX_CHARS = 120

# How far from a frame a sentence may be and still be called what was being
# said over it.
#
# Most frames land INSIDE a sentence and this never comes up. It decides the
# others: a still cut from a silent establishing shot, a music sting, the beat
# between two lines. Two and a half seconds is about the longest gap that
# still reads as the same moment; past it the nearest words are from a
# different part of the video, and no caption is better than a confident wrong
# one, so the frame is left blank for the user to type into.
MATCH_MAX_GAP_S = 2.5

# What one caption should weigh, in characters.
#
# It is the length a caption is WRITTEN to, not a limit discovered afterwards:
# narrator.py is told this number and asked for lines that fit it (see SYSTEM
# there), and this module holds it because both halves need it and the
# dependency runs this way — narrator imports transcriber, never the reverse.
#
# Eighty, because that is a caption a viewer takes in at a glance on a phone
# and it is roughly what four lines of this app's caption type can hold at
# twenty characters a line (see `maxChars` in js/channels.js). It used to be
# 132, which is a paragraph: what that produced was a frame carrying two
# thoughts, of which the picture underneath illustrated one.
CAPTION_CHARS = 80

# ...and the length one may reach when a thought will not go into 80.
#
# Twice it, and the doubling is not a rounding — it is the second frame. A
# caption over CAPTION_CHARS is not set on one still; it is split across two
# consecutive ones (see captions_for), so what this really says is "a thought
# may take two images and never three". Past 160 the rewrite is cut back at a
# clause boundary rather than spread further, because a sentence that needs
# three frames is a sentence nobody reads to the end of: by the third the
# picture has changed twice.
CAPTION_CHARS_MAX = CAPTION_CHARS * 2

# Where one caption may be cut in two when it will not fit on one still — see
# split_caption. Both want a mark with a SPACE after it: the comma inside
# "3,000" and the point inside "9.5" are not places a sentence comes apart.
#
# One pattern and not two, because which MARK it is turns out not to be the
# question — see split_caption, where what a mark is followed by decides
# whether it is a place to cut.
CAPTION_BREAK = re.compile(r"[.!?\u2026,;:\u2014\u2013][\"'\u201d\u2019)\]]*(?=\s)")


@dataclass
class Sentence:
    """One thing said, whole, and when it was said."""
    text: str
    start: float
    end: float


@dataclass
class _Word:
    """One word as the model timed it — the raw material sentences are cut from."""
    text: str
    start: float
    end: float


_model = None
_load_failed = False
# The language of the last transcription, as Whisper detected it, or None.
#
# Module state rather than a return value, because `transcribe` answers a list
# of Sentence and that is an interface other things read. One video is
# processed at a time (see the process lock in api.py), so "the last
# transcription" is never ambiguous while anybody is asking.
#
# It exists for narrator.py. Whisper works the language out for free on its
# way to the words and this used to be discarded on the line that called it;
# telling a language model the name of the language it is writing in turns out
# to be the difference between a caption and a translation. See the language
# rule in narrator.SYSTEM.
_language = None
# Serialises the LOAD, not the transcription: on an install that has yet to
# fetch the weights, two runs arriving together would otherwise start the same
# multi-hundred-megabyte download into the same directory. Transcription
# itself needs no lock — one video is processed at a time (see the process
# lock in api.py).
_load_lock = threading.Lock()


def _ensure_loaded(download: bool = True) -> bool:
    """
    Builds the model once, downloading the weights if this machine has not got
    them yet. False means transcription is unavailable and captions will
    simply not appear.
    """
    global _model, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False

    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            # The one failure worth its own message: it is not a broken
            # machine, it is an install that predates this feature, and
            # installing faster-whisper is the whole fix.
            log.warning("transcription: faster-whisper is not installed (%s) — "
                        "frames will arrive without captions", e)
            _load_failed = True
            return False

        # Nothing on disk and nothing allowed to arrive: report unavailable
        # rather than letting the library reach for the network on a caller
        # that asked it not to.
        if not download and not any(MODEL_DIR.rglob("*.bin")):
            return False

        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            log.info("transcription: loading Whisper %s (%s/%s)", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
            _model = WhisperModel(
                MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE,
                download_root=str(MODEL_DIR),
            )
            return True
        except Exception as e:
            log.warning("transcription: Whisper %s could not be loaded (%s)", MODEL_SIZE, e)
            _load_failed = True
            return False


def preload(download: bool = True) -> bool:
    """
    Build the model now rather than inside the run that first needs it. Called
    at server startup alongside the other four; returns whether transcription
    is available at all.
    """
    return _ensure_loaded(download)


def is_available() -> bool:
    """Whether a capture run can expect captions. False means frames come back bare."""
    return _model is not None or not _load_failed


def _audio(video_path: str) -> np.ndarray | None:
    """
    The voice track as mono float32 at SAMPLE_RATE, or None for a video that
    has no audio in it at all.

    Decoded through the same ffmpeg every frame in this app comes out of
    rather than through the library's own reader: faster-whisper decodes with
    PyAV, which is a second media stack with a second set of codecs to be
    missing, on a machine that already has a working one. Handing it a numpy
    array skips that path entirely.

    Written to a pipe rather than a temp file because the whole of it is
    wanted at once anyway — 32 KB per second of video, so a ten-minute clip is
    19 MB, which is a fraction of what a single frame's restoration holds.
    """
    try:
        result = subprocess.run(
            [FFMPEG, "-nostdin", "-i", video_path, "-vn",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True, creationflags=_NO_WINDOW,
        )
    except Exception as e:
        log.warning("transcription: could not run ffmpeg (%s)", e)
        return None

    # A video with no audio stream fails here, and that is not an error worth
    # reporting as one: a silent clip is a clip with nothing to transcribe.
    if result.returncode != 0 or not result.stdout:
        log.info("transcription: no audio track in %s", os.path.basename(video_path))
        return None

    # Trimmed to whole samples before the view is taken: a pipe closed mid-
    # sample would otherwise raise on a buffer that is not a multiple of 2.
    raw = result.stdout
    pcm = np.frombuffer(raw[:len(raw) - (len(raw) % 2)], dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _words(audio: np.ndarray) -> list[_Word]:
    """
    Every word the model heard, with the seconds it was spoken at.

    `word_timestamps` is what makes the assembly below possible at all: a
    sentence cut out of the middle of a segment has to be able to say when it
    started, and segment-level timing can only say when the segment did.

    `vad_filter` keeps the music beds, room tone and applause these clips are
    full of from being decoded into the confident nonsense Whisper produces
    over non-speech — which, left in, would caption an establishing shot with
    a sentence nobody said.
    """
    global _language
    # `info` is answered before the first segment is decoded, so this costs
    # nothing beyond keeping it.
    segments, info = _model.transcribe(audio, word_timestamps=True, vad_filter=True)
    _language = info.language
    words: list[_Word] = []
    for segment in segments:
        for word in (segment.words or []):
            text = word.word.strip()
            if text:
                words.append(_Word(text=text, start=float(word.start), end=float(word.end)))
    return words


def _cut(words: list[_Word]) -> list[list[_Word]]:
    """
    The first pass: words split into runs at the two boundaries that are
    visible without understanding anything — a terminal mark, and a pause.

    Length is deliberately not one of them. Cutting at a word count here is
    cutting at a number, and a number lands wherever it lands: on real footage
    it produced "...from Beyonce to Sir Elton" with "John, and now an entire
    nation..." beginning the next caption. Runs are allowed to grow instead,
    and the one place that knows how to choose a break decides where they come
    apart — see _split_overlong, which looks for the speaker's own comma
    first and falls back to the audio's widest pause.
    """
    runs: list[list[_Word]] = []
    current: list[_Word] = []

    for i, word in enumerate(words):
        gap = word.start - words[i - 1].end if i else 0.0
        if current and gap >= PAUSE_SPLIT_S and len(current) >= MIN_WORDS:
            runs.append(current)
            current = []
        current.append(word)
        if TERMINAL.search(word.text):
            runs.append(current)
            current = []

    if current:
        runs.append(current)
    return runs


def _too_long(run: list[_Word]) -> bool:
    """Whether a run has outgrown what four lines of type can hold."""
    return len(run) > MAX_WORDS or len(" ".join(w.text for w in run)) > MAX_CHARS


def _merge_continuations(runs: list[list[_Word]]) -> list[list[_Word]]:
    """
    The first repair: a run cut off mid-clause by a hesitation, put back
    together with the words that finish it.

    This is the failure the whole module is aimed at — "the council said on
    Tuesday that the bridge" is eight words and passes every length test in
    here, so nothing else would ever look at it again, and it is not a
    sentence. What gives it away is that it does not END: no full stop, no
    question mark, nothing. A speaker who paused mid-thought leaves exactly
    that trace.

    Two guards keep it from swallowing the video whole. The pause has to be
    short enough to be a hesitation rather than a sentence boundary
    (PAUSE_HARD_S), and the join is refused if it would produce a caption
    nobody can set — a run-on of two sentences is not an improvement on a
    truncated one.

    One forward pass rather than a fixpoint: each run is offered to the one
    being built, and a chain of three cut-off pieces is joined a piece at a
    time by the same test.
    """
    merged: list[list[_Word]] = []
    for run in runs:
        previous = merged[-1] if merged else None
        if (previous
                and not TERMINAL.search(previous[-1].text)
                and run[0].start - previous[-1].end < PAUSE_HARD_S
                and not _too_long(previous + run)):
            previous.extend(run)
        else:
            merged.append(list(run))
    return merged


def _merge_fragments(runs: list[list[_Word]]) -> list[list[_Word]]:
    """
    The second repair: anything still too short to be read on its own is put
    back with the run it was cut from.

    WHICH neighbour is the whole of the decision, and the run's own last word
    answers it. A fragment that ends in a full stop is the tail of the thought
    before it — "...until spring." belongs to the sentence that started "The
    council said". One that ends in nothing is the head of the thought after
    it, cut off by a pause somebody took mid-clause, and belongs forwards.

    A merge can leave the result short again (two two-word fragments make a
    four-word one, which is only just standing), so this runs until nothing
    moves rather than once over the list.
    """
    merged = [list(run) for run in runs]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, run in enumerate(merged):
            if len(run) >= MIN_WORDS:
                continue
            backwards = bool(TERMINAL.search(run[-1].text))
            # ...unless there is no neighbour that way, in which case the only
            # neighbour there is takes it. A fragment at either end of the
            # video still has to go somewhere.
            if backwards and i > 0:
                merged[i - 1].extend(run)
            elif not backwards and i < len(merged) - 1:
                merged[i + 1][:0] = run
            elif i > 0:
                merged[i - 1].extend(run)
            else:
                merged[1][:0] = run
            merged.pop(i)
            changed = True
            break
    return merged


def _split_overlong(run: list[_Word]) -> list[list[_Word]]:
    """
    A run that no four lines of type can hold, broken at the best boundary it
    still has left.

    This is the last resort and it shows: the punctuation rule and the two
    pause rules have all already been spent, so what is left is a speaker who
    said twenty-five words without stopping. The break has to go somewhere, and
    where it goes is the difference between two readable halves and a caption
    ending "...from Beyonce to Sir Elton" with the name finished on the next
    frame. That was a real result on real footage, and it is why this looks at
    two things rather than one:

      * a COMMA (or a semicolon, colon or dash) the model wrote down. Speech
        has clause boundaries even when it has no sentence boundaries, and this
        is the only place they are written. Preferred whatever the timing says,
        and the one nearest the middle is taken so both halves come out worth
        reading.
      * failing that, the widest pause — the nearest thing to a clause
        boundary the audio itself has left. Ties go to the break nearest the
        middle, and that tie-break is not a nicety: delivered narration paces
        its words evenly, so the pauses inside one long run are often equal to
        the hundredth of a second, and a plain "widest wins" then always
        picked the earliest candidate. That left MIN_WORDS on the first line
        and everything else on the second, which the recursion below then
        did again, and again — a run of stubs down one edge.

    Recursive, because one break is not always enough, and bounded by MIN_WORDS
    on both halves so the split cannot manufacture the fragments the pass above
    just finished removing.
    """
    if not _too_long(run):
        return [run]
    if len(run) < MIN_WORDS * 2:
        return [run]

    span = range(MIN_WORDS, len(run) - MIN_WORDS + 1)
    middle = len(run) / 2

    # A break AFTER a word ending in one of these is a break at a clause the
    # speaker marked themselves.
    clause = [i for i in span if INTERNAL.search(run[i - 1].text)]
    best = (min(clause, key=lambda i: abs(i - middle)) if clause
            else max(span, key=lambda i: (run[i].start - run[i - 1].end, -abs(i - middle))))

    return _split_overlong(run[:best]) + _split_overlong(run[best:])


def _sentence(run: list[_Word]) -> Sentence:
    """One run of words written out as the caption it will become."""
    # Collapsed rather than joined blindly: the model emits its own spacing
    # inside a word for some languages, and a caption with a double space in
    # it is a caption somebody has to edit for no reason.
    text = re.sub(r"\s+", " ", " ".join(w.text for w in run)).strip()
    return Sentence(text=DANGLING.sub("", text), start=run[0].start, end=run[-1].end)


def transcribe(video_path: str) -> list[Sentence]:
    """
    The video's voice track as whole sentences, in the order they were spoken.

    Empty for anything that did not work — no audio, no model, no speech, a
    decode that failed halfway. See the module header: a frame fetch that
    produced good stills is not allowed to fail over the words on them.
    """
    global _language
    _language = None
    if not _ensure_loaded():
        return []
    try:
        audio = _audio(video_path)
        if audio is None or not audio.size:
            return []
        words = _words(audio)
    except Exception as e:
        log.warning("transcription: failed (%s) — frames will have no captions", e)
        return []

    if not words:
        log.info("transcription: no speech found")
        return []

    runs = _merge_fragments(_merge_continuations(_cut(words)))
    sentences = [_sentence(part) for run in runs for part in _split_overlong(run)]
    sentences = [s for s in sentences if s.text]
    if sentences:
        log.info("transcription: %d sentences over %.1fs of speech",
                 len(sentences), sentences[-1].end - sentences[0].start)
    return sentences


def detected_language() -> str | None:
    """
    The language the last transcription came back in, as an ISO code ("pt",
    "fr", "en"), or None when nothing has been transcribed or the attempt
    failed.

    Whisper decides this from the audio itself, which is the only thing in the
    app that knows: the caption is the video's own words, and no channel pack
    or user setting says what language a given video was shot in.
    """
    return _language


def _match(sentences: list[Sentence], seconds: float) -> int | None:
    """
    Which sentence was being said at `seconds`, by index, or None.

    A frame is an instant and a sentence is an interval, so the answer is the
    interval the instant falls in. A frame that falls in a gap takes the
    nearest sentence, but only within MATCH_MAX_GAP_S: past that the words are
    from another part of the video, and no caption is better than a confident
    wrong one.
    """
    best, best_distance = None, float("inf")
    for i, sentence in enumerate(sentences):
        if sentence.start <= seconds <= sentence.end:
            return i
        distance = sentence.start - seconds if seconds < sentence.start else seconds - sentence.end
        if distance < best_distance:
            best, best_distance = i, distance
    return best if best is not None and best_distance <= MATCH_MAX_GAP_S else None


def split_caption(text: str, limit: int, most: int = 2) -> list[str]:
    """
    One caption divided into the screens it takes: one, or two, and never a
    third.

    Divided where the writer STARTED SOMETHING NEW, or not at all. That is the
    whole of the rule, and it is the third version of this function. The first
    cut at the nearest word to the middle whenever no punctuation was handy:

        "Brian Siever, son of Cassie Parton and representing the Parton"
        "and announced the passing of his Aunt Dolly, Rebecca Parton Dean."

    Two stills, seconds apart, and the second opens with "and". Neither half
    is a caption — the first stops in the middle of a phrase, the second has
    no subject, and no reader watching one frame at a time can repair either.

    The second version cut only at punctuation, which fixed the first half and
    not the second: a comma is a real mark and

        "...for the film The Bodyguard"
        "and it became the best selling single by a woman."

    still puts a verb with no subject on its own still. The mark was never the
    problem. What matters is whether what FOLLOWS it begins something, and
    there is a signal for that which costs nothing and belongs to no language
    in particular: case. A piece that starts with a capital is a new sentence,
    or a name — either way it says who or what it is about. A piece that
    starts in lower case is the back half of something, and always reads as
    one.

    That is `_continues` run backwards, and it inherits its limits knowingly:
    in a script with no case at all — Japanese, Chinese, Arabic — no character
    is lower case, the test never fires, and every mark is a candidate. A
    weaker pass, not a broken one.

    Both halves have to fit `limit` and both have to be at least MIN_WORDS
    long; among the cuts that qualify, the most even one wins. Evenness is the
    last question asked and not the first — a screen holding eighty characters
    followed by one holding nine reads as a caption that ran out, but a cut in
    the wrong place reads as a mistake.

    A caption with nowhere to come apart comes back WHOLE, on one still. It is
    over the limit and the renderer sets it anyway (a long line wraps; it is
    not lost), which is a caption that is too long rather than two that are
    not sentences. Which of those is worse is not a close call — and it is not
    a case that should arise, because narrator._fit puts every line it writes
    through this function and refuses to keep one over a screen that does not
    survive it.

    A DANGLING mark left at the end of the first half is stripped, for the
    reason the sentence assembly strips it: a comma says "this goes on", which
    is true of the sentence and false of the still it is now the whole of.
    """
    if most < 2 or len(text) <= limit:
        return [text]

    best = None
    for mark in CAPTION_BREAK.finditer(text):
        head, tail = text[:mark.end()].strip(), text[mark.end():].strip()
        if len(head) > limit or len(tail) > limit:
            continue
        if len(re.findall(r"\S+", head)) < MIN_WORDS:
            continue
        if len(re.findall(r"\S+", tail)) < MIN_WORDS:
            continue
        if tail[:1].islower():
            continue
        spread = abs(len(head) - len(tail))
        if best is None or spread < best[0]:
            best = (spread, head, tail)

    return [DANGLING.sub("", best[1]), best[2]] if best else [text]


def captions_for(sentences: list[Sentence], moments: list[float]) -> list[str]:
    """
    The caption for each frame of a run, decided for the grid as a whole.

    Answered for every frame at once rather than one at a time, and that is
    the whole of what this does that matching each frame on its own does not.
    Answered separately, several frames landing inside one sentence all get
    the same words, and a grid where four stills in a row carry the same line
    is a grid the user has to retype: what it looks like is not four captions
    but one caption that failed to advance.

    Two ways of laying the account over the grid, and which one runs is
    decided by whether the account was WRITTEN for this grid.

    ## In order, when there is a caption for every still

    narrator.py is told how many stills the run is about to deliver and writes
    that many captions (see `_group` there). When it has — when there are no
    more captions than there are frames — they are dealt out in order: the
    first caption to the first still, the second to the second, a caption that
    needs two screens taking two of them.

    That is the arrangement the whole pipeline is pointed at. Every still
    carries a line, every line the model wrote is seen, and the account reads
    down the grid in the order it was written. Matching by timestamp cannot
    promise any of it: frames are chosen on picture quality, so two of them
    fall inside one sentence while the next sentence has none, and the run
    comes back with duplicated blanks in the middle of a story that had
    exactly enough words to fill it.

    What it gives up is the last of the timing. A caption no longer sits over
    the instant its words were spoken; it sits at its own place in the
    account. Both lists are in time order, so the two stay close — caption ten
    of forty lands on still ten of forty, and both are about a quarter of the
    way through the video — but "close" is the whole of the promise now. That
    is a deliberate trade and the second half of one that started when a
    sentence was first split across two frames.

    ## By timestamp, when there is not

    A run with more captions than frames was not written for this grid: the
    rewrite was unavailable and these are the sentences as spoken, or the
    transcript made more thoughts than the model could condense. Dealing those
    out in order would caption the whole grid with the opening minute of the
    video and never reach the rest, so each frame takes the sentence that was
    being said at the moment it was cut from, and the sentences that no frame
    lands on are not shown.

    ## Both ways

    No two frames ever carry the same text. A caption over CAPTION_CHARS is
    split across two consecutive stills rather than repeated on both, and one
    with nowhere to come apart is left whole on a single still (see
    split_caption). A frame with nothing left to say comes back "" — an empty
    text box the user types into, which is what a frame over silence already
    gets.
    """
    captions = [""] * len(moments)
    if not sentences or not moments:
        return captions

    order = sorted(range(len(moments)), key=lambda k: moments[k])

    # Written for this grid: deal the account out over it, in order.
    if len(sentences) <= len(moments):
        frame = 0
        for sentence in sentences:
            if frame >= len(order):
                break
            pieces = split_caption(sentence.text, CAPTION_CHARS,
                                   min(2, len(order) - frame))
            for piece in pieces:
                captions[order[frame]] = piece
                frame += 1
        return captions

    # Not written for this grid: each still takes what was being said over it.
    claims: dict[int, list[int]] = {}
    for frame in order:
        index = _match(sentences, moments[frame])
        if index is not None:
            claims.setdefault(index, []).append(frame)

    for index, frames in claims.items():
        pieces = split_caption(sentences[index].text, CAPTION_CHARS, min(len(frames), 2))
        for frame, piece in zip(frames, pieces):
            captions[frame] = piece
    return captions
