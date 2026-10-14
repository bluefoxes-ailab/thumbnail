"""
Previews the captions a Static Studio run would set - from a video, a link,
or a transcript - without extracting a frame, touching the GPU or opening
the browser.

    python preview_captions.py "https://www.youtube.com/watch?v=..." --frames 40
    python preview_captions.py video.mp4 --frames 40
    python preview_captions.py transcript.txt --frames 40 --language pt

It runs the two steps a channel with `rewrite` runs, and nothing else: the
voice track transcribed by backend/transcriber.py (or a transcript read from a
text file, one sentence per line), then backend/narrator.py writing the
account for the given number of frames. What it prints is what the frames
would carry, in order, each with its length, and the totals against the
budget - 80 characters a frame.

Why this and not a run of the app: a capture run spends nearly all of its time
on the pictures - extraction, scoring, restoration - and the rewrite is the
last thing it does, so judging a change to the wording that way costs a whole
run per look. This is the same rewrite, reached directly, with the same
models.

The first run downloads what the app downloads - Whisper (about 500 MB) when
there is a video to transcribe, and the narration model (about 3 GB) - into
backend/models, the folder the backend itself uses. A checkout that has run
the backend already has them.

`--frames` stands in for the number of frames a run DELIVERS, which in the app
is confirmed after the frames are prepared and can be lower than the number
asked for. Here it is simply the number given.
"""

import argparse
import logging
import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND)


def _sentences(source: str, language: str | None) -> tuple[list[str], str | None]:
    """The transcript as text, and its language - from a file, a link or a video."""
    import transcriber

    if source.lower().endswith(".txt"):
        with open(source, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()], language

    if source.startswith(("http://", "https://")):
        import video_downloader
        print(f"Downloading {source} ...", flush=True)
        source = video_downloader.download_video(source)

    print(f"Transcribing {source} ...", flush=True)
    sentences = [s.text for s in transcriber.transcribe(source)]
    # Whisper's answer, unless the caller insisted on one - which is there for
    # trying a language the detection got wrong, not for normal use.
    return sentences, language or transcriber.detected_language()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview the captions a rewritten Static Studio run would set.")
    parser.add_argument("source", help="a video file, a YouTube/Drive/Dropbox link, "
                                       "or a .txt transcript with one sentence per line")
    parser.add_argument("--frames", type=int, default=40,
                        help="how many frames the run delivers (default 40)")
    parser.add_argument("--language",
                        help="the transcript's language as an ISO code (pt, en, fr...); "
                             "a video's is detected")
    parser.add_argument("--transcript", action="store_true",
                        help="also print the transcript as spoken, for comparison")
    args = parser.parse_args(argv)

    # The captions are in whatever language the video was, and a Windows
    # console left on its default code page would stop the whole preview at
    # the first accented letter rather than show it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # The backend logs through uvicorn's logger; with no uvicorn here, this is
    # what makes the "narration:" and "transcription:" lines show.
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    import narrator
    from transcriber import CAPTION_CHARS

    sentences, language = _sentences(args.source, args.language)
    if not sentences:
        print("No speech found - there is nothing to rewrite.")
        return 1

    print(f"\nTranscript: {len(sentences)} sentences, "
          f"{sum(len(s) for s in sentences):,} characters, "
          f"language {language or 'unknown'}")
    if args.transcript:
        for sentence in sentences:
            print(f"    {sentence}")

    budget = args.frames * CAPTION_CHARS
    print(f"Rewriting for {args.frames} frames, up to {budget:,} characters ...\n", flush=True)
    captions = narrator.rewrite(sentences, language, frames=args.frames)

    if captions is None:
        why = ("the narration model ran but produced no usable line"
               if narrator.loaded()
               else "the narration model is not available on this machine")
        print(f"\nNo rewrite: {why} - see the 'narration:' lines above. "
              "A run would set the transcript as spoken.")
        return 1

    print()
    for n, caption in enumerate(captions, 1):
        flag = "  <- over" if len(caption) > CAPTION_CHARS else "  <- empty" if not caption else ""
        print(f"{n:4d}  [{len(caption):3d}]  {caption}{flag}")

    print(f"\n{len(captions)} frames: {sum(len(c) for c in captions):,} of {budget:,} characters, "
          f"{sum(1 for c in captions if len(c) > CAPTION_CHARS)} over {CAPTION_CHARS}, "
          f"{sum(1 for c in captions if not c)} empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
