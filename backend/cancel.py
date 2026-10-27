"""
Stopping a run part-way, when the user who started it has gone.

A run takes minutes and holds the backend for all of them: one run at a time
(see api.process_video), and the proxy hands each backend to one user. When
that user closes the page or reloads it and says yes to leaving, the run is
working for nobody — and the next user given this backend would be told it is
busy until it ends. So the run is asked to stop, and stops at the next point
it checks: every progress update (api._set_progress), every chunk of a
download (video_downloader), and while a Drive download runs in its own
process.

Cooperative rather than forced because a Python thread cannot be killed from
outside, and because stopping between steps is what keeps the session's state
whole: a run cut off in the middle of writing a frame record would leave the
next one to trip over it. The cost is that a long step already under way —
scoring every sampled frame, transcribing the voice track — finishes before
the run notices. Downloads are the exception, and the step most worth it:
they are the longest and the least predictable, and they are cut mid-file.
"""

import threading


class RunCancelled(BaseException):
    """
    Raised at a checkpoint once the run has been asked to stop.

    BaseException, not Exception, on purpose — the way KeyboardInterrupt is.
    The pipeline is full of `except Exception` that turn a failure into a
    message or a retry (a download that fails is tried again; a fetch that
    fails becomes "Download failed: ..."), and every one of them would
    otherwise catch the request to stop and carry on, or report it to nobody
    as an error.
    """


_requested = threading.Event()


def request() -> None:
    """Asks the run in flight to stop at its next checkpoint."""
    _requested.set()


def clear() -> None:
    """Called as a run starts, so a request aimed at the last one cannot stop it."""
    _requested.clear()


def requested() -> bool:
    return _requested.is_set()


def check() -> None:
    """A checkpoint: returns if the run should go on, raises RunCancelled if not."""
    if _requested.is_set():
        raise RunCancelled()


def wait(seconds: float) -> bool:
    """Sleeps up to `seconds`, waking early if a stop is requested. True if it was."""
    return _requested.wait(seconds)
