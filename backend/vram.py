"""
vram.py — Handing GPU memory back to the driver between jobs.

PyTorch's caching allocator never returns a freed block to the driver on its
own: it keeps it for the next allocation of a similar size. That is the right
default for a training loop feeding one fixed shape, and the wrong one here,
where every frame is a different size and the pool only ever ratchets up to
the largest thing the session has ever seen. Measured on this app's own
pipeline, on a 4GB card:

    after model preload              reserved =    732 MB
    after a 1280x720  frame          reserved =  1,672 MB
    after a 2560x1440 frame          reserved =  2,350 MB
    after a 3840x2160 frame          reserved =  3,874 MB
    back down to a 1280x720 frame    reserved =  3,874 MB   <- never falls
    torch.cuda.empty_cache()         reserved =    744 MB

A full editing session (twenty frames restored, then reframed) drove it to
10,756 MB on that same 4GB card — the Windows driver silently spills the
overflow into system RAM, which is why it never surfaced as an OOM and why
the process sat at ~15GB of commit. None of it was released between videos,
so the second video started from wherever the first one had left the pool.

`release()` is deliberately not called after every inference. Emptying the
cache costs a device synchronise and hands back blocks the very next frame
would probably have reused, so doing it constantly trades the leak for a
slower editing loop. It fires only when the pool is holding a meaningful
amount of memory that nothing is using — see RELEASE_SLACK_MB — with
`force=True` at the points where the app genuinely has nothing in flight
(between videos).
"""

import sys
import logging

log = logging.getLogger("uvicorn.error")

# Unused reserved memory (reserved minus allocated) above which release()
# actually calls into the allocator. Below this the pool is doing its job —
# holding blocks the next frame will reuse — and emptying it would only buy a
# slower next allocation.
RELEASE_SLACK_MB = 512


def _torch():
    """
    torch, but only if something has already imported it AND CUDA is usable.

    Looked up through sys.modules rather than imported: on a machine with no
    restoration backend available, torch is never loaded at all, and this
    module must not be the thing that pulls tens of seconds of import cost
    into a request purely to free memory that was never allocated.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def stats() -> tuple[float, float]:
    """(allocated, reserved) in MB — (0, 0) when there's no CUDA in play."""
    torch = _torch()
    if torch is None:
        return (0.0, 0.0)
    return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)


def release(force: bool = False) -> float:
    """
    Returns the caching allocator's unused blocks to the driver, and reports
    how many MB that freed.

    A no-op when there's no CUDA, and (unless `force`) when the pool is
    holding less than RELEASE_SLACK_MB of slack — see this module's header for
    why this isn't unconditional.
    """
    torch = _torch()
    if torch is None:
        return 0.0

    reserved = torch.cuda.memory_reserved()
    slack = reserved - torch.cuda.memory_allocated()
    if not force and slack < RELEASE_SLACK_MB * 2**20:
        return 0.0

    torch.cuda.empty_cache()
    freed = (reserved - torch.cuda.memory_reserved()) / 2**20
    if freed >= 1.0:
        log.info("vram: released %.0f MB back to the driver (%.0f MB still reserved)",
                 freed, torch.cuda.memory_reserved() / 2**20)
    return freed
