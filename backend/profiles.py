"""
profiles.py — What this machine can actually afford to run, decided once at
startup instead of discovered accidentally per request.

Before this existed, degradation was silent and invisible: face_restorer
returned the source image unchanged if no restorer backend imported, and
inpainter fell back from LaMa to OpenCV Telea if the model wouldn't load.
Both are reasonable fallbacks, but nothing anywhere said which one was in
effect — a user on a small GPU got quietly worse output with no indication
that it wasn't the intended pipeline, and no way to trade quality for speed
deliberately.

A profile is that decision made explicit. It's chosen automatically from the
detected hardware and can be overridden with the TNMAKER_PROFILE environment
variable ("full", "balanced", "lite").

    full      Discrete CUDA GPU with comfortable VRAM. Everything on, face
              restoration at the widest context.
    balanced  CUDA GPU with limited VRAM (the common case: 4-6GB laptop and
              entry desktop cards), or a capable CPU. Face restoration is
              scoped to face ROIs rather than the whole frame — the restorer
              only ever touches aligned face crops anyway, so this is close
              to free in quality and is what keeps the model inside a small
              VRAM budget. Inpainting still runs.
    lite      No usable torch at all, or explicitly requested. The GAN stage
              is skipped entirely and only the classical pipeline (denoise,
              tone, clarity, sharpen) runs — which accounts for a good share
              of the visible improvement and is pure CPU/OpenCV. Inpainting
              falls back to Telea.
"""

import os
import logging
from dataclasses import dataclass

log = logging.getLogger("uvicorn.error")

# VRAM (GiB) at or above which "full" is chosen automatically. Below it, the
# ROI-scoped restoration path ("balanced") is both faster and the difference
# between fitting and an out-of-memory crash — GFPGAN over a full ~3.5MP
# pre-crop frame plus its own detector is what pushes a 4GB card over.
FULL_PROFILE_MIN_VRAM_GB = 6.0


@dataclass(frozen=True)
class Profile:
    name: str
    use_gan: bool             # run GFPGAN/CodeFormer at all
    gan_scope: str            # "roi" (face regions only) or "frame" (whole image)
    use_lama: bool            # LaMa inpainting, vs OpenCV Telea
    use_half_precision: bool  # fp16 weights on CUDA
    device: str               # "cuda" | "cpu" | "none"
    detail: str               # human-readable reason, for the startup log


PROFILES = {
    "full":     dict(use_gan=True,  gan_scope="frame", use_lama=True,  use_half_precision=False),
    "balanced": dict(use_gan=True,  gan_scope="roi",   use_lama=True,  use_half_precision=True),
    "lite":     dict(use_gan=False, gan_scope="roi",   use_lama=False, use_half_precision=False),
}

_profile: Profile | None = None


def _probe_torch() -> tuple[str, float, str]:
    """Returns (device, vram_gb, description) without raising if torch is absent."""
    try:
        import torch
    except Exception as e:
        return "none", 0.0, f"torch unavailable ({type(e).__name__})"

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            gb = props.total_memory / (1024 ** 3)
            return "cuda", gb, f"{props.name}, {gb:.1f}GB VRAM"
        except Exception:
            return "cuda", 0.0, "CUDA device (properties unavailable)"
    return "cpu", 0.0, "CPU only (no CUDA device)"


def detect() -> Profile:
    """
    Picks (and caches) this process's profile. Called once at startup; every
    later call returns the same decision, so nothing can change pipeline
    behavior mid-session.
    """
    global _profile
    if _profile is not None:
        return _profile

    device, vram_gb, description = _probe_torch()

    requested = (os.environ.get("TNMAKER_PROFILE") or "").strip().lower()
    if requested in PROFILES:
        name = requested
        why = f"TNMAKER_PROFILE={requested}"
    elif device == "cuda" and vram_gb >= FULL_PROFILE_MIN_VRAM_GB:
        name, why = "full", "discrete GPU with ample VRAM"
    elif device in ("cuda", "cpu"):
        name, why = "balanced", "limited VRAM" if device == "cuda" else "no GPU, using CPU"
    else:
        name, why = "lite", "no usable torch install"

    settings = dict(PROFILES[name])
    # fp16 is only meaningful on CUDA. Note it buys VRAM headroom rather than
    # much speed on cards without tensor cores (e.g. GTX 16-series) — which
    # is exactly the case that needs the headroom.
    if device != "cuda":
        settings["use_half_precision"] = False

    _profile = Profile(name=name, device=device, detail=f"{description} — {why}", **settings)
    return _profile


def get() -> Profile:
    return detect()


def log_startup() -> None:
    p = detect()
    log.info(
        "profile=%s (%s) | face restoration: %s | inpainting: %s | fp16: %s",
        p.name, p.detail,
        f"{p.gan_scope}-scoped GAN" if p.use_gan else "classical pipeline only (no GAN)",
        "LaMa" if p.use_lama else "OpenCV Telea",
        "on" if p.use_half_precision else "off",
    )


def as_dict() -> dict:
    p = detect()
    return {
        "profile": p.name,
        "device": p.device,
        "detail": p.detail,
        "use_gan": p.use_gan,
        "gan_scope": p.gan_scope,
        "use_lama": p.use_lama,
        "half_precision": p.use_half_precision,
    }
