"""
appinfo.py — what version this is, and what it knows about updates.

Small on purpose. The backend does not update anything and must not try: the
files it would be replacing are the ones it is running from, and the one
process in a position to do it safely is the launcher, before either server
exists (see installer/updater.py). What this does is answer the question, so
the page can show a version number and say when a restart would bring a newer
one.

Two files, and they say different things:

    app/release.json      what this BUILD is — ships inside the app directory,
                          so it is replaced along with it and can never
                          disagree with the code sitting next to it
    <install>/version.json  what this INSTALL is — lives outside app/, records
                            what has been applied to this machine and where it
                            looks for what comes next

Running from the repo there is no install root and no version.json, which is
not an error: the answer is then just the release file, and `installed` is
null. The paths below land on the right things either way — `app/backend`'s
parent is `app` in an install and the repo root out of it, and release.json
sits in both.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger("uvicorn.error")

HERE = Path(__file__).resolve().parent          # <install>/app/backend
APP_DIR = HERE.parent                            # <install>/app
INSTALL_DIR = APP_DIR.parent                     # <install>


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def release():
    """The build's own identity. Never None — an unknown version is still an answer."""
    return _read(APP_DIR / "release.json") or {"version": "unknown"}


def install_state():
    """The machine's record of itself, or None when running from the repo."""
    return _read(INSTALL_DIR / "version.json")


def as_dict():
    """
    Everything the page is told about versions, in one object.

    `update_available` is a comparison of two numbers already on disk — what
    the last check found against what is installed — and involves no network
    call. Deliberately: this endpoint is hit on every page load, and a page
    load is not a reason to talk to an update server. The check that does that
    runs once, in the launcher, at startup.
    """
    rel = release()
    state = install_state() or {}
    history = state.get("history") or []
    latest_applied = history[-1] if history else None

    return {
        "app": "Thumbnail Maker",
        "version": state.get("app_version") or rel.get("version") or "unknown",
        "build": rel.get("version"),
        "installed": state.get("installed"),
        "updated": state.get("updated"),
        "updates_configured": bool(state.get("feed")),
        "signed_updates": bool(state.get("public_key")),
        "last_check": state.get("last_check"),
        "last_result": state.get("last_result"),
        "last_update": latest_applied,
        # Where a user's own channel packs go, so the UI can point at it
        # rather than describing it. None when there is no install to point
        # into.
        "content_dir": str(INSTALL_DIR / "content") if (INSTALL_DIR / "content").is_dir() else None,
    }


def log_startup():
    info = as_dict()
    log.info(
        "version=%s | updates: %s | own channels: %s",
        info["version"],
        "configured" if info["updates_configured"] else "not configured",
        info["content_dir"] or "none",
    )
