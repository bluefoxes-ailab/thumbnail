"""
The static server the page is loaded from — plus the one thing it generates.

It serves `frontend/` as files, and it serves `/content/` out of two
directories at once: the packs that shipped with the app, and the packs this
machine added. The merge of those two is published at
`/content/channels.json`, which `js/channels.js` fetches once at startup.

Two roots, because an update replaces the first one wholesale:

    <install>/app/frontend/content/   what shipped — a patch overwrites it
    <install>/content/               what this machine added — never touched

Both are served under the same `/content/` URL space, so a pack's own assets
are reachable by the same path whichever root it came from, and neither the
page nor a pack has to know which one it is in.

Serving them from HERE rather than from the backend on :8000 is deliberate.
Channel textures are drawn into the export canvas, and a canvas that has been
painted with an image from another origin is tainted: `toDataURL` throws and
every download fails. Same-origin keeps that whole class of problem out.
"""

import functools
import http.server
import io
import json
import os
import time
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent

# Where this machine's own packs live. The launcher points it at
# <install>/content; running from the repo it is <repo>/content if that
# exists, so a pack can be tried out without an install.
_external = os.environ.get("TNMAKER_CONTENT_DIR", "").strip()
EXTERNAL_CONTENT = Path(_external).resolve() if _external else (FRONTEND.parent / "content")

BUNDLED_CONTENT = FRONTEND / "content"

# Lowest first; a later root wins, so the machine's own packs override the
# shipped ones of the same id.
CONTENT_ROOTS = [BUNDLED_CONTENT, EXTERNAL_CONTENT]

# The generated document, and how long it may be reused. Rebuilt on a timer
# rather than per request because the page asks for it once per load, and
# rather than never because dropping a folder in should not need a restart.
CONTENT_DOC_PATH = "/content/channels.json"
CONTENT_DOC_TTL_S = 2.0


def _read_json(path, problems, label=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        # Named, not swallowed: a pack with a trailing comma in it silently
        # disappearing from the dropdown is the failure mode this whole
        # data-driven arrangement could most easily hide. The label is the
        # folder as well as the file, because "channel.json could not be read"
        # is true of one of a dozen identically named files.
        problems.append(f"{label or path.name} could not be read ({e}) — "
                        "the channel it defines is missing")
        return None


def _collect_singles(kind, problems):
    """
    Reads `<root>/<kind>/*.json` from every root into one id -> spec map.

    Used for faces/ and presets/, which are one file per thing. The id comes
    from the file's own `id` if it has one, and from its filename otherwise,
    so a pack that omits it still lands somewhere predictable.
    """
    out = {}
    for root in CONTENT_ROOTS:
        folder = root / kind
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            spec = _read_json(path, problems, f"{kind}/{path.name}")
            if spec is None:
                continue
            spec_id = str(spec.get("id") or path.stem)
            spec["id"] = spec_id
            spec["base"] = f"content/{kind}"
            out[spec_id] = spec
    return out


def _collect_channels(problems):
    """
    Reads `<root>/channels/<id>/channel.json` from every root.

    A folder name is the id. A channel present in both roots is taken from the
    later one WHOLE rather than merged with the earlier: half of one brand
    laid over half of another is not a thing anyone means, and "my copy of
    this channel" is.
    """
    found = {}
    for root in CONTENT_ROOTS:
        folder = root / "channels"
        if not folder.is_dir():
            continue
        for pack_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            manifest = pack_dir / "channel.json"
            if not manifest.is_file():
                problems.append(f"{pack_dir.name}/ has no channel.json, so it is not a channel pack")
                continue
            spec = _read_json(manifest, problems, f"{pack_dir.name}/channel.json")
            if spec is None:
                continue
            declared = spec.get("id")
            if declared and declared != pack_dir.name:
                problems.append(
                    f'the pack in {pack_dir.name}/ calls itself "{declared}". The folder name is '
                    "what identifies a channel, so that is what it is being loaded as."
                )
            spec["id"] = pack_dir.name
            # What a "./" path inside this pack is relative to. Always the URL
            # space, never the disk path — the page has no idea which root the
            # pack came from and does not need one.
            spec["base"] = f"content/channels/{pack_dir.name}"
            found[pack_dir.name] = spec

    # Order in the dropdown. `order` is what a pack uses to place itself
    # among the others; name is the tiebreak, so packs that all left it out
    # are at least alphabetical rather than in whatever order the filesystem
    # happened to list them.
    #
    # The tiebreak is not a fallback for the Snapchat packs, it is the whole
    # arrangement: every one of them declares the SAME `order`, so the group
    # is sorted by name and stays that way. Forty shows numbered one by one
    # is a list that goes wrong the moment a show is added in the middle --
    # either the new pack takes a number already spoken for, or every pack
    # after it is renumbered in the same commit. Sharing one number costs a
    # channel the ability to sit at a chosen spot inside its group, which no
    # Snapchat pack wants, and buys a new folder landing in its alphabetical
    # place with nothing else edited.
    return sorted(found.values(), key=lambda c: (c.get("order", 1000), str(c.get("name", c["id"])).lower()))


def content_document():
    """The merged view of every root — faces, presets, channels, and what went wrong."""
    problems = []
    doc = {
        "schema": 1,
        "generated": time.time(),
        "roots": [str(r) for r in CONTENT_ROOTS if r.is_dir()],
        "faces": _collect_singles("faces", problems),
        "presets": _collect_singles("presets", problems),
        "channels": _collect_channels(problems),
    }
    doc["problems"] = problems
    return doc


class ContentServingHandler(http.server.SimpleHTTPRequestHandler):
    """
    Disables caching entirely. This is a dev server for actively-edited
    frontend code — SimpleHTTPRequestHandler's default Last-Modified/ETag
    conditional-caching means a browser can keep serving a pre-edit copy of
    index.html after a real reload, making a just-shipped fix look like it
    silently didn't take effect. It applies just as well to an installed copy
    that has been patched underneath a browser tab that stayed open.
    """

    # Pinned rather than left to the platform: on Windows, mimetypes seeds
    # itself from the registry, where .js is routinely registered as
    # text/plain. A module script served with that type is rejected outright
    # by the browser's strict MIME checking, so the whole app would fail to
    # start on some machines and work on others.
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }

    _doc_cache = None
    _doc_cached_at = 0.0

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def translate_path(self, path):
        """
        Maps a URL to a file, looking through the content roots in turn.

        The base class is asked first and its answer is used as the sanitised
        form — that is where `..` and friends are neutralised, and redoing it
        here would mean maintaining a second copy of a security check. Only
        after it has produced a path under `frontend/` is the same relative
        path tried in the other roots.
        """
        local = super().translate_path(path)
        try:
            rel = Path(local).resolve().relative_to(FRONTEND)
        except ValueError:
            return local
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "content":
            return local

        inner = Path(*parts[1:])
        # Later roots win, so the machine's own copy of a file is preferred
        # over the shipped one — same rule as the packs themselves.
        for root in reversed(CONTENT_ROOTS):
            candidate = root / inner
            if candidate.exists():
                return str(candidate)
        return local

    def send_head(self):
        """Intercepts the one path that is generated rather than read off disk."""
        if self.path.split("?", 1)[0].rstrip("/") == CONTENT_DOC_PATH.rstrip("/"):
            return self._send_content_document()
        return super().send_head()

    def _send_content_document(self):
        cls = type(self)
        now = time.monotonic()
        if cls._doc_cache is None or now - cls._doc_cached_at > CONTENT_DOC_TTL_S:
            cls._doc_cache = json.dumps(content_document()).encode("utf-8")
            cls._doc_cached_at = now
        body = cls._doc_cache

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # send_head's contract is to return an open stream for do_GET to copy
        # and for do_HEAD to close, which is why this is a BytesIO rather than
        # a direct write.
        return io.BytesIO(body)


# Kept under the old name so anything that imported it still works.
NoCacheRequestHandler = ContentServingHandler


def main():
    port = int(os.environ.get("PORT", 3000))
    # Empty means every interface, which is what a dev box wants. The
    # installed app passes 127.0.0.1 instead: it is a single-machine tool, and
    # binding publicly only serves to raise a Windows Firewall prompt on first
    # run.
    host = os.environ.get("HOST", "")
    handler = functools.partial(ContentServingHandler, directory=str(FRONTEND))

    # Threaded, and with the address reusable. A page load is one document and
    # a dozen assets, and a single-threaded server hands them out strictly one
    # at a time behind whichever connection the browser opened first — which
    # is fine until one of them is slow and the whole page waits on it.
    with http.server.ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"Serving {FRONTEND} on port {port}")
        for root in CONTENT_ROOTS:
            print(f"  content: {root}" + ("" if root.is_dir() else "  (absent)"))
        httpd.serve_forever()


if __name__ == "__main__":
    main()
