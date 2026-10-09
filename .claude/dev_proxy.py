"""
Local preview of the frontend, with the processing done by a remote server.

Serves `frontend/` from this checkout — so whatever is being edited here is
what the page runs — and forwards every request that is not a file in it
(processing a video, fetching frames, uploads) to the server named on the
command line, which runs the real backends.

    python .claude/dev_proxy.py http://34.13.214.196:3000 3000

This exists because this machine has no backend of its own: the frontend's
API is same-origin (config.API is ""), so a page served by a plain static
server answers every POST with 501 and the status line reads "Error: Error".
frontend/serve.py is the production proxy and cannot stand in for this one —
its backends are fixed to 127.0.0.1:8000-8003 and it creates its log folder
under /home/... on start.

Usage events (POST /api/track/...) are answered here and NOT forwarded, so a
morning of testing does not land in the production adoption logs as real use.
"""

import http.server
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
REMOTE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://34.13.214.196:3000"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

# Hop-by-hop headers, and the ones this proxy rewrites itself. Content-Length
# is recomputed because the body is re-sent whole; Accept-Encoding is dropped
# on the way out so the remote answers uncompressed and nothing here has to
# decode it.
DROP_REQUEST = {"host", "connection", "accept-encoding", "content-length"}
DROP_RESPONSE = {"transfer-encoding", "connection", "content-encoding", "content-length"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def _local_file(self):
        path = urlparse(self.path).path
        if path in ("", "/"):
            return True
        target = (FRONTEND / path.lstrip("/")).resolve()
        return target.is_file() and FRONTEND in target.parents

    def _forward(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP_REQUEST}
        req = urllib.request.Request(REMOTE + self.path, data=body, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            self.send_error(502, f"Remote server unreachable: {e}")
            return
        data = resp.read()
        self.send_response(resp.status if hasattr(resp, "status") else resp.code)
        for k, v in resp.headers.items():
            if k.lower() not in DROP_RESPONSE:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        if self._local_file():
            super().do_GET()
        else:
            self._forward("GET")

    def do_HEAD(self):
        if self._local_file():
            super().do_HEAD()
        else:
            self._forward("HEAD")

    def do_POST(self):
        if urlparse(self.path).path.startswith("/api/track"):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(204)
            self.end_headers()
            return
        self._forward("POST")

    def do_PUT(self):
        self._forward("PUT")

    def do_DELETE(self):
        self._forward("DELETE")


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving {FRONTEND} on http://127.0.0.1:{PORT}, forwarding the rest to {REMOTE}", flush=True)
    server.serve_forever()
