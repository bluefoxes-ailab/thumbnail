import sys, os, urllib.request, urllib.error, json, uuid, time, datetime
from http import cookies
from urllib.parse import urlparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
    "http://127.0.0.1:8004",
    "http://127.0.0.1:8005"
]

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

PROXY_PREFIXES = (
    "/process-video", "/upload", "/upload-video", "/upload_video",
    "/compose-frame", "/reframe-frame", "/enhance-frame", "/vary-frame",
    "/cutout", "/cutout-status", "/warm-cutouts", "/cleanup", "/export-pack",
    "/frame-wide"
)

SESSION_TIMEOUT_SECONDS = 45 * 60
LOGS_DIR = "/home/bluefoxesfreelancer/thumbnail/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class MultiUserProxyHandler(SimpleHTTPRequestHandler):
    user_backends = {}
    last_activity = {}

    @classmethod
    def cleanup_stale_sessions(cls):
        now = time.time()
        stale = [sid for sid, t in cls.last_activity.items() if now - t > SESSION_TIMEOUT_SECONDS]
        for sid in stale:
            backend = cls.user_backends.pop(sid, None)
            cls.last_activity.pop(sid, None)
            if backend:
                print(f"[PROXY] Cleaned up inactive session {sid}, freed {backend}")

    def get_session_id(self):
        cookie_hdr = self.headers.get("Cookie", "")
        if cookie_hdr:
            c = cookies.SimpleCookie()
            try:
                c.load(cookie_hdr)
                if "user_session" in c:
                    return c["user_session"].value
            except Exception:
                pass
        return self.client_address[0]

    def allocate_worker(self, session_id):
        MultiUserProxyHandler.cleanup_stale_sessions()

        if session_id in MultiUserProxyHandler.user_backends:
            MultiUserProxyHandler.last_activity[session_id] = time.time()
            return MultiUserProxyHandler.user_backends[session_id]

        used = set(MultiUserProxyHandler.user_backends.values())
        for b in BACKENDS:
            if b not in used:
                MultiUserProxyHandler.user_backends[session_id] = b
                MultiUserProxyHandler.last_activity[session_id] = time.time()
                print(f"[PROXY] Assigned user {session_id} -> {b}")
                return b

        return None

    def get_backend_url(self):
        session_id = self.get_session_id()
        return self.allocate_worker(session_id)

    def log_adoption_event(self, tool_name, payload):
        log_file = os.path.join(LOGS_DIR, f"adoption_{tool_name}.jsonl")
        client_ip = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "client_ip": client_ip,
            "channel": payload.get("channel", "unknown"),
            "user": payload.get("user", client_ip),
            "action": payload.get("action", "activity"),
            "details": payload.get("details", {})
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()

    def forward(self, method):
        parsed = urlparse(self.path)
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query

        backend = self.get_backend_url()
        if not backend:
            msg = json.dumps({
                "detail": "All thumbnail slots are currently busy. Please wait until another user finishes to avoid conflicting frames."
            }).encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        target_url = backend + path
        session_id = self.get_session_id()
        cookie_hdr = self.headers.get("Cookie", "")
        needs_cookie = "user_session" not in cookie_hdr

        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        body = None
        if "Content-Length" in self.headers:
            body = self.rfile.read(int(self.headers["Content-Length"]))

        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                if needs_cookie:
                    self.send_header("Set-Cookie", f"user_session={session_id}; Path=/; SameSite=Lax")
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                        self.send_header(k, v)
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            if needs_cookie:
                self.send_header("Set-Cookie", f"user_session={session_id}; Path=/; SameSite=Lax")
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                    self.send_header(k, v)
            err_data = e.read()
            self.send_header("Content-Length", str(len(err_data)))
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as ex:
            self.send_error(502, f"Bad Gateway: {ex}")

    def do_GET(self):
        req_path = urlparse(self.path).path
        if req_path.startswith(PROXY_PREFIXES) or req_path.startswith(("/frames/", "/cache/", "/frame-wide/")):
            self.forward("GET")
        else:
            session_id = self.get_session_id()
            self.allocate_worker(session_id)
            super().do_GET()

    def do_POST(self):
        req_path = urlparse(self.path).path
        if req_path.startswith("/api/track/"):
            tool_name = req_path.replace("/api/track/", "").strip("/") or "thumbnail_maker"
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = {}
            self.log_adoption_event(tool_name, payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return

        self.forward("POST")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MultiUserProxyHandler)
    print(f"Multi-user proxy serving on {PORT} with workers: {BACKENDS}")
    server.serve_forever()
