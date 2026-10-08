import datetime
import sys, os, urllib.request, urllib.error, json
import uuid
from http import cookies
import urllib.request
import urllib.error
from urllib.parse import urlparse
from http.server import SimpleHTTPRequestHandler, HTTPServer
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003"
]

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

PROXY_PREFIXES = (
    "/process-video", "/upload", "/upload-video", "/upload_video",
    "/frame", "/vary-frame", "/enhance-frame", "/reframe-frame",
    "/cleanup", "/selection", "/api")
LOG_DIR = "/home/bluefoxesfreelancer/thumbnail/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def record_adoption(tool_name, payload, client_ip):
    # Separate log files per tool
    clean_tool = "static_studio" if "static" in tool_name.lower() else "thumbnail_maker"
    log_file = os.path.join(LOG_DIR, f"adoption_{clean_tool}.jsonl")

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
def get_idle_backend():
    for b in BACKENDS:
        try:
            req = urllib.request.Request(f"{b}/process-video/progress", headers={"User-Agent": "proxy"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") in ("idle", "done", "error"):
                    return b
        except Exception:
            continue
    return BACKENDS[0]

class MultiUserProxyHandler(SimpleHTTPRequestHandler):
    user_backends = {}

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
        return None

    def get_backend_url(self):
        session_id = self.get_session_id()
        
        # When initiating a new run or upload, always route to an idle worker
        if self.path.startswith(("/process-video", "/upload", "/upload-video", "/upload_video")) and not self.path.startswith(("/process-video/progress", "/process-video/result")):
            chosen = get_idle_backend()
            if session_id:
                MultiUserProxyHandler.user_backends[session_id] = chosen
            return chosen

        # Otherwise stick to the assigned backend for polling, frames, and edits
        if session_id and session_id in MultiUserProxyHandler.user_backends:
            return MultiUserProxyHandler.user_backends[session_id]

        chosen = get_idle_backend()
        if session_id:
            MultiUserProxyHandler.user_backends[session_id] = chosen
        return chosen

    def forward(self, method):
        # Extract pure relative path in case full URL was passed in self.path
        parsed = urlparse(self.path)
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        backend = self.get_backend_url()
        target_url = backend + path

        session_id = self.get_session_id()
        new_session = False
        if not session_id:
            session_id = str(uuid.uuid4())
            new_session = True
            MultiUserProxyHandler.user_backends[session_id] = backend

        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        body = None
        if "Content-Length" in self.headers:
            body = self.rfile.read(int(self.headers["Content-Length"]))

        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                if new_session:
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
            if new_session:
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
        clean_path = req_path.lstrip("/")

        if clean_path == "" or os.path.exists(clean_path):
            self.path = req_path
            super().do_GET()
        else:
            self.forward("GET")

    def do_POST(self):
        req_path = urlparse(self.path).path
        if req_path.startswith("/api/track"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(body.decode("utf-8"))
                tool = data.get("tool", "thumbnail_maker")
                log_file = f"/home/bluefoxesfreelancer/thumbnail/logs/adoption_{tool}.jsonl"
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
        else:
            self.forward("POST")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MultiUserProxyHandler)
    server.active_backend = BACKENDS[0]
    print(f"Multi-user proxy serving on {PORT} with workers {BACKENDS}...")
    server.serve_forever()
