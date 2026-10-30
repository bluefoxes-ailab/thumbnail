import sys, os, re, threading, urllib.request, urllib.error, json, uuid, time, datetime
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

# A seat is held until its page says goodbye (/api/leave), and for no other
# reason. There is no timeout of any kind.
#
# There used to be two - three minutes without a request, twenty without use -
# and both stopped the user's run when they fired. Going quiet is not leaving:
# browsers freeze or discard a tab that has sat in the background a while
# (Memory Saver, Energy Saver), a Windows PC with the screen off stops its
# network, a phone suspends the browser. Every one of those took the video
# down minutes in, with the tab still open and the user fully intending to
# come back to the frames. Only the goodbye comes after the user said yes to
# losing their work (see the leaving section of main.js), so only it frees a
# seat and stops a run.
#
# The price is knowingly paid: a browser that vanishes without saying goodbye
# - a crash, a killed process, a laptop that never wakes up - keeps its seat
# until this proxy restarts. The same browser coming back carries the same
# cookie and sits straight back down in it.
#
# DRAIN: a backend just taken back may still be stopping its run (it stops at
# the next checkpoint, not at once - see backend/cancel.py), so for this long
# it is handed out only when nothing else is free.
DRAIN_SECONDS = 60
LOGS_DIR = "/home/bluefoxesfreelancer/thumbnail/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# What a session id this proxy issued looks like: uuid4().hex.
SESSION_ID = re.compile(r"[0-9a-f]{32}")

ALLOCATION_LOCK = threading.Lock()

# Pages, as opposed to the scripts, styles and images they pull in. Only these
# hand out a session id - see do_GET.
PAGE_PATHS = ("/", "/index.html", "/maintenance.html", "/maintenance-game.html")

def stop_run(backend, run_id=None):
    """
    Asks a backend to stop the run it is working on, without waiting for it.

    Sent whenever a seat is taken back: whatever that backend is processing
    belongs to the user who has just gone, and left running it would hold the
    backend for the next one. `run_id` is the run the page said was its own,
    when it said; without one the backend stops whatever it has (see
    /process-video/cancel). On its own thread because this is called while
    seats are being handed out under ALLOCATION_LOCK, and nobody arriving
    should wait on a backend's answer.
    """
    def send():
        body = json.dumps({"id": run_id}).encode("utf-8")
        req = urllib.request.Request(backend + "/process-video/cancel", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[PROXY] Stop sent to {backend}: {resp.read().decode('utf-8', 'replace')}")
        except Exception as ex:
            print(f"[PROXY] Could not reach {backend} to stop its run: {ex}")
    threading.Thread(target=send, daemon=True).start()


class MultiUserProxyHandler(SimpleHTTPRequestHandler):
    user_backends = {}
    # Backends taken back recently, and when - see DRAIN_SECONDS.
    draining = {}
    # The id to send with this response, when it is the first a browser gets.
    issue_session = None

    @classmethod
    def release(cls, session_id, run_id=None, reason=""):
        """
        Takes a seat back, and stops the run its backend was doing for it.
        Callers hold ALLOCATION_LOCK.
        """
        backend = cls.user_backends.pop(session_id, None)
        if backend:
            cls.draining[backend] = time.time()
            stop_run(backend, run_id)
            print(f"[PROXY] Released {session_id} ({reason}), freed {backend}")

    def cookie_session(self):
        """
        The session id this browser was issued, or None.

        Only an id this proxy could have issued counts. The cookie used to hold
        the visitor's IP address, so everyone behind one router - an office, a
        house, a phone network's shared address - was one "user" to this
        proxy: one backend, one run, one set of frames, each of them answering
        409 to the others and overwriting their work. A browser still carrying
        one of those old values is treated as having none and given a real id
        with the next page it loads.
        """
        cookie_hdr = self.headers.get("Cookie", "")
        if cookie_hdr:
            c = cookies.SimpleCookie()
            try:
                c.load(cookie_hdr)
                if "user_session" in c:
                    value = c["user_session"].value
                    if SESSION_ID.fullmatch(value):
                        return value
            except Exception:
                pass
        return None

    def get_session_id(self):
        # The IP only for a request that arrives with no id at all, which a
        # browser that loaded the page never does (see do_GET). Something has
        # to stand in: an id invented per request would take a new backend per
        # request and fill every slot in a few clicks.
        return self.cookie_session() or f"ip:{self.client_address[0]}"

    def allocate_worker(self, session_id):
        # Under a lock because the server answers each request on its own
        # thread: two sessions arriving together could otherwise both see the
        # same backend free and both be given it - two users on one backend,
        # which is the one thing this class exists to prevent.
        cls = MultiUserProxyHandler
        with ALLOCATION_LOCK:
            if session_id in cls.user_backends:
                return cls.user_backends[session_id]

            # A backend still stopping the last user's run is the last choice,
            # not excluded: it is free a few seconds from now, and a busy
            # message is worse than a short wait for the next user's run.
            now = time.time()
            taken = set(cls.user_backends.values())
            free = [b for b in BACKENDS if b not in taken]
            free.sort(key=lambda b: now - cls.draining.get(b, 0) < DRAIN_SECONDS)
            if free:
                b = free[0]
                cls.draining.pop(b, None)
                cls.user_backends[session_id] = b
                print(f"[PROXY] Assigned user {session_id} -> {b}")
                return b

            return None

    def end_headers(self):
        # Where a page response picks up the id do_GET decided to issue - the
        # static file handler builds its own headers, so this is the one point
        # every response passes through.
        if self.issue_session:
            self.send_header("Set-Cookie", f"user_session={self.issue_session}; "
                                           "Path=/; SameSite=Lax; HttpOnly")
            self.issue_session = None
        super().end_headers()

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

        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        body = None
        if "Content-Length" in self.headers:
            body = self.rfile.read(int(self.headers["Content-Length"]))

        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                        self.send_header(k, v)
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
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
            # The page is where a browser gets its session id, and the only
            # place: everything it then asks for carries the cookie. Handing
            # one out on any request without it would mint several for a single
            # visitor - the scripts, styles and icon of a first visit are all
            # requested before the first answer lands - and each id takes a
            # backend of its own.
            #
            # A seat is still taken on opening the page, as it always was;
            # what changed is only whose name it is in.
            self.issue_session = None
            session_id = self.cookie_session()
            if session_id is None and req_path in PAGE_PATHS:
                session_id = self.issue_session = uuid.uuid4().hex
            if session_id is not None:
                self.allocate_worker(session_id)
            super().do_GET()

    def read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
        except Exception:
            return {}

    def answer_empty(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        req_path = urlparse(self.path).path

        # The heartbeat pages used to send when seats timed out. Nothing reads
        # it any more, but a page loaded before this proxy was updated still
        # sends one a minute, and forwarded it would only earn a 404 from a
        # backend - so it is answered here and dropped.
        if req_path == "/api/heartbeat":
            self.read_json()
            self.answer_empty()
            return

        # The page closing or reloading, after the user said yes to leaving
        # (or with nothing to lose, when no question was asked). Only ever sent
        # by the LAST tab of this browser: the others share the seat and the
        # run, and are still using both. The one way a seat is freed and a run
        # stopped (see DRAIN_SECONDS for why there is no other).
        if req_path == "/api/leave":
            payload = self.read_json()
            session_id = self.cookie_session()
            run_id = payload.get("runId")
            if session_id:
                with ALLOCATION_LOCK:
                    MultiUserProxyHandler.release(
                        session_id, run_id if isinstance(run_id, str) else None, reason="left")
            self.answer_empty()
            return

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
