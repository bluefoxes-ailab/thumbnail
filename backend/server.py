import os
import sys

# Add backend directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

print("Starting server setup...", flush=True)

try:
    from fastapi.staticfiles import StaticFiles
    from api import app
    import uvicorn
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}", flush=True)
    print("Run: pip install fastapi uvicorn", flush=True)
    sys.exit(1)

# Path to frontend
FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))

if os.path.isdir(FRONTEND_DIR):
    print(f"Mounting frontend from: {FRONTEND_DIR}", flush=True)
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"[WARNING] Frontend directory not found at: {FRONTEND_DIR}", flush=True)

if __name__ == "__main__":
    print("\nStarting Thumbnail Maker on http://localhost:8000 ...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")