"""Browser launcher for toocomfy.

Starts the local server in a background thread and opens a browser tab.
For the windowed desktop experience, use the Electron app in `desktop/`:

    cd desktop && npm install && npm start

Usage:
    pip install -r requirements.txt
    python main.py
"""

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from backend.app import app  # noqa: E402
from backend.config import HOST, PORT  # noqa: E402


def serve() -> None:
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"Starting toocomfy on {url}")
    print("Use the desktop app instead? Run: cd desktop && npm install && npm start")

    server = threading.Thread(target=serve, daemon=True)
    server.start()

    # Wait briefly for the server to come up.
    import time
    import urllib.request
    for _ in range(40):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=2)
            break
        except Exception:
            time.sleep(0.5)

    try:
        webbrowser = __import__("webbrowser")
        webbrowser.open(url)
    except Exception:
        print(f"Open {url} manually.")

    print("Browser mode. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
