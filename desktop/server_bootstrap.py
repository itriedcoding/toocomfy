"""Bridge that boots the Python backend for the desktop app.

Spawned by the Electron main process (desktop/main.js). Reads
SMALLER_* environment variables from the parent, starts the FastAPI
app on a free port, and prints the chosen port to stdout as the first
line. The Electron side parses that line and loads it into the window.

When packaged, this file lives at process.resourcesPath; the backend
package is copied next to it via electron-builder's extraResources.
"""

import os
import socket
import sys
from pathlib import Path

# In development, the project root is one level up from desktop/.
# When packaged, everything sits in process.resourcesPath.
if hasattr(sys, "_MEIPASS"):
    ROOT = Path(os.environ.get("SMALLER_APP_ROOT", "."))
else:
    ROOT = Path(__file__).resolve().parent.parent

for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    port = int(os.environ.get("SMALLER_PORT") or free_port())
    os.environ["SMALLER_HOST"] = "127.0.0.1"
    os.environ["SMALLER_PORT"] = str(port)

    import uvicorn

    from backend.app import app

    # First stdout line: the port. The Electron parent reads this.
    print(port, flush=True)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
