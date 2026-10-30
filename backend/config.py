import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("SMALLER_MODELS_DIR") or (BASE_DIR / "models"))
OUTPUT_DIR = Path(os.environ.get("SMALLER_OUTPUT_DIR") or (BASE_DIR / "outputs"))
FRONTEND_DIR = BASE_DIR / "frontend"

HOST = os.environ.get("SMALLER_HOST") or "127.0.0.1"
PORT = int(os.environ.get("SMALLER_PORT") or "7860")
DEVICE = (os.environ.get("SMALLER_DEVICE") or "").strip().lower() or None
