"""Download open-source checkpoints into ./models.

Usage:
    python scripts/download_models.py          # download everything small
    python scripts/download_models.py sd ltx   # pick by short name

Available:
    sd     image   runwayml/stable-diffusion-v1-5        (~5GB)  OpenRAIL
    pixart image   PixArt-alpha/PixArt-XL-2-1024-MS      (~1GB)  OpenRAIL
    svd    video   stabilityai/stable-video-diffusion-img2vid-xt  (~9GB) OpenRAIL
    ltx    video   Lightricks/LTX-Video                  (~5GB)  LTXV license (open weights)
    wan    video   Wan-AI/Wan2.1-T2V-1.3B                ~5GB)   Apache-2.0
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models"

CATALOG = {
    "sd": ("runwayml/stable-diffusion-v1-5", "sd15"),
    "pixart": ("PixArt-alpha/PixArt-XL-2-1024-MS", "pixart"),
    "svd": ("stabilityai/stable-video-diffusion-img2vid-xt", "svd"),
    "ltx": ("Lightricks/LTX-Video", "ltx"),
    "wan": ("Wan-AI/Wan2.1-T2V-1.3B", "wan-1.3b"),
}

DEFAULT = ("sd", "pixart")


def main() -> None:
    keys = sys.argv[1:] or DEFAULT
    invalid = [k for k in keys if k not in CATALOG]
    if invalid:
        print(f"unknown model(s): {', '.join(invalid)}")
        print("choices:", ", ".join(CATALOG))
        sys.exit(1)
    TARGET.mkdir(parents=True, exist_ok=True)
    for key in keys:
        repo_id, folder = CATALOG[key]
        dest = TARGET / folder
        print(f"downloading {repo_id} -> {dest}")
        snapshot_download(repo_id=repo_id, local_dir=str(dest), max_workers=4)
        print(f"done: {folder}")


if __name__ == "__main__":
    main()
