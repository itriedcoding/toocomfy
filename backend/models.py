"""Discovery of locally installed diffusion checkpoints.

Two supported layouts:
- A diffusers-format folder (contains model_index.json + component subfolders),
- A single-file .safetensors/.bin/.ckpt/.gguf checkpoint.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .config import MODELS_DIR

CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".gguf")

VIDEO_TOKENS = (
    "anim",
    "motion",
    "svd",
    "video",
    "hunyuan",
    "wan",
    "ltx",
    "sana",
    "mochi",
    "cosmos",
    "kling",
    "pixverse",
)

XL_TOKENS = ("sdxl", "xl", "playground", "fotor", "kontext", "wdr")


@dataclass
class ModelInfo:
    id: str
    path: Path
    kind: str  # "image" | "video"
    pipeline: str
    estimated_gb: float


def _is_diffusers_folder(path: Path) -> bool:
    return (path / "model_index.json").is_file()


def _estimate_gb(path: Path) -> float:
    total = 0.0
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in CHECKPOINT_SUFFIXES:
            try:
                total += candidate.stat().st_size / 1e9
            except OSError:
                pass
    return round(total, 2)


def classify(path: Path) -> Tuple[str, str]:
    """Return (kind, pipeline_class_name) for a checkpoint path.

    Kind is detected from name + folder layout; pipeline from a small
    family table. Fallback is the base StableDiffusionPipeline.
    """
    name = path.name.lower()
    kind = "video" if any(token in name for token in VIDEO_TOKENS) else "image"
    if kind == "video":
        pipeline = _video_pipeline(name)
    else:
        pipeline = _image_pipeline(name)
    return kind, pipeline


def _image_pipeline(name: str) -> str:
    if "flux" in name:
        return "FluxPipeline"
    if "sd3" in name or "stable-diffusion-3" in name:
        return "StableDiffusion3Pipeline"
    if any(token in name for token in XL_TOKENS):
        return "StableDiffusionXLPipeline"
    if "pixart" in name:
        return "PixArtAlphaPipeline"
    if "kandinsky" in name:
        return "KandinskyPipeline"
    if "lumina" in name:
        return "Lumina2Pipeline"
    return "StableDiffusionPipeline"


def _video_pipeline(name: str) -> str:
    if "svd" in name or "stable-video" in name:
        return "StableVideoDiffusionPipeline"
    if "hunyuan" in name:
        return "HunyuanVideoPipeline"
    if "wan" in name:
        return "WanPipeline"
    if "ltx" in name:
        return "LTXPipeline"
    if "sana" in name:
        return "SanaVideoPipeline"
    if any(token in name for token in ("anim", "motion", "video")):
        return "AnimateDiffPipeline"
    return "AnimateDiffPipeline"


def _diffusers_kind(folder: Path) -> str:
    """Determine kind for a diffusers folder from model_index.json."""
    try:
        with open(folder / "model_index.json", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "image"
    # SVD and AnimateDiff folders ship a motion_* key; video models use
    # "image_encoder" (SVD) or "unet" with a temporal layer.
    keys = set(index.keys())
    if "image_encoder" in keys or any(k.startswith("motion") for k in keys):
        return "video"
    # Temporal VAE / UNet3D is a strong video signal.
    if any("Temporal" in str(v) for v in index.values()):
        return "video"
    return "image"


def _diffusers_pipeline(folder: Path) -> str:
    name = folder.name.lower()
    if "svd" in name or "stable-video" in name:
        return "StableVideoDiffusionPipeline"
    if "hunyuan" in name:
        return "HunyuanVideoPipeline"
    if "wan" in name:
        return "WanPipeline"
    if "ltx" in name:
        return "LTXPipeline"
    if "sana" in name:
        return "SanaVideoPipeline"
    if "flux" in name:
        return "FluxPipeline"
    if "sd3" in name or "stable-diffusion-3" in name:
        return "StableDiffusion3Pipeline"
    if any(token in name for token in XL_TOKENS):
        return "StableDiffusionXLPipeline"
    if "pixart" in name:
        return "PixArtAlphaPipeline"
    if "kandinsky" in name:
        return "KandinskyPipeline"
    if "lumina" in name:
        return "Lumina2Pipeline"
    if any(token in name for token in ("anim", "motion", "video")):
        return "AnimateDiffPipeline"
    return "StableDiffusionPipeline"


def discover_models() -> list:
    """Walk MODELS_DIR and return a list of usable ModelInfo entries."""
    found: list = []
    if not MODELS_DIR.exists():
        return found

    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.is_dir() and _is_diffusers_folder(entry):
            kind = _diffusers_kind(entry)
            pipeline = _diffusers_pipeline(entry)
            found.append(ModelInfo(entry.name, entry, kind, pipeline, _estimate_gb(entry)))
        elif entry.is_file() and entry.suffix.lower() in CHECKPOINT_SUFFIXES:
            kind, pipeline = classify(entry)
            found.append(ModelInfo(entry.stem, entry, kind, pipeline, _estimate_gb(entry)))
    return found
