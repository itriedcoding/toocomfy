"""Diffusion inference engine for smaller-comfyui.

Wraps diffusers pipelines with the small set of memory optimizations required
to keep generation under ~6GB VRAM on ordinary consumer hardware:

- model CPU offload via accelerate (CUDA/XPU) so oversized checkpoints spill
  over to system RAM instead of exhausting VRAM,
- attention slicing and VAE slicing/tiling to cut activation memory.

Works on CUDA (NVIDIA), MPS (Apple Silicon) and CPU (fallback).
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import (
    AnimateDiffPipeline,
    FluxPipeline,
    HunyuanVideoPipeline,
    KandinskyPipeline,
    Lumina2Pipeline,
    PixArtAlphaPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    WanPipeline,
)
from diffusers.utils import export_to_video
from PIL import Image

from .config import DEVICE, OUTPUT_DIR

IMAGE_KIND = "image"
VIDEO_KIND = "video"


def get_device() -> str:
    if DEVICE:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cpu_threads() -> int:
    """Cap thread count for CPU inference.

    torch defaults to all logical cores, which on 16+ core CPUs causes
    diminishing returns and high RAM load. 8 threads hits the sweet spot
    for most diffusion models on consumer hardware.
    """
    return int(os.environ.get("SMALLER_CPU_THREADS") or 8)


def get_torch_dtype(device: str):
    if device in ("cuda", "xpu"):
        return torch.float16
    if device == "mps":
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    return torch.float32


def apply_cpu_settings() -> None:
    """Enable CPU inference optimizations on a CPU-only build."""
    if get_device() != "cpu":
        return
    torch.set_num_threads(cpu_threads())
    torch.set_num_interop_threads(1)


def vram_gb() -> float:
    """Total dedicated VRAM in GB (0.0 when unavailable)."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def low_vram() -> bool:
    """True when we should use the strongest memory-saving settings.

    Auto-detected for GPUs with 6GB or less; override with
    SMALLER_LOWVRAM=1|0 (small models stay resident, big ones spill to RAM).
    """
    env = (os.environ.get("SMALLER_LOWVRAM") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return 0.0 < vram_gb() <= 6.0


def _enable_memory_optimizations(pipe) -> None:
    try:
        if low_vram():
            # Strongest tier: every module streams to GPU one layer at a time.
            pipe.enable_sequential_cpu_offload()
        elif get_device() not in ("cpu", ""):
            # Standard tier: whole components swap between CPU RAM and GPU.
            pipe.enable_model_cpu_offload()
    except Exception:
        pass
    if low_vram():
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pass
    else:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    for method in ("enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, method):
            try:
                getattr(pipe, method)()
            except Exception:
                pass


def _to_pil(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (3, 4):
            array = array.transpose(1, 2, 0)
        return Image.fromarray(array.astype("uint8"))
    return None


def _extract_frames(result: Any) -> list:
    frames = None
    if hasattr(result, "frames"):
        frames = result.frames
    elif hasattr(result, "images"):
        frames = result.images
    elif isinstance(result, tuple):
        frames = result[0]
    else:
        frames = result
    flat: list = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
        elif node is not None:
            flat.append(node)

    walk(frames)
    return flat


def _generator(seed) -> Optional[torch.Generator]:
    seed_value = int(seed) if seed is not None else 0
    if seed_value <= 0:
        return None
    return torch.Generator(device=get_device()).manual_seed(seed_value)


def _init_image(params: dict):
    data_url = params.get("init_image") or ""
    if not data_url:
        return None
    if "," in data_url and "base64" in data_url.lower():
        header, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
        image = Image.open(BytesIO(raw)).convert("RGB")
        return image
    return None


def build_ready_pipeline(model_path: str, kind: str):
    apply_cpu_settings()
    path_str = str(model_path)
    dtype = get_torch_dtype(get_device())
    if kind == VIDEO_KIND:
        pipe = _build_video_pipeline(path_str, dtype)
    else:
        pipe = _build_image_pipeline(path_str, dtype, path_str.lower())
    _enable_memory_optimizations(pipe)
    _prepare_pipeline(pipe)
    return pipe


def _establish_execution_device(pipe) -> None:
    """Ensure _execution_device tracking exists for manually-built pipelines."""
    if hasattr(pipe, "_execution_device"):
        return
    if get_device() == "cpu":
        object.__setattr__(pipe, "hf_device_map", None)
        object.__setattr__(pipe, "_offload_gpu_id", None)


def _warmup(pipe) -> None:
    """First-inference JIT/allocator warmup (significant on CPU)."""
    if get_device() != "cpu":
        return
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 32, 32)
            _ = dummy + 1.0
        del dummy
    except Exception:
        pass


def _prepare_pipeline(pipe) -> None:
    """Make a pipeline runnable on the target device.

    from_pretrained pipelines on GPU keep their accelerate hooks; CPU and
    in-process pipelines need a plain .to(device) plus first-run warmup.
    """
    if get_device() == "cpu":
        _establish_execution_device(pipe)
        pipe.to("cpu")
        _warmup(pipe)


def _build_image_pipeline(path_str: str, dtype, name: str):
    if "flux" in name:
        return FluxPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "sd3" in name or "stable-diffusion-3" in name:
        return StableDiffusion3Pipeline.from_pretrained(path_str, torch_dtype=dtype)
    if any(token in name for token in ("sdxl", "xl", "playground", "fotor", "kontext", "wdr")):
        return StableDiffusionXLPipeline.from_pretrained(
            path_str, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
    if "pixart" in name:
        return PixArtAlphaPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "kandinsky" in name:
        return KandinskyPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "lumina" in name:
        return Lumina2Pipeline.from_pretrained(path_str, torch_dtype=dtype)
    return StableDiffusionPipeline.from_pretrained(
        path_str, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )


def _build_video_pipeline(path_str: str, dtype):
    name = path_str.lower()
    if "svd" in name or "stable-video" in name:
        return StableVideoDiffusionPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "hunyuan" in name:
        return HunyuanVideoPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "wan" in name:
        return WanPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "ltx" in name:
        from diffusers import LTXPipeline

        return LTXPipeline.from_pretrained(path_str, torch_dtype=dtype)
    if "sana" in name:
        from diffusers import SanaVideoPipeline

        return SanaVideoPipeline.from_pretrained(path_str, torch_dtype=dtype)
    unet_dir = Path(path_str) / "unet"
    motion_control = Path(path_str) / "motion_controller"
    motion_scheduler = Path(path_str) / "motion_scheduler"
    kwargs = {}
    if motion_control.exists():
        kwargs["motion_control"] = str(motion_control)
    if motion_scheduler.exists():
        kwargs["motion_scheduler"] = str(motion_scheduler)
    unet_path = str(unet_dir) if unet_dir.exists() else path_str
    try:
        return AnimateDiffPipeline.from_pretrained(unet_path, torch_dtype=dtype, **kwargs)
    except Exception:
        return AnimateDiffPipeline.from_pretrained(path_str, torch_dtype=dtype, **kwargs)


def generate_image(pipe, params: dict) -> Any:
    kwargs = {
        "prompt": params.get("prompt") or "",
        "negative_prompt": params.get("negative_prompt") or None,
        "num_inference_steps": int(params.get("num_inference_steps") or 25),
        "guidance_scale": float(params.get("guidance_scale") or 7.5),
        "generator": _generator(params.get("seed")),
        "num_images_per_prompt": 1,
        "return_dict": True,
    }
    cls = type(pipe)
    if not (issubclass(cls, AnimateDiffPipeline) or issubclass(cls, StableVideoDiffusionPipeline)):
        kwargs["width"] = int(params.get("width") or 0) or None
        kwargs["height"] = int(params.get("height") or 0) or None
    result = pipe(**kwargs)
    return _extract_frames(result)


def generate_video(pipe, params: dict) -> tuple:
    kwargs = {
        "prompt": params.get("prompt") or "",
        "negative_prompt": params.get("negative_prompt") or None,
        "num_frames": int(params.get("num_frames") or 16),
        "num_inference_steps": int(params.get("num_inference_steps") or 25),
        "fps": int(params.get("fps") or 7),
        "guidance_scale": float(params.get("guidance_scale") or 7.5),
        "generator": _generator(params.get("seed")),
        "return_dict": True,
    }
    if isinstance(pipe, StableVideoDiffusionPipeline):
        kwargs["min_guidance_scale"] = 1.0
        kwargs["max_guidance_scale"] = 3.0
        image = _init_image(params)
        if image is not None:
            kwargs["image"] = image
    elif not issubclass(type(pipe), AnimateDiffPipeline):
        # Text-to-video pipelines without image conditioning take width/height.
        kwargs["width"] = int(params.get("width") or 0) or None
        kwargs["height"] = int(params.get("height") or 0) or None
    result = pipe(**kwargs)
    frames = _extract_frames(result)
    return frames, int(kwargs.get("fps") or 7)


def run_generation(model_path: str, kind: str, params: dict, job_id: str) -> dict:
    """Run one generation job end to end and save the output."""
    pipe = build_ready_pipeline(model_path, kind)
    if kind == VIDEO_KIND:
        frames, fps = generate_video(pipe, params)
        frame_list = [frame for frame in frames if frame is not None]
        filename = f"{job_id}.mp4"
        output_path = str(OUTPUT_DIR / filename)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames=frame_list, output_video_path=output_path, fps=fps, quality=5.0, bitrate=None, macro_block_size=16)
        return {"kind": VIDEO_KIND, "video_path": filename, "frames": len(frame_list), "fps": fps}
    images = generate_image(pipe, params)
    pil_image = _to_pil(images[0]) if images else None
    filename = f"{job_id}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pil_image.save(str(OUTPUT_DIR / filename))
    return {"kind": IMAGE_KIND, "image_path": filename, "width": params.get("width"), "height": params.get("height")}
