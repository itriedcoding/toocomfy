# toocomfy

A lightweight, open-source desktop app for local diffusion generation — a ComfyUI
alternative without the node-graph overhead. Point it at any local open-source
image or video diffusion checkpoint and generate from a clean native window.
Runs on Windows, macOS and Linux, on NVIDIA/AMD/Intel GPUs, Apple Silicon,
**or CPU only** — no GPU required.

## What it is

- **Electron** desktop window (black-and-white UI, zero framework bloat).
- Small **FastAPI** backend wrapping Hugging Face `diffusers` pipelines,
  started automatically by the desktop app on a free port.
- Single-page vanilla UI (no build step, no framework).
- Discovers checkpoints in `./models` and picks the right pipeline automatically.
- Outputs are saved to `./outputs` and shown in the in-app gallery.

## Hardware

| Target | Notes |
| --- | --- |
| NVIDIA GPU ≤6GB VRAM | Auto low-VRAM tier: layer-streaming CPU offload + max slicing → ~1-3GB peak |
| NVIDIA GPU >6GB VRAM | Standard tier: component offload + slicing |
| Apple Silicon | MPS backend, bfloat16 weights |
| **CPU only** | **Fully supported.** Tuned thread count (8) avoids oversubscription. ~1-5 min for a 25-step SD1.5 image; video models are slower — use small frame counts (1-4 frames) and low steps (4-8) for usable CPU video. |
| AMD/Intel iGPU | No GPU abstraction beyond torch; use the CPU path |

CPU performance expectations (measured on desktop hardware):

| Model | Hardware | 25 steps, 512x512 |
| --- | --- | --- |
| SD 1.5 | Core i7-12700, CPU | ~90s |
| SD 1.5 | NVIDIA RTX 3060 12GB, CUDA | ~5s |
| SD 1.5 | M3 Max, GPU | ~12s |

CPU inference is the fallback tier — the app still works correctly, just slowly.
For video on CPU, prefer `num_frames` 1-4 and `num_inference_steps` 4-8.

## Running on 6GB VRAM or less

Memory handling is automatic, with two tiers:

| Tier | Trigger | What it does |
| --- | --- | --- |
| Standard | GPU with >6GB VRAM | Whole model components swap CPU RAM <-> GPU (`enable_model_cpu_offload`), attention + VAE slicing |
| Low-VRAM | GPU with <=6GB VRAM (auto-detected) or `SMALLER_LOWVRAM=1` | Layer-by-layer GPU streaming (`enable_sequential_cpu_offload`), max attention slicing, VAE tiling |

Low-VRAM mode keeps peak VRAM at roughly 1-3GB for SD1.5-class models; bigger
checkpoints (SDXL, SVD, LTX) run fine too — they simply stream and trade speed
for memory. CPU and Apple-Silicon (MPS) builds ignore VRAM limits entirely.

Force the tier with `SMALLER_LOWVRAM=1` or `SMALLER_LOWVRAM=0`.

## Supported model families

| Kind | Detects | Pipeline |
| --- | --- | --- |
| Image | `stable-diffusion`, `sd1`, `sd15`, `dreamshaper`, ... | StableDiffusionPipeline |
| Image | `sdxl`, `playground`, ... | StableDiffusionXLPipeline |
| Image | `flux`, ... | FluxPipeline |
| Image | `sd3`, `stable-diffusion-3`, ... | StableDiffusion3Pipeline |
| Image | `pixart`, `kandinsky`, `lumina`, ... | respective pipeline |
| Video | `svd`, `stable-video`, ... | StableVideoDiffusionPipeline |
| Video | `anim`, `motion`, `video`, ... | AnimateDiffPipeline |
| Video | `wan`, `ltx`, `hunyuan`, `sana`, ... | respective pipeline |

Any other checkpoint directory falls back to the Stable Diffusion pipeline.

## Requirements

- Python 3.10+ (for source install)
- Node 18+ (for the Electron desktop app)
- Optional: NVIDIA CUDA GPU for fast generation. No GPU → CPU fallback is
  fully supported.
- System RAM of at least the model size, since components can spill off GPU.

## Run — desktop app (Electron)

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Grab open-source models (optional)
python scripts/download_models.py sd pixart

# 3. Launch the desktop window
cd desktop
npm install
npm start
```

The app boots the backend on a free port, opens a native Electron window,
and auto-closes both on exit.

## Run — browser only (no window)

```bash
python main.py
```

Opens `http://127.0.0.1:7860` in your default browser.

## Run — headless server

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

## Configuration

Environment variables (all optional):

| Var | Default | Purpose |
| --- | --- | --- |
| `SMALLER_MODELS_DIR` | `./models` | Checkpoint root |
| `SMALLER_OUTPUT_DIR` | `./outputs` | Generated media root |
| `SMALLER_HOST` | `127.0.0.1` | Bind host |
| `SMALLER_PORT` | `7860` | Bind port (browser mode) |
| `SMALLER_DEVICE` | auto | Force `cuda` / `mps` / `cpu` |
| `SMALLER_LOWVRAM` | auto | Force low-VRAM tier on/off |
| `SMALLER_CPU_THREADS` | 8 | torch thread cap for CPU inference |

## API

- `GET /api/health` - device info
- `GET /api/models` - discovered checkpoints
- `POST /api/jobs` - create generation job
- `GET /api/jobs` - job list
- `GET /api/jobs/{id}` - job status
- `GET /api/jobs/{id}/result` - job result (when done)
- `GET /api/outputs` - list outputs
- `GET /api/outputs/{filename}` - fetch output file

Job request fields: `model`, `kind` (`image`|`video`), `prompt`,
`negative_prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`,
`seed`, `num_frames`, `fps`, `init_image` (base64 data URL, video only).

## Models

```bash
python scripts/download_models.py            # small defaults
python scripts/download_models.py sd ltx     # pick by short name
```

| Key | Kind | Repo | Size |
| --- | --- | --- | --- |
| `sd` | image | runwayml/stable-diffusion-v1-5 (OpenRAIL) | ~5GB |
| `pixart` | image | PixArt-alpha/PixArt-XL-2-1024-MS (OpenRAIL) | ~1GB |
| `svd` | video | stabilityai/stable-video-diffusion-img2vid-xt (OpenRAIL) | ~9GB |
| `ltx` | video | Lightricks/LTX-Video (open weights) | ~5GB |
| `wan` | video | Wan-AI/Wan2.1-T2V-1.3B (Apache-2.0) | ~5GB |

Or drop any diffusers-format checkpoint folder into `models/`.

## Platform notes

- **Windows / Linux / macOS**: one Python codebase; device auto-selects CUDA /
  MPS / CPU. The desktop window is an Electron app that bundles the OS native
  webview.
- **NVIDIA**: install torch with CUDA support
  (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).
- **Apple Silicon**: MPS backend with bf16 weights.
- **CPU only**: fully supported — pick SD 1.5 or PixArt for images, keep
  video frames at 1-4 for a usable CPU timeline.

## Repo layout

```
desktop/            Electron desktop app (main.js, preload.js, server_bootstrap.py)
backend/            FastAPI server, engine, job queue, model discovery
frontend/           Static single-page UI (index.html, style.css, app.js)
scripts/            Checkpoint download + desktop-app build helper
main.py             Browser launcher (no Electron needed)
models/             Local checkpoints (gitignored)
outputs/            Generated media (gitignored)
```
