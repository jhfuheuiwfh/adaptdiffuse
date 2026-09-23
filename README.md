# AdaptDiffuse

**Plug-and-play Stable Diffusion engine** with automatic GPU backend cascade and a clean WebUI.

Primary engine is **sd-cli** ([stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)) —
one small native binary, no torch needed:

```
NVIDIA (CUDA)  →  AMD/Intel (Vulkan)  →  Forced ROCm(HIP)  →  DirectML (very old)  →  CPU
                     └── all sd-cli builds, DirectML is the only torch fallback
```

## Quick start (Windows)

1. Install [Python 3.10+](https://www.python.org/downloads/) (check *Add to PATH*).
2. Double-click **`launch.bat`**.
3. First run creates a private `.venv`, detects your GPU, downloads the matching sd-cli
   build (+ Vulkan SDK only if the loader is missing), and opens **http://127.0.0.1:7860**.

That's it — no manual CUDA/ROCm/Vulkan setup for the default paths.

## Backend cascade

| Priority | Hardware | Engine | Notes |
|---|---|---|---|
| 1 | NVIDIA (CUDA) | **sd-cli CUDA build** | Falls back to the Vulkan build if download fails |
| 2 | AMD / Intel / any GPU with Vulkan | **sd-cli Vulkan build** | Auto-downloads binary; Vulkan SDK if loader missing |
| 3 | Forced `rocm` | **sd-cli ROCm(HIP) build** | `win-rocm` asset; needs a ROCm-capable GPU |
| 4 | Very old Windows GPU, no Vulkan | `torch-directml` (**torch 2.4.1** + **torch-directml 0.2.5.dev240914**) | Broad DX12 coverage; only torch path |
| 5 | Anything else | **sd-cli CPU build** | Always works |

**FP16 policy:** the torch/DirectML path only enables `float16` on known-safe GPU
generations. sd-cli is immune to this — ggml applies whatever quantization the model
file already has (prefer **GGUF** for weak GPUs).

**Weak-GPU guards (sd-cli):** VAE tiling always; `--max-vram` on ≤6 GB cards;
`--offload-to-cpu` when weights exceed ~85% of VRAM or the card has ≤2 GB.

## Default model

The sd-cli path defaults to a single **GGUF** file (~1.6 GB), downloaded on first use:

`second-state/stable-diffusion-v1-5-GGUF` → `stable-diffusion-v1-5-pruned-emaonly-Q5_1.gguf`

The DirectML/torch path defaults to `stable-diffusion-v1-5/stable-diffusion-v1-5` (diffusers layout).

## Features

- **WebUI (Gradio)** — prompt, advanced params, gallery, model manager, backend switcher
- **LoRA** — pick a file under `models/loras/` + scale; applied via sd-cli `<lora:name:scale>` syntax
- **Model downloader** — Hugging Face repos + CivitAI URLs/IDs/search (type `LORA` → `models/loras/`)
- **Local models** — scans `models/` for `.safetensors` / `.ckpt` / `.gguf`
- **Weak-GPU optimizations** — attention slicing, VAE slicing, DPM++ 2M Karras, conservative resolutions
- **No telemetry** — prompts and images stay on your machine

## CivitAI / HF tokens

Optional environment variables (never written to disk by AdaptDiffuse):

```bat
set HF_TOKEN=hf_...
set CIVITAI_API_KEY=...
```

Age-restricted CivitAI models require your own API key.

## Manual CLI

```bat
.venv\Scripts\python -m app.backend          # detect + save backend.json
.venv\Scripts\python -m app.main --setup     # install deps for backend
.venv\Scripts\python -m app.main --port 7860
.venv\Scripts\python -m app.main --backend vulkan
```

Backend choices: `auto` · `cuda` · `rocm` · `vulkan` · `directml` · `cpu`

## Project layout

```
adaptdiffuse/
├── launch.bat           # plug-and-play entry (venv + deps + WebUI)
├── requirements.txt
├── app/
│   ├── backend.py       # GPU detect + cascade + installers
│   ├── pipeline.py      # diffusers + stable-diffusion.cpp engines
│   ├── downloader.py    # HF / CivitAI (checkpoints, LoRAs)
│   ├── webui.py         # Gradio UI (incl. LoRA dropdown + scale)
│   └── main.py          # CLI entry
├── models/              # checkpoints/, loras/, vae (gitignored)
└── outputs/             # generations (gitignored)
```

## Uninstall

Delete the project folder (includes `.venv`, `models/`, `outputs/`).

## License

MIT
