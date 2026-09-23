"""Inference engines: sd-cli (stable-diffusion.cpp) primary + diffusers fallback.

The primary engine is always the native `sd-cli` binary from
leejet/stable-diffusion.cpp (CUDA / Vulkan / HIP / CPU). The `diffusers`
(torch) path is only used for the DirectML fallback on very old GPUs.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .backend import BackendInfo, ensure_sdcpp_binary, ensure_vulkan_sdk, sync_sdcpp_device
from .downloader import list_local_models

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

# Torch/diffusers fallback default (full diffusers-layout repo)
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# sd-cli default: a single GGUF file (~1.6 GB, works out of the box)
DEFAULT_GGUF_REPO = "second-state/stable-diffusion-v1-5-GGUF"
DEFAULT_GGUF_FILE = "stable-diffusion-v1-5-pruned-emaonly-Q5_1.gguf"

_DEVICE_PREFIX = {"vulkan": "Vulkan", "cuda": "CUDA", "hip": "HIP"}


def default_model_for(engine: str) -> str:
    if engine == "sdcpp":
        return DEFAULT_GGUF_REPO
    return DEFAULT_MODEL


def _sd_cli_list_devices(binary: Path, timeout: int = 30) -> list[str]:
    """Return device names from `sd-cli --list-devices` (e.g. ['Vulkan0', 'CPU'])."""
    try:
        proc = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(binary.parent),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        names = []
        for line in out.splitlines():
            name = line.strip().split()[0] if line.strip() else ""
            if name and not name.startswith("-") and "usage" not in name.lower():
                names.append(name)
        return names
    except Exception:
        return []


@dataclass
class GenParams:
    prompt: str
    negative_prompt: str = ""
    steps: int = 28
    width: int = 512
    height: int = 512
    cfg_scale: float = 7.0
    seed: int = -1
    samples: int = 1
    model: str = ""
    scheduler: str = "DPM++ 2M Karras"


class DiffusionEngine:
    def __init__(self, backend: BackendInfo):
        self.backend = backend
        self.pipe = None
        self.loaded_model = ""
        self.device_str = self._torch_device(backend)

    @staticmethod
    def _torch_device(b: BackendInfo) -> str:
        if b.device == "cuda":
            return "cuda"
        if b.device == "hip":
            return "cuda"  # ROCm uses the cuda API surface
        if b.device == "directml":
            try:
                import torch_directml

                return str(torch_directml.device())
            except Exception:
                return "cpu"
        return "cpu"

    def _import_torch(self):
        import torch

        return torch

    def load_model(self, model: str) -> None:
        if self.backend.engine == "sdcpp":
            self.loaded_model = model
            return
        if self.pipe is not None and self.loaded_model == model:
            return

        torch = self._import_torch()
        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionPipeline,
        )

        device = self.device_str
        dtype = torch.float32
        if self.backend.dtype == "float16":
            # Extra runtime guard: query the device before committing to fp16
            try:
                if device.startswith("cuda") or "privateuseone" in device or device == "hip":
                    dtype = torch.float16
            except Exception:
                dtype = torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            model if "/" in model and not Path(model).exists() else model,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=Path(model).exists() if Path(str(model)).exists() else False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe = pipe.to(device)
        # Memory optimizations for weak GPUs
        try:
            pipe.enable_attention_slicing(1)
        except Exception:
            pass
        if self.backend.device in ("directml", "cpu"):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self.pipe = pipe
        self.loaded_model = model

    def _resolve_model_path(self, model: str) -> str:
        """Local path for the selected model; downloads the default if needed."""
        if model and Path(model).exists():
            return model
        local = [m for m in list_local_models() if model and (model in m["name"] or model in m["path"])]
        if local:
            return local[0]["path"]

        target = model or default_model_for(self.backend.engine)
        if self.backend.engine == "sdcpp":
            return self._resolve_sdcpp_model(target)
        return target

    def _resolve_sdcpp_model(self, target: str) -> str:
        """Resolve a GGUF/single-file checkpoint for sd-cli, downloading if needed."""
        name = Path(target).name
        if name.lower().endswith(".gguf") and "/" in target:
            repo_id, filename = target.split("/", 1) if target.count("/") == 1 else (None, None)
            # HF repo ids look like org/name (exactly one slash)
            if repo_id and "/" not in filename:
                dest = MODELS / "checkpoints" / repo_id.replace("/", "__") / filename
                if dest.exists():
                    return str(dest)
                from .downloader import download_huggingface

                files = download_huggingface(repo_id, filename)
                return str(files[0])

        # Bare repo id (no file) -> default GGUF file from that repo
        if "/" in target and not target.lower().endswith((".gguf", ".safetensors", ".ckpt")):
            repo_id = target
            dest_dir = MODELS / "checkpoints" / repo_id.replace("/", "__")
            existing = next(
                (p for p in dest_dir.glob("*.gguf") if p.is_file()),
                None,
            ) if dest_dir.exists() else None
            if existing:
                return str(existing)
            from .downloader import download_huggingface

            files = download_huggingface(repo_id, DEFAULT_GGUF_FILE)
            return str(files[0])

        if target.lower().endswith((".gguf", ".safetensors", ".ckpt")):
            # Single-file HF repo with standard layout, or DEFAULT
            repo_id = target if "/" in target else DEFAULT_GGUF_REPO
            from .downloader import download_huggingface

            try:
                files = download_huggingface(repo_id, Path(target).name if "/" not in target else None)
                ckpt = next(
                    (f for f in files if f.suffix.lower() in (".gguf", ".safetensors", ".ckpt")),
                    None,
                )
                if ckpt:
                    return str(ckpt)
            except Exception:
                pass

        # Last resort: default GGUF
        from .downloader import download_huggingface

        files = download_huggingface(DEFAULT_GGUF_REPO, DEFAULT_GGUF_FILE)
        return str(files[0])

    def generate(self, params: GenParams, progress_cb=None) -> list[Path]:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        seed = params.seed
        if seed is None or int(seed) < 0:
            seed = int(time.time_ns() % (2**31))

        if self.backend.engine == "sdcpp":
            return self._generate_sdcpp(params, seed, stamp, progress_cb)

        return self._generate_torch(params, seed, stamp, progress_cb)

    def _generate_torch(self, params: GenParams, seed: int, stamp: str, progress_cb) -> list[Path]:
        model = self._resolve_model_path(params.model)
        self.load_model(model)
        torch = self._import_torch()

        generator = None
        try:
            generator = torch.Generator(device=self.device_str if "cpu" not in self.device_str else "cpu")
            generator.manual_seed(seed)
        except Exception:
            generator = torch.Generator()
            generator.manual_seed(seed)

        outputs: list[Path] = []
        prompts = [params.prompt] * max(1, params.samples)

        def _cb(pipe, step, timestep, kw):
            if progress_cb:
                progress_cb(step + 1, params.steps)

        try:
            result = self.pipe(
                prompts,
                negative_prompt=params.negative_prompt or None,
                num_inference_steps=int(params.steps),
                width=int(params.width),
                height=int(params.height),
                guidance_scale=float(params.cfg_scale),
                generator=generator,
                callback=_cb,
                callback_steps=1,
            )
        except TypeError:
            result = self.pipe(
                prompts,
                negative_prompt=params.negative_prompt or None,
                num_inference_steps=int(params.steps),
                width=int(params.width),
                height=int(params.height),
                guidance_scale=float(params.cfg_scale),
                generator=generator,
            )

        images = result.images
        for i, img in enumerate(images):
            path = OUTPUTS / f"{stamp}_{seed}_{i:02d}.png"
            img.save(path)
            outputs.append(path)
        return outputs

    def _sd_backend_args(self, binary: Path) -> list[str]:
        """Pick the ggml device name via --list-devices (e.g. diffusion=Vulkan0)."""
        prefix = _DEVICE_PREFIX.get(self.backend.device)
        if not prefix:
            return []
        for name in _sd_cli_list_devices(binary):
            if name.lower().startswith(prefix.lower()):
                return ["--backend", f"diffusion={name}"]
        # Fallback to the conventional name
        return ["--backend", f"diffusion={prefix}0"]

    def _vram_args(self, model_path: str) -> list[str]:
        """Low-VRAM guards for weak GPUs (RX 580 4 GB, GTX 1050, iGPU, ...)."""
        vram = self.backend.gpu.vram_mb or 0
        if vram <= 0:
            return ["--vae-tiling"]
        args: list[str] = ["--vae-tiling"]
        try:
            model_mb = Path(model_path).stat().st_size // (1 << 20)
        except OSError:
            model_mb = 0
        # Weights barely fit or spill -> offload everything heavy to RAM
        if vram <= 2048 or (model_mb and model_mb > vram * 0.85):
            args.append("--offload-to-cpu")
        elif vram <= 6144:
            args.append("--max-vram")
            args.append(f"{max(1.0, (vram - 512) / 1024):.1f}")
        return args

    def _generate_sdcpp(self, params: GenParams, seed: int, stamp: str, progress_cb) -> list[Path]:
        # Keep BackendInfo.device in sync with the sd-cli build on disk
        # (e.g. requested CUDA but only the Vulkan build could be downloaded).
        binary = ensure_sdcpp_binary(self.backend.device)
        if not binary:
            raise RuntimeError(
                "stable-diffusion.cpp binary not found. Check network / GitHub releases."
            )
        self.backend = sync_sdcpp_device(self.backend)

        if self.backend.device == "vulkan":
            ensure_vulkan_sdk()

        model = self._resolve_model_path(params.model)

        out_dir = OUTPUTS
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        n = max(1, int(params.samples))
        backend_args = self._sd_backend_args(binary)
        vram_args = self._vram_args(model)

        # Stream sd-cli output so failures are visible immediately
        env = os.environ.copy()
        cwd = str(binary.parent)
        tail: list[str] = []

        for i in range(n):
            if progress_cb and n > 1:
                progress_cb(i, n)
            out_file = out_dir / f"{stamp}_{seed}_{i:02d}.png"
            cmd = [
                str(binary),
                "-m", model,
                "-p", params.prompt,
                "-n", params.negative_prompt or "",
                "-W", str(int(params.width)),
                "-H", str(int(params.height)),
                "--steps", str(int(params.steps)),
                "--cfg-scale", str(float(params.cfg_scale)),
                "-s", str(int(seed) + i),
                "-o", str(out_file),
                "-b", "1",
                "--sampling-method", "dpm++2m",
                "--scheduler", "karras",
            ] + backend_args + vram_args

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=cwd,
                bufsize=1,
            )
            step_re = re.compile(r"(\d+)\s*/\s*(\d+)")
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if len(tail) > 80:
                    tail.pop(0)
                print(line, flush=True)
                if progress_cb and n == 1:
                    m = step_re.search(line)
                    if m:
                        try:
                            progress_cb(int(m.group(1)), int(m.group(2)))
                        except Exception:
                            pass
            proc.wait()

            if proc.returncode != 0 and not out_file.exists():
                raise RuntimeError(
                    f"stable-diffusion.cpp failed (exit {proc.returncode}):\n"
                    + "\n".join(tail[-40:])
                )
            if not out_file.exists():
                candidates = sorted(
                    [p for p in out_dir.glob(f"{stamp}*") if p.suffix.lower() in (".png", ".jpg", ".webp")]
                )
                if candidates:
                    outputs.append(candidates[-1])
                else:
                    raise RuntimeError(
                        "sd-cli finished but produced no image:\n" + "\n".join(tail[-40:])
                    )
            else:
                outputs.append(out_file)
            if progress_cb:
                progress_cb(i + 1, n)
        return outputs


_engine: Optional[DiffusionEngine] = None


def get_engine(backend: BackendInfo) -> DiffusionEngine:
    global _engine
    if _engine is None or _engine.backend.device != backend.device or _engine.backend.engine != backend.engine:
        _engine = DiffusionEngine(backend)
    return _engine
