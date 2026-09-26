"""Hardware detection and backend selection.

Primary engine is always **sd-cli** (stable-diffusion.cpp): one small native
binary that runs on CUDA / ROCm(HIP) / Vulkan / CPU without installing torch.

Cascade (in order):
  1. NVIDIA            -> sd-cli + CUDA build
  2. AMD / Intel       -> sd-cli + Vulkan build (auto Vulkan SDK if missing)
  3. Forced rocm       -> sd-cli + ROCm(HIP) build
  4. Very old Windows  -> torch + DirectML (torch 2.4.1 stack)
  5. Fallback          -> sd-cli + CPU build
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "backend.json"
BIN_DIR = ROOT / "bin"
SDCPP_DIR = BIN_DIR / "sdcpp"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "backend.log"

GPU_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"

# ---------------------------------------------------------------------------
# Diagnostics: every swallowed error ends up here instead of disappearing
# ---------------------------------------------------------------------------

_failures: list[str] = []


def log(msg: str) -> None:
    """Append a timestamped line to logs/backend.log (never raises)."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def note_failure(stage: str, err: object) -> None:
    """Record a failure reason so the UI / troubleshooting can show it."""
    text = f"{stage}: {err}"
    if text not in _failures:
        _failures.append(text)
    if len(_failures) > 50:
        del _failures[: len(_failures) - 50]
    log("ERROR " + text)


def failures() -> list[str]:
    """Failure reasons collected during this process (newest last)."""
    return list(_failures)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GPUInfo:
    vendor: str = "unknown"  # nvidia | amd | intel | unknown
    name: str = "unknown"
    vram_mb: int = 0
    driver_version: str = ""
    vulkan: bool = False
    dx12: bool = True


@dataclass
class BackendInfo:
    engine: str = "sdcpp"  # sdcpp (sd-cli, primary) | torch (fallback)
    device: str = "cpu"  # cuda | hip | vulkan | directml | cpu
    dtype: str = "float32"
    gpu: GPUInfo = field(default_factory=GPUInfo)
    reason: str = ""
    fp16: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls) -> Optional["BackendInfo"]:
        if not STATE_FILE.exists():
            return None
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            gpu = GPUInfo(**raw.pop("gpu", {}))
            return cls(gpu=gpu, **raw)
        except Exception as e:
            note_failure(f"backend.json corrompido ({STATE_FILE.name})", e)
            return None

    def save(self) -> None:
        STATE_FILE.write_text(self.to_json(), encoding="utf-8")


def _run_rc(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    """Run a command and return (returncode, combined output)."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            tail = (out.strip()[-500:] if out.strip() else "(sem saída)")
            log(f"cmd rc={r.returncode}: {' '.join(cmd[:4])}... -> {tail}")
        return r.returncode, out
    except Exception as e:
        note_failure("cmd " + " ".join(cmd[:4]), e)
        return -1, ""


def _run(cmd: list[str], timeout: int = 20) -> str:
    return _run_rc(cmd, timeout)[1]


def detect_gpus() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []

    # NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        out = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    vram = int(float(parts[1]))
                except ValueError:
                    vram = 0
                gpus.append(
                    GPUInfo(
                        vendor="nvidia",
                        name=parts[0],
                        vram_mb=vram,
                        driver_version=parts[2],
                    )
                )

    # WMI for AMD / Intel / others
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { "
        "\"$($_.Name)|$($_.DriverVersion)|$($_.AdapterRAM)\" "
        "}"
    )
    out = _run(["powershell", "-NoProfile", "-Command", ps])
    for line in out.strip().splitlines():
        if "|" not in line:
            continue
        parts = line.split("|")
        name = parts[0].strip()
        if not name or name.lower() == "virtualmonitor device":
            continue
        driver = parts[1].strip() if len(parts) > 1 else ""
        try:
            vram = int(float(parts[2])) // (1024 * 1024) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            vram = 0
        low = name.lower()
        if any(x in low for x in ("nvidia", "geforce", "rtx ", "gtx", "quadro")):
            vendor = "nvidia"
        elif any(x in low for x in ("amd", "radeon", "rx ", "vega")):
            vendor = "amd"
        elif "intel" in low:
            vendor = "intel"
        else:
            vendor = "unknown"
        if vendor == "nvidia" and any(g.vendor == "nvidia" for g in gpus):
            continue
        if gpus and any(g.name.lower() == name.lower() for g in gpus):
            continue
        gpus.append(GPUInfo(vendor=vendor, name=name, vram_mb=vram, driver_version=driver))

    vulkan_ok = (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll").exists()
    if platform.system() == "Linux":
        vulkan_ok = bool(shutil.which("vulkaninfo")) or Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists()
    for g in gpus:
        g.vulkan = vulkan_ok
    if not gpus:
        gpus.append(GPUInfo(vendor="unknown", name="No dedicated GPU", vulkan=vulkan_ok))
    return gpus


def has_vulkan_loader() -> bool:
    if platform.system() == "Linux":
        return bool(shutil.which("vulkaninfo")) or Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists()
    return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll").exists()


def _fp16_supported(gpu: GPUInfo, device: str) -> bool:
    """Conservative FP16 gate for the torch fallback path only."""
    name = gpu.name.lower()
    if device == "cuda":
        if any(k in name for k in ("rtx", "a100", "h100", "l40", "tesla")):
            return True
        if any(k in name for k in ("gtx 10", "p100", "p40")):
            return True
        return False
    if device == "hip":
        if any(k in name for k in ("rx 6", "rx 7", "rx 9", "w6", "mi1", "mi2", "mi3")):
            return True
        return False
    if device == "directml":
        if any(k in name for k in ("rtx", "arc a", "rx 6", "rx 7", "rx 9")):
            return True
        return False
    return False


def select_backend(prefer: Optional[str] = None) -> BackendInfo:
    """Pick the best backend for this machine (or honour `prefer`).

    sd-cli (engine ``sdcpp``) is the primary engine; torch/diffusers is only
    used for DirectML (very old Windows GPUs) or when explicitly requested.
    """
    gpus = detect_gpus()
    gpu = gpus[0]
    vendor = gpu.vendor

    def sdcpp(device: str, reason: str) -> BackendInfo:
        # sd-cli stores weights with ggml quantization from the model file;
        # the torch dtype/FP16 concept does not apply to it.
        return BackendInfo(
            engine="sdcpp",
            device=device,
            dtype="float32",
            gpu=gpu,
            reason=reason,
            fp16=False,
        )

    def torch_backend(device: str, reason: str) -> BackendInfo:
        dtype = "float16" if _fp16_supported(gpu, device) else "float32"
        return BackendInfo(
            engine="torch",
            device=device,
            dtype=dtype,
            gpu=gpu,
            reason=reason,
            fp16=dtype == "float16",
        )

    # ---- explicit preferences -------------------------------------------
    if prefer == "cpu":
        return sdcpp("cpu", "Forced CPU (sd-cli)")
    if prefer == "directml":
        return torch_backend("directml", "DirectML forced (torch-directml)")
    if prefer == "vulkan":
        return sdcpp("vulkan", "Vulkan forced (sd-cli / stable-diffusion.cpp)")
    if prefer == "rocm":
        return sdcpp("hip", "ROCm/HIP forced (sd-cli)")
    if prefer == "cuda":
        return sdcpp("cuda", "CUDA forced (sd-cli)")

    # ---- automatic cascade ----------------------------------------------
    # 1) NVIDIA -> sd-cli CUDA build (falls back to Vulkan inside the downloader)
    if vendor == "nvidia":
        return sdcpp("cuda", "NVIDIA GPU -> sd-cli CUDA")

    # 2) AMD / Intel / anything with a Vulkan loader -> sd-cli Vulkan build
    if has_vulkan_loader():
        return sdcpp(
            "vulkan",
            "Vulkan compute via sd-cli (stable-diffusion.cpp) - weak/old GPU path",
        )

    # 3) Very old Windows GPU without Vulkan -> DirectML (torch 2.4.1 stack)
    if platform.system() == "Windows" and vendor != "unknown":
        return torch_backend("directml", "No Vulkan loader -> DirectML fallback (torch)")

    # 4) CPU fallback via sd-cli
    return sdcpp("cpu", "CPU fallback (sd-cli)")


def ensure_backend(prefer: Optional[str] = None) -> BackendInfo:
    cached = BackendInfo.load()
    if cached and not prefer:
        return cached
    info = select_backend(prefer)
    info.save()
    return info


def install_payload(info: BackendInfo) -> None:
    """Install the Python deps that match the selected backend."""
    py = sys.executable
    req = str(ROOT / "requirements.txt")
    rc, out = _run_rc([py, "-m", "pip", "install", "-q", "-r", req], timeout=1800)
    if rc != 0:
        note_failure("pip install -r requirements.txt", f"rc={rc} {out.strip()[-300:]}")

    if info.engine != "torch":
        # sd-cli needs no torch stack at all
        return

    if info.device == "cuda":
        rc, out = _run_rc(
            [
                py, "-m", "pip", "install", "-q",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ],
            timeout=3600,
        )
        if rc != 0:
            note_failure("pip install torch (cu124)", f"rc={rc} {out.strip()[-300:]}")
    elif info.device == "hip":
        if platform.system() == "Linux":
            rc, out = _run_rc(
                [
                    py, "-m", "pip", "install", "-q",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/rocm6.2",
                ],
                timeout=3600,
            )
            if rc != 0:
                note_failure("pip install torch (rocm6.2)", f"rc={rc} {out.strip()[-300:]}")
        else:
            info.reason += " (ROCm torch unavailable on Windows -> DirectML stack)"
            info.device = "directml"
            info.save()
            _install_directml(py)
    elif info.device == "directml":
        _install_directml(py)
    else:
        rc, out = _run_rc(
            [
                py, "-m", "pip", "install", "-q", "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ],
            timeout=3600,
        )
        if rc != 0:
            note_failure("pip install torch (cpu)", f"rc={rc} {out.strip()[-300:]}")


def _install_directml(py: str) -> None:
    """DirectML stack pinned to the stable torch 2.4.1 + torch-directml combo."""
    _run([py, "-m", "pip", "install", "-q", "torch==2.4.1", "torchvision==0.19.1"], timeout=3600)
    _run([py, "-m", "pip", "install", "-q", "torch-directml==0.2.5.dev240914"], timeout=1800)


def ensure_vulkan_sdk() -> bool:
    """Download + install LunarG Vulkan SDK silently when vulkan-1.dll is missing."""
    if has_vulkan_loader():
        return True
    if platform.system() != "Windows":
        return bool(shutil.which("vulkaninfo"))
    try:
        import requests

        SDCPP_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://sdk.lunarg.com/sdk/download/latest/windows/vulkan-sdk.exe"
        installer = BIN_DIR / "vulkan-sdk.exe"
        if not installer.exists():
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(installer, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            log(f"Vulkan SDK installer baixado ({installer.stat().st_size // (1 << 20)} MB)")
        rc, out = _run_rc(
            [
                str(installer),
                "--accept-licenses",
                "--default-answer",
                "--confirm-command",
                "install",
            ],
            timeout=1800,
        )
        if rc != 0:
            note_failure("instalacao do Vulkan SDK", f"rc={rc} {out.strip()[-300:]}")
        ok = has_vulkan_loader()
        if not ok:
            note_failure("Vulkan SDK", "vulkan-1.dll ausente apos instalacao (reinicie o app/PC?)")
        return ok
    except Exception as e:
        note_failure("download/instalacao do Vulkan SDK", e)
        return False


# ---------------------------------------------------------------------------
# sd-cli (stable-diffusion.cpp) binary management
# ---------------------------------------------------------------------------

_SDCPP_ASSETS = {
    "cuda": ["bin-win-cuda12-x64.zip", "win-cuda12"],
    "hip": ["bin-win-rocm", "win-rocm"],
    "vulkan": ["bin-win-vulkan", "vulkan-x64"],
    "cpu": ["bin-win-cpu", "win-cpu"],
}


def _sdcpp_exe() -> Optional[Path]:
    for name in ("sd-cli.exe", "sd-cli", "sd.exe", "sd"):
        cand = SDCPP_DIR / name
        if cand.is_file():
            return cand
    if SDCPP_DIR.exists():
        for p in sorted(SDCPP_DIR.rglob("sd*.exe")):
            if p.is_file() and "server" not in p.name.lower():
                return p
    return None


def _sdcpp_installed_device() -> Optional[str]:
    """Which ggml backend DLL is currently present in bin/sdcpp."""
    if not SDCPP_DIR.exists():
        return None
    names = [p.name.lower() for p in SDCPP_DIR.glob("ggml-*.dll")]
    if any(n.startswith("ggml-cuda") for n in names):
        return "cuda"
    if any(n.startswith("ggml-vulkan") for n in names):
        return "vulkan"
    if any(n.startswith("ggml-hip") or n.startswith("ggml-rocm") for n in names):
        return "hip"
    if _sdcpp_exe():
        return "cpu"
    return None


def _download_asset(asset: dict, dest_zip: Path) -> None:
    """Download a release asset to a temp file, then verify size + digest.

    GitHub exposes the asset SHA-256 in `asset["digest"]` ("sha256:<hex>");
    when present the file is rejected unless it matches.
    """
    import requests

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_suffix(dest_zip.suffix + ".part")
    expected_size = int(asset.get("size") or 0)
    digest = (asset.get("digest") or "")
    with requests.get(asset["browser_download_url"], stream=True, timeout=600) as r:
        r.raise_for_status()
        header_len = int(r.headers.get("content-length") or 0)
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)

    got = tmp.stat().st_size
    want = expected_size or header_len
    if want and got != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download incompleto de {asset['name']}: {got} de {want} bytes")

    if digest.startswith("sha256:"):
        want_hex = digest.split(":", 1)[1].strip().lower()
        got_hex = _sha256(tmp)
        if got_hex != want_hex:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 nao confere em {asset['name']}: esperado {want_hex[:16]}..., obtido {got_hex[:16]}..."
            )
        log(f"SHA-256 verificado: {asset['name']} ({got_hex[:16]}...)")
    else:
        log(f"AVISO: {asset['name']} sem digest publicado; verificado apenas o tamanho")

    tmp.replace(dest_zip)


def _extract(zip_path: Path, target: Path) -> None:
    """Extract a zip safely: block path traversal (Zip Slip) outside `target`."""
    target.mkdir(parents=True, exist_ok=True)
    base = target.resolve()
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                dest = (base / member.filename).resolve()
                if dest != base and base not in dest.parents:
                    raise RuntimeError(f"entrada insegura no zip bloqueada: {member.filename}")
            z.extractall(target)
    except zipfile.BadZipFile as e:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"zip corrompido ({zip_path.name}) - download interrompido ou asset mudou; "
            "delete o arquivo e tente de novo"
        ) from e
    zip_path.unlink(missing_ok=True)


def ensure_sdcpp_binary(device: str = "vulkan") -> Optional[Path]:
    """Download the sd-cli build that matches `device` (with safe fallbacks).

    Returns the path to sd-cli, or None when nothing could be downloaded.
    """
    SDCPP_DIR.mkdir(parents=True, exist_ok=True)
    installed = _sdcpp_installed_device()

    # Already have a usable build for this device (any build serves "cpu")
    if _sdcpp_exe() and installed and (installed == device or device == "cpu"):
        return _sdcpp_exe()

    try:
        import requests

        rel = requests.get(GPU_API, timeout=60)
        rel.raise_for_status()
        assets = rel.json().get("assets", [])

        # Build preference chain: requested device, then vulkan, then cpu
        keys: list[str] = []
        for d in (device, "vulkan", "cpu"):
            for k in _SDCPP_ASSETS.get(d, []):
                if k not in keys:
                    keys.append(k)

        asset = None
        for key in keys:
            for a in assets:
                if key in a["name"] and a["name"].endswith(".zip") and "cudart" not in a["name"]:
                    asset = a
                    break
            if asset:
                break
        if not asset:
            note_failure(
                "busca de binario sd-cli",
                f"nenhum asset compativel com '{device}' na release; "
                f"nomes disponiveis: {', '.join(a['name'] for a in assets[:8])}",
            )
            return _sdcpp_exe()

        zip_path = BIN_DIR / asset["name"]
        try:
            if not zip_path.exists():
                log(f"baixando sd-cli [{asset['name']}] ({asset.get('size', 0) // (1 << 20)} MB)")
                _download_asset(asset, zip_path)
            _extract(zip_path, SDCPP_DIR)
        except Exception as e:
            note_failure(f"instalacao do sd-cli ({asset['name']})", e)
            return _sdcpp_exe()

        # CUDA builds may need the redistributable runtime (best effort)
        if device == "cuda":
            for a in assets:
                if "cudart" in a["name"] and a["name"].endswith(".zip"):
                    try:
                        crt = BIN_DIR / a["name"]
                        if not crt.exists():
                            _download_asset(a, crt)
                        _extract(crt, SDCPP_DIR)
                    except Exception as e:
                        note_failure("cudart (runtime CUDA)", e)
                    break
    except Exception as e:
        # GitHub API unreachable, rate limit, proxy block, etc.
        note_failure("API do GitHub (release sd-cli)", e)
        return _sdcpp_exe()

    return _sdcpp_exe()


def sync_sdcpp_device(info: BackendInfo) -> BackendInfo:
    """Make BackendInfo.device agree with the sd-cli build actually on disk."""
    if info.engine != "sdcpp":
        return info
    installed = _sdcpp_installed_device()
    if installed and installed != info.device:
        info.device = installed
        info.reason = f"{info.reason} (binary backend: {installed})"
        info.save()
    return info


if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else None
    info = ensure_backend(pref)
    print(info.to_json())
