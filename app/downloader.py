"""Model download helpers for Hugging Face and CivitAI.

API notes / safety:
  - No tokens are ever written to disk or logs.
  - CivitAI optional token is read only from the CIVITAI_API_KEY env var.
  - Filenames are sanitized; downloads stream to a temp path then rename.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORA_DIR = MODELS_DIR / "loras"
VAE_DIR = MODELS_DIR / "vae"
# Ensure model kind folders exist so the UI lists them and sd-cli can scan them
for _d in (CHECKPOINTS_DIR, LORA_DIR, VAE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HF_DIRECT_RE = re.compile(r"^https?://huggingface\.co/.+")
CIVIT_URL = "https://civitai.com/api/v1/models"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "AdaptDiffuse/1.0"})
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf:
        s.headers["Authorization"] = f"Bearer {hf}"
    return s


def _safe_name(url: str, fallback: str = "model") -> str:
    name = Path(urlparse(url).path).name or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or fallback


def _stream_to(url: str, dest: Path, headers: Optional[dict] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300, headers=headers or {}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total and done % (16 << 20) < (1 << 20):
                    pct = done * 100 // total
                    print(f"  {dest.name}: {pct}% ({done // (1 << 20)} MB / {total // (1 << 20)} MB)", flush=True)
    tmp.replace(dest)
    return dest


def download_huggingface(repo_id: str, filename: Optional[str] = None, subfolder: str = "") -> list[Path]:
    """Download file(s) from a Hugging Face repo into models/."""
    from huggingface_hub import hf_hub_download, snapshot_download

    results: list[Path] = []
    if filename:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder or None,
            local_dir=str(CHECKPOINTS_DIR / repo_id.replace("/", "__")),
        )
        results.append(Path(p))
        return results

    dest = MODELS_DIR / "hf" / repo_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=repo_id, local_dir=str(dest))
    for p in Path(path).rglob("*"):
        if p.suffix.lower() in (".safetensors", ".ckpt", ".pt", ".gguf", ".bin"):
            results.append(p)
    return results


def download_url(url: str, dest_dir: Optional[Path] = None, filename: Optional[str] = None) -> Path:
    """Download an arbitrary model URL (CivitAI direct links, HF resolve, etc.)."""
    name = filename or _safe_name(url)
    if not any(name.lower().endswith(ext) for ext in (".safetensors", ".ckpt", ".pt", ".gguf", ".bin", ".vae")):
        # CivitAI sometimes returns a model file without extension
        ctype_guess = ""
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            ctype_guess = head.headers.get("content-type", "")
        except Exception:
            pass
        if "octet-stream" in ctype_guess or not name:
            name = name + ".safetensors"
    target_dir = dest_dir or CHECKPOINTS_DIR
    return _stream_to(url, target_dir / name)


def search_civitai(query: str, limit: int = 8, civit_type: str = "Checkpoint") -> list[dict]:
    """Search CivitAI public model API (no token required for SFW listings)."""
    headers = {}
    key = os.environ.get("CIVITAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    params = {"limit": limit, "sort": "Highest Rated"}
    if civit_type and civit_type.strip():
        params["types"] = civit_type.strip()
    else:
        params["types"] = "Checkpoint"
    if query and query.strip():
        params["query"] = query.strip()
    r = requests.get(CIVIT_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    items = []
    for model in data.get("items", []):
        stats = model.get("stats", {}) or {}
        model_version = (model.get("modelVersions") or [{}])[0]
        files = model_version.get("files") or [{}]
        link = files[0].get("downloadUrl") or ""
        items.append(
            {
                "id": model.get("id"),
                "name": model.get("name", "untitled"),
                "type": model.get("type", ""),
                "downloads": stats.get("downloadCount", 0),
                "rating": stats.get("rating", 0),
                "base_model": model_version.get("baseModel", ""),
                "download_url": link,
                "ssd": files[0].get("sizeKB"),
            }
        )
    return items


def download_civitai(url_or_id: str) -> Path:
    """Download a CivitAI model from a page URL, API model id, or direct download URL."""
    if url_or_id.isdigit():
        # Resolve latest version file
        headers = {}
        key = os.environ.get("CIVITAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.get(f"{CIVIT_URL}/{url_or_id}", headers=headers, timeout=60)
        r.raise_for_status()
        model = r.json()
        mtype = (model.get("type") or "").lower()
        dest_dir = LORA_DIR if mtype == "lora" else CHECKPOINTS_DIR
        versions = model.get("modelVersions") or []
        if not versions:
            raise RuntimeError("No model versions found on CivitAI")
        files = versions[0].get("files") or []
        if not files:
            raise RuntimeError("No files on CivitAI version")
        dl = files[0].get("downloadUrl")
        name = files[0].get("name") or f"civitai_{url_or_id}.safetensors"
        if not dl:
            raise RuntimeError("CivitAI file has no downloadUrl")
        return download_url(dl, dest_dir=dest_dir, filename=_safe_name(name))

    if "civitai.com" in url_or_id and "/models/" in url_or_id and "download" not in url_or_id:
        m = re.search(r"/models/(\d+)", url_or_id)
        if m:
            return download_civitai(m.group(1))

    return download_url(url_or_id)


def list_local_models() -> list[dict]:
    out = []
    for folder, kind in (
        (CHECKPOINTS_DIR, "checkpoint"),
        (LORA_DIR, "lora"),
        (VAE_DIR, "vae"),
        (MODELS_DIR / "hf", "checkpoint"),
    ):
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in (".safetensors", ".ckpt", ".pt", ".gguf", ".bin") and p.is_file():
                out.append(
                    {
                        "name": p.stem,
                        "path": str(p),
                        "kind": kind,
                        "size_mb": p.stat().st_size // (1 << 20),
                    }
                )
    return out
