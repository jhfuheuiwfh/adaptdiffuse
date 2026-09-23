"""Entry point: python -m app.main [--port 7860] [--share] [--backend auto|cuda|...]"""

from __future__ import annotations

import argparse
import sys

from .backend import (
    ensure_backend,
    ensure_sdcpp_binary,
    ensure_vulkan_sdk,
    install_payload,
    sync_sdcpp_device,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="AdaptDiffuse WebUI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--backend", default=None, help="auto|cuda|rocm|vulkan|directml|cpu")
    ap.add_argument("--setup", action="store_true", help="Install backend deps then exit")
    args = ap.parse_args()

    pref = None if (args.backend in (None, "auto")) else args.backend
    info = ensure_backend(pref)

    # First-run dependency install
    need_install = args.setup
    try:
        if info.engine == "torch":
            import torch  # noqa: F401

            if info.device == "cuda" and not torch.cuda.is_available():
                need_install = True
            if info.device == "directml":
                import torch_directml  # noqa: F401

                _ = torch_directml.device()
    except Exception:
        need_install = True

    if need_install:
        print(f"[setup] Installing backend: {info.engine}/{info.device} ...", flush=True)
        install_payload(info)
        info = ensure_backend(pref)

    if info.engine == "sdcpp":
        if info.device == "vulkan" and not ensure_vulkan_sdk():
            print("[warn] Vulkan SDK/loader not detected; trying GPU driver runtime...", flush=True)
        bin_path = ensure_sdcpp_binary(info.device)
        info = sync_sdcpp_device(info)
        if bin_path:
            print(f"[setup] sd-cli (stable-diffusion.cpp) ready: {bin_path}", flush=True)
        else:
            print("[warn] sd-cli binary not downloaded; generation will retry on demand.", flush=True)

    print(f"[backend] {info.engine}/{info.device} | {info.gpu.name} | {info.reason}", flush=True)

    if args.setup:
        return 0

    from .webui import build_ui

    # Avoid corporate/system proxies breaking the local Gradio socket
    import os

    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)

    demo = build_ui()
    demo.queue(max_size=16).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        inbrowser=True,
        prevent_thread_lock=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
