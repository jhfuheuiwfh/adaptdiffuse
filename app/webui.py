"""AdaptDiffuse WebUI (Gradio)."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import gradio as gr

from . import __version__
from .backend import BackendInfo, ensure_backend, ensure_sdcpp_binary, ensure_vulkan_sdk, install_payload, select_backend, sync_sdcpp_device
from .downloader import download_civitai, download_huggingface, list_local_models, search_civitai
from .pipeline import DEFAULT_MODEL, GenParams, default_model_for, get_engine

ROOT = Path(__file__).resolve().parent.parent

CSS = """
:root {
  --bg: #0b0f17;
  --panel: #121826;
  --panel-2: #0e1420;
  --line: #1f2a3f;
  --text: #e7eefc;
  --muted: #93a4c3;
  --accent: #6ea8fe;
  --accent-2: #8b5cf6;
}
.gradio-container {
  max-width: 1200px !important;
  margin: auto !important;
  background: radial-gradient(1200px 600px at 10% -10%, #1a2340 0%, var(--bg) 55%) !important;
  color: var(--text) !important;
}
header#banner { display: none !important; }
.tabs, .tab-nav { background: transparent !important; }
.tab-nav button {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  color: var(--muted) !important;
  border-radius: 10px !important;
  margin: 0 4px !important;
}
.tab-nav button.selected {
  color: white !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px rgba(110,168,254,.35), 0 8px 24px rgba(0,0,0,.35) !important;
  background: linear-gradient(180deg, #172033, #121a2b) !important;
}
.gr-panel, .gr-box, .gr-button, .gr-input, .gr-textarea, .gr-dropdown {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  color: var(--text) !important;
}
.gr-button.primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
}
.status-pill {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#132038; border:1px solid #27406b; color:#9ec5ff; font-size:12px;
}
footer { display: none !important; }
"""


def _fmt_backend(b: BackendInfo) -> str:
    if b.engine == "sdcpp":
        prec = "ggml (from model file)"
    else:
        prec = "FP16" if b.fp16 else "FP32"
    return (
        f"**Engine:** `{b.engine}` (sd-cli)" if b.engine == "sdcpp" else f"**Engine:** `{b.engine}` (torch)"
    ) + (
        f"  ·  **Device:** `{b.device}`  ·  **Precision:** `{prec}`\n\n"
        f"**GPU:** {b.gpu.name} ({b.gpu.vendor})  ·  **VRAM:** {b.gpu.vram_mb or '?'} MB\n\n"
        f"**Why:** {b.reason}"
    )


def build_ui() -> gr.Blocks:
    backend = ensure_backend()

    with gr.Blocks(title="AdaptDiffuse", css=CSS, theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            f"""
# AdaptDiffuse
### Plug-and-play Stable Diffusion · sd-cli first (CUDA / ROCm / Vulkan / CPU)
<span class="status-pill">v{__version__}</span>
<span class="status-pill">{backend.engine}:{backend.device}</span>
<span class="status-pill">{backend.gpu.name}</span>
            """,
            elem_id="title",
        )

        with gr.Row():
            backend_md = gr.Markdown(_fmt_backend(backend), elem_id="backendinfo")
            refresh_btn = gr.Button("Redisetect hardware", variant="secondary", scale=0)

        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        model_dd = gr.Dropdown(
                            choices=_model_choices(),
                            value=backend_reasonable_default(),
                            label="Model",
                            allow_custom_value=True,
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            placeholder="a serene mountain lake at sunrise, ultra detailed, photorealistic",
                        )
                        negative = gr.Textbox(
                            label="Negative prompt",
                            lines=2,
                            value="blurry, low quality, watermark, text, deformed",
                        )
                        lora_dd = gr.Dropdown(
                            choices=_lora_choices(),
                            value="",
                            label="LoRA (none if empty)",
                            allow_custom_value=True,
                        )
                        lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA scale")
                        with gr.Accordion("Advanced", open=False):
                            steps = gr.Slider(4, 80, value=26, step=1, label="Steps")
                            cfg = gr.Slider(1.0, 14.0, value=7.0, step=0.5, label="CFG Scale")
                            width = gr.Dropdown([512, 576, 640, 768, 832, 896, 1024], value=512, label="Width")
                            height = gr.Dropdown([512, 576, 640, 768, 832, 896, 1024], value=512, label="Height")
                            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                            samples = gr.Slider(1, 4, value=1, step=1, label="Images")
                        run_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label="Results", columns=2, height=480, object_fit="contain")
                        status = gr.Markdown("Ready.")
                        open_out = gr.Markdown("")

                def _run(pr, neg, st, w, h, cfg_s, sd, n, model, lora, lora_w):
                    backend_in = ensure_backend()
                    eng = get_engine(backend_in)
                    params = GenParams(
                        prompt=pr or "",
                        negative_prompt=neg or "",
                        steps=int(st),
                        width=int(w),
                        height=int(h),
                        cfg_scale=float(cfg_s),
                        seed=int(sd),
                        samples=int(n),
                        model=model or "",
                        lora=(lora or "").strip(),
                        lora_scale=float(lora_w),
                    )
                    t0 = time.time()
                    try:
                        paths = eng.generate(params)
                        imgs = [str(p) for p in paths]
                        dt = time.time() - t0
                        used_seed = params.seed if params.seed is not None and int(params.seed) >= 0 else "auto"
                        lora_note = f" · LoRA `{Path(params.lora).stem}` ×{params.lora_scale:g}" if params.lora else ""
                        status = f"Done in **{dt:.1f}s** · seed `{used_seed}`{lora_note}"
                        folder = str(ROOT / "outputs")
                        return imgs, status, f"Saved to `{folder}`"
                    except Exception as e:
                        return [], f"**Error:** {e}", ""

                run_btn.click(
                    _run,
                    inputs=[prompt, negative, steps, width, height, cfg, seed, samples, model_dd, lora_dd, lora_scale],
                    outputs=[gallery, status, open_out],
                )

            with gr.TabItem("Models"):
                gr.Markdown("Download models from **Hugging Face** or **CivitAI**. Checkpoints go to `models/checkpoints/`; LoRA (type **LORA**) goes to `models/loras/`. CivitAI NSFW/age-gated models need `CIVITAI_API_KEY` in the environment — the key is never stored.")
                with gr.Row():
                    hf_repo = gr.Textbox(label="Hugging Face repo id", placeholder="second-state/stable-diffusion-v1-5-GGUF")
                    hf_file = gr.Textbox(label="Filename (optional)", placeholder="stable-diffusion-v1-5-pruned-emaonly-Q5_1.gguf")
                    hf_btn = gr.Button("Download from HF", variant="primary")
                with gr.Row():
                    civit_in = gr.Textbox(label="CivitAI URL or Model ID", placeholder="https://civitai.com/models/xxxx or 12345")
                    civit_btn = gr.Button("Download from CivitAI", variant="primary")
                with gr.Row():
                    search_in = gr.Textbox(label="Search CivitAI", placeholder="e.g. anime pastel")
                    search_type = gr.Dropdown(
                        ["Checkpoint", "LORA", "Embedding", "VAE"],
                        value="Checkpoint",
                        label="Type",
                    )
                    search_btn = gr.Button("Search")
                search_table = gr.Dataframe(headers=["name", "type", "base", "downloads", "rating", "url"], label="Results", interactive=False)
                dl_status = gr.Markdown("")
                local_table = gr.Dataframe(
                    headers=["name", "kind", "size_mb", "path"],
                    value=_local_rows(),
                    label="Local models",
                    interactive=False,
                )
                refresh_local = gr.Button("Refresh local list")

                def _hf(repo, filename):
                    try:
                        files = download_huggingface(repo.strip(), filename.strip() or None)
                        return f"Downloaded {len(files)} file(s):\n" + "\n".join(f"- `{f}`" for f in files), _local_rows(), _lora_choices()
                    except Exception as e:
                        return f"**HF error:** {e}", _local_rows(), _lora_choices()

                def _civit(link):
                    try:
                        p = download_civitai(link.strip())
                        return f"Downloaded `{p}` ({p.stat().st_size // (1<<20)} MB)", _local_rows(), _lora_choices()
                    except Exception as e:
                        return f"**CivitAI error:** {e}", _local_rows(), _lora_choices()

                def _search(q, typ):
                    try:
                        rows = [
                            [i["name"], i["type"], i["base_model"], i["downloads"], i["rating"], i["download_url"] or ""]
                            for i in search_civitai(q.strip(), civit_type=typ or "Checkpoint")
                        ]
                        return rows or [["No results", "", "", "", "", ""]]
                    except Exception as e:
                        return [[f"Error: {e}", "", "", "", "", ""]]

                hf_btn.click(_hf, inputs=[hf_repo, hf_file], outputs=[dl_status, local_table, lora_dd])
                civit_btn.click(_civit, inputs=[civit_in], outputs=[dl_status, local_table, lora_dd])
                search_btn.click(_search, inputs=[search_in, search_type], outputs=[search_table])
                refresh_local.click(lambda: (_local_rows(), _lora_choices()), outputs=[local_table, lora_dd])

            with gr.TabItem("Backend"):
                gr.Markdown(
                    """
Primary engine: **sd-cli** (stable-diffusion.cpp) - one small native binary,
no torch needed. Auto cascade:

1. **NVIDIA** -> sd-cli CUDA build (falls back to Vulkan)
2. **AMD / Intel / anything with Vulkan** -> sd-cli Vulkan build (auto Vulkan SDK)
3. Forced **rocm** -> sd-cli ROCm(HIP) build
4. Very old Windows GPU without Vulkan -> **DirectML** (torch 2.4.1 + torch-directml)
5. Fallback -> sd-cli **CPU** build

FP16 is only enabled on the torch/DirectML path when the GPU is known-safe;
sd-cli uses ggml quantization straight from the model file (GGUF recommended).
                    """
                )
                with gr.Row():
                    choice = gr.Radio(
                        choices=["auto", "cuda", "rocm", "vulkan", "directml", "cpu"],
                        value="auto",
                        label="Backend preference",
                    )
                    apply_btn = gr.Button("Apply & reinstall deps", variant="primary")
                backend_log = gr.Textbox(label="Log", lines=16, interactive=False)

                def _apply(pref):
                    try:
                        p = None if pref == "auto" else pref
                        info = select_backend(p)
                        info.save()
                        if info.engine == "sdcpp":
                            if info.device == "vulkan":
                                ensure_vulkan_sdk()
                            ensure_sdcpp_binary(info.device)
                            info = sync_sdcpp_device(info)
                        install_payload(info)
                        info = ensure_backend(p)
                        eng = get_engine(info)
                        eng.backend = info
                        eng.loaded_model = ""
                        eng.pipe = None
                        txt = info.to_json()
                        return txt, _fmt_backend(info), _model_choices()
                    except Exception as e:
                        return f"error: {e}", _fmt_backend(ensure_backend()), _model_choices()

                apply_btn.click(
                    _apply,
                    inputs=[choice],
                    outputs=[backend_log, backend_md, model_dd],
                )

            with gr.TabItem("About"):
                gr.Markdown(
                    f"""
**AdaptDiffuse v{__version__}** — Stable Diffusion engine tuned for weak GPUs
(Polaris, GTX 10-series, iGPU-class) while scaling to RTX / RDNA.

- Engines: **sd-cli / stable-diffusion.cpp** primary (CUDA / ROCm / Vulkan / CPU) + `diffusers` fallback (DirectML / CPU)
- DirectML stack (fallback only): torch **2.4.1** + torch-directml **0.2.5.dev240914**
- Weak-GPU guards: VAE tiling, CPU offload, max-VRAM budget
- FP16 auto-gated on the torch path; sd-cli uses GGUF quantization from the model file
- Models: Hugging Face + CivitAI downloader (GGUF one-click default)
- Everything is local — prompts/images never leave this machine
                    """
                )

            with gr.TabItem("Support"):
                gr.Markdown(
                    """
### Found a bug or have feedback?

**Email:** [stableuser1223@gmail.com](mailto:stableuser1223@gmail.com?subject=AdaptDiffuse%20feedback)

**GitHub:** open a **Feedback** or **Bug report** issue (templates below) or comment on
[github.com/jhfuheuiwfh/adaptdiffuse](https://github.com/jhfuheuiwfh/adaptdiffuse/issues/new/choose)

When reporting a problem, please include:

- What you tried (prompt / model / LoRA / backend)
- The exact error message from the **status** area or console
- Your GPU + backend shown at the top of this page (`Engine` / `Device` pills)
                    """
                )
                with gr.Row():
                    gr.HTML(
                        '<a href="mailto:stableuser1223@gmail.com?subject=AdaptDiffuse%20feedback" '
                        'style="display:inline-block;padding:10px 18px;border-radius:10px;'
                        'background:linear-gradient(135deg,#6ea8fe,#8b5cf6);color:#fff;'
                        'text-decoration:none;font-weight:600;">Email support</a>&nbsp;&nbsp;'
                        '<a href="https://github.com/jhfuheuiwfh/adaptdiffuse/issues/new?template=feedback.yml" '
                        'target="_blank" '
                        'style="display:inline-block;padding:10px 18px;border-radius:10px;'
                        'background:var(--panel);border:1px solid var(--line);color:#e7eefc;'
                        'text-decoration:none;font-weight:600;">Feedback on GitHub</a>&nbsp;&nbsp;'
                        '<a href="https://github.com/jhfuheuiwfh/adaptdiffuse/issues/new?template=bug_report.yml" '
                        'target="_blank" '
                        'style="display:inline-block;padding:10px 18px;border-radius:10px;'
                        'background:var(--panel);border:1px solid var(--line);color:#e7eefc;'
                        'text-decoration:none;font-weight:600;">Bug report on GitHub</a>'
                    )

        refresh_btn.click(
            lambda: _fmt_backend(ensure_backend()),
            outputs=[backend_md],
        )

    return demo


def backend_reasonable_default() -> str:
    models = list_local_models()
    if models:
        # Prefer GGUF for the sd-cli engine
        backend = ensure_backend()
        if backend.engine == "sdcpp":
            ggufs = [m for m in models if m["path"].lower().endswith(".gguf")]
            if ggufs:
                return ggufs[0]["path"]
        return models[0]["path"]
    return default_model_for(backend.engine)


def _model_choices() -> list[str]:
    backend = ensure_backend()
    default = default_model_for(backend.engine)
    choices = [m["path"] for m in list_local_models() if m["kind"] == "checkpoint"]
    if default not in choices:
        choices = [default] + choices
    return choices


def _lora_choices() -> list[str]:
    """Paths under models/loras (sd-cli style) plus a blank for none."""
    choices = [""] + [m["path"] for m in list_local_models() if m["kind"] == "lora"]
    return choices


def _local_rows():
    return [[m["name"], m["kind"], m["size_mb"], m["path"]] for m in list_local_models()] or [["(empty)", "", "", ""]]
