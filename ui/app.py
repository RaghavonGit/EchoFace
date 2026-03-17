import sys
import os
import warnings
warnings.filterwarnings("ignore", message="Failed to build CUDA kernels")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import queue
import threading
import time
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image

import gpu_utils
from generator.styleclip_wrapper import StyleCLIPWrapper
from generator.accessories import apply_accessories
from generator.landmark_detector import detect_landmarks
from asr.audio_processor import process_audio
from asr.transcriber import Transcriber

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #07090f !important;
    font-family: 'Outfit', sans-serif !important;
    color: #b8c8e0 !important;
    min-height: 100vh;
}
.gradio-container {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, #0a1628 0%, #07090f 60%) !important;
}

/* ── Header ──────────────────────────────────────────────── */
.ef-header { text-align: center; padding: 2.8rem 0 2rem; }
.ef-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.8rem, 6vw, 4.2rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1 !important;
    background: linear-gradient(140deg, #e8f0ff 0%, #8ab4ff 45%, #00cfff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.6rem !important;
}
.ef-tagline {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #2e4a6a !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
}
.ef-rule {
    width: 80px; height: 1px;
    background: linear-gradient(90deg, transparent, #00cfff 50%, transparent);
    margin: 1.4rem auto 0;
    animation: rule-pulse 4s ease-in-out infinite;
}
@keyframes rule-pulse {
    0%,100% { opacity:0.3; width:80px; }
    50%      { opacity:0.9; width:140px; }
}

/* ── Panels ───────────────────────────────────────────────── */
.ef-panel {
    background: #0b1120 !important;
    border: 1px solid #13233a !important;
    border-radius: 18px !important;
    padding: 1.6rem 1.5rem !important;
    position: relative;
}
.ef-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #00cfff;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    opacity: 0.8;
}
.ef-sub-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #1e4060;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

/* ── Gradio overrides ────────────────────────────────────── */
.gradio-container .gr-box,
.gradio-container .gr-form { background: transparent !important; border: none !important; }

label.block span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: #2e4a6a !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
textarea, input[type=text] {
    background: #050811 !important;
    border: 1px solid #13233a !important;
    border-radius: 10px !important;
    color: #c0d0e8 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    resize: vertical !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #00cfff !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,207,255,0.07) !important;
}

/* ── Audio ────────────────────────────────────────────────── */
.gr-audio, .gr-audio * { background: #050811 !important; }
.gr-audio { border: 1px solid #13233a !important; border-radius: 10px !important; overflow: hidden; }

/* ── Generate button ─────────────────────────────────────── */
#gen-btn {
    background: linear-gradient(135deg, #ff5c2a 0%, #e8001a 100%) !important;
    border: none !important; border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.95rem !important; width: 100% !important;
    cursor: pointer !important;
    transition: box-shadow 0.25s, transform 0.15s !important;
    position: relative; overflow: hidden;
}
#gen-btn::before {
    content: ''; position: absolute; top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    transition: left 0.5s;
}
#gen-btn:hover::before { left: 140%; }
#gen-btn:hover { box-shadow: 0 0 40px rgba(255,80,40,0.45) !important; transform: translateY(-2px) !important; }
#gen-btn:active { transform: translateY(0) !important; }

/* ── Export button ───────────────────────────────────────── */
#export-btn {
    background: transparent !important;
    border: 1px solid #00cfff !important; border-radius: 10px !important;
    color: #00cfff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    padding: 0.7rem !important; width: 100% !important;
    cursor: pointer !important; margin-top: 0.8rem !important;
    transition: all 0.25s !important; text-transform: uppercase;
}
#export-btn:hover {
    background: rgba(0,207,255,0.07) !important;
    box-shadow: 0 0 24px rgba(0,207,255,0.2) !important;
}

/* ── Main output image ───────────────────────────────────── */
#main-image {
    border-radius: 14px !important; border: 1px solid #13233a !important;
    overflow: hidden !important; background: #050811 !important;
    min-height: 380px !important;
}
#main-image img { border-radius: 13px !important; width: 100% !important; transition: filter 0.5s ease !important; }
#main-image .wrap { min-height: 380px !important; }

/* ── Loader overlay ──────────────────────────────────────── */
#loader-wrap {
    height: 0 !important; overflow: visible !important;
    position: relative !important; z-index: 20 !important;
    pointer-events: none !important;
}
.ef-loader-center {
    position: absolute; top: -205px; left: 50%;
    transform: translateX(-50%);
    display: flex; flex-direction: column; align-items: center;
    gap: 10px; white-space: nowrap;
}
.ef-loader-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: rgba(200,220,255,0.7);
    letter-spacing: 0.2em; text-transform: uppercase;
}
.loader {
    width: 50px; height: 28px;
    --_g: no-repeat radial-gradient(farthest-side,#00cfff 94%,#0000);
    background: var(--_g) 50% 0, var(--_g) 100% 0;
    background-size: 12px 12px; position: relative;
    animation: l23-0 1.5s linear infinite;
}
.loader:before {
    content: ""; position: absolute; height: 12px; aspect-ratio: 1;
    border-radius: 50%; background: #00cfff; left: 0; top: 0;
    animation: l23-1 1.5s linear infinite, l23-2 0.5s cubic-bezier(0,200,.8,200) infinite;
}
@keyframes l23-0 {
    0%,31%  { background-position:50% 0,100% 0 }
    33%     { background-position:50% 100%,100% 0 }
    43%,64% { background-position:50% 0,100% 0 }
    66%     { background-position:50% 0,100% 100% }
    79%     { background-position:50% 0,100% 0 }
    100%    { transform:translateX(calc(-100%/3)) }
}
@keyframes l23-1 { 100% { left: calc(100% + 7px) } }
@keyframes l23-2 { 100% { top: -0.1px } }

/* ── Status / score ──────────────────────────────────────── */
#status-box, #score-box { margin-top: 0.5rem; }
#status-box textarea {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
    color: #5a7a9a !important; background: #050811 !important;
    border: 1px solid #0e1d30 !important; border-radius: 8px !important;
    min-height: unset !important; padding: 0.5rem 0.75rem !important;
}
#score-box textarea {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
    font-weight: 500 !important; color: #00cfff !important;
    background: #050811 !important; border: 1px solid #0e1d30 !important;
    border-radius: 8px !important; min-height: unset !important;
    padding: 0.5rem 0.75rem !important;
}

/* ── Export path ─────────────────────────────────────────── */
#export-path textarea {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.68rem !important;
    color: #2e4a6a !important; background: #050811 !important;
    border: 1px solid #0e1d30 !important; border-radius: 8px !important;
    padding: 0.4rem 0.75rem !important;
}

/* ── Divider ─────────────────────────────────────────────── */
.ef-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #13233a, transparent);
    margin: 1.2rem 0;
}

/* ── Footer ──────────────────────────────────────────────── */
footer, .built-with { display: none !important; }
.ef-footer {
    text-align: center; padding: 2rem 0 1.5rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    color: #141e2e; letter-spacing: 0.18em; text-transform: uppercase;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: #13233a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00cfff; }
"""

_LOADER_ON = """
<style>#main-image img { filter: blur(7px) brightness(0.65) !important; }</style>
<div class="ef-loader-center">
    <div class="loader"></div>
    <span class="ef-loader-label">Generating</span>
</div>
"""
_LOADER_OFF = """
<style>#main-image img { filter: none !important; }</style>
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _startup_notice(cfg):
    return (
        "No GPU detected — generation will be slow (CPU mode)"
        if not cfg["use_fp16"] else ""
    )


# ── Core generator function ───────────────────────────────────────────────────

def generate_fn(mic_audio, file_audio, text_input, wrapper, cfg):
    """
    Yields 13-element tuples:
      0  status_out    (str)
      1  text_in       (str)
      2  score_out     (str)
      3  image_out     (np.ndarray | None)
      4  stored_image  (PIL.Image | None)
      5  export_btn    (gr.update)
      6  loader_html   (gr.update)
      7  base_face     (PIL.Image | gr.update) — clean face for accessories
      8  dd_glasses    (str | gr.update)       — reset to "none" on done
      9  dd_eye        (str | gr.update)       — reset to "none" on done
     10  landmarks     (dict | None)           — landmark cache for accessories
     11  dd_frame_color (str | gr.update)      — reset to "original" on done
     12  dd_lens_tint  (str | gr.update)       — reset to "none" on done
    """
    def _y(status, text, score, img, pil, export, loader,
           base_face=gr.update(), dd_g=gr.update(), dd_e=gr.update(),
           landmarks=gr.update(), dd_fc=gr.update(), dd_lt=gr.update()):
        return (status, text, score, img, pil, export,
                gr.update(value=loader), base_face, dd_g, dd_e,
                landmarks, dd_fc, dd_lt)

    audio_path = file_audio if file_audio is not None else mic_audio
    prompt     = text_input.strip() if text_input else ""

    # ── Guard: nothing provided ────────────────────────────────────────────
    if not prompt and audio_path is None:
        yield _y("Please provide a description or record audio.",
                 "", "", None, None, gr.update(visible=False), _LOADER_OFF)
        return

    # ── Stage 1: ASR ──────────────────────────────────────────────────────
    if not prompt and audio_path is not None:
        yield _y("Transcribing audio...", "", "", None, None,
                 gr.update(visible=False), _LOADER_ON)
        audio_array   = process_audio(audio_path)
        transcription = Transcriber(cfg).transcribe(audio_array)
        note = "  ⚠ Short — consider editing." if len(transcription.split()) < 3 else ""
        yield _y(f"Transcription complete.{note} Review and click Generate again.",
                 transcription, "", None, None, gr.update(visible=False), _LOADER_OFF)
        return

    # ── Guard: prompt too short ────────────────────────────────────────────
    if len(prompt.split()) < 4:
        yield _y("⚠ Prompt is too short — add more detail (hair, eyes, age, skin tone).",
                 prompt, "", None, None, gr.update(visible=False), _LOADER_OFF)
        return

    # ── Stage 2: StyleCLIP optimisation ───────────────────────────────────
    score_q       = queue.Queue()
    result_holder = {}
    t_start       = time.time()

    def _cb(step, loss, pil_img):
        score_q.put(("progress", step, loss, pil_img))

    def _run():
        try:
            pil_image, final_sim = wrapper.generate(prompt, callback=_cb)
            result_holder["success"] = (pil_image, final_sim)
            score_q.put(("done", None, None, None))
        except Exception as exc:
            import traceback; traceback.print_exc()
            result_holder["error"] = exc
            score_q.put(("error", None, None, None))

    threading.Thread(target=_run, daemon=True).start()

    last_img_arr = None

    while True:
        msg, step, loss, pil_img = score_q.get()

        if msg == "error":
            yield _y(f"Generation failed: {result_holder['error']}",
                     "", "", None, None, gr.update(visible=False), _LOADER_OFF)
            return

        elif msg == "done":
            elapsed   = time.time() - t_start
            pil_image, final_sim = result_holder["success"]
            final_arr = np.array(pil_image) if pil_image is not None else last_img_arr

            lms = detect_landmarks(pil_image) if pil_image is not None else None
            if lms is None:
                print("[EchoFace] WARNING: landmark detection failed — "
                      "accessories will use FFHQ fallback coordinates")

            yield _y(f"Generation complete.  {elapsed:.0f}s",
                     "", f"CLIP Score: {final_sim:.4f}",
                     final_arr, pil_image, gr.update(visible=True), _LOADER_OFF,
                     base_face=pil_image, dd_g="none", dd_e="none",
                     landmarks=lms, dd_fc="original", dd_lt="none")
            return

        elif msg == "progress":
            elapsed = time.time() - t_start
            if pil_img is not None:
                last_img_arr = np.array(pil_img)
            yield _y(f"Optimizing...  step {step} / 60  ({elapsed:.0f}s)",
                     prompt, f"Loss: {loss:.4f}",
                     last_img_arr, pil_img, gr.update(visible=False), _LOADER_ON)


# ── Accessories handler ───────────────────────────────────────────────────────

def accessories_fn(base_pil, landmarks, glasses_style: str,
                   frame_color: str, lens_tint: str, eye_color: str):
    """
    Apply accessory overlays on the stored clean face.
    Called whenever any accessories dropdown changes.

    Returns 5-element tuple:
      0  image_out     (np.ndarray)
      1  status_out    (str)
      2  score_out     (gr.update — unchanged)
      3  stored_image  (PIL.Image)
      4  export_btn    (gr.update)
    """
    if base_pil is None:
        return (None, "Generate a face first.", gr.update(),
                None, gr.update(visible=False))

    result = apply_accessories(base_pil, landmarks,
                               glasses=glasses_style,
                               frame_color=frame_color,
                               lens_tint=lens_tint,
                               eye_color=eye_color)

    parts = []
    if glasses_style != "none":
        parts.append(f"{glasses_style} glasses")
    if frame_color not in ("none", "original"):
        parts.append(f"{frame_color} frame")
    if lens_tint != "none":
        parts.append(f"{lens_tint} lens tint")
    if eye_color != "none":
        parts.append(f"{eye_color} eyes")
    msg = ("Applied: " + ", ".join(parts)) if parts else "Accessories cleared."

    return (np.array(result), msg, gr.update(), result, gr.update(visible=True))


# ── Export ────────────────────────────────────────────────────────────────────

def export_image(pil_image, outputs_dir=None):
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    if pil_image is None:
        return (gr.update(visible=False), gr.update(value="No image to export."))
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(outputs_dir, f"{timestamp}.png")
    pil_image.save(path)
    return (gr.update(visible=True), gr.update(visible=True, value=path))


# ── UI Build ──────────────────────────────────────────────────────────────────

def _build_demo(wrapper, cfg):
    import functools
    gen = functools.partial(generate_fn, wrapper=wrapper, cfg=cfg)
    exp = functools.partial(export_image)

    with gr.Blocks(title="EchoFace", css=CUSTOM_CSS) as demo:

        gr.HTML("""
        <div class="ef-header">
            <div class="ef-title">EchoFace</div>
            <div class="ef-tagline">Speak a face into existence</div>
            <div class="ef-rule"></div>
        </div>
        """)

        notice = _startup_notice(cfg)
        if notice:
            gr.HTML(
                f'<div style="text-align:center;font-family:JetBrains Mono,monospace;'
                f'font-size:0.7rem;color:#ff5c2a;margin-bottom:1rem;letter-spacing:0.1em;">'
                f'{notice}</div>'
            )

        # ── Persistent state ──────────────────────────────────────────────
        stored_image    = gr.State(value=None)
        base_face_state = gr.State(value=None)   # always the clean generated face
        landmarks_state = gr.State(value=None)

        with gr.Row(equal_height=False):

            # ── Panel 1: Input ────────────────────────────────────────────
            with gr.Column(scale=3, elem_classes=["ef-panel"]):
                gr.HTML('<div class="ef-section-label">Input</div>')

                mic_in  = gr.Audio(source="microphone", type="filepath",
                                   label="Microphone")
                file_in = gr.Audio(source="upload",     type="filepath",
                                   label="Upload Audio")
                gr.HTML('<div class="ef-divider"></div>')
                text_in = gr.Textbox(
                    lines=3, label="Face Description",
                    placeholder="e.g.  a young woman with dark curly hair and blue eyes",
                )
                gen_btn = gr.Button("Generate", variant="primary",
                                    elem_id="gen-btn")

            # ── Panel 2: Output ───────────────────────────────────────────
            with gr.Column(scale=5, elem_classes=["ef-panel"]):
                gr.HTML('<div class="ef-section-label">Output</div>')

                image_out  = gr.Image(type="numpy", show_label=False,
                                      elem_id="main-image")
                loader_html = gr.HTML(value=_LOADER_OFF, elem_id="loader-wrap")
                status_out  = gr.Textbox(label="Status", interactive=False,
                                         elem_id="status-box")
                score_out   = gr.Textbox(label="CLIP Score", interactive=False,
                                         elem_id="score-box")
                gr.HTML('<div class="ef-divider"></div>')
                export_btn  = gr.Button("Export PNG", visible=False,
                                        elem_id="export-btn")
                export_path = gr.Textbox(label="Saved to", interactive=False,
                                         visible=False, elem_id="export-path")

            # ── Panel 3: Accessories ──────────────────────────────────────
            with gr.Column(scale=3, elem_classes=["ef-panel"]):
                gr.HTML('<div class="ef-section-label">&#10022; Accessories</div>')

                # Placeholder — visible before first generate
                with gr.Column(visible=True) as acc_placeholder:
                    gr.HTML(
                        '<div style="text-align:center;padding:3rem 0.5rem;'
                        'font-family:JetBrains Mono,monospace;font-size:0.60rem;'
                        'color:#1a2a3a;line-height:2;letter-spacing:0.12em;">'
                        'Generate a face first<br>to apply accessories</div>'
                    )

                # Controls — visible after first generate
                with gr.Column(visible=False) as acc_inner:
                    gr.HTML('<div class="ef-sub-label">Frame Style</div>')
                    dd_glasses = gr.Dropdown(
                        choices=["none", "aviator", "round", "wayfarer",
                                 "clubmaster", "cat-eye"],
                        value="none", label="Glasses", interactive=True,
                    )
                    gr.HTML('<div class="ef-sub-label">Frame Colour</div>')
                    dd_frame_color = gr.Dropdown(
                        choices=["original", "black", "gold", "silver", "tortoise"],
                        value="original", label="Frame Color", interactive=True,
                    )
                    gr.HTML('<div class="ef-sub-label">Lens Tint</div>')
                    dd_lens_tint = gr.Dropdown(
                        choices=["none", "grey", "brown", "blue", "green"],
                        value="none", label="Lens Tint", interactive=True,
                    )
                    gr.HTML('<div class="ef-sub-label">Eye Colour</div>')
                    dd_eye = gr.Dropdown(
                        choices=["none", "blue", "green", "hazel", "grey", "brown"],
                        value="none", label="Eye Color", interactive=True,
                    )

        gr.HTML(
            '<div class="ef-footer">EchoFace &nbsp;&middot;&nbsp; '
            'FFHQ StyleGAN2-ADA + CLIP ViT-B/32 + Whisper '
            '&nbsp;&middot;&nbsp; Fully Local</div>'
        )

        # ── Wiring ────────────────────────────────────────────────────────
        _acc_out = [image_out, status_out, score_out, stored_image, export_btn]

        # Generate
        gen_btn.click(
            fn=gen,
            inputs=[mic_in, file_in, text_in],
            outputs=[status_out, text_in, score_out, image_out, stored_image,
                     export_btn, loader_html, base_face_state,
                     dd_glasses, dd_eye,
                     landmarks_state, dd_frame_color, dd_lens_tint],
        ).then(
            # Show accessory controls once we have a face
            fn=lambda bf: (
                gr.update(visible=bf is not None),  # acc_inner
                gr.update(visible=bf is None),      # acc_placeholder
            ),
            inputs=[base_face_state],
            outputs=[acc_inner, acc_placeholder],
        )

        # Accessories — all 4 dropdowns use the same handler;
        # Gradio passes current values of ALL listed inputs automatically.
        for _dd in (dd_glasses, dd_frame_color, dd_lens_tint, dd_eye):
            _dd.change(
                fn=accessories_fn,
                inputs=[base_face_state, landmarks_state,
                        dd_glasses, dd_frame_color, dd_lens_tint, dd_eye],
                outputs=_acc_out,
            )

        # Export
        export_btn.click(
            fn=exp,
            inputs=[stored_image],
            outputs=[export_btn, export_path],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg      = gpu_utils.get_device_config()
    print("[EchoFace] Loading FFHQ StyleGAN2-ADA + CLIP ...")
    PKL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ffhq", "ffhq.pkl")
    wrapper  = StyleCLIPWrapper(PKL_PATH, cfg)
    demo     = _build_demo(wrapper, cfg)
    demo.queue(concurrency_count=1).launch(
        server_name="127.0.0.1", share=False, show_error=True
    )
