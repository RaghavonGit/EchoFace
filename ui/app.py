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

/* ── Header ─────────────────────────────────────────────────── */
.ef-header {
    text-align: center;
    padding: 2.8rem 0 2rem;
}
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
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00cfff 50%, transparent);
    margin: 1.4rem auto 0;
    animation: rule-pulse 4s ease-in-out infinite;
}
@keyframes rule-pulse {
    0%,100% { opacity:0.3; width:80px; }
    50%      { opacity:0.9; width:140px; }
}

/* ── Panels ──────────────────────────────────────────────────── */
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

/* ── Gradio overrides ──────────────────────────────────────── */
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

/* ── Audio components ─────────────────────────────────────── */
.gr-audio, .gr-audio * { background: #050811 !important; }
.gr-audio { border: 1px solid #13233a !important; border-radius: 10px !important; overflow: hidden; }

/* ── Generate button ──────────────────────────────────────── */
#gen-btn {
    background: linear-gradient(135deg, #ff5c2a 0%, #e8001a 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.95rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: box-shadow 0.25s, transform 0.15s !important;
    position: relative; overflow: hidden;
}
#gen-btn::before {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    transition: left 0.5s;
}
#gen-btn:hover::before { left: 140%; }
#gen-btn:hover {
    box-shadow: 0 0 40px rgba(255,80,40,0.45) !important;
    transform: translateY(-2px) !important;
}
#gen-btn:active { transform: translateY(0) !important; }

/* ── Export button ───────────────────────────────────────── */
#export-btn {
    background: transparent !important;
    border: 1px solid #00cfff !important;
    border-radius: 10px !important;
    color: #00cfff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    padding: 0.7rem !important;
    width: 100% !important;
    cursor: pointer !important;
    margin-top: 0.8rem !important;
    transition: all 0.25s !important;
    text-transform: uppercase;
}
#export-btn:hover {
    background: rgba(0,207,255,0.07) !important;
    box-shadow: 0 0 24px rgba(0,207,255,0.2) !important;
}

/* ── Main output image ───────────────────────────────────── */
#main-image {
    border-radius: 14px !important;
    border: 1px solid #13233a !important;
    overflow: hidden !important;
    background: #050811 !important;
    min-height: 380px !important;
}
#main-image img {
    border-radius: 13px !important;
    width: 100% !important;
    transition: filter 0.5s ease !important;
}
#main-image .wrap { min-height: 380px !important; }

/* ── Loader overlay — zero-height, overflows up into the image ── */
#loader-wrap {
    height: 0 !important;
    overflow: visible !important;
    position: relative !important;
    z-index: 20 !important;
    pointer-events: none !important;
}
.ef-loader-center {
    position: absolute;
    top: -205px;          /* centers in the ~380px image block above */
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
}
.ef-loader-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(200, 220, 255, 0.7);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

/* Bouncing dots loader — cyan theme */
.loader {
    width: 50px;
    height: 28px;
    --_g: no-repeat radial-gradient(farthest-side, #00cfff 94%, #0000);
    background:
        var(--_g) 50%  0,
        var(--_g) 100% 0;
    background-size: 12px 12px;
    position: relative;
    animation: l23-0 1.5s linear infinite;
}
.loader:before {
    content: "";
    position: absolute;
    height: 12px;
    aspect-ratio: 1;
    border-radius: 50%;
    background: #00cfff;
    left: 0;
    top: 0;
    animation:
        l23-1 1.5s linear infinite,
        l23-2 0.5s cubic-bezier(0,200,.8,200) infinite;
}
@keyframes l23-0 {
    0%,31%  { background-position: 50% 0,    100% 0 }
    33%     { background-position: 50% 100%, 100% 0 }
    43%,64% { background-position: 50% 0,    100% 0 }
    66%     { background-position: 50% 0,    100% 100% }
    79%     { background-position: 50% 0,    100% 0 }
    100%    { transform: translateX(calc(-100%/3)) }
}
@keyframes l23-1 { 100% { left: calc(100% + 7px) } }
@keyframes l23-2 { 100% { top: -0.1px } }

/* ── Status + score boxes ────────────────────────────────── */
#status-box, #score-box { margin-top: 0.5rem; }
#status-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #5a7a9a !important;
    background: #050811 !important;
    border: 1px solid #0e1d30 !important;
    border-radius: 8px !important;
    min-height: unset !important;
    padding: 0.5rem 0.75rem !important;
}
#score-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #00cfff !important;
    background: #050811 !important;
    border: 1px solid #0e1d30 !important;
    border-radius: 8px !important;
    min-height: unset !important;
    padding: 0.5rem 0.75rem !important;
}

/* ── Export path ─────────────────────────────────────────── */
#export-path textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #2e4a6a !important;
    background: #050811 !important;
    border: 1px solid #0e1d30 !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.75rem !important;
}

/* ── Edit section ─────────────────────────────────────────── */
#edit-section {
    border-top: 1px solid #1a3a2a !important;
    margin-top: 0.8rem !important;
    padding-top: 0.8rem !important;
}
.ef-edit-label { color: #00ff99 !important; }

#apply-btn {
    background: linear-gradient(135deg, #0a6a3a 0%, #00884a 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #00ff99 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.8rem !important;
    cursor: pointer !important;
    transition: box-shadow 0.25s !important;
}
#apply-btn:hover {
    box-shadow: 0 0 24px rgba(0,180,80,0.3) !important;
}
#reset-btn {
    background: transparent !important;
    border: 1px solid #2e4a6a !important;
    border-radius: 10px !important;
    color: #5a7a9a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.8rem !important;
    cursor: pointer !important;
    transition: all 0.25s !important;
    text-transform: uppercase;
}
#reset-btn:hover {
    border-color: #5a7a9a !important;
    color: #8aaaca !important;
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
    text-align: center;
    padding: 2rem 0 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #141e2e;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: #13233a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00cfff; }
"""

# Loader HTML — zero-height element; content overflows up onto the image
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
    Yields 8-element tuples:
      0  status_out    (str)
      1  text_in       (str)
      2  score_out     (str)
      3  image_out     (np.ndarray | None)
      4  stored_image  (PIL.Image | None)
      5  export_btn    (gr.update)
      6  loader_html   (gr.update)
      7  edit_section  (gr.update)  — visible=True only on done yield
    """
    def _y(status, text, score, img, pil, export, loader, edit_sec=gr.update()):
        return (status, text, score, img, pil, export, gr.update(value=loader), edit_sec)

    audio_path = file_audio if file_audio is not None else mic_audio
    prompt = text_input.strip() if text_input else ""

    # ── Guard: nothing provided ────────────────────────────────────────────
    if not prompt and audio_path is None:
        yield _y("Please provide a description or record audio.",
                 "", "", None, None, gr.update(visible=False), _LOADER_OFF)
        return

    # ── Stage 1: ASR ──────────────────────────────────────────────────────
    if not prompt and audio_path is not None:
        yield _y("Transcribing audio...", "", "", None, None,
                 gr.update(visible=False), _LOADER_ON)
        audio_array = process_audio(audio_path)
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

    # ── Stage 2: StyleCLIP optimization ───────────────────────────────────
    score_q = queue.Queue()
    result_holder = {}
    t_start = time.time()

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
            elapsed = time.time() - t_start
            pil_image, final_sim = result_holder["success"]
            final_arr = np.array(pil_image) if pil_image is not None else last_img_arr
            yield _y(f"Generation complete.  {elapsed:.0f}s",
                     "", f"CLIP Score: {final_sim:.4f}",
                     final_arr, pil_image, gr.update(visible=True), _LOADER_OFF,
                     gr.update(visible=True))
            return

        elif msg == "progress":
            elapsed = time.time() - t_start
            if pil_img is not None:
                last_img_arr = np.array(pil_img)
            yield _y(f"Optimizing...  step {step} / 60  ({elapsed:.0f}s)",
                     prompt, f"Loss: {loss:.4f}",
                     last_img_arr, pil_img, gr.update(visible=False), _LOADER_ON)


# ── Edit function ─────────────────────────────────────────────────────────────

def edit_fn(mic_audio, edit_text, wrapper, cfg):
    """
    Yields 8-element tuples (same schema as generate_fn):
      0  status_out    (str)
      1  edit_text_in  (str)
      2  score_out     (str)
      3  image_out     (np.ndarray | None)
      4  stored_image  (PIL.Image | None)
      5  apply_btn     (gr.update)  — hidden during progress, shown on done
      6  loader_html   (gr.update)
      7  edit_section  (gr.update)  — always gr.update() (already visible)
    """
    def _y(status, text, score, img, pil, btn, loader):
        return (status, text, score, img, pil, btn, gr.update(value=loader), gr.update())

    edit_instruction = edit_text.strip() if edit_text else ""
    audio_path = mic_audio

    # ── Guard: nothing provided ────────────────────────────────────────────
    if not edit_instruction and audio_path is None:
        yield _y("Please provide an edit instruction or record audio.",
                 "", "", None, None, gr.update(visible=True), _LOADER_OFF)
        return

    # ── Stage 1: ASR ──────────────────────────────────────────────────────
    if not edit_instruction and audio_path is not None:
        yield _y("Transcribing...", "", "", None, None,
                 gr.update(visible=False), _LOADER_ON)
        audio_array = process_audio(audio_path)
        edit_instruction = Transcriber(cfg).transcribe(audio_array)
        yield _y("Transcribing... Review and click Apply Edit again.",
                 edit_instruction, "", None, None, gr.update(visible=True), _LOADER_OFF)
        return

    # ── Guard: instruction too short ───────────────────────────────────────
    if len(edit_instruction.split()) < 4:
        yield _y("⚠ Edit too short — be specific (e.g. 'change eyes to blue').",
                 edit_instruction, "", None, None, gr.update(visible=True), _LOADER_OFF)
        return

    # ── Stage 2: StyleCLIP edit ────────────────────────────────────────────
    score_q = queue.Queue()
    result_holder = {}
    t_start = time.time()

    def _cb(step, loss, pil_img):
        score_q.put(("progress", step, loss, pil_img))

    def _run():
        try:
            pil_image, final_sim = wrapper.edit(edit_instruction, callback=_cb)
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
            yield _y(f"Edit failed: {result_holder['error']}",
                     edit_instruction, "", None, None,
                     gr.update(visible=True), _LOADER_OFF)
            return

        elif msg == "done":
            elapsed = time.time() - t_start
            pil_image, final_sim = result_holder["success"]
            final_arr = np.array(pil_image) if pil_image is not None else last_img_arr
            yield _y(f"Edit applied.  {elapsed:.0f}s",
                     edit_instruction, f"CLIP Score: {final_sim:.4f}",
                     final_arr, pil_image, gr.update(visible=True), _LOADER_OFF)
            return

        elif msg == "progress":
            elapsed = time.time() - t_start
            if pil_img is not None:
                last_img_arr = np.array(pil_img)
            yield _y(f"Optimizing edit...  step {step} / 80  ({elapsed:.0f}s)",
                     edit_instruction, f"Loss: {loss:.4f}",
                     last_img_arr, pil_img, gr.update(visible=False), _LOADER_ON)


# ── Reset function ─────────────────────────────────────────────────────────────

def reset_fn(wrapper):
    """
    Restore the face to the original generated state.

    Returns 5-element tuple:
      0  image_out    (np.ndarray)
      1  status_out   (str)
      2  score_out    (str)
      3  stored_image (PIL.Image)
      4  export_btn   (gr.update)
    """
    try:
        pil_image, sim = wrapper.reset()
        return (
            np.array(pil_image),
            "Reset to original.",
            f"CLIP Score: {sim:.4f}",
            pil_image,
            gr.update(visible=True),
        )
    except RuntimeError as e:
        return (None, str(e), "", None, gr.update(visible=False))


# ── Export ────────────────────────────────────────────────────────────────────

def export_image(pil_image, outputs_dir=None):
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    if pil_image is None:
        return (gr.update(visible=False), gr.update(value="No image to export."))
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outputs_dir, f"{timestamp}.png")
    pil_image.save(path)
    return (gr.update(visible=True), gr.update(visible=True, value=path))


# ── UI Build ──────────────────────────────────────────────────────────────────

def _build_demo(wrapper, cfg):
    import functools
    gen  = functools.partial(generate_fn, wrapper=wrapper, cfg=cfg)
    edit = functools.partial(edit_fn,     wrapper=wrapper, cfg=cfg)
    exp  = functools.partial(export_image)


    with gr.Blocks(title="EchoFace", css=CUSTOM_CSS) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="ef-header">
            <div class="ef-title">EchoFace</div>
            <div class="ef-tagline">Speak a face into existence</div>
            <div class="ef-rule"></div>
        </div>
        """)

        notice = _startup_notice(cfg)
        if notice:
            gr.HTML(f'<div style="text-align:center;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#ff5c2a;margin-bottom:1rem;letter-spacing:0.1em;">{notice}</div>')

        stored_image = gr.State(value=None)

        with gr.Row(equal_height=False):

            # ── Left panel — inputs ─────────────────────────────────────
            with gr.Column(scale=4, elem_classes=["ef-panel"]):
                gr.HTML('<div class="ef-section-label">Input</div>')

                mic_in  = gr.Audio(source="microphone", type="filepath", label="Microphone")
                file_in = gr.Audio(source="upload",     type="filepath", label="Upload Audio")

                gr.HTML('<div class="ef-divider"></div>')

                text_in = gr.Textbox(
                    lines=4,
                    label="Face Description",
                    placeholder="e.g.  a young woman with dark curly hair and blue eyes",
                )
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")

                # ── Edit Details — hidden until first generation ─────────
                with gr.Column(visible=False, elem_id="edit-section") as edit_section_col:
                    gr.HTML('<div class="ef-section-label ef-edit-label">✦ Edit Details</div>')
                    edit_mic_in  = gr.Audio(source="microphone", type="filepath", label="Microphone")
                    edit_text_in = gr.Textbox(
                        lines=2,
                        label="Edit Instruction — one attribute at a time works best",
                        placeholder="e.g.  wearing glasses  |  blonde hair  |  with a beard  |  older face",
                    )
                    with gr.Row():
                        apply_btn = gr.Button("Apply Edit", elem_id="apply-btn")
                        reset_btn = gr.Button("Reset",      elem_id="reset-btn")

            # ── Right panel — outputs ───────────────────────────────────
            with gr.Column(scale=6, elem_classes=["ef-panel"]):
                gr.HTML('<div class="ef-section-label">Output</div>')

                image_out = gr.Image(
                    type="numpy",
                    show_label=False,
                    elem_id="main-image",
                )

                # Loader: zero-height, overflows up onto the image
                loader_html = gr.HTML(value=_LOADER_OFF, elem_id="loader-wrap")

                status_out = gr.Textbox(label="Status",     interactive=False, elem_id="status-box")
                score_out  = gr.Textbox(label="CLIP Score", interactive=False, elem_id="score-box")

                gr.HTML('<div class="ef-divider"></div>')

                export_btn = gr.Button("Export PNG", visible=False, elem_id="export-btn")
                export_path = gr.Textbox(
                    label="Saved to", interactive=False, visible=False, elem_id="export-path"
                )

        gr.HTML('<div class="ef-footer">EchoFace &nbsp;·&nbsp; FFHQ StyleGAN2-ADA + CLIP ViT-B/32 + Whisper &nbsp;·&nbsp; Fully Local</div>')

        # ── Wiring ───────────────────────────────────────────────────────
        gen_btn.click(
            fn=gen,
            inputs=[mic_in, file_in, text_in],
            outputs=[status_out, text_in, score_out, image_out, stored_image,
                     export_btn, loader_html, edit_section_col],
        )
        apply_btn.click(
            fn=edit,
            inputs=[edit_mic_in, edit_text_in],
            outputs=[status_out, edit_text_in, score_out, image_out, stored_image,
                     apply_btn, loader_html, edit_section_col],
        )
        reset_btn.click(
            fn=lambda: reset_fn(wrapper),
            inputs=[],
            outputs=[image_out, status_out, score_out, stored_image, export_btn],
        )
        export_btn.click(
            fn=exp,
            inputs=[stored_image],
            outputs=[export_btn, export_path],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = gpu_utils.get_device_config()
    print("[EchoFace] Loading FFHQ StyleGAN2-ADA + CLIP ...")
    PKL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ffhq", "ffhq.pkl")
    wrapper  = StyleCLIPWrapper(PKL_PATH, cfg)
    demo     = _build_demo(wrapper, cfg)
    demo.queue(concurrency_count=1).launch(
        server_name="127.0.0.1", share=False, show_error=True
    )
