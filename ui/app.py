import sys
import os
import warnings
# Suppress the MSVC/CUDA kernel build spam from StyleGAN-Human
warnings.filterwarnings("ignore", message="Failed to build CUDA kernels")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import queue
import threading
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


def _startup_notice(cfg):
    """Return a notice string if running in CPU mode, else empty string."""
    return (
        "No GPU detected — generation will be slow (CPU mode, 256x256)"
        if not cfg["use_fp16"]
        else ""
    )


def generate_fn(mic_audio, file_audio, text_input, wrapper, cfg):
    """Generator function wiring ASR + StyleCLIP into a single Gradio event handler.

    Yield schema (6-element tuples):
      index 0: status_out  (str)
      index 1: text_in     (str — ASR path fills this; other paths yield "")
      index 2: score_out   (str)
      index 3: image_out   (np.ndarray | None)
      index 4: stored_image (PIL.Image | None)
      index 5: export_btn  (gr.update)
    """
    # STAGE 0 — Resolve audio path and prompt
    audio_path = file_audio if file_audio is not None else mic_audio
    prompt = text_input.strip() if text_input else ""

    # STAGE 1 — Empty input guard
    if not prompt and audio_path is None:
        yield (
            "Please provide a description or record audio.",
            "",
            "",
            None,
            None,
            gr.update(visible=False),
        )
        return

    # STAGE 2 — ASR path (two-stage flow)
    if not prompt and audio_path is not None:
        audio_array = process_audio(audio_path)
        transcription = Transcriber(cfg).transcribe(audio_array)
        status_msg = "Transcription complete — review and click Generate again."
        if len(transcription.split()) < 3:
            status_msg += " [WARNING: very short transcription — consider editing]"
        yield (
            status_msg,
            transcription,
            "",
            None,
            None,
            gr.update(visible=False),
        )
        return  # STOP — do not call optimize(); user must click Generate again

    # STAGE 3 — Text bypass path (prompt is non-empty)

    # ── Legacy path: StyleCLIP 50-step latent optimisation ────────────────────
    
    score_q = queue.Queue()
    result_holder = {}

    def _cb(step, loss, img):
        score_q.put(("progress", step, loss, img))

    def _run():
        try:
            pil_image, final_sim = wrapper.generate(prompt, callback=_cb)
            result_holder["success"] = (pil_image, final_sim)
            score_q.put(("done", None, None, None))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_holder["error"] = exc
            score_q.put(("error", None, None, None))
            
    threading.Thread(target=_run, daemon=True).start()
    
    last_img = None
    while True:
        msg, step, loss, img = score_q.get()
        if msg == "error":
            yield (
                f"Generation failed: {result_holder['error']}",
                "",
                "",
                None,
                None,
                gr.update(visible=False),
            )
            return
        elif msg == "done":
            pil_image, final_sim = result_holder["success"]
            # Always convert to numpy - Gradio `gr.Image(type='numpy')` needs ndarray
            img_arr = np.array(pil_image) if pil_image is not None else last_img
            yield (
                "✅ Generation complete!",
                "",
                f"CLIP Score: {final_sim:.4f}",
                img_arr,
                pil_image,
                gr.update(visible=True),
            )
            return
        elif msg == "progress":
            last_img = img if img is not None else last_img
            arr = np.array(last_img) if last_img is not None else None
            yield (
                f"🔄 Optimizing... Step {step}/100",
                "",
                f"Current Loss: {loss:.4f}",
                arr,
                last_img,
                gr.update(visible=False),
            )


def export_image(pil_image, outputs_dir=None):
    """Save pil_image to outputs_dir as a timestamped PNG.

    Returns:
        (gr.update for export_btn, gr.update for export_path textbox)
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")

    if pil_image is None:
        return (gr.update(visible=False), gr.update(value="No image to export"))

    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outputs_dir, f"{timestamp}.png")
    pil_image.save(path)
    return (gr.update(visible=True), gr.update(visible=True, value=path))


def _build_demo(wrapper, cfg):
    """Build and return the Gradio Blocks demo object.

    Separated from __main__ so tests can call it with mock wrapper/cfg
    without triggering pkl load.

    Args:
        wrapper:     StyleCLIPWrapper
        cfg:         device config dict
    """
    import functools

    startup_notice = _startup_notice(cfg)

    gen = functools.partial(generate_fn, wrapper=wrapper, cfg=cfg)
    exp = functools.partial(export_image)

    with gr.Blocks(title="EchoFace") as demo:
        gr.Markdown("# EchoFace")
        if startup_notice:
            gr.Markdown(f"> **Notice:** {startup_notice}")

        stored_image = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                mic_in = gr.Audio(source="microphone", type="filepath", label="Record audio")
                file_in = gr.Audio(source="upload", type="filepath", label="Upload audio file")
                text_in = gr.Textbox(
                    lines=3,
                    label="Face description — type here or record audio above",
                    placeholder="e.g. a young woman with dark curly hair wearing a red dress",
                )
                gen_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                status_out = gr.Textbox(label="Status", interactive=False)
                score_out = gr.Textbox(label="CLIP Similarity Score", interactive=False)
                image_out = gr.Image(type="numpy", label="Generated Face")
                export_btn = gr.Button("Export PNG", visible=False)
                export_path = gr.Textbox(
                    label="Saved path", interactive=False, visible=False
                )

        # text_in is BOTH an input AND an output — ASR path writes transcription back
        gen_btn.click(
            fn=gen,
            inputs=[mic_in, file_in, text_in],
            outputs=[status_out, text_in, score_out, image_out, stored_image, export_btn],
        )
        export_btn.click(
            fn=exp,
            inputs=[stored_image],
            outputs=[export_btn, export_path],
        )

    return demo


if __name__ == "__main__":
    cfg = gpu_utils.get_device_config()

    print("[app] Initializing StyleCLIP Optimizer (FFHQ StyleGAN2)...")
    PKL_PATH = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "ffhq",
        "ffhq.pkl",
    )
    wrapper = StyleCLIPWrapper(PKL_PATH, cfg)

    demo = _build_demo(wrapper, cfg)
    demo.queue(concurrency_count=1).launch(
        server_name="127.0.0.1", share=False, show_error=True
    )
