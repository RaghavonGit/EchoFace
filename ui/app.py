import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import queue
import threading
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image

import gpu_utils
from generator.stylegan_wrapper import StyleGANWrapper
from optimizer.clip_optimizer import optimize
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


def generate_fn(mic_audio, file_audio, text_input, wrapper, cfg, gan_wrapper=None):
    """Generator function wiring ASR + optimizer (or GAN) into a single Gradio event handler.

    When gan_wrapper is provided, STAGE 3 uses the CLIP-conditioned FaceGAN
    (instant single-pass inference) instead of the 150-step CLIP optimizer.
    When gan_wrapper is None, falls back to the legacy optimize() path.

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

    # ── Fast path: CLIP-conditioned FaceGAN (trained model) ───────────────────
    if gan_wrapper is not None:
        yield (
            "Generating face...",
            "",
            "",
            None,
            None,
            gr.update(visible=False),
        )
        try:
            pil_image, final_sim = gan_wrapper.generate(prompt)
        except Exception as exc:
            yield (
                f"Generation failed: {exc}",
                "",
                "",
                None,
                None,
                gr.update(visible=False),
            )
            return
        yield (
            "Done.",
            "",
            f"CLIP similarity: {final_sim:.4f}",
            np.array(pil_image),
            pil_image,
            gr.update(visible=True),
        )
        return

    # ── Legacy path: 150-step CLIP latent optimisation ────────────────────────
    yield (
        "Starting optimization...",
        "",
        "",
        None,
        None,
        gr.update(visible=False),
    )

    score_q = queue.Queue()
    result_holder = {}

    def _cb(step, loss, sim_score):
        score_q.put((step, sim_score))

    def _run():
        try:
            result_holder["ok"] = optimize(prompt, wrapper, cfg, callback=_cb)
        except Exception as exc:
            result_holder["err"] = exc
        finally:
            score_q.put(None)  # sentinel

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    while True:
        item = score_q.get()
        if item is None:
            break
        step, sim = item
        yield (
            f"Optimizing... step {step}/150",
            "",
            f"CLIP similarity: {sim:.4f}",
            None,
            None,
            gr.update(visible=False),
        )

    t.join()

    if "err" in result_holder:
        err = result_holder["err"]
        yield (
            f"Generation failed: {err}",
            "",
            "",
            None,
            None,
            gr.update(visible=False),
        )
        return

    pil_image, w_tensor, final_sim = result_holder["ok"]
    yield (
        "Done.",
        "",
        f"CLIP similarity: {final_sim:.4f}",
        np.array(pil_image),
        pil_image,
        gr.update(visible=True),
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


def _build_demo(wrapper, cfg, gan_wrapper=None):
    """Build and return the Gradio Blocks demo object.

    Separated from __main__ so tests can call it with mock wrapper/cfg
    without triggering pkl load.

    Args:
        wrapper:     StyleGANWrapper (used when gan_wrapper is None)
        cfg:         device config dict
        gan_wrapper: FaceGANWrapper (when available, used instead of optimize())
    """
    import functools

    startup_notice = _startup_notice(cfg)

    gen = functools.partial(generate_fn, wrapper=wrapper, cfg=cfg, gan_wrapper=gan_wrapper)
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
                    placeholder="e.g. a young woman with dark curly hair and brown eyes",
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

    # ── Prefer FaceGAN if a trained checkpoint exists ─────────────────────────
    GAN_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "face_gan", "gen_best.pth")
    gan_wrapper = None
    wrapper = None

    if os.path.exists(GAN_CHECKPOINT):
        from generator.face_gan_wrapper import FaceGANWrapper
        print(f"[app] Loading FaceGANWrapper from {GAN_CHECKPOINT}")
        gan_wrapper = FaceGANWrapper(GAN_CHECKPOINT, cfg)
        print("[app] FaceGAN ready — using fast CLIP-conditioned generation")
    else:
        print("[app] No FaceGAN checkpoint found — using CLIP latent optimiser (StyleGAN-Human)")
        PKL_PATH = os.path.join(
            PROJECT_ROOT,
            "StyleGAN-Human",
            "pretrained_models",
            "stylegan_human_v2_1024.pkl",
        )
        wrapper = StyleGANWrapper(PKL_PATH, cfg)

    demo = _build_demo(wrapper, cfg, gan_wrapper=gan_wrapper)
    demo.queue(concurrency_count=1).launch(
        server_name="127.0.0.1", share=False, show_error=True
    )
