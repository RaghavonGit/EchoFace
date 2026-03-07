# Phase 6: Gradio UI Integration - Research

**Researched:** 2026-03-07
**Domain:** Gradio 3.50.2 Blocks API, Python UI event handling, pipeline integration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Input trigger**
- Single "Generate" button triggers the pipeline — no auto-start on upload or recording stop
- User sets up all inputs, then explicitly clicks Generate

**Input mode precedence**
- Three input sources: mic recording (`gr.Audio(source="microphone")`), WAV/MP3 file upload, manual text box
- If text box is non-empty: fully bypass ASR — no Whisper load, no audio preprocessing; use text directly
- If text box is empty and audio is provided: run ASR (process_audio → Transcriber.transcribe), display transcription result in the text box so user can review and edit before the optimizer runs
- Mic vs file upload conflict: not expected simultaneously in Gradio 3.x (single Audio component handles both sources); if both somehow present, file upload takes precedence
- Mic component: `gr.Audio(source="microphone")` — returns audio file path compatible with `process_audio()`

**VRAM sequencing**
- Unchanged from Phase 4 decision: Whisper unloads (del model → gc.collect() → torch.cuda.empty_cache()) before optimizer runs
- Text-only path skips Whisper entirely — no VRAM impact

**Progress display**
- Callback fires every 10 steps with `(step, loss, sim_score)` — Phase 3 locked this signature
- Display live CLIP similarity score updating every 10 steps during optimization
- No intermediate image previews during optimization (avoids synthesis overhead; Phase 3 decision)
- After optimization completes: display final image + final cosine similarity score (CLIP-03)

**Attribute direction sliders**
- Out of scope for Phase 6 — deferred to v2
- ATTR-01 (latent direction discovery via SVM/PCA) was explicitly deferred in Phase 5
- No stub slider UI — keep the interface clean and functional

**Export behavior**
- Export button saves the generated image as a timestamped PNG to `outputs/`
- Filename format: `outputs/YYYYMMDD_HHMMSS.png` (Claude's discretion on exact format)
- Button appears after generation completes; not visible/enabled before that

**Error handling**
- Empty input (no audio, no text): show inline error message — "Please provide a description or record audio." — keep button enabled for immediate retry
- Short ASR result (< 3 words): display the transcription in the text box with a visible warning label; user edits before generating — do not block the pipeline
- Optimization failure (exception mid-run): catch all exceptions, display a readable error message in the UI (e.g., "Generation failed: out of memory"), keep Gradio app alive for retry
- CPU-only (no CUDA): on app launch, detect CUDA availability and display a startup notice — "No GPU detected — generation will be slow (CPU mode, 256x256)"

### Claude's Discretion
- Exact Gradio layout (tab structure, column arrangement)
- Progress bar vs step counter display during optimization
- Exact warning label styling for short ASR result
- Timestamp format for export filename
- Logger name and log level for UI events

### Deferred Ideas (OUT OF SCOPE)
- Attribute direction sliders (Age, Smile, Facial Hair, Glasses, Pose) — requires ATTR-01 (latent direction discovery); v2 scope
- FID scoring button in UI — Phase 5 noted Gradio could call compute_fid(); deferred, not in Phase 6 success criteria
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UI-01 | gradio_app.py runs on localhost with mic record button, text description override box, and .wav/.mp3 file upload — all wired to the ASR + generation pipeline | Gradio 3.50.2 Blocks confirmed installed; `gr.Audio(source="microphone", type="filepath")` returns tmp filepath; generator-with-yield pattern enables live CLIP score updates |
| UI-02 | User can export the final generated image as a PNG file via an export button in the Gradio UI | `PIL.Image.save()` to `outputs/YYYYMMDD_HHMMSS.png`; export button shown post-generation via `gr.update(visible=True)`; `gr.State` holds the PIL image between events |
| CLIP-03 | User can see a live cosine similarity score between their text description and the current generated image, updated after optimization completes | Optimizer callback fires every 10 steps; generator function yields updated score string to `gr.Textbox`; final score displayed after `optimize()` returns |
</phase_requirements>

---

## Summary

Phase 6 wires all five existing EchoFace modules into a single-file Gradio 3.50.2 Blocks application. The installed version (confirmed `3.50.2`) uses the Gradio 3.x string-based `source=` API, not the Gradio 4.x `sources=[]` list API — this distinction matters for `gr.Audio`.

The core architectural challenge is streaming CLIP similarity score updates to the UI during a 150-step optimization loop that runs in a regular Python thread. The correct approach is a **generator function** that calls `optimize()` with a callback which appends to a shared list, then yields updated score text to a `gr.Textbox` output component. The `.queue(concurrency_count=1)` call before `.launch()` is required for generator streaming to work. The Generate button is disabled during generation and re-enabled via `.then()` chaining.

The `gr.State` component holds the final PIL image and w_tensor between the Generate callback and the Export button callback, since Gradio does not share function-local variables between events.

**Primary recommendation:** Single-file `ui/app.py` using `gr.Blocks` with a generator function for live score updates, `gr.State` for image persistence, `gr.update(visible=...)` for the export button reveal, and `.queue(concurrency_count=1).launch(server_name="127.0.0.1", share=False)`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | 3.50.2 (pinned >=3.40,<4.0) | UI framework — Blocks layout, event listeners, Audio component | Already installed; Python 3.8 compatible; 4.x requires Python 3.10+ |
| PIL (Pillow) | installed | Save generated image as PNG | Already used by `optimize()` return value; `Image.save()` is the export path |
| datetime | stdlib | Generate timestamped export filename | No extra install |
| logging | stdlib | UI-level event logging | Consistent with existing module pattern |
| os | stdlib | Path joining for `outputs/` directory | Confirmed scaffolded in Phase 1 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | <1.24 (pinned 1.23.5) | Convert PIL Image to numpy array for `gr.Image` display | `gr.Image` expects numpy HWC array; `numpy.array(pil_image)` |
| sys | stdlib | sys.path injection for project-root imports | Same pattern as `utils/metrics.py` and `generator/stylegan_wrapper.py` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Generator function + yield | Threading + queue polling | Generator is native Gradio; threading requires custom polling loop outside standard patterns |
| `gr.State` for image persistence | Global variable | `gr.State` is session-scoped; global variable breaks if two users ever use the app simultaneously |
| `gr.update(visible=True)` for export button | Always-visible button | Always-visible button enables export before generation; UX decision is locked: button appears after |

**Installation:** Nothing new to install. All dependencies are in the `stylehuman` conda env already.

---

## Architecture Patterns

### Recommended Project Structure

```
ui/
├── .gitkeep          # (remove after app.py is created)
└── app.py            # Single file — the entire Phase 6 deliverable
outputs/              # Already scaffolded in Phase 1
```

The single-file constraint comes from the scope: one Blocks app, no helper modules needed.

### Pattern 1: Startup Initialization (module-level)

**What:** Load heavyweight objects once at app start, not per-request.
**When to use:** Always for StyleGANWrapper — pkl load takes ~5 seconds.

```python
# Source: Established EchoFace pattern (gpu_utils.py, stylegan_wrapper.py)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_utils
from generator.stylegan_wrapper import StyleGANWrapper

cfg = gpu_utils.get_device_config()
wrapper = StyleGANWrapper("StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl", cfg)

STARTUP_NOTICE = (
    "No GPU detected — generation will be slow (CPU mode, 256x256)"
    if not cfg["use_fp16"] else ""
)
```

**Why module-level:** Gradio 3.x runs the module once then serves requests; module-level init fires exactly once.

### Pattern 2: Generator Function for Live Score Updates (CRITICAL)

**What:** A `def generate(...)` that `yield`s intermediate outputs so Gradio can stream them to the UI.
**When to use:** Required for CLIP-03 — live sim score during 150 optimization steps.

```python
# Source: Gradio 3.x streaming guide + verified via gr.Blocks help()
def generate(audio_input, text_input):
    """Generator: yields (status_text, score_text, image, export_btn_update) tuples."""
    # --- Input validation ---
    prompt = text_input.strip() if text_input else ""
    if not prompt and audio_input is None:
        yield "Please provide a description or record audio.", "", None, gr.update(visible=False)
        return

    # --- ASR path (only if text box is empty) ---
    if not prompt:
        yield "Transcribing audio...", "", None, gr.update(visible=False)
        from asr.audio_processor import process_audio
        from asr.transcriber import Transcriber
        audio_array = process_audio(audio_input)   # audio_input is a filepath string
        transcriber = Transcriber(cfg)
        prompt = transcriber.transcribe(audio_array)
        # Display transcription — user sees it but generate continues
        yield f"Transcription: {prompt}", "", None, gr.update(visible=False)

    # --- Optimization ---
    score_log = []

    def _callback(step, loss, sim_score):
        score_log.append(sim_score)

    yield "Optimizing...", "", None, gr.update(visible=False)

    # Run optimize() — this blocks the thread (150 steps)
    # Because .queue(concurrency_count=1) is set, Gradio handles this correctly
    try:
        pil_image, w_tensor, final_sim = optimize(prompt, wrapper, cfg, callback=_callback)
    except Exception as exc:
        yield f"Generation failed: {exc}", "", None, gr.update(visible=False)
        return

    # Final result
    import numpy as np
    img_array = np.array(pil_image)
    yield "Done.", f"CLIP similarity: {final_sim:.4f}", img_array, gr.update(visible=True)
```

**Key constraint:** `optimize()` is synchronous. The callback cannot yield — it only records values. The live score updates happen via intermediate `yield` statements in the generator. To get per-10-step updates, an alternative approach uses a thread:

```python
# Alternative: threading for true per-step yields (if live score needed mid-run)
# Use a queue.Queue as the callback's output channel
import queue, threading

def generate_with_live_score(audio_input, text_input):
    score_queue = queue.Queue()

    def _callback(step, loss, sim_score):
        score_queue.put((step, sim_score))

    # Run optimize in a thread
    result_holder = {}
    def _run():
        try:
            result_holder['result'] = optimize(prompt, wrapper, cfg, callback=_callback)
        except Exception as e:
            result_holder['error'] = e
        finally:
            score_queue.put(None)  # sentinel

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    while True:
        item = score_queue.get()
        if item is None:
            break
        step, sim = item
        yield f"Step {step}/150", f"CLIP similarity: {sim:.4f}", None, gr.update(visible=False)

    if 'error' in result_holder:
        yield f"Generation failed: {result_holder['error']}", "", None, gr.update(visible=False)
        return

    pil_image, w_tensor, final_sim = result_holder['result']
    import numpy as np
    yield "Done.", f"CLIP similarity: {final_sim:.4f}", np.array(pil_image), gr.update(visible=True)
```

**Decision for Phase 6:** The threading approach enables true per-step score streaming and should be used since it satisfies CLIP-03 most directly. The simpler non-threading approach only shows the final score.

### Pattern 3: gr.State for Cross-Event Image Persistence

**What:** Store the PIL image between the generate callback and the export button callback.
**When to use:** Any time data needs to persist between two separate button-click events.

```python
# Source: Gradio 3.x state guide
stored_image = gr.State(value=None)  # stores PIL.Image after generation

# In the export callback:
def export_image(pil_image):
    if pil_image is None:
        return
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("outputs", f"{timestamp}.png")
    pil_image.save(path)
    return path
```

**Note:** `gr.State` holds the PIL image (not the numpy array). Convert to numpy only for `gr.Image` display. Store PIL in State for the export path.

### Pattern 4: Component Visibility Control

**What:** Hide the export button initially; reveal it after generation via `gr.update(visible=True)`.
**When to use:** Export button must only appear after a successful generation (locked decision).

```python
# In Blocks definition:
export_btn = gr.Button("Export PNG", visible=False)

# In generate() generator — last yield:
yield status, score_text, img_array, gr.update(visible=True)  # reveals export_btn

# In Blocks event wiring:
gen_btn.click(
    fn=generate,
    inputs=[audio_in, text_in],
    outputs=[status_text, score_text, image_out, export_btn],
)
```

### Pattern 5: .queue() + .launch() for Generator Support

**What:** `.queue()` must be called before `.launch()` for generator functions to stream.
**When to use:** Always when the prediction function uses `yield`.

```python
# Source: Confirmed via conda run -n stylehuman python -c "help(gr.Blocks.queue)"
demo.queue(concurrency_count=1).launch(
    server_name="127.0.0.1",
    share=False,
    show_error=True,
)
```

`concurrency_count=1` is correct for EchoFace — only one optimization can run at a time (VRAM constraint).

### Pattern 6: gr.Audio in Gradio 3.50.2

**What:** The `source=` parameter in 3.x is a string, not a list (that is the 4.x API).
**When to use:** All Audio component declarations in the app.

```python
# Confirmed from live introspection: conda run -n stylehuman python -c "help(gr.Audio.__init__)"
# source: Literal[('upload', 'microphone')] | None = None

# For mic-only input:
mic_in = gr.Audio(source="microphone", type="filepath", label="Record description")

# For file upload:
file_in = gr.Audio(source="upload", type="filepath", label="Upload audio file (.wav / .mp3)")
```

`type="filepath"` returns a string path to a temp file — directly compatible with `process_audio(path)`.

### Anti-Patterns to Avoid

- **Two gr.Audio components for mic vs file:** Gradio 3.x Audio with `source="microphone"` already handles both mic and file upload via its UI toggle in some versions. Having two separate components adds complexity. If a single component is insufficient, use two separate `gr.Audio` components with explicit source — but do not try to merge their outputs without checking which is non-None.
- **Calling `optimize()` in a non-queued Blocks app:** Without `.queue()`, generator functions do not stream — Gradio will wait for all yields and return only the final one.
- **Storing mutable state in global variables:** Use `gr.State` instead. Global variables are shared across all concurrent users (even if concurrency=1 today, this is fragile).
- **numpy array in gr.State:** Store PIL.Image in State; convert to numpy only for `gr.Image` output. `gr.State(numpy_array)` works but is wasteful since `gr.Image` conversion is O(1).
- **Catching only specific exceptions in generate():** Catch `Exception` broadly and display a readable message. The optimizer, ASR, and synthesis all raise different exception types.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Live streaming UI updates | Custom WebSocket server, polling endpoint | `gr.Blocks` generator function with `yield` + `.queue()` | Gradio 3.x handles threading, SSE, client sync internally |
| Audio format conversion | Custom librosa conversion in UI | `gr.Audio(type="filepath")` + existing `process_audio()` | Gradio converts microphone output to wav already; process_audio() handles the rest |
| Session state | Global dict keyed by session ID | `gr.State` | Gradio manages per-session state automatically |
| Component show/hide logic | CSS injection, JavaScript | `gr.update(visible=...)` | Native Gradio component update mechanism |
| Export file naming | UUID or random string | `datetime.now().strftime("%Y%m%d_%H%M%S")` | Timestamp is human-readable, sortable, and collision-safe for single-user app |

**Key insight:** Gradio 3.x already solves the hardest UI problems (streaming, state, audio capture). The app.py should be almost entirely wiring — no custom UI logic.

---

## Common Pitfalls

### Pitfall 1: Wrong gr.Audio source= API (Gradio 4.x vs 3.x)

**What goes wrong:** Developer reads current Gradio docs (which describe 4.x) and writes `gr.Audio(sources=["microphone", "upload"])` — this crashes on import in 3.50.2 because `sources` (plural, list) does not exist.
**Why it happens:** Gradio docs at gradio.app default to the latest version (4.x).
**How to avoid:** Always use `source=` (singular, string) in Gradio 3.x. Verified from live `help(gr.Audio.__init__)`: `source: Literal[('upload', 'microphone')] | None`.
**Warning signs:** `TypeError: __init__() got an unexpected keyword argument 'sources'` on app start.

### Pitfall 2: Generator Functions Silently Not Streaming (Missing .queue())

**What goes wrong:** Generator function with `yield` is written correctly but Gradio displays only the final yielded value — no live updates.
**Why it happens:** Without `.queue()`, Gradio 3.x treats the generator as a regular function and buffers all yields.
**How to avoid:** Always call `demo.queue(concurrency_count=1)` before `demo.launch()` when any function uses `yield`.
**Warning signs:** UI freezes during generation, then shows final result in one update.

### Pitfall 3: optimize() Callback Cannot yield (Thread Boundary)

**What goes wrong:** Developer tries to `yield` from inside the `_callback` passed to `optimize()` — `yield` inside a regular function called from another thread has no effect on the outer generator's stream.
**Why it happens:** `optimize()` calls the callback from its own for-loop, not from the Gradio generator's frame.
**How to avoid:** Use `queue.Queue` as a bridge: callback puts items into the queue; generator consumes items from the queue and yields them (see threading pattern above).
**Warning signs:** No live score updates during optimization even though callback is being called.

### Pitfall 4: PIL Image Closed Before Export

**What goes wrong:** `pil_image.save(path)` raises `ValueError: I/O operation on closed image` when the export button is clicked.
**Why it happens:** Gradio may garbage-collect or the generator may close the image if it is not stored persistently.
**How to avoid:** Store the original PIL.Image in `gr.State` immediately after `optimize()` returns. Do not close it explicitly.
**Warning signs:** Export works the first click but fails on retry.

### Pitfall 5: sys.path Not Set — Module Import Failure

**What goes wrong:** `from optimizer.clip_optimizer import optimize` raises `ModuleNotFoundError` when running `python ui/app.py` from the `ui/` directory or from the project root without path setup.
**Why it happens:** `ui/app.py` is one directory below the project root; Python does not automatically add the parent to `sys.path`.
**How to avoid:** Add sys.path injection at the very top of `ui/app.py`, before all other imports:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This is the same pattern used in `utils/metrics.py` and `generator/stylegan_wrapper.py`.
**Warning signs:** `ModuleNotFoundError: No module named 'optimizer'` or similar on app launch.

### Pitfall 6: Audio Input is None When Text Is Non-Empty

**What goes wrong:** The generate function receives `audio_input=None` when the user types text and clicks Generate without providing audio. Passing `None` to `process_audio()` raises `ValueError`.
**Why it happens:** `gr.Audio` returns `None` when no audio has been recorded or uploaded.
**How to avoid:** The locked text-bypass decision already handles this — check `text_input.strip()` first; only call `process_audio(audio_input)` when text is empty AND `audio_input is not None`.
**Warning signs:** `ValueError: Cannot load audio: ...` on text-only generation.

### Pitfall 7: gr.State Receives numpy Array Instead of PIL Image

**What goes wrong:** Export callback receives the numpy array that was displayed in `gr.Image`, not a PIL Image, making `.save()` unavailable.
**Why it happens:** Developer stores the output fed to `gr.Image` (numpy) rather than the PIL image from `optimize()`.
**How to avoid:** Store `pil_image` (the PIL.Image returned by `optimize()`) in `gr.State`. Convert to numpy separately only for the `gr.Image` display output.

---

## Code Examples

Verified patterns from confirmed sources (live Gradio 3.50.2 help output and project codebase):

### Full App Skeleton

```python
# ui/app.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import queue
import threading
from datetime import datetime

import gradio as gr
import numpy as np

import gpu_utils
from generator.stylegan_wrapper import StyleGANWrapper
from optimizer.clip_optimizer import optimize

logger = logging.getLogger("echoface.ui")
logging.basicConfig(level=logging.INFO)

# --- Startup init (runs once at module load) ---
cfg = gpu_utils.get_device_config()
PKL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "StyleGAN-Human", "pretrained_models", "stylegan_human_v2_1024.pkl"
)
wrapper = StyleGANWrapper(PKL_PATH, cfg)

STARTUP_NOTICE = (
    "No GPU detected — generation will be slow (CPU mode, 256x256)"
    if not cfg["use_fp16"] else ""
)

# --- Build Blocks UI ---
with gr.Blocks(title="EchoFace") as demo:
    if STARTUP_NOTICE:
        gr.Markdown(f"> **Notice:** {STARTUP_NOTICE}")

    stored_image = gr.State(value=None)  # holds PIL.Image between events

    with gr.Row():
        with gr.Column():
            mic_in   = gr.Audio(source="microphone", type="filepath", label="Record description")
            file_in  = gr.Audio(source="upload",     type="filepath", label="Upload audio (.wav/.mp3)")
            text_in  = gr.Textbox(label="Text description (bypasses ASR if non-empty)", lines=3)
            gen_btn  = gr.Button("Generate", variant="primary")

        with gr.Column():
            status_out = gr.Textbox(label="Status", interactive=False)
            score_out  = gr.Textbox(label="CLIP Similarity Score", interactive=False)
            image_out  = gr.Image(label="Generated Face", type="numpy")
            export_btn = gr.Button("Export PNG", visible=False)
            export_path = gr.Textbox(label="Saved to", interactive=False, visible=False)

    def generate(mic_audio, file_audio, text_input):
        # Input precedence: text > file_upload > mic
        prompt = text_input.strip() if text_input else ""
        audio_path = file_audio if file_audio is not None else mic_audio

        if not prompt and audio_path is None:
            yield (
                "Please provide a description or record audio.",
                "", None, None,
                gr.update(visible=False)
            )
            return

        if not prompt:
            # ASR path
            from asr.audio_processor import process_audio
            from asr.transcriber import Transcriber
            yield "Transcribing audio...", "", None, None, gr.update(visible=False)
            try:
                audio_array = process_audio(audio_path)
                transcriber = Transcriber(cfg)
                prompt = transcriber.transcribe(audio_array)
            except Exception as exc:
                yield f"Transcription failed: {exc}", "", None, None, gr.update(visible=False)
                return
            word_count = len(prompt.split())
            status_msg = f"Transcribed: {prompt}"
            if word_count < 3:
                status_msg += " [WARNING: very short transcription — consider editing]"
            yield status_msg, "", None, None, gr.update(visible=False)

        # Optimization with threading for live score updates
        score_q = queue.Queue()

        def _cb(step, loss, sim_score):
            score_q.put((step, sim_score))

        result_holder = {}

        def _run():
            try:
                result_holder['ok'] = optimize(prompt, wrapper, cfg, callback=_cb)
            except Exception as exc:
                result_holder['err'] = exc
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
                f"CLIP similarity: {sim:.4f}",
                None, None,
                gr.update(visible=False)
            )

        if 'err' in result_holder:
            yield (
                f"Generation failed: {result_holder['err']}",
                "", None, None,
                gr.update(visible=False)
            )
            return

        pil_image, _, final_sim = result_holder['ok']
        img_array = np.array(pil_image)
        yield (
            "Done.",
            f"CLIP similarity: {final_sim:.4f}",
            img_array,
            pil_image,          # stored in gr.State
            gr.update(visible=True)
        )

    gen_btn.click(
        fn=generate,
        inputs=[mic_in, file_in, text_in],
        outputs=[status_out, score_out, image_out, stored_image, export_btn],
    )

    def export_image(pil_image):
        if pil_image is None:
            return gr.update(visible=False), gr.update(value="No image to export")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs", f"{timestamp}.png"
        )
        pil_image.save(out_path)
        logger.info("Exported image to %s", out_path)
        return gr.update(visible=True), gr.update(visible=True, value=out_path)

    export_btn.click(
        fn=export_image,
        inputs=[stored_image],
        outputs=[export_btn, export_path],
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=1).launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
    )
```

### gr.Audio Confirmed Signature (Gradio 3.50.2)

```python
# Source: conda run -n stylehuman python -c "help(gr.Audio.__init__)"
# source: Literal[('upload', 'microphone')] | None = None  — STRING, not list
# type: Literal[('numpy', 'filepath')] = 'numpy'
# format: Literal[('wav', 'mp3')] = 'wav'

gr.Audio(source="microphone", type="filepath")  # CORRECT for Gradio 3.x
gr.Audio(sources=["microphone"])                # WRONG — Gradio 4.x only
```

### Confirmed .queue() + .launch() Signatures

```python
# Source: conda run -n stylehuman python -c "help(gr.Blocks.queue)"
demo.queue(concurrency_count=1)  # required for generator streaming

# Source: conda run -n stylehuman python -c "help(gr.Blocks.launch)"
demo.launch(
    server_name="127.0.0.1",  # default; localhost only
    share=False,               # no public tunnel
    show_error=True,           # display exceptions in UI
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `gr.Interface()` single function | `gr.Blocks()` for custom layout | Gradio 2.x → 3.x | Blocks allows multi-input, multi-step, and conditional UI |
| `enable_queue=True` in `.launch()` | `.queue()` method before `.launch()` | Gradio 3.x | `enable_queue` is deprecated; `.queue()` is the correct API |
| `gr.Audio(sources=[...])` (list) | `gr.Audio(source="...")` (string) | Gradio 3.x vs 4.x | 3.x uses singular string; 4.x uses list — this project is pinned to 3.x |
| Global mutable state | `gr.State` per session | Gradio 3.x | Session-scoped; safe for multi-user (even if concurrency=1 now) |

**Deprecated/outdated:**
- `enable_queue=True` in `.launch()`: deprecated in 3.x, use `.queue()` method
- `gr.Audio(sources=[...])`: Gradio 4.x API — not available in 3.50.2

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | unittest (stdlib, Python 3.8.5) |
| Config file | none — tests run via `python -m pytest tests/` |
| Quick run command | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py -x -q` |
| Full suite command | `conda run -n stylehuman python -m pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UI-01 | generate() bypasses ASR when text is non-empty | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_text_bypass -x` | Wave 0 |
| UI-01 | generate() calls process_audio + transcribe when text is empty and audio provided | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_asr_path -x` | Wave 0 |
| UI-01 | generate() yields error message on empty input (no audio, no text) | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_empty_input -x` | Wave 0 |
| UI-01 | generate() catches exceptions and yields readable error message | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_exception_handling -x` | Wave 0 |
| UI-02 | export_image() saves PIL image to outputs/ with timestamp filename | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestExport::test_export_saves_file -x` | Wave 0 |
| UI-02 | export_image() returns gr.update(visible=True) for export_path | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestExport::test_export_shows_path -x` | Wave 0 |
| CLIP-03 | generate() yields CLIP sim score text during optimization | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_score_yielded -x` | Wave 0 |
| CLIP-03 | generate() yields final sim score in Done state | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_final_score -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `conda run -n stylehuman python -m pytest tests/test_gradio_app.py -x -q`
- **Per wave merge:** `conda run -n stylehuman python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_gradio_app.py` — covers UI-01, UI-02, CLIP-03 (all 8 tests above)

**Testing strategy for Gradio functions without launching server:**

```python
# Key insight: test generate() as a pure Python generator; do NOT import the full module
# because module-level StyleGANWrapper init would try to load the pkl file.
# Solution: patch at module level BEFORE import, or test only the logic functions in isolation.

# Pattern (established in this project):
from unittest.mock import patch, MagicMock
import unittest

class TestGenerate(unittest.TestCase):
    def setUp(self):
        # Patch StyleGANWrapper and gpu_utils before the module's module-level code runs
        # OR extract generate() logic into a testable function that accepts wrapper + cfg
        self.mock_wrapper = MagicMock()
        self.mock_cfg = {
            'device': 'cpu', 'dtype': torch.float32,
            'use_fp16': False, 'resolution': 256, 'truncation_psi': 0.5
        }

    def test_text_bypass(self):
        """generate() does not call process_audio or transcribe when text is non-empty."""
        with patch('ui.app.optimize') as mock_opt:
            mock_opt.return_value = (MagicMock(), MagicMock(), 0.42)
            results = list(generate_fn(
                mic_audio=None, file_audio=None, text_input="a young woman",
                wrapper=self.mock_wrapper, cfg=self.mock_cfg
            ))
        # Last yielded status should be "Done."
        final_status = results[-1][0]
        assert final_status == "Done."
```

**Critical:** To make `generate()` testable without server launch, define it as a standalone function that accepts `wrapper` and `cfg` as parameters (not closing over module-level globals). The Blocks wiring uses `functools.partial` or a lambda to bind the globals.

---

## Open Questions

1. **Single vs Dual Audio Component**
   - What we know: CONTEXT.md says "single Audio component handles both sources"; Gradio 3.50.2 `source=` only accepts a single string (not a list)
   - What's unclear: Whether a single `gr.Audio` component in Gradio 3.50.2 can show both mic and upload in its UI, or whether two separate components are needed
   - Recommendation: Use two separate `gr.Audio` components (one with `source="microphone"`, one with `source="upload"`) and implement file-upload-takes-precedence logic in the generate function. This is safer than relying on a single component's internal toggle behavior.

2. **Testability of module-level init**
   - What we know: `StyleGANWrapper(pkl_path, cfg)` runs at module import time; this will fail in tests unless patched
   - What's unclear: Whether `importlib.reload` + patching is cleaner than refactoring `app.py` to have an `init()` function
   - Recommendation: Extract generate logic into a standalone function with explicit `wrapper, cfg` parameters. Use `if __name__ == "__main__"` guard for the demo init. This keeps the module importable in tests without triggering pkl load.

---

## Sources

### Primary (HIGH confidence)
- Live `conda run -n stylehuman python -c "help(gr.Audio.__init__)"` — confirmed `source=` string API, `type="filepath"` behavior
- Live `conda run -n stylehuman python -c "help(gr.Blocks.queue)"` — confirmed `concurrency_count`, `.queue()` method signature
- Live `conda run -n stylehuman python -c "help(gr.Blocks.launch)"` — confirmed `server_name`, `share`, `show_error` parameters
- Live `conda run -n stylehuman python -c "import gradio; print(gradio.__version__)"` — confirmed 3.50.2
- Existing project modules (optimizer/clip_optimizer.py, asr/transcriber.py, gpu_utils.py, asr/audio_processor.py) — confirmed APIs from source

### Secondary (MEDIUM confidence)
- [Gradio Blocks and Event Listeners Guide](https://www.gradio.app/guides/blocks-and-event-listeners) — `.then()` chaining, `gr.update()` pattern, generator yield streaming
- [Gradio Streaming Outputs Guide](https://www.gradio.app/guides/streaming-outputs) — generator function pattern for progressive updates

### Tertiary (LOW confidence)
- WebSearch results on Gradio 3.x threading patterns — verified conceptually against streaming guide; threading + queue.Queue pattern for callback bridge is community-standard but not in official docs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Gradio 3.50.2 confirmed installed; all APIs verified via live introspection
- Architecture: HIGH — generator + threading + gr.State patterns verified against live Gradio 3.50.2 help; project module APIs verified from source files
- Pitfalls: HIGH — source= vs sources= confirmed from live help(); .queue() requirement confirmed; other pitfalls derived from verified API behavior
- Testing: MEDIUM — testing Gradio functions in isolation is a documented community pattern; exact test file structure follows established project convention

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (Gradio 3.x is stable; pinned version won't change)
