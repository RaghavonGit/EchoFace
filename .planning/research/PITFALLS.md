# Domain Pitfalls

**Domain:** CLIP-guided GAN face synthesis — fully local multimodal pipeline
**Project:** EchoFace
**Researched:** 2026-03-06
**Confidence:** MEDIUM (web tools unavailable; based on documented ecosystem patterns from training data through August 2025 — flag all claims for verification before implementation)

---

## Critical Pitfalls

Mistakes that cause rewrites, silent failures, or environment destruction.

---

### Pitfall C1: Importing StyleGAN-Human internals directly breaks on every git pull

**What goes wrong:** Developers `import` from `StyleGAN-Human/training/networks.py` or
`StyleGAN-Human/dnnlib/` by adding the subdir to `sys.path`. Any upstream change to the
cloned repo (git pull, manual file edit) silently breaks those import paths. The failure
often surfaces as `AttributeError` or `ModuleNotFoundError` that looks like a conda
environment issue rather than a path issue.

**Why it happens:** StyleGAN-Human has no `setup.py` / `pyproject.toml`. Its modules are
only importable if the repo root is on `sys.path`. Developers add it at the top of scripts
and forget.

**Consequences:** Tight coupling between EchoFace code and StyleGAN-Human internals — any
upstream change forces EchoFace refactoring. Repo is supposed to be cloned into project
root and left untouched.

**Prevention:**
- Isolate all StyleGAN-Human interaction behind a single adapter module: `echoface/stylegan_adapter.py`
- That module alone does the `sys.path.insert(0, "StyleGAN-Human")` and imports
- Every other EchoFace file imports from the adapter only
- Document exactly which StyleGAN-Human version hash was tested

**Detection:** Any `from training.networks import ...` or `from dnnlib import ...` outside `stylegan_adapter.py` is a red flag.

**Phase:** Phase 1 (StyleGAN-Human integration) — establish this boundary on day one.

---

### Pitfall C2: Loading the .pkl with `open()` instead of `dnnlib.util.open_url()`

**What goes wrong:** `pickle.load(open("pretrained_models/stylegan_human_v2_1024.pkl", "rb"))` fails with `_pickle.UnpicklingError` or `AttributeError: Can't get attribute 'SynthesisNetwork'` because the pickle references custom classes in `dnnlib` and `training.networks` that must be registered before deserialization.

**Why it happens:** StyleGAN2/StyleGAN-Human pickles are not plain Python objects — they embed `legacy` module references. The correct loader (`legacy.load_network_pkl`) registers those classes first.

**Consequences:** The generator never loads; the error message is cryptic and misleads toward environment issues.

**Prevention:**
```python
# CORRECT — via legacy loader
import sys
sys.path.insert(0, "StyleGAN-Human")
import dnnlib
import legacy

with dnnlib.util.open_url("StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"].eval()
```
Never use raw `pickle.load` on StyleGAN pkl files.

**Detection:** Raw `pickle.load` call anywhere in the codebase.

**Phase:** Phase 1 (StyleGAN-Human integration).

---

### Pitfall C3: Optimizing in Z-space instead of W-space gives poor attribute control

**What goes wrong:** Feeding gradients directly into a random `z` vector and re-running mapping network each step appears to work (loss decreases) but produces faces where attribute edits interfere with each other (smile changes age, glasses change pose). The disentanglement that StyleGAN2/StyleGAN-Human is known for is only available in W-space.

**Why it happens:** The mapping network `G.mapping(z, None)` is non-linear; gradients in Z-space navigate a highly entangled manifold. W-space is the linearized, more disentangled representation.

**Consequences:** Attribute sliders (age, smile, glasses) produce inconsistent or coupled results. CLIP optimization converges but the semantic fidelity is lower. Fixing this later requires rewriting the entire optimization loop.

**Prevention:**
```python
# CORRECT — optimize w directly
w_init = G.mapping(z_init, None)          # shape [1, 18, 512] for 1024px
w = w_init.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([w], lr=0.01)

# Each step: synthesize from w directly
img = G.synthesis(w, noise_mode="const")  # bypass mapping on each step
```
Pin W-space as the optimization target in the architecture decision, not as an afterthought.

**Detection:** Any optimization loop that calls `G.mapping()` on every step (not just initialization) is likely Z-space or mixed-space.

**Phase:** Phase 2 (CLIP optimization loop) — document the design decision explicitly.

---

### Pitfall C4: CLIP cosine similarity gradient explosion during optimization

**What goes wrong:** Raw `-torch.cosine_similarity(text_features, image_features)` loss explodes at early steps when the generated image is far from the text embedding. Gradients through the synthesis network can become NaN after 5-20 steps, silently producing a black or gray image.

**Why it happens:** Cosine similarity is bounded [-1, 1] but the gradient of the synthesis network w.r.t. W can be large. Without clipping or normalization, Adam's adaptive step can amplify early gradients before the optimization converges.

**Consequences:** NaN in `w` propagates silently; subsequent steps generate garbage with no error. The UI may display a corrupted image without any exception being raised.

**Prevention:**
- Normalize CLIP embeddings before computing similarity: `F.normalize(features, dim=-1)`
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)` before optimizer step
- Check for NaN explicitly: `if torch.isnan(w).any(): raise RuntimeError("Latent NaN detected")`
- Use a warmup: start with `lr=0.001` for first 10 steps, then increase to `0.01`

**Detection:** Loss becomes `nan` or stays exactly `0.0` after a few steps; generated images are gray/black.

**Phase:** Phase 2 (CLIP optimization loop).

---

### Pitfall C5: CLIP semantic drift — optimization converges to texture, not semantics

**What goes wrong:** After 150 steps, the CLIP score is high but the face looks photorealistic-but-wrong: background textures, clothing colors, or hair patterns dominate the match rather than face geometry described in the prompt. Saying "young woman with blue eyes" produces a face with blue tinting everywhere.

**Why it happens:** CLIP ViT-B/32 is sensitive to low-frequency color/texture patterns. Without a regularization term, optimization finds adversarial directions in W-space that maximize cosine similarity via texture rather than semantic content.

**Consequences:** Generated faces do not match the spoken description semantically. Users experience this as the tool "not working" even though loss is low.

**Prevention:**
- The regularization term `λ||w - w_init||²` (λ=0.1) in the project spec is the correct defense — do not remove it or reduce λ below 0.05
- Augment the image before CLIP encoding: random crops, color jitter, resizing. This makes texture gaming harder.
- Preprocess the generated image to face-crop region before CLIP encoding (center crop to face bounding box)
- Validate against a qualitative test set of prompts (blonde, dark hair, elderly, young) before declaring the optimizer "done"

**Detection:** High CLIP score but descriptions do not match generated faces visually. Test with "an elderly man with a beard" — if result looks young/smooth, drift is occurring.

**Phase:** Phase 2 (CLIP optimization loop) and Phase 3 (evaluation/FID).

---

### Pitfall C6: openai-whisper model size vs VRAM — silent OOM on shared GPU

**What goes wrong:** `whisper.load_model("medium")` loads a 1.5 GB model. If StyleGAN-Human synthesis network is already resident in VRAM (also ~1-2 GB for 1024px), loading Whisper at app startup pushes total VRAM usage over the GPU's limit. PyTorch silently moves tensors to CPU for Whisper (if CUDA runs out) which appears to work but is 10-50x slower. On smaller GPUs (4 GB), the entire pipeline crashes with `CUDA out of memory`.

**Why it happens:** Both models are large; Whisper is loaded at `import` time if `whisper.load_model()` is called at module scope.

**Consequences:** Whisper transcription takes 30-120 seconds instead of 2-5 seconds. On small GPUs, the pipeline is unusable.

**Prevention:**
- Load Whisper lazily (on first transcription call, not at import)
- Unload or move Whisper to CPU after transcription before running StyleGAN optimization: `whisper_model.cpu(); torch.cuda.empty_cache()`
- Document minimum VRAM requirements in README: 6 GB minimum (Whisper medium on CPU + StyleGAN on GPU), 8 GB recommended (both on GPU)
- Provide a `--whisper-size small` flag for users with limited VRAM

**Detection:** `torch.cuda.memory_allocated()` before and after model loads; watch for silent CPU fallback in Whisper.

**Phase:** Phase 1 (pipeline integration) and README/setup documentation.

---

### Pitfall C7: Audio format mismatch silently degrades Whisper transcription quality

**What goes wrong:** Whisper's internal resampler expects 16 kHz mono float32. Feeding 44.1 kHz stereo audio (default microphone output on many systems) produces transcriptions that are subtly wrong (words dropped, garbled) with no error or warning. The librosa preprocessing step is critical but easy to skip during prototyping.

**Why it happens:** Whisper does internal resampling but it degrades quality compared to pre-resampling with librosa. Stereo audio causes double-transcription artifacts.

**Consequences:** "young woman with brown hair" transcribes as "young woman with brown air" or similar; the CLIP query becomes wrong, producing wrong faces. The bug is nearly invisible because the pipeline continues without errors.

**Prevention:**
```python
import librosa
import soundfile as sf

def preprocess_audio(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=16000, mono=True)  # force 16kHz mono
    # normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

# Pass numpy array directly, not file path
result = whisper_model.transcribe(audio_array, fp16=False)
```
Never pass raw mic-captured audio files directly to `whisper_model.transcribe()` without resampling.

**Detection:** Test with a known phrase recorded at 44.1 kHz stereo; compare transcription against expected text. Any degradation indicates missing preprocessing.

**Phase:** Phase 1 (audio preprocessing pipeline).

---

### Pitfall C8: Python 3.8.5 + PyTorch 1.9.1 incompatibility with Gradio 4.x

**What goes wrong:** Gradio 4.x requires Python >=3.10 (as of Gradio 4.0+). Installing Gradio 4.x into the StyleGAN-Human conda environment (Python 3.8.5) either fails outright with a resolver error, or pip downgrades/upgrades core packages that break PyTorch 1.9.1 compatibility.

**Why it happens:** Gradio 4.x uses `match` statements (Python 3.10+ syntax) and type hints that are invalid under 3.8. Pip's dependency resolver may not catch this cleanly if wheels are available for 3.8.

**Consequences:** Either the installation fails (best case) or Gradio installs but crashes at runtime with `SyntaxError: invalid syntax` (worst case — hard to debug). Alternatively, pip silently installs an incompatible numpy/httpx version that breaks whisper or torch.

**Prevention:**
- Pin Gradio to the highest 3.x version compatible with Python 3.8: `gradio>=3.40,<4.0`
- Validate with `python -c "import gradio; print(gradio.__version__)"` in the conda env
- Document exact versions in `requirements_extra.txt` with pinned upper bounds
- Test the full import chain after setup: `python -c "import gradio, whisper, clip, torch"`

**Detection:** Run `pip check` after all installs — any dependency conflict is a warning that something will break at runtime.

**Phase:** Phase 0 (environment setup) — this must be resolved before any code is written.

---

### Pitfall C9: CLIP pip install confusion — openai SDK vs CLIP library

**What goes wrong:** `pip install clip` or `pip install openai-clip` installs unofficial or wrong packages. `pip install openai` installs the OpenAI Python SDK which requires an API key. The correct package is installed via GitHub URL only.

**Why it happens:** The official CLIP library was never published to PyPI under a maintained name. Multiple third-party forks use similar names.

**Consequences:** Wrong `clip` package is imported; the API differs (no `clip.load()`, no `clip.tokenize()`, etc.). Failures appear as `AttributeError: module 'clip' has no attribute 'load'`.

**Prevention:**
```bash
# CORRECT — only this installs the real OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# Verify
python -c "import clip; model, preprocess = clip.load('ViT-B/32'); print('CLIP OK')"
```
This is already documented in PROJECT.md but must be enforced in `requirements_extra.txt`:
```
# requirements_extra.txt
clip @ git+https://github.com/openai/CLIP.git
```

**Detection:** `import clip; clip.load("ViT-B/32")` — if it raises `AttributeError` or `TypeError`, the wrong package is installed.

**Phase:** Phase 0 (environment setup).

---

### Pitfall C10: Conda env conflicts between StyleGAN-Human numpy/Pillow and Whisper/CLIP

**What goes wrong:** StyleGAN-Human's `environment.yml` pins `numpy=1.19.x` and `Pillow=8.x`. Whisper requires `numpy>=1.20`. CLIP requires `Pillow>=9.1` for certain image preprocessing functions. Conda/pip resolvers can silently upgrade these, breaking StyleGAN's training utilities (even though EchoFace only uses inference).

**Why it happens:** StyleGAN-Human's environment was locked against a 2021 environment. Two years of dependency drift means extra deps want newer versions of shared transitive dependencies.

**Consequences:**
- `numpy` upgrade breaks `dnnlib` or causes subtle numerical differences in synthesis output
- `Pillow` upgrade causes deprecation warnings that become errors in Python 3.8+ (`ANTIALIAS` removed in Pillow 10.x)
- Environment appears healthy but specific operations fail with cryptic errors at runtime

**Prevention:**
- Install `requirements_extra.txt` with `pip install --no-deps` where possible, then manually install transitive deps at compatible versions
- After all installs, verify: `python -c "import numpy; print(numpy.__version__)"`  — should match what StyleGAN-Human tests document
- Use `Pillow>=9.1,<10.0` to avoid the ANTIALIAS removal breaking existing StyleGAN image save calls
- Pin `numpy>=1.20,<1.24` (compatible with both Whisper and StyleGAN-Human inference path)

**Detection:** `pip check` after install; any conflict is a latent bug. `from PIL import Image; Image.ANTIALIAS` — if `AttributeError`, Pillow is too new.

**Phase:** Phase 0 (environment setup) and ongoing (re-test after any dep change).

---

### Pitfall C11: Gradio live update threading deadlock with optimization loop

**What goes wrong:** The CLIP optimization loop runs for 150 steps on the main thread. Gradio's update callbacks (`gr.Progress()` or `yield` from a generator function) attempt to communicate with Gradio's internal server via a separate thread. If the optimization loop blocks the event loop or holds the GIL continuously, live preview updates freeze — the UI appears hung even though optimization is running.

**Why it happens:** PyTorch operations (especially CUDA kernels) release the GIL but CPU-bound Python code (loss computation, logging) does not. Gradio's internal server uses asyncio; a blocking function prevents event loop iterations.

**Consequences:** Users see a frozen UI for 30-60 seconds with no feedback, then the final image appears. The "live preview" feature does not work as designed.

**Prevention:**
- Use Gradio's generator function pattern: `yield` intermediate results every N steps rather than using callbacks
- Run the optimization in a separate thread with `concurrent.futures.ThreadPoolExecutor` and poll for updates
- Use `torch.cuda.synchronize()` only when needed (before image capture for display), not on every step
- Throttle updates to every 10 steps (as specified in PROJECT.md) — do not yield every step

```python
def optimize_with_updates(text_embedding, G, steps=150):
    w = initialize_w(G)
    for step in range(steps):
        loss = optimization_step(w, text_embedding, G)
        if step % 10 == 0:
            preview_img = synthesize_for_display(G, w)
            yield preview_img, step, float(loss)
    yield synthesize_final(G, w), steps, 0.0
```

**Detection:** If Gradio UI freezes during optimization and only updates at the end, the generator pattern is not being used correctly.

**Phase:** Phase 3 (Gradio UI integration).

---

### Pitfall C12: CUDA/CPU fallback produces different latent behaviors, not just slower output

**What goes wrong:** The CPU fallback path uses `truncation=0.5` and `256x256` output as specified. However, the W-space regularization term `λ||w - w_init||²` behaves differently on CPU: Adam optimizer accumulates float64 gradients by default on CPU vs float16 on GPU. The optimization produces different faces for identical prompts depending on whether CUDA is available.

**Why it happens:** PyTorch's default dtype differs between CPU and CUDA paths. Adam's epsilon term also interacts differently with float16 vs float32 precision.

**Consequences:** Users on CPU-only machines get different (often lower quality) results than GPU users for the same text prompt. Not a crash, but a correctness issue that is hard to reproduce and debug.

**Prevention:**
- Explicitly set dtype everywhere: `torch.set_default_dtype(torch.float32)` regardless of device
- Keep `fp16=True` only for the Whisper transcription, not for the optimization loop
- Test the optimization loop on CPU with `device = torch.device("cpu")` explicitly and compare output characteristics
- Document in README that CPU results will differ in quality (256px, more truncated) but not in semantic alignment direction

**Detection:** Run the same seed + prompt on GPU vs CPU; compare faces. If they are dramatically different in structure (not just resolution), a dtype inconsistency exists.

**Phase:** Phase 2 (optimization loop) and Phase 4 (CPU fallback testing).

---

### Pitfall C13: `generate.py` `--outdir` side effects pollute the project filesystem

**What goes wrong:** If EchoFace calls StyleGAN-Human's `generate.py` as a subprocess (instead of importing its generator directly), every invocation writes output images and a `log.txt` to the `--outdir` directory. On repeated calls (150 optimization steps), this fills disk rapidly and creates thousands of intermediate files.

**Why it happens:** `generate.py` is designed as a CLI script with filesystem side effects. Using it as a subprocess is the most "safe from internals" approach but is deeply wrong for an optimization loop.

**Consequences:** 150 optimization steps × ~4 MB per 1024px image = ~600 MB written to disk per single face generation. Disk fills; temp files are never cleaned up.

**Prevention:**
- Do NOT use `generate.py` as a subprocess for the optimization loop
- Import the generator directly through `stylegan_adapter.py` and call `G.synthesis()` in memory
- `generate.py` is only relevant for understanding the interface; the actual API is `G.synthesis(ws, noise_mode="const")`
- If subprocess is used for any reason, use `/dev/null` equivalent or a temp dir with cleanup

**Detection:** Check if any code path calls `subprocess.run(["python", "StyleGAN-Human/generate.py", ...])` in a loop.

**Phase:** Phase 1 (StyleGAN-Human integration) — establish the in-process API immediately.

---

## Moderate Pitfalls

### Pitfall M1: Whisper `fp16=False` required on CPU but forgotten

**What goes wrong:** `whisper_model.transcribe(audio, fp16=True)` on a CPU-only machine raises `RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'`. This is a well-known Whisper gotcha.

**Prevention:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
result = model.transcribe(audio, fp16=(device == "cuda"))
```
Always pass `fp16` as a computed value, never hardcode `True`.

**Phase:** Phase 1 (Whisper integration).

---

### Pitfall M2: CLIP model download happens at first `clip.load()` call — blocks UI

**What goes wrong:** If CLIP weights (~350 MB for ViT-B/32) are not cached, `clip.load("ViT-B/32")` blocks for 30-120 seconds while downloading. If this is called inside a Gradio request handler, the UI hangs with no feedback.

**Prevention:**
- Pre-warm both Whisper and CLIP models at application startup (before any Gradio handler fires)
- Add a startup progress indicator: `print("Loading CLIP... "); model, preprocess = clip.load("ViT-B/32"); print("Done.")`
- Document in README that first run requires network access for model download; subsequent runs are fully offline

**Phase:** Phase 1 (model loading) and README documentation.

---

### Pitfall M3: Latent direction sliders applied incorrectly — direction not normalized

**What goes wrong:** If latent attribute directions (age, smile, etc.) discovered from the dataset are not L2-normalized before being added to `w`, the effective step size varies wildly between attributes. A smile slider at `+1.0` changes pose drastically while an age slider at `+1.0` does nothing.

**Prevention:**
```python
direction = direction / (np.linalg.norm(direction) + 1e-8)
w_edited = w_base + slider_value * direction
```
Always normalize attribute directions before storing them. Store normalized directions, not raw ones.

**Phase:** Phase 4 (latent direction sliders).

---

### Pitfall M4: FID scoring requires at least 2048 samples — Kaggle dataset may be too small

**What goes wrong:** FID (Fréchet Inception Distance) becomes statistically unreliable with fewer than 2048 samples. If the Kaggle Human Faces Dataset has fewer images, FID scores will have high variance and mislead evaluation.

**Prevention:**
- Count images in `data/human_faces/` before computing FID: `assert len(images) >= 2048, "Need >= 2048 images for reliable FID"`
- If dataset is smaller, use a different metric (IS score or LPIPS) or document the limitation
- Check the Kaggle dataset size before designing the FID pipeline

**Phase:** Phase 3 (evaluation/FID utility).

---

### Pitfall M5: `noise_mode` in `G.synthesis()` affects reproducibility

**What goes wrong:** Using `noise_mode="random"` means two calls with the same `w` produce different images (different stochastic noise). This makes debugging and evaluation impossible — you cannot tell if a change improved quality or just hit a better noise sample.

**Prevention:** Always use `noise_mode="const"` during optimization and evaluation. Only use `noise_mode="random"` if intentional diversity is wanted (not during CLIP optimization).

**Phase:** Phase 2 (CLIP optimization loop).

---

## Minor Pitfalls

### Pitfall Mi1: Gradio's `gr.Audio` component default sample rate does not match Whisper

**What goes wrong:** `gr.Audio(source="microphone")` captures audio at the browser's native sample rate (often 48000 Hz). The numpy array passed to the Python handler may be at 48 kHz, not 16 kHz.

**Prevention:** Always resample in the handler:
```python
def handle_audio(audio_tuple):
    sr, data = audio_tuple  # Gradio returns (sample_rate, numpy_array)
    if sr != 16000:
        data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=16000)
```

**Phase:** Phase 3 (Gradio UI integration).

---

### Pitfall Mi2: PIL `Image.fromarray()` requires uint8, not float32

**What goes wrong:** StyleGAN synthesis output is typically a float32 tensor in [-1, 1]. Passing it directly to `Image.fromarray()` produces a black image or `TypeError`. The conversion step is easy to forget during prototyping.

**Prevention:**
```python
img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2 * 255
img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
pil_img = Image.fromarray(img_np)
```

**Phase:** Phase 1/2 (anywhere a StyleGAN output is displayed or saved).

---

### Pitfall Mi3: `torch.cuda.empty_cache()` does not immediately free memory

**What goes wrong:** Calling `torch.cuda.empty_cache()` after deleting a model does not immediately free VRAM for OS-level reuse. The memory shows as freed in `torch.cuda.memory_allocated()` but `nvidia-smi` still shows it reserved. Subsequent large allocations may still OOM.

**Prevention:** Do not rely on `empty_cache()` for memory management between Whisper and StyleGAN. Instead, design the pipeline so both models are loaded at startup and coexist within the VRAM budget.

**Phase:** Phase 1 (pipeline design).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Environment setup | Gradio 4.x incompatible with Python 3.8.5 (C8) | Pin `gradio<4.0` in requirements_extra.txt |
| Environment setup | Wrong CLIP package from PyPI (C9) | GitHub URL install, verification step |
| Environment setup | numpy/Pillow version conflicts (C10) | `pip check` after all installs; pin upper bounds |
| StyleGAN integration | Raw pickle.load fails on .pkl (C2) | Use `legacy.load_network_pkl()` exclusively |
| StyleGAN integration | Direct imports from StyleGAN internals (C1) | Adapter module pattern |
| StyleGAN integration | generate.py subprocess in optimization loop (C13) | In-process `G.synthesis()` only |
| Audio preprocessing | Mic audio at wrong sample rate (C6, Mi1) | librosa resample to 16 kHz mono before Whisper |
| Whisper integration | fp16 on CPU crashes (M1) | Compute fp16 flag from device availability |
| Whisper integration | Model download blocks UI (M2) | Pre-warm at startup |
| CLIP optimization | Z-space vs W-space (C3) | Optimize `w` directly; bypass `G.mapping()` in loop |
| CLIP optimization | Gradient explosion / NaN (C4) | Gradient clipping + NaN assertion |
| CLIP optimization | Semantic drift via texture (C5) | Augmentation + regularization λ=0.1 |
| CLIP optimization | noise_mode reproducibility (M5) | Use `noise_mode="const"` |
| Gradio UI | Optimization loop freezes UI (C11) | Generator function with `yield` every 10 steps |
| Gradio UI | Audio sample rate mismatch (Mi1) | Resample in Gradio handler |
| Latent sliders | Unnormalized directions (M3) | L2-normalize all attribute directions before storage |
| Evaluation/FID | Too few samples for reliable FID (M4) | Assert >= 2048 samples; document limitation |
| CPU fallback | dtype differences GPU vs CPU (C12) | Explicit float32 everywhere; test CPU path |
| Image export | float32 tensor to PIL (Mi2) | Clamp + scale + uint8 cast pipeline |

---

## Confidence Notes

All pitfalls above are derived from well-established patterns in the StyleGAN2, CLIP, Whisper, and Gradio ecosystems documented through August 2025. Web research tools were unavailable during this session. Confidence levels:

| Pitfall Area | Confidence | Notes |
|---|---|---|
| StyleGAN pkl loading (C2) | HIGH | Documented in StyleGAN2/StyleGAN-Human READMEs and GitHub issues |
| W-space vs Z-space (C3) | HIGH | Fundamental to StyleGAN architecture; well-documented |
| CLIP gradient explosion (C4) | MEDIUM | Common in CLIP-guided optimization literature; verify exact clip threshold |
| Semantic drift (C5) | MEDIUM | Documented in CLIPDraw and text-to-image papers; augmentation strategy is standard |
| Gradio Python 3.8 compat (C8) | MEDIUM | Verify exact Gradio version cutoff — may be 3.x that still works; check release notes |
| Whisper fp16 on CPU (M1) | HIGH | Explicitly documented in Whisper repo FAQ |
| numpy/Pillow conflicts (C10) | MEDIUM | Version numbers may have shifted; run `pip check` to verify actual state |
| Gradio threading (C11) | MEDIUM | Generator pattern is documented Gradio approach; exact behavior under Python 3.8 needs verification |

## Sources

- StyleGAN-Human GitHub: https://github.com/stylegan-human/StyleGAN-Human (training knowledge, not fetched)
- OpenAI CLIP GitHub: https://github.com/openai/CLIP (training knowledge, not fetched)
- OpenAI Whisper GitHub: https://github.com/openai/whisper (training knowledge, not fetched)
- Gradio documentation: https://gradio.app/docs (training knowledge, not fetched)
- Note: All URLs are provided for human verification. Web fetch was unavailable during this research session.
