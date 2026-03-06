# Architecture Patterns

**Domain:** Speech-to-image / CLIP-guided GAN synthesis (EchoFace)
**Researched:** 2026-03-06
**Confidence:** HIGH (grounded in PROJECT.md spec + established CLIP-GAN optimization literature)

---

## Recommended Architecture

EchoFace is a sequential pipeline with a feedback loop. Audio enters one end; a generated face exits the other. The optimization loop is the only non-linear section: it iterates over latent space until CLIP similarity converges or step budget is exhausted.

```
Audio Input (mic / file)
        |
        v
   [ASR Module]          asr/
        | text transcript
        v
  [CLIP Encoder]         encoder/
        | text embedding (512-d float tensor)
        v
  [Latent Optimizer]     optimizer/
        |   ^--- CLIP image embedding feedback
        |        (every 10 steps → UI preview + score)
        v
 [StyleGAN Wrapper]      generator/
        | 1024x1024 RGB tensor
        v
  [Gradio UI]            ui/
        | PNG export / slider edits
        v
      Output
```

### Component Boundaries

| Component | Module Path | Responsibility | Inputs | Outputs |
|-----------|-------------|----------------|--------|---------|
| ASR | `asr/` | Convert audio bytes to text transcript | Raw audio bytes (16kHz PCM) or file path | UTF-8 text string |
| CLIP Encoder | `encoder/` | Encode text and images into CLIP embedding space | Text string OR PIL Image / tensor | 512-d normalized float tensor |
| StyleGAN Wrapper | `generator/` | Wrap StyleGAN-Human generate.py; synthesize image from W-space latent | W-space latent tensor [1, 18, 512] | RGB image tensor [1, 3, 1024, 1024] |
| Latent Optimizer | `optimizer/` | Run CLIP-guided gradient descent on W-space latent; emit progress callbacks | Text CLIP embedding, initial latent w_init, hyperparams | Final optimized latent; per-step callback with (step, image, similarity score) |
| Gradio UI | `ui/` | Orchestrate pipeline, display live preview, expose sliders, export PNG | User audio/text input events; optimizer callbacks | Gradio Blocks app on localhost |
| Utils | `utils/` | Dataset loading, FID scoring, latent direction discovery, audio preprocessing | Various | Various |

---

## Data Flow (Explicit Direction)

### Stage 1: Audio to Text

```
mic bytes / .wav file
  → librosa resample + normalize + silence trim   [utils/audio_preprocess.py]
  → 16kHz float32 PCM array
  → whisper.load_model("medium").transcribe()     [asr/transcriber.py]
  → {"text": "a middle-aged man with curly hair"} string
```

Manual text override short-circuits this entire stage; the text string is injected directly.

### Stage 2: Text to CLIP Embedding

```
text string
  → clip.tokenize([text])                         [encoder/text_encoder.py]
  → token tensor [1, 77]
  → model.encode_text(tokens).float()
  → L2-normalize
  → text_embedding: Tensor [1, 512]               # fixed, computed once
```

The text embedding is computed once before the optimization loop starts and is held constant throughout.

### Stage 3: Latent Initialization

```
                                                  [generator/stylegan_wrapper.py]
  → sample w_init from StyleGAN W-space
    (or use mean latent for deterministic start)
  → w: Tensor [1, 18, 512]   (requires_grad=True)
```

W-space (18 layers × 512 dims) is used instead of Z-space because it is more disentangled, making attribute edits via direction vectors meaningful.

### Stage 4: Optimization Loop (Core Feedback Loop)

This is the non-linear section. It runs for up to 150 gradient steps.

```
LOOP (step = 0..149):

  w [1, 18, 512]  (differentiable leaf variable)
      |
      v
  StyleGAN synthesis (G.synthesis(w))              [generator/stylegan_wrapper.py]
      |
      v
  image_tensor [1, 3, 1024, 1024]
      |
      v
  resize + normalize for CLIP (224x224)            [encoder/image_encoder.py]
      |
      v
  model.encode_image(image_tensor).float()
      |
      v
  image_embedding: Tensor [1, 512]
      |
      v
  L_clip = -cosine_similarity(text_emb, image_emb)   # CLIP loss
  L_reg  = lambda * ||w - w_init||^2                  # regularization (λ=0.1)
  L_total = L_clip + L_reg
      |
      v
  L_total.backward()                               # gradients flow back into w
  optimizer.step()                                 # Adam(lr=0.01) updates w
  optimizer.zero_grad()
      |
      v
  if step % 10 == 0:
      callback(step, image_tensor, similarity_score)  # UI preview update

END LOOP
  → final_w [1, 18, 512]
```

Key insight: StyleGAN's synthesis network must be run in fp16 on GPU but `w` is kept in fp32 for stable gradient accumulation. The generator's weights are frozen — only `w` receives gradients.

### Stage 5: Image Export and Attribute Editing

```
final_w [1, 18, 512]
  → slider adjustment: w' = final_w + alpha * direction_vector
  → G.synthesis(w')
  → image_tensor → PIL.Image → PNG
```

Attribute sliders (age, smile, facial hair, glasses, pose) apply pre-discovered direction vectors in W-space. These directions are discovered offline from the Kaggle Human Faces dataset using `utils/latent_directions.py`.

---

## Component Communication Contract

All inter-component communication uses plain Python objects (tensors, strings, callbacks). No message queues, no shared state beyond the Gradio session object.

| From | To | Medium | Type |
|------|----|--------|------|
| UI | ASR | direct function call | `bytes` or `str` (file path) |
| ASR | UI | return value | `str` (transcript) |
| UI | Encoder | direct function call | `str` (text) |
| Encoder | Optimizer | return value | `torch.Tensor [1, 512]` |
| Optimizer | Generator | direct function call (per step) | `torch.Tensor [1, 18, 512]` |
| Generator | Optimizer | return value (per step) | `torch.Tensor [1, 3, 1024, 1024]` |
| Optimizer | Encoder | direct function call (per step) | `PIL.Image` or tensor |
| Optimizer | UI | progress callback (every 10 steps) | `(int, PIL.Image, float)` |
| UI | Generator | direct call for slider edits | `torch.Tensor [1, 18, 512]` |

---

## Patterns to Follow

### Pattern 1: Generator as Stateless Callable

**What:** The StyleGAN wrapper exposes a single `synthesize(w) -> image` function. No state is stored between calls. The StyleGAN-Human repo is never modified.

**When:** Always. The generator's weights are frozen at load time. All state lives in `w`.

**Example:**
```python
# generator/stylegan_wrapper.py
class StyleGANWrapper:
    def __init__(self, pkl_path: str, device: str):
        with open(pkl_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device).eval()
        for p in self.G.parameters():
            p.requires_grad_(False)

    def synthesize(self, w: torch.Tensor) -> torch.Tensor:
        # w: [1, 18, 512], returns [1, 3, 1024, 1024]
        return self.G.synthesis(w, noise_mode='const')
```

### Pattern 2: CLIP Encoder as Dual-Mode Service

**What:** One CLIP model instance handles both text and image encoding. Loading it twice wastes VRAM.

**When:** Always. The model is loaded once at startup and shared across both text encoding (pre-loop) and image encoding (per step in loop).

**Example:**
```python
# encoder/clip_encoder.py
class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens).float()
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # image already preprocessed to [1, 3, 224, 224]
        emb = self.model.encode_image(image).float()
        return emb / emb.norm(dim=-1, keepdim=True)
```

### Pattern 3: Optimizer as Generator-of-Callbacks

**What:** The latent optimizer is a Python generator (or takes a callback function). It yields/calls on every 10th step, providing the current image and similarity score. This lets Gradio update the UI without threading complexity.

**When:** For Gradio integration. Gradio's event handlers are synchronous; yielding from a generator allows `gr.Progress` updates without a separate thread.

**Example:**
```python
# optimizer/latent_optimizer.py
def optimize(
    generator, encoder, text_emb, w_init,
    steps=150, lr=0.01, lam=0.1,
    callback=None
):
    w = w_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for step in range(steps):
        image = generator.synthesize(w)
        image_clip = preprocess_for_clip(image)
        image_emb = encoder.encode_image(image_clip)
        sim = torch.cosine_similarity(text_emb, image_emb)
        loss = -sim + lam * (w - w_init).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if callback and step % 10 == 0:
            callback(step, to_pil(image.detach()), sim.item())
    return w.detach()
```

### Pattern 4: Gradio Streaming via Generator Yield

**What:** Gradio's `gr.Progress` and generator-based event functions allow streaming updates.

**When:** In the Gradio event handler that runs the optimization. Yield intermediate PIL images to update the preview gallery.

**Example:**
```python
# ui/app.py
def run_pipeline(audio, text_override, progress=gr.Progress()):
    text = text_override or asr.transcribe(audio)
    text_emb = encoder.encode_text(text)
    w_init = generator.sample_w()
    previews = []
    def on_step(step, image, score):
        previews.append(image)
        progress(step / 150, desc=f"Step {step}/150 — similarity: {score:.3f}")
    final_w = optimizer.optimize(generator, encoder, text_emb, w_init,
                                  callback=on_step)
    final_image = generator.synthesize(final_w)
    return to_pil(final_image), previews, text
```

Note: If Gradio 4.x does not support generator yields in all event types, use `gr.Progress` with the callback approach above. Gradio 4.x `gr.Interface` and `gr.Blocks` both support generator functions with `yield` for streaming — this is the preferred pattern.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying StyleGAN-Human Internals

**What:** Editing files inside the cloned `StyleGAN-Human/` directory.

**Why bad:** Upstream updates become impossible to merge; the internal API is undocumented and may change. The project constraint explicitly forbids this.

**Instead:** Wrap the external API only. If `generate.py` does not expose what you need, call `G.synthesis()` directly through the wrapper after loading the pickle — do not patch the source.

### Anti-Pattern 2: Reloading CLIP or StyleGAN Per Request

**What:** Calling `clip.load()` or `pickle.load()` inside the Gradio event handler (once per user request).

**Why bad:** Model load time is 5-30 seconds. User experience degrades. VRAM spikes during load can OOM.

**Instead:** Load all models once at application startup. Keep them as module-level or class-level singletons. Pass references to the pipeline.

### Anti-Pattern 3: Computing Gradients Through the Full StyleGAN Graph

**What:** Accidentally calling `G.synthesis()` without freezing generator parameters, causing memory-expensive gradient tape through millions of parameters.

**Why bad:** OOM on consumer GPUs. Only `w` should be in the gradient graph.

**Instead:** Always `p.requires_grad_(False)` for all generator parameters at load time. Only `w` has `requires_grad=True`.

### Anti-Pattern 4: Z-space Optimization

**What:** Optimizing the initial noise vector Z instead of the W-space latent.

**Why bad:** Z-space is entangled — moving along any axis changes multiple attributes simultaneously. Attribute direction sliders become meaningless.

**Instead:** Map Z → W via `G.mapping()` once (to get `w_init`), then optimize in W-space. The mapping network is not re-run during optimization.

### Anti-Pattern 5: Running the Full 1024px Image Through CLIP

**What:** Passing the raw 1024px StyleGAN output directly to CLIP without resizing.

**Why bad:** CLIP ViT-B/32 expects 224x224 input. Passing 1024px will error or produce wrong results.

**Instead:** Apply `torchvision.transforms.Resize(224)` + CLIP's normalization transform before `encode_image`. The display image is a separate copy.

### Anti-Pattern 6: No CPU Fallback Guard

**What:** Hard-coding `device="cuda"` everywhere.

**Why bad:** Crashes on machines without CUDA, alienating the developer audience.

**Instead:** `device = "cuda" if torch.cuda.is_available() else "cpu"`. On CPU, use truncation=0.5 and 256x256 output (configurable). Document expected runtime difference clearly.

---

## Suggested Build Order

The dependency graph drives build order. Each item can only be built after its dependencies exist.

```
Level 0 (no deps):
  utils/audio_preprocess.py     — pure librosa/soundfile, no model deps
  utils/dataset_loader.py       — pure filesystem/PIL

Level 1 (depends on: nothing ML):
  asr/transcriber.py            — depends on whisper (model load), no other components
  generator/stylegan_wrapper.py — depends on StyleGAN-Human clone + pickle

Level 2 (depends on: Level 1):
  encoder/clip_encoder.py       — depends on CLIP library (model load), no other components

Level 3 (depends on: Level 1 + 2):
  optimizer/latent_optimizer.py — depends on generator + encoder contracts being stable

Level 4 (depends on: Level 0-3):
  ui/app.py                     — depends on all above; integration point

Level 5 (depends on: generator + dataset):
  utils/latent_directions.py    — can be developed in parallel with Level 4
  utils/fid_scorer.py           — depends on generator + dataset_loader
```

**Recommended milestone sequence:**

1. **Repo scaffold and environment** — conda env, StyleGAN-Human clone, directory structure, `requirements_extra.txt`, CPU/GPU device detection utility. Validates the environment before any ML code.

2. **Generator wrapper** — Load the .pkl, synthesize from a random W-space sample, save PNG. Verifies StyleGAN-Human is callable before building anything that depends on it.

3. **CLIP encoder** — Load ViT-B/32, encode a hardcoded text string, encode a generated image, compute similarity. Verifies the CLIP contract before the optimizer depends on it.

4. **Latent optimizer (no UI)** — Wire generator + encoder into the optimization loop. Run from a script with a hardcoded text prompt. Validate that loss decreases and output image improves. This is the hardest component to debug; isolate it from UI complexity.

5. **ASR module** — Load Whisper medium, transcribe a sample audio file. Keep separate from optimizer until optimizer is proven.

6. **Gradio UI** — Wire all components into the Blocks app. Add mic input, text override, progress callbacks, live preview, similarity score display.

7. **Attribute sliders** — Discover latent directions from dataset, expose sliders in UI.

8. **Evaluation utilities** — FID scorer, latent direction discovery. These are independent of the main pipeline and can be done last.

---

## GPU/CPU Architecture Considerations

| Concern | GPU (CUDA) | CPU Fallback |
|---------|-----------|--------------|
| StyleGAN output resolution | 1024x1024 | 256x256 (truncation=0.5) |
| Optimization steps | 150 | 150 (but ~10x slower — consider 50) |
| fp16 inference | Yes (StyleGAN synthesis) | No (fp32 only) |
| VRAM budget | ~8GB minimum (StyleGAN 1024px + CLIP + w gradient) | N/A |
| Model loading | All models on GPU | All models on CPU |
| Expected time per run | ~30-60 seconds | ~5-15 minutes |

fp16 note: StyleGAN-Human synthesis can run in fp16 to save VRAM. `w` must remain fp32 for gradient stability. Use `autocast` context manager around synthesis only.

---

## Scalability Considerations

EchoFace is single-user localhost — scalability is not a primary concern. However, these design choices matter for developer experience:

| Concern | At 1 user (localhost) | If multi-user (future) |
|---------|----------------------|------------------------|
| Model loading | Once at startup, shared | Per-process or model server |
| VRAM contention | No contention | CUDA OOM if concurrent |
| Optimization speed | Full GPU budget | Need queuing (not in scope) |
| State isolation | Single Gradio session | Session-scoped state required |

For the current scope: single-process, all models as module-level singletons, no concurrency handling needed.

---

## How Gradio Integrates with the Optimizer

Gradio 4.x `gr.Blocks` with generator-based event functions is the integration mechanism.

```
User clicks "Generate"
        |
        v
Gradio fires event handler (Python generator function)
        |
        v
Handler calls asr.transcribe() → text
        |
        v
Handler calls encoder.encode_text(text) → text_emb
        |
        v
Handler calls optimizer.optimize(..., callback=on_step)
        |  (synchronous loop inside optimizer)
        |
        +-- every 10 steps: callback(step, image, score)
        |       |
        |       v
        |   Gradio gr.Progress updates progress bar
        |   Gradio Image component updated with latest preview
        |   Gradio Number component updated with similarity score
        |
        v  (after 150 steps)
Handler returns (final_image, transcript)
        |
        v
Gradio renders final output panel
```

The key constraint: Gradio event handlers run in the same thread as the optimization loop. The callback approach (not threading) keeps this simple and avoids race conditions. Gradio's `yield` support in generator functions allows streaming intermediate results without explicit threading.

If the optimization loop blocks for 30-60 seconds without progress updates, the browser may show a stale state. The every-10-step callback prevents this.

---

## File Structure

```
EchoFace/
├── asr/
│   ├── __init__.py
│   └── transcriber.py          # WhisperTranscriber class
├── encoder/
│   ├── __init__.py
│   └── clip_encoder.py         # CLIPEncoder class (text + image)
├── generator/
│   ├── __init__.py
│   └── stylegan_wrapper.py     # StyleGANWrapper class
├── optimizer/
│   ├── __init__.py
│   └── latent_optimizer.py     # optimize() function
├── ui/
│   ├── __init__.py
│   └── app.py                  # Gradio Blocks app
├── utils/
│   ├── __init__.py
│   ├── audio_preprocess.py     # librosa preprocessing
│   ├── dataset_loader.py       # Kaggle Human Faces loader
│   ├── latent_directions.py    # Direction discovery from dataset
│   └── fid_scorer.py           # FID evaluation utility
├── StyleGAN-Human/             # Cloned repo (do not modify)
│   └── pretrained_models/
│       └── stylegan_human_v2_1024.pkl
├── data/
│   └── human_faces/            # Manually placed by user
├── environment.yml             # From StyleGAN-Human (base)
├── requirements_extra.txt      # openai-whisper, CLIP, Gradio 4.x
└── README.md
```

---

## Sources

- PROJECT.md (EchoFace project specification, 2026-03-06) — HIGH confidence
- CLIP-guided GAN optimization: established pattern from StyleCLIP (Patashnik et al., 2021), CLIPDraw, Text2Image-GAN literature — HIGH confidence for loop structure
- StyleGAN-Human architecture: https://github.com/stylegan-human/StyleGAN-Human — HIGH confidence for W-space API
- OpenAI CLIP API: https://github.com/openai/CLIP — HIGH confidence for encode_text/encode_image signatures
- Gradio 4.x generator functions: Gradio documentation — MEDIUM confidence (Gradio 4.x streaming via yield is documented but Gradio versions evolve rapidly; verify against actual installed version)
- W-space vs Z-space analysis: StyleGAN2 paper (Karras et al., 2020) — HIGH confidence
