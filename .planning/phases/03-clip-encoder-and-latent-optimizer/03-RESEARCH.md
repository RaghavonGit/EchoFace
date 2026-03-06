# Phase 3: CLIP Encoder and Latent Optimizer - Research

**Researched:** 2026-03-06
**Domain:** CLIP embedding, W-space gradient optimization, PyTorch autograd stability
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Optimizer API shape**
- Entry point: `optimize(text_prompt, wrapper, cfg, callback=None) -> (PIL.Image, w_tensor, final_sim_score)`
- Single function in `optimizer/clip_optimizer.py` — no class wrapper needed
- Returns a 3-tuple: final PIL image, final W-space tensor, and final cosine similarity score
- `callback` is an optional keyword argument (default None); when provided it is called every 10 steps

**Progress callback payload**
- Callback signature: `callback(step: int, loss: float, sim_score: float)`
- No intermediate image in callback — avoids synthesis overhead at every checkpoint
- Phase 6 reads `final_sim_score` from the return tuple for CLIP-03 display

**NaN/instability recovery**
- On NaN detection: stop the loop immediately, synthesize from the best latent seen so far (highest cosine similarity across all completed steps), and return normally
- Signal instability via `logging.warning(...)` — no change to return type or function signature
- "Best latent" tracked by argmax of sim_score across completed steps

**CLIP image preprocessing**
- Use `clip.load()` built-in `preprocess` transform for image encoding — handles resize + center-crop + normalize internally (224x224 square crop from center of portrait)
- During the optimization loop: call `wrapper.synthesize(w)` to get uint8 HWC tensor, convert to PIL Image, then apply `clip.preprocess()` — gradient flows through W-space latent only (not through preprocessing)

**CLIPEncoder class**
- `encoder/clip_encoder.py` exposes a `CLIPEncoder` class (consistent with `StyleGANWrapper` pattern)
- CLIP model loads once on `__init__`; shared between `encode_text()` and `encode_image()` calls
- Constructor accepts `cfg` dict from `gpu_utils.get_device_config()`

**File locations**
- `encoder/clip_encoder.py` — CLIPEncoder class
- `optimizer/clip_optimizer.py` — `optimize()` function
- Both use existing Phase 1 scaffold directories

### Claude's Discretion
- Gradient clipping max_norm value (starting point: 1.0 per STATE.md blocker note)
- Regularization implementation details (L2 norm of w - w_init)
- Adam optimizer internal settings beyond lr=0.01
- Exact logging format for NaN warning

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLIP-01 | clip_encoder.py loads CLIP ViT-B/32 locally (no API key, no openai Python SDK), encodes text descriptions into normalized embeddings | CLIP API verified: `clip.load()` + `clip.tokenize()` + `model.encode_text()` + explicit L2-norm; local weights cache at `~/.cache/clip/` |
| CLIP-02 | clip_encoder.py encodes generated images into CLIP image embeddings for use in optimization loss computation | `model.encode_image()` + explicit L2-norm; `preprocess` transform handles PIL→tensor resize/crop/normalize |
| OPT-01 | Optimizer samples w_init from Gaussian distribution in W-space as the starting latent | `wrapper.sample_w()` (Phase 2) already does this; optimizer stores w_init for regularization |
| OPT-02 | clip_optimizer.py runs 150-step gradient descent on W-space latent: L_total = -Sim(E_text, E_image) + λ\|\|w - w_init\|\|² (λ=0.1, lr=0.01), generator frozen | Adam on `w` with `requires_grad=True`; CLIP model in `torch.no_grad()` or frozen; synthesis outside grad graph except through `w` |
| OPT-03 | Optimizer applies gradient clipping and NaN detection/recovery to prevent optimization instability | `clip_grad_norm_(max_norm=1.0)` + `torch.isnan(loss)` check; best-latent tracking via sim_score list; logging.warning on NaN |
</phase_requirements>

---

## Summary

Phase 3 implements two tightly coupled modules. The `CLIPEncoder` class wraps the local OpenAI CLIP ViT-B/32 model (installed from GitHub, not the openai Python SDK) and exposes `encode_text()` and `encode_image()` with explicit L2-normalization applied by the caller, not by CLIP internally. The `optimize()` function in `clip_optimizer.py` runs 150 Adam steps on the W-space latent, computing CLIP cosine similarity loss plus an L2 regularization term against w_init on each step.

The most important technical discovery concerns CLIP's internal precision. The CLIP model's `encode_text()` and `encode_image()` methods return **raw, unnormalized embeddings** — L2 normalization must be applied by the caller before computing cosine similarity. Additionally, CLIP loads its weights as FP16, and gradient flow through the CLIP model produces NaN values within a few steps. The correct mitigation is to freeze the CLIP model entirely (`requires_grad_(False)`) and run the optimization gradient only through the W-space latent; this is already implied by the architecture but must be explicitly enforced in the optimizer loop.

The gradient flows exclusively through `w` (the W-space latent). CLIP, StyleGAN-G, and the `preprocess` transform all sit outside the gradient graph. The only autograd-tracked operation is the loss computation that reads image embeddings as `.detach()` values and text embeddings as pre-computed constants.

**Primary recommendation:** Freeze CLIP model weights, normalize embeddings explicitly after `encode_*()`, and compute gradient only through W-space latent. Apply `clip_grad_norm_` with max_norm=1.0 before `optimizer.step()` and check `torch.isnan(loss)` at the top of each step.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| clip (openai/CLIP) | GitHub HEAD | Text and image embedding | Already installed in stylehuman env; local inference, no API key |
| torch.optim.Adam | PyTorch 1.9.1 | W-space latent optimizer | Standard for latent optimization; adaptive lr handles non-uniform gradient scales |
| torch.nn.utils.clip_grad_norm_ | PyTorch 1.9.1 | Gradient clipping | Built-in; prevents exploding gradients in CLIP-guided loops |
| PIL.Image | Pillow (installed) | uint8 HWC tensor → PIL for preprocess | Required by CLIP preprocess transform |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.nn.functional.cosine_similarity | PyTorch 1.9.1 | Similarity loss computation | After explicit L2-norm of text and image embeddings |
| logging (stdlib) | stdlib | NaN warning, step logging | NaN recovery signal and debug output |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Adam (lr=0.01) | SGD or LBFGS | Adam is locked by CONTEXT.md; LBFGS requires closure re-evaluation, higher per-step cost |
| clip_grad_norm_ (L2) | clip_grad_value_ | Norm clipping preserves gradient direction; value clipping does not |

**Installation:** Already installed in the `stylehuman` conda environment. No new packages needed for Phase 3.

---

## Architecture Patterns

### Recommended Project Structure

```
encoder/
└── clip_encoder.py      # CLIPEncoder class — load once, encode many times

optimizer/
├── clip_optimizer.py    # optimize() function — main optimization loop
└── attribute_directions.py  # Phase 5, do not touch

tests/
├── test_clip_encoder.py     # Wave 0 gap — must create
└── test_clip_optimizer.py   # Wave 0 gap — must create
```

### Pattern 1: CLIPEncoder Class

**What:** Load CLIP ViT-B/32 once, freeze weights, expose `encode_text()`/`encode_image()` with explicit L2-normalization applied by caller.

**When to use:** Whenever embedding is needed. Single instance shared between optimizer and any future callers.

**Critical:** `encode_text()` and `encode_image()` in the CLIP library return **unnormalized** embeddings. The `forward()` method normalizes them, but calling `encode_*()` directly skips normalization. Callers must apply `F.normalize(feat, dim=-1)` explicitly.

**Example:**
```python
# Source: github.com/openai/CLIP + verified against model.py
import clip
import torch
import torch.nn.functional as F
from PIL import Image

class CLIPEncoder:
    def __init__(self, cfg: dict) -> None:
        self.device = cfg['device']
        # jit=False required when CLIP weights may be cast to float32
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # Freeze CLIP — gradients must NOT flow through CLIP weights
        # FP16 CLIP weights overflow immediately under gradient updates (github.com/openai/CLIP/issues/40)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, text: str) -> torch.Tensor:
        """Returns L2-normalized text embedding, shape [1, 512], float32."""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
        return F.normalize(feat.float(), dim=-1)

    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Returns L2-normalized image embedding, shape [1, 512], float32."""
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img_tensor)
        return F.normalize(feat.float(), dim=-1)
```

### Pattern 2: Optimization Loop Structure

**What:** Adam gradient descent on W-space latent, with CLIP and StyleGAN generator completely outside the gradient graph.

**When to use:** The core of `optimize()`.

**Gradient flow diagram:**
```
w (requires_grad=True)
  └─→ wrapper.synthesize(w)     [StyleGAN synthesis — gradients DO flow through w here]
        └─→ PIL.Image            [uint8 conversion — gradient STOPS here]
              └─→ preprocess()   [CLIP transform — no grad, outside graph]
                    └─→ encoder.encode_image()  [CLIP model frozen — returns detached tensor]
                          └─→ cosine_similarity(text_feat, img_feat)  [loss scalar]
                                └─→ loss.backward()                    [grad accumulates on w]
```

**Important:** `synthesize()` returns `uint8` tensor on CPU. The uint8 conversion (`clamp(0,255).to(torch.uint8)`) **breaks the gradient graph**. This means gradient does NOT flow back through the synthesis → but wait — the loss is computed from the **CLIP embedding of the rendered image**, not from the raw synthesis output. The gradient flows through w → G.synthesis → (no grad through uint8/PIL/CLIP-preprocess/frozen-CLIP). This means the gradient through the loss IS zero unless a differentiable rendering path is used.

**Resolution of this critical issue:** The W-space optimization gradient cannot flow through `synthesize()` → PIL → preprocess → encode_image() because of the uint8 conversion and detached CLIP model. The correct approach used in StyleCLIP and similar works is one of:
1. Use the synthesis output **before** the uint8 clamp to compute a differentiable image feature.
2. Compute the CLIP loss using the raw float output of `G.synthesis()` directly (bypassing the uint8 path), then run the PIL/CLIP path separately as a monitoring step.

The CONTEXT.md decision states: "Gradient flows through W-space latent only (not through preprocessing)." This correctly describes approach (2): the float synthesis tensor feeds through a differentiable path for gradient computation, while the uint8 PIL path is used only for the callback image and final output.

**Differentiable preprocessing pattern (for loss computation only):**
```python
# Source: StyleCLIP approach — differentiable path for gradient; no uint8 conversion
# Run inside the gradient context:
img_float = wrapper.G.synthesis(w, noise_mode='const', force_fp32=True)
# img_float: [1, 3, H, W] float32, range [-1, 1]

# Resize to 224x224 for CLIP using differentiable interpolation
img_clip = F.interpolate(img_float, size=(224, 224), mode='bicubic', align_corners=False)
# Normalize to CLIP expected range: mean=(0.481, 0.457, 0.408) std=(0.268, 0.261, 0.275)
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
img_norm = ((img_float * 0.5 + 0.5) - mean) / std  # convert [-1,1] → [0,1] → normalize
img_clip = F.interpolate(img_norm, size=(224, 224), mode='bicubic', align_corners=False)

img_feat = encoder.model.encode_image(img_clip)   # CLIP frozen, but differentiable forward pass
img_feat = F.normalize(img_feat.float(), dim=-1)
```

This is the **key architecture insight**: the optimizer must call `G.synthesis()` directly (not `wrapper.synthesize()`) for the gradient path, then separately call `wrapper.synthesize()` → PIL for the final image and for monitoring/callback images.

### Pattern 3: NaN Recovery

**What:** Detect NaN loss, stop loop, return from best seen latent.

```python
# Source: PyTorch docs + github.com/openai/CLIP/issues/40 mitigation
best_w = w_init.clone().detach()
best_sim = -1.0
sim_history = []

for step in range(n_steps):
    optimizer.zero_grad()

    # Differentiable CLIP loss path (see Pattern 2)
    loss = compute_loss(w, text_feat, encoder, wrapper, cfg, w_init, lam)

    if torch.isnan(loss):
        logging.warning(
            f"NaN loss at step {step}; stopping early and returning best latent "
            f"(step {sim_history.index(best_sim)}, sim={best_sim:.4f})"
        )
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        sim = compute_sim(w, text_feat, encoder, wrapper)
    sim_history.append(sim)
    if sim > best_sim:
        best_sim = sim
        best_w = w.detach().clone()

    if callback is not None and (step + 1) % 10 == 0:
        callback(step=step + 1, loss=loss.item(), sim_score=sim)
```

### Anti-Patterns to Avoid

- **Using `wrapper.synthesize()` in the gradient path:** The uint8 conversion breaks the graph. Use `G.synthesis()` directly for the loss computation.
- **Calling `encode_text()`/`encode_image()` without L2-normalizing:** These methods return raw projections; cosine similarity computed on unnormalized vectors gives wrong values.
- **Letting CLIP model have `requires_grad=True`:** FP16 CLIP weights overflow under gradient updates (confirmed: github.com/openai/CLIP/issues/40). Always freeze CLIP immediately after load.
- **Storing `w_init` as a leaf variable with `requires_grad=True`:** `w_init` must be detached; only `w` (the optimization variable) carries gradients.
- **Clipping gradients AFTER `optimizer.step()`:** `clip_grad_norm_` must be called BEFORE `optimizer.step()` and AFTER `loss.backward()`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image resize + center-crop + normalize for CLIP | Custom PIL transform | CLIP's built-in `preprocess` (for final output/callback) | CLIP's transform uses exact mean/std values baked into model training |
| Differentiable 224×224 resize | Manual loop | `F.interpolate(size=(224,224), mode='bicubic')` | Built-in PyTorch, gradient-safe, handles batches |
| Cosine similarity | Manual dot product | `F.cosine_similarity()` or explicit L2-norm then dot | Handles edge cases, numerically stable |
| Gradient clipping | Manual norm computation | `torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)` | Correct in-place mutation, handles None gradients |
| Text tokenization | Manual tokenizer | `clip.tokenize([text])` | CLIP-specific BPE tokenizer with 77-token context limit |

**Key insight:** The preprocessing pipeline has exact numeric constants baked into the CLIP model weights. Any custom preprocessing that differs by even one pixel-level transform will produce incorrect similarity scores.

---

## Common Pitfalls

### Pitfall 1: encode_text/encode_image Return Unnormalized Embeddings

**What goes wrong:** Cosine similarity computed on raw CLIP embeddings gives values in non-standard range; loss does not behave correctly.

**Why it happens:** The CLIP library's `forward()` normalizes, but `encode_text()` and `encode_image()` do NOT normalize. This is an easy mistake because the model docs show the full forward pass with normalization.

**How to avoid:** Always apply `F.normalize(feat.float(), dim=-1)` after any `encode_*()` call.

**Warning signs:** Cosine similarity values outside [0, 1] or not starting near 0 at step 0.

### Pitfall 2: CLIP FP16 Weights + Gradient Flow = NaN

**What goes wrong:** Gradients through the CLIP model (even if you don't intend them) cause NaN within 1-5 steps.

**Why it happens:** CLIP weights are loaded in FP16. Gradient updates through FP16 weights overflow/underflow without loss scaling, producing NaN.

**How to avoid:** Freeze CLIP immediately after `clip.load()` with `requires_grad_(False)` on all parameters. Use `torch.no_grad()` when calling `encoder.model.encode_image()` for monitoring.

**Warning signs:** NaN loss at step 1 or 2, or `clip_grad_norm_` returning nan/inf before the first step.

### Pitfall 3: Gradient Graph Broken by uint8 Conversion

**What goes wrong:** Optimizer runs 150 steps without error, but generated image never changes (gradient is always zero).

**Why it happens:** `wrapper.synthesize()` calls `.to(torch.uint8)` which is non-differentiable; the gradient graph is severed before the loss computation.

**How to avoid:** Call `wrapper.G.synthesis(w, noise_mode='const', force_fp32=True)` directly inside the gradient context for the loss path. Use `wrapper.synthesize()` only for final image output and monitoring (outside gradient context).

**Warning signs:** Loss is nonzero but `w` never changes from `w_init`; `w.grad` is None or all-zero after `loss.backward()`.

### Pitfall 4: w_init Carries Gradients

**What goes wrong:** Regularization loss `||w - w_init||²` propagates gradients back through `w_init`, corrupting the optimization objective.

**Why it happens:** `w_init = wrapper.sample_w()` returns a tensor that came through `G.mapping()` which has `requires_grad=False` for its parameters but the output tensor may still be part of a computation graph.

**How to avoid:** Always `w_init = wrapper.sample_w().detach().clone()` before creating the optimizer variable `w`.

**Warning signs:** Both `w` and `w_init` move together during optimization; regularization loss grows unexpectedly.

### Pitfall 5: CLIP Tokenizer 77-Token Limit

**What goes wrong:** `clip.tokenize()` silently truncates text descriptions longer than 77 tokens; the truncated portion of the prompt is ignored.

**Why it happens:** CLIP's text transformer has a fixed context length of 77 tokens.

**How to avoid:** Keep text prompts under 77 tokens (typical face descriptions are 10-30 tokens). No mitigation needed for Phase 3's simple prompts.

**Warning signs:** Long prompts produce same embedding as their first 77 tokens.

---

## Code Examples

Verified patterns from official sources:

### CLIP Load and Tokenize
```python
# Source: github.com/openai/CLIP (README)
import clip

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokens = clip.tokenize(["a photo of a person with brown hair"]).to(device)
```

### Explicit L2-Normalization (Required After encode_*)
```python
# Source: github.com/openai/CLIP/blob/main/clip/model.py — forward() shows normalization
# encode_text/encode_image do NOT normalize; caller must normalize explicitly
import torch.nn.functional as F

text_feat_raw = model.encode_text(tokens)
text_feat = F.normalize(text_feat_raw.float(), dim=-1)  # shape [1, 512], L2-norm = 1.0

img_feat_raw = model.encode_image(img_tensor)
img_feat = F.normalize(img_feat_raw.float(), dim=-1)    # shape [1, 512], L2-norm = 1.0

# Cosine similarity: dot product of unit vectors
sim = (text_feat * img_feat).sum(dim=-1)  # scalar in [-1, 1]
```

### Differentiable Synthesis Path for Gradient
```python
# Source: StyleCLIP approach (github.com/orpatashnik/StyleCLIP) + CLIP normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=device).view(1,3,1,1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

def synth_for_clip(w, wrapper, device):
    """Differentiable synthesis → CLIP-ready tensor. Gradient flows through w."""
    # G.synthesis raw float output: [1, 3, H, W], range approx [-1, 1]
    img = wrapper.G.synthesis(w, noise_mode='const', force_fp32=True)
    # Convert to [0, 1]
    img = (img * 0.5 + 0.5).clamp(0, 1)
    # Resize to 224x224 (CLIP input size)
    img = F.interpolate(img, size=(224, 224), mode='bicubic', align_corners=False)
    # Normalize with CLIP training stats
    img = (img - CLIP_MEAN) / CLIP_STD
    return img
```

### Full Optimizer Loop Skeleton
```python
# Source: derived from OPT-02/OPT-03 requirements + above patterns
def optimize(text_prompt, wrapper, cfg, callback=None):
    device = cfg['device']
    encoder = CLIPEncoder(cfg)

    # Pre-compute text embedding (constant throughout optimization)
    text_feat = encoder.encode_text(text_prompt)  # [1, 512], normalized, no grad

    # W-space starting latent
    w_init = wrapper.sample_w().detach()
    w = w_init.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=0.01)

    best_w = w_init.clone()
    best_sim = -float('inf')

    for step in range(150):
        optimizer.zero_grad()

        # Differentiable path: grad flows through w → G.synthesis → CLIP
        img_clip = synth_for_clip(w, wrapper, device)
        img_feat_raw = encoder.model.encode_image(img_clip)
        img_feat = F.normalize(img_feat_raw.float(), dim=-1)

        sim = (text_feat * img_feat).sum(dim=-1)
        reg = ((w - w_init) ** 2).sum()
        loss = -sim + 0.1 * reg

        if torch.isnan(loss):
            logging.warning(f"NaN at step {step}; returning best latent (sim={best_sim:.4f})")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
        optimizer.step()

        sim_val = sim.item()
        if sim_val > best_sim:
            best_sim = sim_val
            best_w = w.detach().clone()

        if callback is not None and (step + 1) % 10 == 0:
            callback(step=step + 1, loss=loss.item(), sim_score=sim_val)

    # Render final image using wrapper.synthesize (non-differentiable, safe)
    with torch.no_grad():
        img_hwc = wrapper.synthesize(best_w)
    final_image = Image.fromarray(img_hwc.numpy(), 'RGB')
    return final_image, best_w, best_sim
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct PIL → CLIP preprocess for gradient | `F.interpolate` + manual CLIP normalization in float32 for gradient path | Established in StyleCLIP 2021 | Allows gradient to flow through synthesis → CLIP |
| Optimize in Z-space | Optimize in W-space (W+) | StyleGAN2 2020 | Better disentanglement, more stable optimization |
| No regularization | L2 reg on ||w - w_init||² | StyleCLIP 2021 | Prevents collapse to adversarial latents |
| CLIP model fine-tuned with gradients | CLIP frozen, only latent optimized | Established practice | Prevents CLIP FP16 NaN failures |

**Deprecated/outdated:**
- Optimizing StyleGAN Z-space: entangled representation; W-space is standard and required by project scope
- Using CLIP `forward()` output directly for similarity: logit scaling makes this inappropriate for gradient optimization; use `encode_*()` + manual norm instead

---

## Open Questions

1. **Gradient magnitude through `G.synthesis()` in fp32 mode**
   - What we know: `force_fp32=True` is used; synthesis output is float32
   - What's unclear: Whether the gradient magnitude through the full synthesis network at step 0 is large enough to produce useful updates, or whether it vanishes in the deep layers of StyleGAN-Human v2
   - Recommendation: In the first integration test, log `w.grad.norm()` at step 1. If < 1e-6, the learning rate may need adjustment upward.

2. **CLIP preprocess mean/std constants accuracy**
   - What we know: CLIP documentation and model code reference constants (0.48145466, 0.4578275, 0.40821073) / (0.26862954, 0.26130258, 0.27577711)
   - What's unclear: Whether these exact values are present in the installed version of CLIP in the stylehuman env
   - Recommendation: Read them from `encoder.preprocess.transforms[-1].mean` rather than hardcoding. If unavailable, use the constants above.

3. **Portrait aspect ratio effect on CLIP similarity**
   - What we know: StyleGAN-Human v2 outputs 512×1024 portrait; CLIP expects 224×224 square via center-crop
   - What's unclear: Whether center-cropping a 512×1024 portrait to 224×224 retains the face region (likely yes for full-body, uncertain for face prominence)
   - Recommendation: The center-crop from `preprocess` will extract the torso/upper body region for a 512×1024 portrait. The differentiable path using `F.interpolate` to 224×224 squashes the portrait — may reduce face visibility. Acceptable for Phase 3 (hardcoded prompt, not UI-driven). Flag for Phase 6 to consider aspect-ratio-preserving crop.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (installed in stylehuman env) |
| Config file | none — uses `conftest.py` at project root |
| Quick run command | `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py tests/test_clip_optimizer.py -x -q` |
| Full suite command | `conda run -n stylehuman python -m pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLIP-01 | `CLIPEncoder` loads ViT-B/32, `encode_text()` returns normalized [1,512] tensor | unit (mock) | `pytest tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_text_shape -x` | ❌ Wave 0 |
| CLIP-01 | Text embedding is L2-normalized (norm ≈ 1.0) | unit (mock) | `pytest tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_text_normalized -x` | ❌ Wave 0 |
| CLIP-02 | `encode_image()` accepts PIL Image, returns normalized [1,512] tensor | unit (mock) | `pytest tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_image_shape -x` | ❌ Wave 0 |
| CLIP-02 | Image embedding is L2-normalized (norm ≈ 1.0) | unit (mock) | `pytest tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_image_normalized -x` | ❌ Wave 0 |
| OPT-01 | `optimize()` calls `wrapper.sample_w()` to get w_init | unit (mock) | `pytest tests/test_clip_optimizer.py::TestOptimize::test_calls_sample_w -x` | ❌ Wave 0 |
| OPT-02 | Optimizer runs exactly 150 steps (no NaN case) | unit (mock) | `pytest tests/test_clip_optimizer.py::TestOptimize::test_runs_150_steps -x` | ❌ Wave 0 |
| OPT-02 | Returns 3-tuple (PIL.Image, tensor, float) | unit (mock) | `pytest tests/test_clip_optimizer.py::TestOptimize::test_return_tuple -x` | ❌ Wave 0 |
| OPT-02 | Generator parameters remain frozen after optimization | unit (mock) | `pytest tests/test_clip_optimizer.py::TestOptimize::test_generator_frozen -x` | ❌ Wave 0 |
| OPT-03 | NaN loss triggers early stop and returns best latent | unit (mock) | `pytest tests/test_clip_optimizer.py::TestOptimize::test_nan_recovery -x` | ❌ Wave 0 |
| OPT-03 | Callback called at steps 10, 20, ..., 150 with correct signature | unit (mock) | `pytest tests/test_clip_optimizer.py::TestOptimize::test_callback_frequency -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py tests/test_clip_optimizer.py -x -q`
- **Per wave merge:** `conda run -n stylehuman python -m pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_clip_encoder.py` — covers CLIP-01, CLIP-02
- [ ] `tests/test_clip_optimizer.py` — covers OPT-01, OPT-02, OPT-03
- [ ] `encoder/clip_encoder.py` — implementation file (currently empty scaffold)
- [ ] `optimizer/clip_optimizer.py` — implementation file (currently empty scaffold)

---

## Sources

### Primary (HIGH confidence)
- `github.com/openai/CLIP` — clip.load(), tokenize(), encode_text(), encode_image() API signatures
- `github.com/openai/CLIP/blob/main/clip/model.py` — confirmed encode_* returns unnormalized embeddings; forward() normalizes
- `github.com/openai/CLIP/issues/40` — FP16 gradient NaN root cause and FP32 freeze solution
- PyTorch 1.9.1 docs — `torch.nn.utils.clip_grad_norm_`, `torch.optim.Adam`, `F.normalize`, `F.interpolate`
- `D:\EchoFace\generator\stylegan_wrapper.py` — confirmed `synthesize()` returns uint8 (gradient-breaking); `G.synthesis()` interface
- `D:\EchoFace\gpu_utils.py` — confirmed cfg dict keys: device, dtype, use_fp16, resolution, truncation_psi

### Secondary (MEDIUM confidence)
- `github.com/orpatashnik/StyleCLIP` — optimization approach (differentiable synthesis path, L2 regularization pattern)
- `github.com/l4rz/stylegan2-clip-approach` — learning rate 0.02 for portraits (reference data point; lr=0.01 is locked)
- CLIP ViT-B/32 normalization constants (0.48145466 / 0.26862954 etc.) — referenced across multiple sources; recommend reading from `preprocess.transforms` at runtime

### Tertiary (LOW confidence)
- `neptune.ai/blog/understanding-gradient-clipping` — max_norm=1.0 as starting point; empirical, project-specific tuning needed

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — CLIP install verified in stylehuman env; PyTorch 1.9.1 confirmed working
- Architecture: HIGH — encode_* normalization behavior confirmed from model.py source; uint8 gradient break confirmed from stylegan_wrapper.py source; CLIP FP16 NaN cause confirmed from official issue
- Pitfalls: HIGH for normalization/FP16/uint8 (verified); MEDIUM for max_norm=1.0 (empirical starting point)

**Research date:** 2026-03-06
**Valid until:** 2026-06-06 (stable APIs; CLIP and PyTorch 1.9.1 are frozen in stylehuman env)
