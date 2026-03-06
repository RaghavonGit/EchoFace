# Phase 3: CLIP Encoder and Latent Optimizer - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement CLIP-guided W-space latent optimization: `encoder/clip_encoder.py` (CLIP-01, CLIP-02) + `optimizer/clip_optimizer.py` (OPT-01, OPT-02, OPT-03). A hardcoded text prompt drives 150 gradient descent steps producing a face semantically aligned with the prompt. ASR, UI, and dataset utilities are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Optimizer API shape
- Entry point: `optimize(text_prompt, wrapper, cfg, callback=None) -> (PIL.Image, w_tensor, final_sim_score)`
- Single function in `optimizer/clip_optimizer.py` — no class wrapper needed
- Returns a 3-tuple: final PIL image, final W-space tensor (for Phase v2 attribute editing), and final cosine similarity score (for Phase 6 CLIP-03 display)
- `callback` is an optional keyword argument (default None); when provided it is called every 10 steps

### Progress callback payload
- Callback signature: `callback(step: int, loss: float, sim_score: float)`
- No intermediate image in callback — avoids synthesis overhead at every checkpoint
- Phase 6 reads `final_sim_score` from the return tuple for CLIP-03 display; it does not need to track callback state

### NaN/instability recovery
- On NaN detection: stop the loop immediately, synthesize from the best latent seen so far (highest cosine similarity across all completed steps), and return normally
- Signal instability via `logging.warning(...)` — no change to return type or function signature
- "Best latent" tracked by argmax of sim_score across completed steps

### CLIP image preprocessing
- Use `clip.load()` built-in `preprocess` transform for image encoding — handles resize + center-crop + normalize internally (224x224 square crop from center of portrait)
- During the optimization loop: call `wrapper.synthesize(w)` to get uint8 HWC tensor, convert to PIL Image, then apply `clip.preprocess()` — gradient flows through W-space latent only (not through preprocessing)

### CLIPEncoder class
- `encoder/clip_encoder.py` exposes a `CLIPEncoder` class (consistent with `StyleGANWrapper` pattern)
- CLIP model loads once on `__init__`; shared between `encode_text()` and `encode_image()` calls
- Constructor accepts `cfg` dict from `gpu_utils.get_device_config()`

### File locations
- `encoder/clip_encoder.py` — CLIPEncoder class
- `optimizer/clip_optimizer.py` — `optimize()` function
- Both use existing Phase 1 scaffold directories

### Claude's Discretion
- Gradient clipping max_norm value (starting point: 1.0 per STATE.md blocker note)
- Regularization implementation details (L2 norm of w - w_init)
- Adam optimizer internal settings beyond lr=0.01
- Exact logging format for NaN warning

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `StyleGANWrapper.sample_w()`: generates w_init via `G.mapping()` from a random z — optimizer calls this to get the starting latent (satisfies OPT-01)
- `StyleGANWrapper.synthesize(w)`: renders W+ latent to uint8 HWC tensor on CPU — optimizer calls this each step to get the image for CLIP encoding
- `gpu_utils.get_device_config()`: returns `{device, dtype, use_fp16, resolution, truncation_psi}` — CLIPEncoder and optimizer both consume this dict

### Established Patterns
- Class-based modules: `StyleGANWrapper` takes `(pkl_path, cfg)` — CLIPEncoder should take `(cfg)` similarly
- Generator frozen via `requires_grad_(False)` in `__init__` — already done, optimizer must not re-enable it
- `sys.path` injection for StyleGAN-Human done in `stylegan_wrapper.py` — CLIPEncoder does not need this, CLIP installs cleanly via pip
- Test mocks: patch module-level namespace bindings (not top-level modules); use `MagicMock` for GPU objects

### Integration Points
- `optimizer/clip_optimizer.py` imports `encoder.clip_encoder.CLIPEncoder` and accepts a `StyleGANWrapper` instance
- `StyleGANWrapper.synthesize()` is called inside the optimization loop to render each candidate latent
- Phase 6 (`ui/`) will call `optimize(text, wrapper, cfg, callback=ui_callback)` and display `final_sim_score`

</code_context>

<specifics>
## Specific Ideas

No specific UI references or third-party examples — open to standard approaches for the optimization loop structure.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-clip-encoder-and-latent-optimizer*
*Context gathered: 2026-03-06*
