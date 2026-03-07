# Phase 5: Dataset Utilities and Evaluation - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement two utilities: `utils/dataset_loader.py` loads the Kaggle Human Faces dataset as a list of PIL images, and `utils/metrics.py` computes a FID score between the real dataset and generated outputs using pytorch-fid. Attribute direction discovery (ATTR-01) is v2 scope — not in this phase.

Requirements covered: DATA-01, EVAL-01.

</domain>

<decisions>
## Implementation Decisions

### Image loading interface
- `load_dataset(path, max_images=None) -> List[PIL.Image]`
- Returns a list of PIL Images (not tensors, not paths) — matches how CLIPEncoder receives images and lets pytorch-fid apply its own preprocessing
- Return original image sizes — no resize (pytorch-fid / InceptionV3 resizes internally; double-resizing degrades FID)
- `max_images=None` by default loads all images; caller can cap for fast dev runs (e.g., max_images=2048 during testing)
- No shuffle — return in filesystem order for determinism and testability

### FID invocation
- `compute_fid(real_dir, gen_dir='outputs/') -> float` — callable function that returns the score, prints nothing
- CLI block: `if __name__ == '__main__'` with argparse accepting two positional args: `python utils/metrics.py data/human_faces/ outputs/`
- CLI block prints the returned float; function stays pure
- `gen_dir` defaults to `'outputs/'` — zero-config for standard use

### Missing dataset behavior
- When `data/human_faces/` is absent or empty: print a human-readable message then raise `FileNotFoundError`
  - Message: `"Dataset not found at {path}. Download the Kaggle 'Human Faces Dataset' by kaustubhdhote and place images there."`
- Invalid/unreadable image files: skip silently, continue loading valid ones, print `"Skipped N unreadable files"` at end
- Both behaviors are testable (exception type is catchable; Gradio UI in Phase 6 can catch FileNotFoundError and show a warning instead of crashing)

### 2048-image minimum check
- Lives in `metrics.py`, not in `dataset_loader`
- `compute_fid()` asserts `len(real_images) >= 2048` before invoking pytorch-fid, with a clear message
- `dataset_loader` stays general-purpose — the 2048 threshold is a FID evaluation concern, not a loader concern

### Claude's Discretion
- Exact error message wording for the 2048-image assertion
- How pytorch-fid is called internally (directory-based vs in-memory tensor path)
- Test mock strategy for PIL.Image.open

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils/gpu_utils.py` — `get_device_config()` returns device/dtype cfg dict; metrics.py may use `cfg['device']` for InceptionV3 placement
- `encoder/clip_encoder.py` — example of how PIL images are consumed; dataset_loader should return the same PIL.Image type

### Established Patterns
- Pure functions that raise on error, callers catch — matches `process_audio()` and `CLIPEncoder` patterns
- `unittest.mock.patch` on module-level namespace bindings — test mocks patch `utils.dataset_loader.PIL` / `utils.metrics.pytorch_fid` not top-level
- Stateless pure functions preferred (no class, no cfg dict in dataset_loader — caller owns the path)
- `outputs/` directory already scaffolded in Phase 1 — default gen_dir path will always exist

### Integration Points
- `metrics.py` consumes output of `dataset_loader.load_dataset()` for the real images side
- Generated images come from `outputs/` — Phase 2 saves to this directory
- Phase 6 (Gradio UI) may call `compute_fid()` as a function after generation completes; `FileNotFoundError` from loader must be catchable

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for pytorch-fid integration.

</specifics>

<deferred>
## Deferred Ideas

- ATTR-01: Latent attribute direction discovery (SVM/PCA on W-space embeddings) — v2 scope, not Phase 5
- STATE.md blocker note: "SVM vs PCA for latent direction discovery is unresolved for StyleGAN-Human v2 specifically" — flag for Phase 5.1 or v2 planning

</deferred>

---

*Phase: 05-dataset-utilities-and-evaluation*
*Context gathered: 2026-03-07*
