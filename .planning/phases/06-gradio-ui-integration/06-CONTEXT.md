# Phase 6: Gradio UI Integration - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire all proven components (ASR, optimizer, export) into a localhost Gradio app. A developer runs `python ui/app.py`, speaks or types a face description, watches the pipeline run, sees the final CLIP similarity score, and exports the result as a timestamped PNG — all offline. No attribute sliders, no cloud calls.

Requirements covered: UI-01, UI-02, CLIP-03.

</domain>

<decisions>
## Implementation Decisions

### Input trigger
- Single "Generate" button triggers the pipeline — no auto-start on upload or recording stop
- User sets up all inputs, then explicitly clicks Generate

### Input mode precedence
- Three input sources: mic recording (`gr.Audio(source="microphone")`), WAV/MP3 file upload, manual text box
- **If text box is non-empty:** fully bypass ASR — no Whisper load, no audio preprocessing; use text directly
- **If text box is empty and audio is provided:** run ASR (process_audio → Transcriber.transcribe), display transcription result in the text box so user can review and edit before the optimizer runs
- **Mic vs file upload conflict:** not expected simultaneously in Gradio 3.x (single Audio component handles both sources); if both somehow present, file upload takes precedence
- Mic component: `gr.Audio(source="microphone")` — returns audio file path compatible with `process_audio()`

### VRAM sequencing
- Unchanged from Phase 4 decision: Whisper unloads (del model → gc.collect() → torch.cuda.empty_cache()) before optimizer runs
- Text-only path skips Whisper entirely — no VRAM impact

### Progress display
- Callback fires every 10 steps with `(step, loss, sim_score)` — Phase 3 locked this signature
- Display live CLIP similarity score updating every 10 steps during optimization
- No intermediate image previews during optimization (avoids synthesis overhead; Phase 3 decision)
- After optimization completes: display final image + final cosine similarity score (CLIP-03)

### Attribute direction sliders
- Out of scope for Phase 6 — deferred to v2
- ATTR-01 (latent direction discovery via SVM/PCA) was explicitly deferred in Phase 5
- No stub slider UI — keep the interface clean and functional

### Export behavior
- Export button saves the generated image as a timestamped PNG to `outputs/`
- Filename format: `outputs/YYYYMMDD_HHMMSS.png` (Claude's discretion on exact format)
- Button appears after generation completes; not visible/enabled before that

### Error handling
- **Empty input (no audio, no text):** show inline error message — "Please provide a description or record audio." — keep button enabled for immediate retry
- **Short ASR result (< 3 words):** display the transcription in the text box with a visible warning label; user edits before generating — do not block the pipeline
- **Optimization failure (exception mid-run):** catch all exceptions, display a readable error message in the UI (e.g., "Generation failed: out of memory"), keep Gradio app alive for retry
- **CPU-only (no CUDA):** on app launch, detect CUDA availability and display a startup notice — "No GPU detected — generation will be slow (CPU mode, 256x256)"

### Claude's Discretion
- Exact Gradio layout (tab structure, column arrangement)
- Progress bar vs step counter display during optimization
- Exact warning label styling for short ASR result
- Timestamp format for export filename
- Logger name and log level for UI events

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `optimizer/clip_optimizer.py`: `optimize(text_prompt, wrapper, cfg, callback=None) -> (PIL.Image, w_tensor, final_sim_score)` — Phase 3, ready to call
- `asr/transcriber.py`: `Transcriber(cfg).transcribe(audio_array) -> str` — Phase 4, handles VRAM lifecycle internally
- `asr/audio_processor.py`: `process_audio(path) -> np.ndarray` — Phase 4, accepts any librosa-supported format
- `utils/gpu_utils.py`: `get_device_config()` — returns `{device, dtype, use_fp16, resolution, truncation_psi}`; single source of truth for device selection
- `generator/stylegan_wrapper.py`: `StyleGANWrapper(pkl_path, cfg)` — Phase 2, loads and freezes generator
- `outputs/` — scaffolded in Phase 1, export target directory

### Established Patterns
- All modules accept `cfg` dict from `get_device_config()` — UI should call this once at startup and pass cfg to StyleGANWrapper, Transcriber, and optimize()
- Errors raised as exceptions (ValueError from process_audio, FileNotFoundError from dataset_loader) — UI should catch at call site and display message
- PIL Image returned by optimize() — convert to numpy array for gr.Image display (`numpy.array(pil_image)`)
- `legacy.load_network_pkl()` writes `ori_model.txt` to CWD unconditionally — gitignored, no action needed

### Integration Points
- `ui/app.py` is the new file — `ui/` directory exists with `.gitkeep` placeholder
- App startup: call `get_device_config()`, instantiate `StyleGANWrapper`, display GPU/CPU startup notice
- Generate button callback: text bypass check → optional ASR path → `optimize()` with progress callback → display image + sim score
- Export button callback: `PIL.Image.save(timestamped_path)` from the stored result

</code_context>

<specifics>
## Specific Ideas

No specific UI references or design examples — open to standard Gradio 3.x layout approaches.

</specifics>

<deferred>
## Deferred Ideas

- Attribute direction sliders (Age, Smile, Facial Hair, Glasses, Pose) — requires ATTR-01 (latent direction discovery); v2 scope
- FID scoring button in UI — Phase 5 noted Gradio could call compute_fid(); deferred, not in Phase 6 success criteria

</deferred>

---

*Phase: 06-gradio-ui-integration*
*Context gathered: 2026-03-07*
