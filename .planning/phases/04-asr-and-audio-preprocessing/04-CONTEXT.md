# Phase 4: ASR and Audio Preprocessing - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Capture audio from a microphone recording or audio file, preprocess it to 16kHz mono float32, transcribe via local Whisper medium model (no API key), unload Whisper from VRAM before the optimization loop begins, and expose a manual text bypass path. Does not include UI wiring or optimizer invocation — those are Phase 6.

</domain>

<decisions>
## Implementation Decisions

### Noise Filtering (audio_processor.py)
- Library: librosa only — no noisereduce or additional deps
- Silence trimming: librosa.effects.trim() with default top_db=20 (auto threshold)
- Amplitude normalization: peak normalize to [-1, 1] (divide by max absolute value)
- Resampling: to 16kHz mono float32
- Input formats: accept any format librosa supports (WAV, MP3, OGG, FLAC, etc.) — not restricted to WAV/MP3 only

### Error & Quality Handling
- Empty/short transcription (< 3 words): return text as-is, log WARNING — do not block the pipeline; manual text override is the safety net
- Corrupt or unsupported audio file: raise ValueError with a descriptive message (e.g., "Cannot load audio: unsupported format")
- Transcriber return type: plain text string only — no Whisper result dict, no segments, no language code

### VRAM Lifecycle (transcriber.py)
- Load strategy: load_model() at the start of each transcribe call, unload after — no singleton, no lazy caching
- Unload sequence: del model → gc.collect() → torch.cuda.empty_cache()
- VRAM verification: log torch.cuda.memory_allocated() before and after unload at DEBUG level (satisfies ASR-03 testability)
- Model size: "medium" hardcoded — not configurable

### Claude's Discretion
- Exact logging format and logger name
- gc import and call placement within unload sequence
- Internal helper structure within audio_processor.py

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils/gpu_utils.get_device_config()`: returns cfg dict with 'device', 'dtype', 'use_fp16', 'resolution', 'truncation_psi' — transcriber should accept this cfg dict for device selection (CPU vs CUDA)
- `encoder/clip_encoder.py`: CLIPEncoder class pattern — takes cfg dict, single responsibility, no state mutations after init
- `generator/stylegan_wrapper.py`: StyleGANWrapper class pattern — takes path + cfg dict, freezes model immediately after load

### Established Patterns
- Class-based modules taking cfg dict (CLIPEncoder, StyleGANWrapper) — transcriber.py should follow the same pattern: `Transcriber(cfg)` with a `transcribe(audio_array)` method
- `optimize()` in optimizer/clip_optimizer.py is a standalone function — audio_processor.py can use a standalone function `process_audio(path) -> np.ndarray` since it has no persistent state
- Mocking pattern: patch module-level namespace bindings (not top-level modules) — tests for transcriber should mock `asr.transcriber.whisper` not `whisper`
- PIL Image must be explicitly closed before tempdir cleanup on Windows (PermissionError) — audio temp files should follow same discipline

### Integration Points
- `asr/` directory is empty — both audio_processor.py and transcriber.py are new files
- `asr/__init__.py` needed (follows package pattern from encoder/ and optimizer/)
- Downstream caller (Phase 6 UI): calls transcriber, gets plain text string, passes to optimizer.clip_optimizer.optimize()
- VRAM sequencing: Whisper unloads → VRAM verified → StyleGANWrapper + optimizer run

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-asr-and-audio-preprocessing*
*Context gathered: 2026-03-07*
