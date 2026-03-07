---
phase: 04-asr-and-audio-preprocessing
plan: 02
subsystem: asr
tags: [whisper, asr, transcription, vram, tdd]

# Dependency graph
requires:
  - phase: 01-environment-and-repo-scaffold
    provides: gpu_utils.get_device_config() and stylehuman conda env with whisper installed
provides:
  - Transcriber(cfg) class with transcribe(audio_array) -> str — load-per-call Whisper medium
  - ASR-04 text bypass contract test — verifies caller can skip Transcriber by passing a str directly
affects:
  - 06-gradio-ui — will call Transcriber.transcribe() or bypass with manual text override

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Load-per-call: Whisper model loaded fresh each transcribe() call — no singleton cached on self"
    - "VRAM unload sequence locked to: del model -> gc.collect() -> torch.cuda.empty_cache()"
    - "Module-level imports in stub: patch() requires names in namespace, not just in implementation"

key-files:
  created:
    - asr/transcriber.py
    - tests/test_transcriber.py
  modified: []

key-decisions:
  - "Whisper load_model takes str device arg, not torch.device — str(cfg['device']) in __init__"
  - "VRAM logging guarded by torch.cuda.is_available() — DEBUG level, two calls per transcribe()"
  - "ASR-04 bypass is absence of call, not code in Transcriber — contract test verifies pipeline logic"
  - "Stub must import whisper at module level even with NotImplementedError body — patch() requires name in namespace"

patterns-established:
  - "Short transcription guard: warn if len(text.split()) < 3, return text unchanged"
  - "VRAM lifecycle: log before load, load, transcribe, del, gc, empty_cache, log after"

requirements-completed: [ASR-02, ASR-04]

# Metrics
duration: 4min
completed: 2026-03-07
---

# Phase 4 Plan 02: Transcriber Summary

**VRAM-safe Whisper medium ASR class with load-per-call lifecycle, CUDA memory logging, and text bypass contract test**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T10:26:18Z
- **Completed:** 2026-03-07T10:30:37Z
- **Tasks:** 2 (TDD: RED then GREEN)
- **Files modified:** 2

## Accomplishments
- `Transcriber(cfg)` class implementing local Whisper medium transcription with no API calls
- VRAM unload sequence (`del model -> gc.collect() -> torch.cuda.empty_cache()`) enforced in locked order
- CUDA memory usage logged at DEBUG level (before Whisper load and after unload) when CUDA available
- Short-transcription WARNING (fewer than 3 words) returns text unchanged — pipeline not blocked
- ASR-04 bypass contract verified: caller can pass `str` directly to pipeline without calling `transcribe()`
- All 6 tests in `TestTranscriber` GREEN; 62 other pre-existing tests unaffected

## Task Commits

Each task was committed atomically:

1. **Task 1: Test scaffold + stub (RED state)** - `3142aec` (test)
2. **Task 2: Implement Transcriber class (GREEN state)** - `9e9ec81` (feat)

_Note: TDD tasks have separate RED (test) and GREEN (feat) commits_

## Files Created/Modified
- `asr/transcriber.py` — Transcriber class with transcribe(audio_array) -> str, full VRAM lifecycle
- `tests/test_transcriber.py` — 6 tests: returns_string, loads_medium, vram_unload, vram_logging, short_transcription_warning, bypass_path

## Decisions Made
- `str(cfg['device'])` in `__init__` — Whisper's `load_model` takes a string ("cpu"/"cuda"), not a `torch.device` object
- Module-level imports (`import whisper`, `import torch`, `import gc`) in the stub — `@patch('asr.transcriber.whisper')` requires the name to exist in the module namespace before patching
- ASR-04 bypass is not implemented inside `Transcriber` — it is the absence of a `transcribe()` call. The contract test simulates the Phase 6 pipeline decision point (`if isinstance(text, str): ...`)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added module-level imports to stub**
- **Found during:** Task 1 (RED state verification)
- **Issue:** Original stub omitted `import whisper`, `import torch`, `import gc` — `@patch('asr.transcriber.whisper')` raised `AttributeError: module does not have attribute 'whisper'`
- **Fix:** Added the three imports to the stub at module level (same imports the full implementation needs)
- **Files modified:** `asr/transcriber.py`
- **Verification:** Tests collected cleanly, 5 failed with NotImplementedError (correct RED), 1 passed (bypass_path by design)
- **Committed in:** `3142aec` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — stub import bug)
**Impact on plan:** Necessary for test collection to work. Consistent with Phase 3 established pattern ("Stub files import clip/CLIPEncoder at module level even with NotImplementedError body").

## Issues Encountered
- `test_audio_processor.py::test_trim_silence` fails in full suite run — this is a pre-existing RED state from Plan 04-01 TDD scaffold, not caused by this plan. Full suite excluding that file: 62 passed, 1 skipped.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `Transcriber` class ready for integration with `audio_processor.py` (Plan 04-01 when GREEN) and Gradio UI (Phase 6)
- Text bypass path contract tested — Phase 6 can implement `if isinstance(text, str): ...` pattern as designed
- No blockers for Phase 4 completion

---
*Phase: 04-asr-and-audio-preprocessing*
*Completed: 2026-03-07*
