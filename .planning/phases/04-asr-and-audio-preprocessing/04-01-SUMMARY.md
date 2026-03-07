---
phase: 04-asr-and-audio-preprocessing
plan: 01
subsystem: asr
tags: [librosa, audio, preprocessing, numpy, tdd, unittest]

# Dependency graph
requires:
  - phase: 01-environment-and-repo-scaffold
    provides: conda env stylehuman with librosa 0.9.2 installed
provides:
  - process_audio(path) -> np.ndarray stateless audio preprocessing function at 16kHz mono float32
  - asr/__init__.py package marker enabling asr.audio_processor imports
  - tests/test_audio_processor.py with 5 unit tests (TestProcessAudio), all GREEN
affects:
  - 04-02 (ASR transcriber — calls process_audio before passing array to Whisper)
  - 06-gradio-ui (UI file-upload path calls process_audio on temp file path)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stateless pure function pattern: process_audio() has no class, no singleton, no cfg dict"
    - "Test mock pattern: @patch('asr.audio_processor.librosa') patches namespace binding, not top-level module"
    - "Near-zero guard: max_val > 1e-6 check before division prevents NaN on silence"

key-files:
  created:
    - asr/__init__.py
    - asr/audio_processor.py
    - tests/test_audio_processor.py
  modified: []

key-decisions:
  - "process_audio() is stateless pure function — no class, no cfg, downstream callers own the path string"
  - "Silence guard threshold 1e-6: below this level audio is treated as near-silent and normalization skipped"
  - "test_trim_silence assertIn fixed to avoid numpy boolean ambiguity — filter numpy arrays from call_args.args before assertIn check"

patterns-established:
  - "ASR module mock pattern: patch 'asr.audio_processor.librosa' not top-level 'librosa'"
  - "Audio pipeline: load (sr=16000, mono=True) -> trim (top_db=20) -> peak normalize with guard"

requirements-completed: [ASR-01, ASR-03]

# Metrics
duration: 5min
completed: 2026-03-07
---

# Phase 4 Plan 01: Audio Preprocessor Summary

**Stateless process_audio() function using librosa — 16kHz mono float32 with silence trim and peak normalization, 5 unit tests all GREEN**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-07T10:26:12Z
- **Completed:** 2026-03-07T10:30:47Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created `asr/__init__.py` package marker enabling `from asr.audio_processor import process_audio`
- Implemented `process_audio(path)` with librosa load+resample (16kHz mono), silence trim (top_db=20), and peak normalization with near-zero guard
- Wrote 5 unit tests in `TestProcessAudio` — all pass GREEN using `@patch('asr.audio_processor.librosa')` mock pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Package marker + test scaffold (RED state)** - `ac8bb11` (test)
2. **Task 2: Implement process_audio() — GREEN state** - `28548c2` (feat)

_Note: TDD tasks — RED commit then GREEN commit_

## Files Created/Modified
- `asr/__init__.py` - Empty package marker (enables asr.audio_processor import)
- `asr/audio_processor.py` - process_audio() implementation: load, trim, normalize
- `tests/test_audio_processor.py` - 5-test TestProcessAudio class, all GREEN

## Decisions Made
- process_audio() is a pure stateless function — no class wrapper, no cfg dict argument. The downstream caller (Phase 6 UI) owns the file path and passes it directly.
- Silence guard threshold `1e-6`: audio with max absolute amplitude below this is returned as-is (near-zero) to avoid NaN from division.
- Fixed `test_trim_silence` assertion: the original plan's `call_kwargs.args or list(call_kwargs.kwargs.values())` expression raises `ValueError` when `call_kwargs.args` contains a numpy array (Python boolean evaluation is ambiguous). Fixed by explicitly separating numpy arrays from scalar positional args before the assertIn check.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_trim_silence assertIn — numpy array boolean ambiguity**
- **Found during:** Task 2 (Implement process_audio() — GREEN state)
- **Issue:** The plan's test assertion `self.assertIn(20, call_kwargs.args or list(call_kwargs.kwargs.values()))` raises `ValueError: The truth value of an array with more than one element is ambiguous` because `call_kwargs.args` is a tuple containing a numpy array. Python's `or` requires boolean evaluation of the tuple contents.
- **Fix:** Replaced with explicit split: `kwargs_values = list(call_kwargs.kwargs.values()); scalar_args = [a for a in call_kwargs.args if not isinstance(a, np.ndarray)]; self.assertIn(20, kwargs_values + scalar_args)`
- **Files modified:** `tests/test_audio_processor.py`
- **Verification:** All 5 tests pass GREEN including test_trim_silence
- **Committed in:** `28548c2` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test assertion)
**Impact on plan:** The fix preserves the test's intent exactly — it still verifies top_db=20 is passed as either a positional or keyword argument. No scope change.

## Issues Encountered
- The `conda run -c` flag cannot accept multiline Python scripts (conda assertion error on newlines in args) — debugged the numpy boolean issue from the pytest traceback directly rather than via interactive Python.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `process_audio()` is ready to call from Phase 4 Plan 02 (ASR transcriber)
- Pattern: `audio_array = process_audio(file_path)` then pass to `whisper_model.transcribe(audio_array)`
- No blockers

---
*Phase: 04-asr-and-audio-preprocessing*
*Completed: 2026-03-07*
