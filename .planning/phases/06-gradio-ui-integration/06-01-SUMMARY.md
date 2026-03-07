---
phase: 06-gradio-ui-integration
plan: 01
subsystem: testing
tags: [gradio, tdd, unittest, pytest, ui, generate_fn, export_image]

# Dependency graph
requires:
  - phase: 05-dataset-utilities-and-evaluation
    provides: complete pipeline (ASR, optimizer, generator) ready for UI wiring
provides:
  - RED test scaffold with 8 unit tests for generate_fn and export_image
  - Acceptance gate for two-stage ASR UX flow (test_asr_path)
  - Acceptance gate for UI-01, UI-02, CLIP-03 requirements
affects:
  - 06-02 (Plan 02 must turn these 8 tests GREEN)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TDD RED scaffold: module-level import fails cleanly on missing ui.app; no try/except
    - patch namespace ui.app.optimize, ui.app.process_audio, ui.app.Transcriber
    - _fake_optimize side_effect calls callback synchronously to exercise queue.Queue bridge

key-files:
  created:
    - tests/test_gradio_app.py
  modified: []

key-decisions:
  - "generate_fn accepts wrapper and cfg as explicit parameters (not closed-over globals) — required for mockability without importing Gradio demo object"
  - "test_asr_path asserts optimize() call_count == 0 — acceptance gate for two-stage UX: ASR fills text box, user reviews, second Generate click runs optimizer"
  - "Module-level import at line 12 fails at collection time (not runtime deferred) — produces clean ModuleNotFoundError RED state without try/except wrapper"

patterns-established:
  - "Pattern: patch ui.app.* namespace bindings (not top-level module names) — follows established project pattern from test_clip_optimizer.py and test_transcriber.py"
  - "Pattern: _fake_optimize side_effect calls callback synchronously — enables queue.Queue bridge testing without threading"

requirements-completed: [UI-01, UI-02, CLIP-03]

# Metrics
duration: 2min
completed: 2026-03-07
---

# Phase 6 Plan 01: Gradio UI Integration RED Test Scaffold Summary

**8-test RED scaffold for generate_fn and export_image covering two-stage ASR UX flow, progress callbacks, exception handling, and PNG export**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-07T15:55:55Z
- **Completed:** 2026-03-07T15:57:46Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `tests/test_gradio_app.py` with 8 test methods across TestGenerate (6) and TestExport (2)
- Established acceptance gate for two-stage ASR UX: test_asr_path asserts transcription fills text box and optimize() is never called in the ASR pass
- Confirmed RED state: `py_compile` exits 0 (no syntax errors), pytest exits non-zero with `ModuleNotFoundError: No module named 'ui.app'`

## Task Commits

Each task was committed atomically:

1. **Task 1: Write RED test scaffold for generate_fn and export_image** - `4231ef6` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `tests/test_gradio_app.py` - 8-test RED scaffold covering UI-01, UI-02, CLIP-03

## Decisions Made

- `generate_fn` accepts `wrapper` and `cfg` as explicit parameters (not module-level globals) so tests can mock without importing the Gradio demo object. This constrains Plan 02's implementation signature.
- `test_asr_path` asserts `mock_opt.call_count == 0` — this is the acceptance gate for the two-stage UX flow locked in CONTEXT.md: ASR pass fills the text box and stops; user reviews; second Generate click runs `optimize()`.
- Module-level import (`from ui.app import generate_fn, export_image`) fails at collection time, producing a clean `ModuleNotFoundError` RED state — no try/except wrapper needed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- RED test scaffold is complete; Plan 02 can now implement `ui/app.py` targeting these 8 tests
- Verify command for Plan 02: `conda run -n stylehuman python -m pytest tests/test_gradio_app.py -v`
- All 8 tests expected to turn GREEN after Plan 02 creates `ui/app.py` with `generate_fn` and `export_image`

---
*Phase: 06-gradio-ui-integration*
*Completed: 2026-03-07*
