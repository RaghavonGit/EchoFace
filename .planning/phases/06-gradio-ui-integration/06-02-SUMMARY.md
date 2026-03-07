---
phase: 06-gradio-ui-integration
plan: 02
subsystem: ui
tags: [gradio, ui, generate_fn, export_image, threading, queue, asr, two-stage-ux]

# Dependency graph
requires:
  - phase: 06-01
    provides: RED test scaffold (8 tests in tests/test_gradio_app.py)
  - phase: 05-dataset-utilities-and-evaluation
    provides: complete pipeline (ASR, optimizer, generator) ready for UI wiring
provides:
  - ui/app.py — complete Gradio 3.50.2 Blocks application
  - generate_fn() — generator wiring ASR + optimizer with two-stage UX flow
  - export_image() — PNG export with timestamped filename
  - _build_demo() — testable Blocks constructor (no pkl load at import time)
affects:
  - Phase 6 end-to-end: app is now launchable with `python ui/app.py`

# Tech tracking
tech-stack:
  added: []
  patterns:
    - threading + queue.Queue bridge for live CLIP similarity yield during optimization
    - functools.partial to bind wrapper/cfg into Gradio event handler (avoids globals)
    - Top-level imports for process_audio and Transcriber (not lazy) — required for patch() to intercept in tests
    - text_in component appears in both inputs= and outputs= of gen_btn.click() — two-stage ASR UX

key-files:
  created:
    - ui/app.py
  modified: []

key-decisions:
  - "Top-level imports for process_audio and Transcriber required — lazy import inside function would not be intercepted by patch('ui.app.process_audio') at test collection time"
  - "ui/.gitkeep kept alongside app.py — test_env.py::test_gitkeep_exists[ui] asserts its presence; removing it breaks Phase 1 scaffold test"
  - "_build_demo() separated from __main__ block — StyleGANWrapper pkl load deferred to launch time, not import time"

requirements-completed: [UI-01, UI-02, CLIP-03]

# Metrics
duration: 4min
completed: 2026-03-07
---

# Phase 6 Plan 02: Gradio UI Integration — Implement ui/app.py Summary

**Complete Gradio 3.50.2 Blocks app wiring ASR, CLIP optimizer, and StyleGAN generator into a single localhost UI with two-stage ASR UX flow and live score updates**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T16:00:46Z
- **Completed:** 2026-03-07T16:04:42Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Created `ui/app.py` (230 lines) implementing `generate_fn()`, `export_image()`, and `_build_demo()`
- All 8 tests in `tests/test_gradio_app.py` turned GREEN
- Full test suite passes: 84 passed, 1 skipped, 0 failures
- Two-stage ASR UX flow: ASR pass fills text box and stops; user reviews; second Generate click runs `optimize()`
- Threading + `queue.Queue` bridge enables live CLIP similarity score yields during 150-step optimization
- `_build_demo()` constructs Gradio Blocks without triggering StyleGANWrapper pkl load (import-safe)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement generate_fn() and export_image()** - `f79b71e` (feat)
2. **Task 2: Smoke-test app launch and pipeline import chain** - no new commit (verification only — working tree clean)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `ui/app.py` - Complete Gradio Blocks application (230 lines)

## Decisions Made

- Top-level imports for `process_audio` and `Transcriber` required — the plan mentioned lazy imports inside the ASR branch, but `patch('ui.app.process_audio')` requires the name to exist in the `ui.app` namespace at patch time. Top-level imports satisfy both requirements.
- `ui/.gitkeep` kept alongside `app.py` — `test_env.py::test_gitkeep_exists[ui]` asserts its presence. The plan said to remove it, but removing it breaks a Phase 1 scaffold test that is still part of the full suite.
- `_build_demo()` separated from `__main__` block so tests can call it with mock wrapper/cfg without triggering the 5-second pkl load.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Kept ui/.gitkeep to avoid test regression**
- **Found during:** Task 2 (full test suite run)
- **Issue:** Plan said to remove `ui/.gitkeep`; `test_env.py::TestScaffold::test_gitkeep_exists[ui]` asserts the file exists. Removing it caused 1 test failure.
- **Fix:** Restored `ui/.gitkeep` alongside `app.py`. Both files coexist without conflict.
- **Files modified:** `ui/.gitkeep` (restored)
- **Commit:** no separate commit — working tree unchanged after restore

**2. [Rule 2 - Missing critical functionality] Top-level imports instead of lazy imports**
- **Found during:** Task 1 (test design analysis)
- **Issue:** Plan specified lazy imports of `process_audio` and `Transcriber` inside the ASR branch. Tests use `patch('ui.app.process_audio')` which requires these names in the `ui.app` namespace at patch time, not just at call time.
- **Fix:** Imported `process_audio` and `Transcriber` at module top level alongside other imports.
- **Files modified:** `ui/app.py`

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 6 is complete. `ui/app.py` is the final deliverable.
- Launch command: `conda activate stylehuman && python ui/app.py` (from `D:\EchoFace`)
- App serves at `http://127.0.0.1:7860`

---
*Phase: 06-gradio-ui-integration*
*Completed: 2026-03-07*
