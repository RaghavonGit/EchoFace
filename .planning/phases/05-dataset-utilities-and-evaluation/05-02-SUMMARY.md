---
phase: 05-dataset-utilities-and-evaluation
plan: 02
subsystem: utils
tags: [tdd, dataset, fid, metrics, green, PIL, pytorch-fid]
dependency_graph:
  requires:
    - phase: 05-01
      provides: "Stub files and RED tests for dataset_loader and metrics"
  provides:
    - "utils/dataset_loader.py — load_dataset() fully implemented, 5/5 tests GREEN"
    - "utils/metrics.py — compute_fid() fully implemented, 4/4 tests GREEN"
  affects: [06-gradio-ui-integration]
tech_stack:
  added: []
  patterns:
    - "sorted(key=str) for deterministic order when PIL Path mocks don't support < comparison"
    - "sys.path injection in utils/ module for project-root imports when run as script"
    - "Path.glob('*') for image counting (no PIL loading, avoids MemoryError on 13k images)"
    - ".convert('RGB').copy() chain — palette-mode handling + Windows file handle release"
key-files:
  created: []
  modified:
    - utils/dataset_loader.py
    - utils/metrics.py
key-decisions:
  - "sorted(key=str) instead of plain sorted() — MagicMock objects don't support < comparison, key=str provides stable sort for both real paths and mocks"
  - "sys.path injection at module level in utils/metrics.py — enables direct CLI execution (python utils/metrics.py) in addition to pytest import path"
  - "num_workers=0 mandatory for calculate_fid_given_paths — Windows raises BrokenPipeError with any higher value"
  - "compute_fid() is a pure function with no print() — CLI block handles printing, function stays testable"
patterns-established:
  - "Path.glob('*') for counting: count image files without loading them into RAM"
  - "project-root sys.path guard: scripts in subdirectories inject _PROJECT_ROOT for sibling module imports"
requirements-completed: [DATA-01, EVAL-01]
duration: 4min
completed: 2026-03-07
---

# Phase 05 Plan 02: Dataset Loader and FID Metrics — Implementations Summary

**load_dataset() with sorted RGB-converted images and compute_fid() with pytorch-fid (num_workers=0) turning 9 RED tests GREEN**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T13:26:13Z
- **Completed:** 2026-03-07T13:30:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced both NotImplementedError stubs with full implementations
- All 9 phase tests GREEN: 5 dataset_loader + 4 metrics
- Full suite clean: 76 passed, 1 skipped, 0 failed across all prior phases
- CLI `python utils/metrics.py --help` reachable with argparse output

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement load_dataset()** - `6c4751f` (feat)
2. **Task 2: Implement compute_fid()** - `545298c` (feat)

## Files Created/Modified

- `utils/dataset_loader.py` — load_dataset() with sorted(key=str), .convert('RGB').copy() chain, skip counting, max_images cap, FileNotFoundError on missing/empty dirs
- `utils/metrics.py` — compute_fid() with _count_images() glob helper, 2048-image guard, empty gen_dir guard, calculate_fid_given_paths(num_workers=0), argparse CLI block, sys.path injection

## Decisions Made

- Used `sorted(key=str)` instead of plain `sorted()` — discovered during first test run that MagicMock objects don't support `<` comparison. `key=str` produces stable sort for real Path objects (`str(Path(...))` gives the full path string) and works with mocks (MagicMock.__str__ returns something unique per mock).
- Added `sys.path` injection to `utils/metrics.py` — when running `python utils/metrics.py real gen` directly (not via pytest), Python adds `utils/` to sys.path but NOT the project root, so `import gpu_utils` fails. The injection follows the same pattern as `generator/stylegan_wrapper.py`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Used sorted(key=str) instead of plain sorted()**
- **Found during:** Task 1 (dataset_loader implementation)
- **Issue:** `sorted(MagicMock, MagicMock)` raises `TypeError: '<' not supported between instances of 'MagicMock' and 'MagicMock'` — 3 of 5 tests failed on first run
- **Fix:** Changed `sorted(...)` to `sorted(..., key=str)` — str() works on both real Path objects and MagicMock instances
- **Files modified:** `utils/dataset_loader.py`
- **Verification:** All 5 dataset_loader tests pass after fix
- **Committed in:** `6c4751f` (Task 1 commit)

**2. [Rule 3 - Blocking] Added sys.path injection for CLI execution**
- **Found during:** Task 2 verification (CLI --help check)
- **Issue:** `python utils/metrics.py --help` failed with `ModuleNotFoundError: No module named 'gpu_utils'` because Python's module search path doesn't include the project root when running a file in a subdirectory
- **Fix:** Added `_PROJECT_ROOT` sys.path injection at top of `utils/metrics.py` following the same pattern as `generator/stylegan_wrapper.py`
- **Files modified:** `utils/metrics.py`
- **Verification:** `python utils/metrics.py --help` prints argparse help without error
- **Committed in:** `545298c` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes required for correctness. sorted(key=str) was necessary for tests to pass at all; sys.path injection was necessary for CLI verification step. No scope creep.

## Issues Encountered

None beyond the two deviation items above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `load_dataset()` and `compute_fid()` are production-ready
- Phase 6 (Gradio UI) can import `load_dataset` for dataset preview and `compute_fid` for evaluation display
- FID scoring requires the full Kaggle Human Faces Dataset (>2048 images) in `data/human_faces/` — smaller dataset returns ValueError with download instructions

## Self-Check: PASSED

Files exist:
- FOUND: utils/dataset_loader.py
- FOUND: utils/metrics.py
- FOUND: .planning/phases/05-dataset-utilities-and-evaluation/05-02-SUMMARY.md

Commits exist:
- FOUND: 6c4751f (feat(05-02): implement load_dataset())
- FOUND: 545298c (feat(05-02): implement compute_fid())

All 9 tests GREEN. Full suite: 76 passed, 1 skipped, 0 failed.

---
*Phase: 05-dataset-utilities-and-evaluation*
*Completed: 2026-03-07*
