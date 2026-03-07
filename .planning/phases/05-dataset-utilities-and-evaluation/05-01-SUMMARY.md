---
phase: 05-dataset-utilities-and-evaluation
plan: 01
subsystem: utils
tags: [tdd, dataset, fid, metrics, stubs, red]
dependency_graph:
  requires: []
  provides: [utils/dataset_loader.py, utils/metrics.py, tests/test_dataset_loader.py, tests/test_metrics.py]
  affects: [05-02-PLAN.md]
tech_stack:
  added: []
  patterns: [unittest.mock.patch, module-level imports for mock patching, TDD RED state]
key_files:
  created:
    - utils/dataset_loader.py
    - utils/metrics.py
    - tests/test_dataset_loader.py
    - tests/test_metrics.py
  modified: []
decisions:
  - "gpu_utils imported as top-level module (import gpu_utils) not utils.gpu_utils — gpu_utils.py lives at project root, not inside utils/"
  - "Module-level imports in stubs are mandatory — patch() resolves names at import time, not at call time"
metrics:
  duration: "~8 minutes"
  completed: "2026-03-07"
  tasks_completed: 2
  files_created: 4
  files_modified: 0
---

# Phase 05 Plan 01: Dataset Loader and FID Metrics — TDD Stubs and RED Tests Summary

**One-liner:** TDD RED scaffolding — two module stubs (NotImplementedError) plus 9 failing unit tests covering load_dataset and compute_fid contracts.

## What Was Built

Two stub modules and two test files establishing the TDD RED state for Phase 5 dataset utilities.

### utils/dataset_loader.py
- Imports `Path` and `Image` at module level (required for mock patching)
- Defines `SUPPORTED_EXTENSIONS` set
- `load_dataset(path, max_images=None)` raises `NotImplementedError`

### utils/metrics.py
- Imports `Path`, `calculate_fid_given_paths`, and `gpu_utils` at module level
- `compute_fid(real_dir, gen_dir='outputs/')` raises `NotImplementedError`
- CLI entry point via `if __name__ == '__main__'`

### tests/test_dataset_loader.py (5 tests — all RED)
- `test_missing_dir_raises` — FileNotFoundError when dir missing
- `test_empty_dir_raises` — FileNotFoundError when no supported extensions found
- `test_skips_unreadable` — 2 images returned, skip count printed when 1 of 3 unreadable
- `test_max_images` — result capped to max_images=3 from 5 available
- `test_returns_pil_list` — returns list of 2 PIL images

### tests/test_metrics.py (4 tests — all RED)
- `test_raises_below_2048` — ValueError with "2048" in message when < 2048 real images
- `test_num_workers_zero` — calculate_fid_given_paths called with num_workers=0
- `test_returns_float` — result is a float
- `test_raises_empty_gen_dir` — raises ValueError or FileNotFoundError for empty gen_dir

## Verification

All 9 tests fail with `NotImplementedError` — RED state confirmed.
Zero `ImportError` or `SyntaxError` in test run.

```
FAILED tests/test_dataset_loader.py::TestLoadDataset::test_empty_dir_raises
FAILED tests/test_dataset_loader.py::TestLoadDataset::test_max_images
FAILED tests/test_dataset_loader.py::TestLoadDataset::test_missing_dir_raises
FAILED tests/test_dataset_loader.py::TestLoadDataset::test_returns_pil_list
FAILED tests/test_dataset_loader.py::TestLoadDataset::test_skips_unreadable
FAILED tests/test_metrics.py::TestComputeFID::test_num_workers_zero
FAILED tests/test_metrics.py::TestComputeFID::test_raises_below_2048
FAILED tests/test_metrics.py::TestComputeFID::test_raises_empty_gen_dir
FAILED tests/test_metrics.py::TestComputeFID::test_returns_float
9 failed in 10.80s
```

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add dataset_loader and metrics stubs | 22d67db |
| 2 | Add failing RED tests for both modules | 4d38d3e |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect gpu_utils import path in metrics stub**
- **Found during:** Task 1 verification
- **Issue:** Plan specified `from utils.gpu_utils import get_device_config` but `gpu_utils.py` lives at the project root (not inside `utils/`). Import caused `ModuleNotFoundError`.
- **Fix:** Changed to `import gpu_utils` matching the pattern used by `generator/stylegan_wrapper.py`
- **Files modified:** `utils/metrics.py`
- **Commit:** 22d67db

## Self-Check: PASSED

Files exist:
- FOUND: utils/dataset_loader.py
- FOUND: utils/metrics.py
- FOUND: tests/test_dataset_loader.py
- FOUND: tests/test_metrics.py

Commits exist:
- FOUND: 22d67db (stubs)
- FOUND: 4d38d3e (RED tests)

All 9 tests RED with NotImplementedError. Zero ImportError or SyntaxError.
