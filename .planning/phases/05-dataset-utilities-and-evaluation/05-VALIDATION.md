---
phase: 5
slug: dataset-utilities-and-evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | unittest (Python stdlib, Python 3.8.5 compatible) |
| **Config file** | none — pytest discovers tests/ automatically |
| **Quick run command** | `conda run -n stylehuman python -m pytest tests/test_dataset_loader.py tests/test_metrics.py -v` |
| **Full suite command** | `conda run -n stylehuman python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds (all mocked, no real images or GPU) |

---

## Sampling Rate

- **After every task commit:** Run `conda run -n stylehuman python -m pytest tests/test_dataset_loader.py tests/test_metrics.py -v`
- **After every plan wave:** Run `conda run -n stylehuman python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 0 | DATA-01 | unit | `python -m pytest tests/test_dataset_loader.py -v` | ❌ W0 | ⬜ pending |
| 5-01-02 | 01 | 0 | EVAL-01 | unit | `python -m pytest tests/test_metrics.py -v` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 1 | DATA-01 | unit | `python -m pytest tests/test_dataset_loader.py -v` | ✅ W0 | ⬜ pending |
| 5-01-04 | 01 | 1 | EVAL-01 | unit | `python -m pytest tests/test_metrics.py -v` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_dataset_loader.py` — 5 tests covering DATA-01 (missing dir, empty dir, skips unreadable, max_images, returns PIL list)
- [ ] `tests/test_metrics.py` — 4 tests covering EVAL-01 (below 2048, num_workers=0, returns float, empty gen_dir)
- [ ] `utils/dataset_loader.py` — stub with `load_dataset` importing `Image` from `PIL` and `Path` from `pathlib` at module level (required for mock patching)
- [ ] `utils/metrics.py` — stub with `compute_fid` importing `calculate_fid_given_paths` from `pytorch_fid.fid_score` and `Path` from `pathlib` at module level (required for mock patching)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| FID score plausibility with real dataset | EVAL-01 | Requires 2048+ real images on disk (not available in CI) | Run `conda run -n stylehuman python utils/metrics.py data/human_faces/ outputs/` after populating dataset; score should be a positive float |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
