---
phase: 1
slug: environment-and-repo-scaffold
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/conftest.py (Wave 0 installs) |
| **Quick run command** | `python -m pytest tests/test_env.py -v` |
| **Full suite command** | `python -m pytest tests/test_env.py tests/test_gpu_utils.py -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_env.py -v`
- **After every plan wave:** Run `python -m pytest tests/test_env.py tests/test_gpu_utils.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | ENV-03 | import | `python -m pytest tests/test_env.py::test_imports -v` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | ENV-03 | install | `python -m pytest tests/test_env.py::test_no_conflicts -v` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | ENV-02 | file | `python -m pytest tests/test_env.py::test_scaffold_dirs -v` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | ENV-04 | unit | `python -m pytest tests/test_gpu_utils.py::test_device_detection -v` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 1 | ENV-01 | integration | `python -m pytest tests/test_env.py::test_validation_script -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_env.py` — stubs for ENV-01, ENV-02, ENV-03
- [ ] `tests/test_gpu_utils.py` — stubs for ENV-04
- [ ] `tests/conftest.py` — shared fixtures (project root path, scaffold dir list)
- [ ] `pytest` install check — `pip install pytest` if not in base env

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `pip check` reports no conflicts after requirements_extra.txt install | ENV-03 | Subprocess output validation is environment-specific | Run `pip install -r requirements_extra.txt && pip check` in stylehuman conda env; confirm "No broken requirements" |
| GPU fp16 resolution reported as 1024px | ENV-04 | Requires actual CUDA-enabled GPU hardware | Run `python utils/gpu_utils.py` on GPU machine; confirm output shows `resolution: 1024, use_fp16: True` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
