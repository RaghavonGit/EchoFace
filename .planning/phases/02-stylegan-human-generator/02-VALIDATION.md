---
phase: 2
slug: stylegan-human-generator
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — existing pytest.ini or inline markers |
| **Quick run command** | `pytest tests/test_stylegan_wrapper.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds (GPU tests auto-skip if no CUDA; CPU tests use mocks) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_stylegan_wrapper.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | GEN-01 | unit (stub) | `pytest tests/test_stylegan_wrapper.py -x -q` | Wave 0 | ⬜ pending |
| 2-01-02 | 01 | 1 | GEN-01 | unit (mock pkl) | `pytest tests/test_stylegan_wrapper.py::test_load_uses_legacy -x` | ✅ W0 | ⬜ pending |
| 2-01-03 | 01 | 1 | GEN-01 | unit (mock G) | `pytest tests/test_stylegan_wrapper.py::test_parameters_frozen -x` | ✅ W0 | ⬜ pending |
| 2-01-04 | 01 | 1 | GEN-01 | unit (mock G) | `pytest tests/test_stylegan_wrapper.py::test_synthesis_signature -x` | ✅ W0 | ⬜ pending |
| 2-02-01 | 02 | 2 | GEN-02 | unit (CUDA skip) | `pytest tests/test_stylegan_wrapper.py::test_gpu_sample_w_shape -x` | ✅ W0 | ⬜ pending |
| 2-02-02 | 02 | 2 | GEN-02 | integration (CUDA skip) | `pytest tests/test_stylegan_wrapper.py::test_gpu_generates_1024_png -x` | ✅ W0 | ⬜ pending |
| 2-02-03 | 02 | 2 | GEN-03 | unit (mock G, CPU) | `pytest tests/test_stylegan_wrapper.py::test_cpu_resize_to_256 -x` | ✅ W0 | ⬜ pending |
| 2-02-04 | 02 | 2 | GEN-03 | unit (mock G, CPU) | `pytest tests/test_stylegan_wrapper.py::test_cpu_generates_256_png -x` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_stylegan_wrapper.py` — stubs for GEN-01, GEN-02, GEN-03 (all tests initially `pytest.skip("not implemented")`)
- [ ] `generator/__init__.py` — empty package marker
- [ ] `ori_model.txt` added to `.gitignore` — side-effect of `legacy.load_network_pkl()`

*Wave 0 creates the test file with all test stubs before implementation begins.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Actual GPU synthesis produces visually valid face (not noise/corruption) | GEN-02 | Visual quality cannot be asserted programmatically | Run `python -c "from generator.stylegan_wrapper import StyleGANWrapper; import gpu_utils; w=StyleGANWrapper('StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl', gpu_utils.get_device_config()); w.generate_and_save()"` then open the PNG in outputs/ |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
