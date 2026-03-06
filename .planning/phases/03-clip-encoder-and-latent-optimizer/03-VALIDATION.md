---
phase: 3
slug: clip-encoder-and-latent-optimizer
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed in stylehuman conda env) |
| **Config file** | none — uses `conftest.py` at project root |
| **Quick run command** | `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py tests/test_clip_optimizer.py -x -q` |
| **Full suite command** | `conda run -n stylehuman python -m pytest tests/ -q` |
| **Estimated runtime** | ~15 seconds (mocked — no GPU needed) |

---

## Sampling Rate

- **After every task commit:** Run `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py tests/test_clip_optimizer.py -x -q`
- **After every plan wave:** Run `conda run -n stylehuman python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | CLIP-01, CLIP-02 | unit (mock) | `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py -x -q` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 0 | OPT-01, OPT-02, OPT-03 | unit (mock) | `conda run -n stylehuman python -m pytest tests/test_clip_optimizer.py -x -q` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 1 | CLIP-01 | unit (mock) | `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_text_shape tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_text_normalized -x -q` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 1 | CLIP-02 | unit (mock) | `conda run -n stylehuman python -m pytest tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_image_shape tests/test_clip_encoder.py::TestCLIPEncoder::test_encode_image_normalized -x -q` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 1 | OPT-01, OPT-02 | unit (mock) | `conda run -n stylehuman python -m pytest tests/test_clip_optimizer.py::TestOptimize::test_calls_sample_w tests/test_clip_optimizer.py::TestOptimize::test_runs_150_steps tests/test_clip_optimizer.py::TestOptimize::test_return_tuple tests/test_clip_optimizer.py::TestOptimize::test_generator_frozen -x -q` | ❌ W0 | ⬜ pending |
| 3-03-02 | 03 | 1 | OPT-03 | unit (mock) | `conda run -n stylehuman python -m pytest tests/test_clip_optimizer.py::TestOptimize::test_nan_recovery tests/test_clip_optimizer.py::TestOptimize::test_callback_frequency -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_clip_encoder.py` — stubs for CLIP-01, CLIP-02
- [ ] `tests/test_clip_optimizer.py` — stubs for OPT-01, OPT-02, OPT-03
- [ ] `encoder/clip_encoder.py` — implementation scaffold (currently empty)
- [ ] `optimizer/clip_optimizer.py` — implementation scaffold (currently empty)

*All test files missing — Wave 0 must create them before implementation begins.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CLIP similarity increases (or is stable) over 150 steps | OPT-02 | Requires real GPU + StyleGAN + CLIP inference; not mockable | Run `conda run -n stylehuman python -c "from optimizer.clip_optimizer import optimize; from generator.stylegan_wrapper import StyleGANWrapper; from gpu_utils import get_device_config; cfg=get_device_config(); w=StyleGANWrapper(cfg); sims=[]; img,_,score=optimize('a young woman with brown hair',w,cfg,callback=lambda step,loss,sim_score: sims.append(sim_score)); print('final sim:', score, 'trend:', sims)"` and verify `score > sims[0]` |
| No NaN raised on 15 callback checkpoints (steps 10,20,...,150) | OPT-03 | Depends on real synthesis path | Same integration run above — verify no NaN warning in logs and all 15 callback invocations complete |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
