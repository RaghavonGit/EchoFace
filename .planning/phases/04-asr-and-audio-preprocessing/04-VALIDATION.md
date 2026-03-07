---
phase: 4
slug: asr-and-audio-preprocessing
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing — installed in `stylehuman` env, used Phases 1-3) |
| **Config file** | none — `conftest.py` handles `sys.path` (already exists) |
| **Quick run command** | `conda run -n stylehuman --no-capture-output python -m pytest tests/test_audio_processor.py tests/test_transcriber.py -x -q` |
| **Full suite command** | `conda run -n stylehuman --no-capture-output python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds (all mocked — no model loading) |

---

## Sampling Rate

- **After every task commit:** Run `conda run -n stylehuman --no-capture-output python -m pytest tests/test_audio_processor.py tests/test_transcriber.py -x -q`
- **After every plan wave:** Run `conda run -n stylehuman --no-capture-output python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | ASR-01, ASR-03 | unit | `pytest tests/test_audio_processor.py -x -q` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 1 | ASR-01 | unit | `pytest tests/test_audio_processor.py::TestProcessAudio::test_resamples_16k -x` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 1 | ASR-01 | unit | `pytest tests/test_audio_processor.py::TestProcessAudio::test_trim_silence -x` | ❌ W0 | ⬜ pending |
| 4-01-04 | 01 | 1 | ASR-01 | unit | `pytest tests/test_audio_processor.py::TestProcessAudio::test_peak_normalize -x` | ❌ W0 | ⬜ pending |
| 4-01-05 | 01 | 1 | ASR-01 | unit | `pytest tests/test_audio_processor.py::TestProcessAudio::test_invalid_file -x` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 0 | ASR-02, ASR-04 | unit | `pytest tests/test_transcriber.py -x -q` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 1 | ASR-02 | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_returns_string -x` | ❌ W0 | ⬜ pending |
| 4-02-03 | 02 | 1 | ASR-02 | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_loads_medium -x` | ❌ W0 | ⬜ pending |
| 4-02-04 | 02 | 1 | ASR-02 | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_vram_unload -x` | ❌ W0 | ⬜ pending |
| 4-02-05 | 02 | 1 | ASR-02 | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_short_transcription_warning -x` | ❌ W0 | ⬜ pending |
| 4-02-06 | 02 | 1 | ASR-04 | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_bypass_path -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `asr/__init__.py` — empty package marker (enables `from asr.audio_processor import process_audio`)
- [ ] `tests/test_audio_processor.py` — stubs for ASR-01, ASR-03
- [ ] `tests/test_transcriber.py` — stubs for ASR-02, ASR-04

*Framework and `tests/conftest.py` already exist from Phase 1.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| VRAM actually drops ~5 GB after transcription on GPU machine | ASR-02 | `torch.cuda.memory_allocated()` returns 0 in CPU-only CI; requires real GPU | Run `python -c "from asr.transcriber import Transcriber; from utils.gpu_utils import get_device_config; import numpy as np; t=Transcriber(get_device_config()); import torch; print(torch.cuda.memory_allocated()); t.transcribe(np.zeros(16000,dtype=np.float32)); print(torch.cuda.memory_allocated())"` and verify second value is lower |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
