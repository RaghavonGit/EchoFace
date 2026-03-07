---
phase: 6
slug: gradio-ui-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (via unittest) |
| **Config file** | none — Wave 0 creates test file |
| **Quick run command** | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py -x -q` |
| **Full suite command** | `conda run -n stylehuman python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `conda run -n stylehuman python -m pytest tests/test_gradio_app.py -x -q`
- **After every plan wave:** Run `conda run -n stylehuman python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | UI-01, UI-02, CLIP-03 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py -x -q` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 1 | UI-01 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_text_bypass -x` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 | 1 | UI-01 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_asr_path -x` | ❌ W0 | ⬜ pending |
| 6-02-03 | 02 | 1 | UI-01 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_empty_input -x` | ❌ W0 | ⬜ pending |
| 6-02-04 | 02 | 1 | UI-01 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_exception_handling -x` | ❌ W0 | ⬜ pending |
| 6-02-05 | 02 | 1 | CLIP-03 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_score_yielded -x` | ❌ W0 | ⬜ pending |
| 6-02-06 | 02 | 1 | CLIP-03 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestGenerate::test_final_score -x` | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 | 2 | UI-02 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestExport::test_export_saves_file -x` | ❌ W0 | ⬜ pending |
| 6-03-02 | 03 | 2 | UI-02 | unit | `conda run -n stylehuman python -m pytest tests/test_gradio_app.py::TestExport::test_export_shows_path -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_gradio_app.py` — stubs for UI-01, UI-02, CLIP-03 (8 test methods)

*Existing infrastructure covers the test runner; only the new test file needs creating.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Mic recording → ASR → generate face end-to-end | UI-01 | Requires physical mic input and running Gradio server | Run `python ui/app.py`, click mic record, speak "a young woman with dark hair", click Generate, verify face image appears |
| CLIP score updates every ~10 steps during optimization | CLIP-03 | Requires real-time UI observation | During generation, observe score_out field updating incrementally |
| Export PNG saved with timestamp filename | UI-02 | File system verification | Click Export PNG after generation; verify file appears in `outputs/` with `YYYYMMDD_HHMMSS.png` name |
| CPU-only startup notice displayed | UI-01 | Environment-dependent | Run on non-CUDA machine; verify startup notice appears in UI |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
