---
phase: 01-environment-and-repo-scaffold
plan: 01
subsystem: infra
tags: [conda, pip, python, numpy, gradio, whisper, clip, librosa, tiktoken, pytorch-fid, stylegan]

# Dependency graph
requires: []
provides:
  - requirements_extra.txt with all nine pinned extra dependencies at project root
  - Nine scaffold directories with .gitkeep placeholders (asr, encoder, generator, optimizer, ui, utils, outputs, data/human_faces, StyleGAN-Human/pretrained_models)
  - stylehuman conda env with all extra packages installed and pip check clean
affects:
  - 01-02-environment-and-repo-scaffold
  - 02-stylegan-generator
  - 03-clip-encoder-optimizer
  - 04-asr-audio
  - 05-utils-dataset
  - 06-gradio-ui

# Tech tracking
tech-stack:
  added:
    - openai-whisper (ASR)
    - tiktoken==0.5.2 (Whisper tokenizer, Python 3.8 safe)
    - clip==1.0 from github.com/openai/CLIP.git (text/image embeddings)
    - librosa==0.9.2 (audio preprocessing)
    - sounddevice>=0.4.4 (microphone capture)
    - soundfile>=0.10.3 (WAV/MP3 I/O)
    - gradio==3.50.2 (3.x pin, Python 3.8 compatible UI)
    - pytorch-fid==0.3.0 (FID scoring)
    - numpy==1.23.5 (<1.24 cap for StyleGAN compatibility)
  patterns:
    - Ordered pip install: whisper first, then tiktoken pin override, then CLIP, audio, gradio, fid, numpy cap last
    - requirements_extra.txt documents reproducible install order alongside conda env
    - .gitkeep placeholders anchor all module directories for git tracking before code exists

key-files:
  created:
    - D:/EchoFace/requirements_extra.txt
    - D:/EchoFace/asr/.gitkeep
    - D:/EchoFace/encoder/.gitkeep
    - D:/EchoFace/generator/.gitkeep
    - D:/EchoFace/optimizer/.gitkeep
    - D:/EchoFace/ui/.gitkeep
    - D:/EchoFace/utils/.gitkeep
    - D:/EchoFace/outputs/.gitkeep
    - D:/EchoFace/data/human_faces/.gitkeep
    - D:/EchoFace/StyleGAN-Human/pretrained_models/.gitkeep
  modified: []

key-decisions:
  - "tiktoken pinned to 0.5.2 immediately after openai-whisper to override 0.7.0 that whisper resolved — prevents Python 3.8 wheel gap"
  - "gradio pinned >=3.40,<4.0 (resolved to 3.50.2) — Gradio 4.x uses Python 3.10+ syntax, would crash on import"
  - "numpy capped to <1.24 as final install step (resolved to 1.23.5) — 1.24 removed np.bool/np.int aliases used in StyleGAN-Human internals"
  - "librosa pinned to 0.9.2 — 0.10+ requires scipy>=1.7.3; env had scipy 1.10.1 (upgraded from base, not a conflict), pin still applied for reproducibility"
  - "CLIP installed via GitHub URL only — PyPI 'clip' package is unrelated"
  - "pip install order matters: single-command requirements_extra.txt install would not enforce tiktoken pin correctly"

patterns-established:
  - "Pattern: ordered pip install sequence — install openai-whisper first, immediately pin tiktoken, then CLIP, audio libs, gradio, fid, numpy cap last"
  - "Pattern: requirements_extra.txt as documentation/reproducibility artifact, not direct install target"

requirements-completed: [ENV-02, ENV-03]

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 1 Plan 01: Environment Scaffold and Extra Dependencies Summary

**requirements_extra.txt with 9 pinned packages installed into stylehuman conda env, and 9-directory repo scaffold created with .gitkeep placeholders — pip check clean with numpy capped at 1.23.5 and gradio downgraded from 4.44.1 to 3.50.2**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-06T03:56:46Z
- **Completed:** 2026-03-06T04:02:48Z
- **Tasks:** 3/3
- **Files modified:** 10 created, 0 modified

## Accomplishments

- requirements_extra.txt written with all nine pinned packages including critical version guards (tiktoken==0.5.2, librosa==0.9.2, gradio>=3.40,<4.0, numpy<1.24, CLIP via GitHub URL)
- Nine module directories created with .gitkeep placeholders anchoring the full EchoFace project structure for phases 2-6
- All extra packages installed into stylehuman conda env in correct dependency order; pip check reports "No broken requirements found"; key imports (gradio, whisper, clip, librosa) all pass; ffmpeg confirmed on system PATH

## Task Commits

Each task was committed atomically:

1. **Task 1: Write requirements_extra.txt with pinned versions** - `e7cc02c` (chore)
2. **Task 2: Create repo directory scaffold with .gitkeep placeholders** - `0228fc2` (chore)
3. **Task 3: Install requirements_extra.txt into stylehuman conda env** - `8a8aae4` (chore)

## Files Created/Modified

- `D:/EchoFace/requirements_extra.txt` - Pinned extra dependency list for pip install on top of StyleGAN-Human base env
- `D:/EchoFace/asr/.gitkeep` - ASR module directory placeholder (Phase 4: Whisper transcriber)
- `D:/EchoFace/encoder/.gitkeep` - CLIP encoder module directory placeholder (Phase 3)
- `D:/EchoFace/generator/.gitkeep` - StyleGAN wrapper module directory placeholder (Phase 2)
- `D:/EchoFace/optimizer/.gitkeep` - Latent optimizer module directory placeholder (Phase 3)
- `D:/EchoFace/ui/.gitkeep` - Gradio app module directory placeholder (Phase 6)
- `D:/EchoFace/utils/.gitkeep` - Utilities module directory placeholder (Phase 5)
- `D:/EchoFace/outputs/.gitkeep` - Generated PNG output directory placeholder (must exist before PIL save())
- `D:/EchoFace/data/human_faces/.gitkeep` - Kaggle dataset directory placeholder (distinguishes missing from empty)
- `D:/EchoFace/StyleGAN-Human/pretrained_models/.gitkeep` - Pretrained weights directory anchored (pkl already present)

## Decisions Made

- tiktoken pinned to 0.5.2 immediately after openai-whisper install to override 0.7.0 that whisper auto-resolved — Python 3.8 wheel guard
- gradio pinned >=3.40,<4.0 (resolved to 3.50.2, downgraded from 4.44.1) — Gradio 4.x uses Python 3.10+ syntax (match statements), would crash on import in Python 3.8.5
- numpy capped to <1.24 as final install step (resolved to 1.23.5, downgraded from 1.24.4) — numpy 1.24 removed np.bool/np.int aliases used in StyleGAN-Human internals
- CLIP installed via GitHub URL only (not PyPI "clip" which is an unrelated package)
- pip install run as ordered steps, not via single `pip install -r` — order is load-bearing for tiktoken pin

## Deviations from Plan

None - plan executed exactly as written. The env already had several packages pre-installed (openai-whisper, tiktoken 0.7.0, librosa 0.11.0, gradio 4.44.1, numpy 1.24.4) which were downgraded as part of the planned pinning steps.

## Issues Encountered

None. Pre-existing package installations (higher versions) were handled as expected by the ordered pip install sequence which overwrote them with the correct pins.

Notable finding: scipy in the env was 1.10.1 (not 1.7.1 as expected from base environment.yml). This does not affect librosa 0.9.2 which requires scipy>=1.2.0. All pins still applied correctly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All nine module directories exist and are git-tracked with .gitkeep
- stylehuman conda env has all extra packages installed with correct version pins
- pip check is clean — no dependency conflicts
- ffmpeg is on system PATH (required for MP3 decode in librosa)
- Ready for Plan 01-02: implement gpu_utils.py, validate_env.py, and pytest test suite

---
*Phase: 01-environment-and-repo-scaffold*
*Completed: 2026-03-06*
