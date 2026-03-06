---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 03-clip-encoder-and-latent-optimizer-03-PLAN.md (all tasks done, smoke test approved)
last_updated: "2026-03-06T17:54:04.204Z"
last_activity: 2026-03-06 — Roadmap created, all 6 phases derived from 21 v1 requirements
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-06)

**Core value:** A developer can speak a face description and receive a photorealistic generated image fully offline — no API keys, no cloud calls, everything runs locally after one-time model download.
**Current focus:** Phase 1 — Environment and Repo Scaffold

## Current Position

Phase: 1 of 6 (Environment and Repo Scaffold)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-03-06 — Roadmap created, all 6 phases derived from 21 v1 requirements

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*
| Phase 01-environment-and-repo-scaffold P01 | 6 | 3 tasks | 10 files |
| Phase 01-environment-and-repo-scaffold P02 | 4 | 3 tasks | 5 files |
| Phase 02-stylegan-human-generator P01 | 3 | 2 tasks | 4 files |
| Phase 02-stylegan-human-generator P02 | 30 | 2 tasks | 1 files |
| Phase 03-clip-encoder-and-latent-optimizer P01 | 2 | 2 tasks | 6 files |
| Phase 03-clip-encoder-and-latent-optimizer P02 | 3 | 1 tasks | 1 files |
| Phase 03-clip-encoder-and-latent-optimizer P03 | 2 | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 1]: gradio must be pinned >=3.40,<4.0 — Gradio 4.x requires Python 3.10+, incompatible with StyleGAN-Human base env (Python 3.8.5)
- [Phase 2]: StyleGAN pkl must be loaded via legacy.load_network_pkl(), not pickle.load() — raw pickle fails with AttributeError on dnnlib custom class references
- [Phase 2]: Generator parameters must be frozen (requires_grad=False) — only W-space latent carries gradients during optimization
- [Phase 4]: Whisper must lazy-load and unload before optimization begins — VRAM budget on smaller GPUs cannot hold both Whisper medium and StyleGAN-Human simultaneously
- [Phase 01-environment-and-repo-scaffold]: tiktoken pinned to 0.5.2 immediately after openai-whisper to override 0.7.0 that whisper resolved — Python 3.8 wheel guard
- [Phase 01-environment-and-repo-scaffold]: gradio pinned >=3.40,<4.0 (resolved to 3.50.2, downgraded from 4.44.1) — Gradio 4.x uses Python 3.10+ syntax, crashes on import in Python 3.8.5
- [Phase 01-environment-and-repo-scaffold]: numpy capped to <1.24 as final install step (resolved to 1.23.5, downgraded from 1.24.4) — 1.24 removed np.bool/np.int aliases used in StyleGAN-Human internals
- [Phase 01-environment-and-repo-scaffold]: gpu_utils returns plain dict (not dataclass) — broadest compatibility with downstream modules importing cfg['device']
- [Phase 01-environment-and-repo-scaffold]: dtype in gpu_utils controls W-space latent and Adam optimizer dtype only; StyleGAN synthesis stays float32 (generate.py force_fp32=True)
- [Phase 02-stylegan-human-generator]: StyleGAN-Human sys.path injected at module level before dnnlib/legacy import
- [Phase 02-stylegan-human-generator]: Test mocks patch generator.stylegan_wrapper namespace bindings rather than top-level modules
- [Phase 02-stylegan-human-generator]: StyleGAN-Human v2 native synthesis output is portrait 512x1024, not square 1024x1024 — GPU test assertion corrected accordingly
- [Phase 02-stylegan-human-generator]: PIL Image must be explicitly closed before tempdir cleanup on Windows to avoid PermissionError file-locking
- [Phase 02-stylegan-human-generator]: Patch generator.stylegan_wrapper namespace bindings (not top-level modules) for correct mock interception in CPU tests
- [Phase 03-clip-encoder-and-latent-optimizer]: Stub files import clip/CLIPEncoder at module level even with NotImplementedError body — patch() requires name in namespace
- [Phase 03-clip-encoder-and-latent-optimizer]: test_runs_150_steps uses encode_image call_count as step proxy (called once per optimization step in gradient path)
- [Phase 03-clip-encoder-and-latent-optimizer]: jit=False required on clip.load() — JIT-compiled model prevents weight access and float32 casting during encode
- [Phase 03-clip-encoder-and-latent-optimizer]: CLIP weights frozen immediately after load (requires_grad_(False)) — FP16 params would produce NaN under gradient updates
- [Phase 03-clip-encoder-and-latent-optimizer]: F.normalize applied explicitly on every encode path — CLIP encode_text/encode_image return unnormalized embeddings
- [Phase 03-clip-encoder-and-latent-optimizer]: wrapper.G.synthesis used in gradient path (not wrapper.synthesize) — uint8 conversion breaks gradient graph
- [Phase 03-clip-encoder-and-latent-optimizer]: encoder.model.encode_image called inside grad context (CLIP weights frozen via requires_grad_(False)) — gradient flows through fwd pass to w
- [Phase 03-clip-encoder-and-latent-optimizer]: NaN check placed before loss.backward() — calling backward on NaN raises or silently corrupts gradients

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3]: CLIP gradient explosion risk — gradient clipping (max_norm=1.0) and NaN detection required; λ=0.1 and lr=0.01 are starting points, empirical validation needed
- [Phase 5]: SVM vs PCA for latent direction discovery is unresolved for StyleGAN-Human v2 specifically; flag for small experiment at Phase 5 planning start

## Session Continuity

Last session: 2026-03-06T17:54:04.199Z
Stopped at: Completed 03-clip-encoder-and-latent-optimizer-03-PLAN.md (all tasks done, smoke test approved)
Resume file: None
