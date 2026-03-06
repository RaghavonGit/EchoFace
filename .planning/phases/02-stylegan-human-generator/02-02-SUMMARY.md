---
phase: 02-stylegan-human-generator
plan: "02"
subsystem: generator
tags: [stylegan, pytorch, synthesis, gpu, cpu, tdd, mock, pil, interpolate]

# Dependency graph
requires:
  - phase: 02-01
    provides: StyleGANWrapper class with load/freeze/sample_w scaffold and GEN-01 tests
  - phase: 01-02
    provides: gpu_utils.get_device_config() returning plain dict with device/dtype/resolution/truncation_psi
provides:
  - StyleGANWrapper.synthesize() — GPU path writes 512x1024 PNG; CPU path resizes 1024->256 and writes 256x256 PNG
  - StyleGANWrapper.generate_and_save() — fully operational on both CUDA and CPU; saves timestamped PNG to outputs/
  - 4 additional passing tests covering GEN-02 and GEN-03 (7 total in test_stylegan_wrapper.py)
  - outputs/ directory with 5 verified 512x1024 PNG human figures from GPU integration run
  - Human-approved visual checkpoint — generated images are recognizable photorealistic full-body figures
affects:
  - 03-clip-optimizer (calls sample_w() and synthesize() at every optimization step)
  - 05-latent-directions (relies on generate_and_save() for direction visualization)

# Tech tracking
tech-stack:
  added: [torch.nn.functional.interpolate (resize path), unittest.mock (MagicMock/patch for CPU tests)]
  patterns:
    - TDD red-green for GPU (real pkl) and CPU (mock G) synthesis paths in same test class
    - Patch generator.stylegan_wrapper namespace bindings (not top-level modules) for mock isolation
    - Float roundtrip for F.interpolate: uint8 -> .float() -> interpolate -> clamp -> uint8
    - PIL image closed before tempdir cleanup on Windows to avoid PermissionError file-locking

key-files:
  created: []
  modified:
    - tests/test_stylegan_wrapper.py
    - outputs/gen_20260306_173311_344968.png
    - outputs/gen_20260306_173331_498454.png
    - outputs/gen_20260306_173425_795491.png
    - outputs/gen_20260306_173518_640196.png
    - outputs/gen_20260306_173554_997182.png

key-decisions:
  - "StyleGAN-Human v2 native synthesis output is portrait 512x1024, not square 1024x1024 — GPU test assertion corrected to (512, 1024)"
  - "PIL Image must be explicitly closed before tempdir cleanup on Windows — PermissionError from open file handle fixed with img.close() before cleanup"
  - "CPU path mock patches generator.stylegan_wrapper.legacy and generator.stylegan_wrapper.dnnlib — top-level module patching does not intercept already-bound names"

patterns-established:
  - "TDD integration pattern: GPU tests load real pkl with CUDA skip guard; CPU tests use MagicMock to eliminate model I/O"
  - "Float roundtrip for resize: img_hwc.permute(2,0,1).unsqueeze(0).float() -> F.interpolate -> squeeze/permute -> clamp(0,255).to(uint8)"
  - "outputs/ directory created via os.makedirs(exist_ok=True) at generate_and_save() call time"

requirements-completed: [GEN-02, GEN-03]

# Metrics
duration: ~30min (Task 1 TDD + GPU integration run); visual checkpoint approved by user
completed: 2026-03-06
---

# Phase 02 Plan 02: StyleGAN Synthesis Paths — GPU 512x1024 and CPU 256x256 Summary

**StyleGANWrapper.synthesize() and generate_and_save() fully operational: GPU path produces 512x1024 photorealistic full-body figures from real pkl; CPU path resizes 1024-native synthesis output to 256x256 via F.interpolate; all 7 wrapper tests pass; visual checkpoint approved.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-03-06T12:06:00Z (estimated from first GPU run timestamp 17:33:11 IST)
- **Completed:** 2026-03-06T12:06:00Z
- **Tasks:** 2 (Task 1: TDD implementation; Task 2: visual checkpoint — approved)
- **Files modified:** 1 (tests/test_stylegan_wrapper.py); 5 PNG outputs generated

## Accomplishments

- Replaced 4 `pytest.skip("not implemented")` stubs with real test implementations covering GEN-02 and GEN-03
- GPU integration test (test_gpu_generates_1024_png) ran against the real 315MB stylegan_human_v2_1024.pkl and produced 5 timestamped PNGs in outputs/
- CPU resize path (test_cpu_resize_to_256) verified with MagicMock — synthesize() correctly resizes 1024x1024 native output to 256x256 using F.interpolate with float roundtrip
- Full test suite: 46 passed, 1 skipped — exit code 0
- Visual checkpoint passed: user confirmed generated images are recognizable photorealistic full-body human figures

## Task Commits

1. **Task 1: Implement and test GPU and CPU synthesis paths** - `c33bc73` (feat)
2. **Task 2: Visual verification checkpoint** - approved (no code change; checkpoint recorded in SUMMARY)

## Files Created/Modified

- `tests/test_stylegan_wrapper.py` — 4 stubs replaced with real GPU and CPU test implementations (93 lines added)
- `outputs/gen_20260306_173311_344968.png` — GPU-generated 512x1024 PNG (visually approved)
- `outputs/gen_20260306_173331_498454.png` — GPU-generated 512x1024 PNG
- `outputs/gen_20260306_173425_795491.png` — GPU-generated 512x1024 PNG
- `outputs/gen_20260306_173518_640196.png` — GPU-generated 512x1024 PNG
- `outputs/gen_20260306_173554_997182.png` — GPU-generated 512x1024 PNG

## Decisions Made

- **StyleGAN-Human v2 native output is 512x1024 portrait, not 1024x1024 square.** The plan spec said "1024x1024 PNG" based on model resolution config, but the actual G.synthesis() output tensor is [1, 3, 1024, 512] in HxW space (portrait orientation). GPU test assertion corrected from `(1024, 1024)` to `(512, 1024)` under Rule 1 (auto-fix bug).
- **PIL image must be closed before tempdir cleanup on Windows.** Windows holds an exclusive file lock on open PIL image objects; calling `img.close()` before the tempdir context exits prevents PermissionError. Fixed under Rule 1.
- **Patch generator.stylegan_wrapper namespace, not top-level modules.** By the time the test runs, `legacy` and `dnnlib` are already bound inside the wrapper module's namespace. Patching `generator.stylegan_wrapper.legacy.load_network_pkl` intercepts the bound name correctly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected GPU test assertion from (1024, 1024) to (512, 1024)**
- **Found during:** Task 1 (test_gpu_generates_1024_png)
- **Issue:** Plan specified `img.size == (1024, 1024)` but StyleGAN-Human v2 synthesizes portrait images; PIL reports size as (width, height) = (512, 1024)
- **Fix:** Changed assertion to `assert img.size == (512, 1024)`
- **Files modified:** tests/test_stylegan_wrapper.py
- **Verification:** Test passes with real pkl on CUDA
- **Committed in:** c33bc73

**2. [Rule 1 - Bug] Added PIL image close before tempdir cleanup to prevent Windows PermissionError**
- **Found during:** Task 1 (test_cpu_generates_256_png and test_gpu_generates_1024_png)
- **Issue:** Windows PermissionError when tempdir tried to delete the PNG while PIL still held the file handle
- **Fix:** Added `img.close()` after size assertion, before tempdir context exit
- **Files modified:** tests/test_stylegan_wrapper.py
- **Verification:** Both CPU and GPU tests pass without PermissionError
- **Committed in:** c33bc73

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs in plan spec vs. actual behavior)
**Impact on plan:** Both fixes necessary for correctness on Windows with the actual model output shape. No scope creep.

## Issues Encountered

- StyleGAN-Human v2 portrait output shape (512x1024 not 1024x1024) was not documented in the plan — discovered during the GPU integration test run. Fixed inline per Rule 1.
- Windows file-locking on PIL Image objects required explicit `.close()` calls in test teardown. Standard Python behavior on POSIX (where GC handles this) differs from Windows. Fixed inline per Rule 1.

## User Setup Required

None - no external service configuration required. The stylegan_human_v2_1024.pkl is already on disk at `D:/EchoFace/StyleGAN-Human/pretrained_models/`.

## Next Phase Readiness

- Phase 3 (CLIP optimizer) is unblocked: `sample_w()` returns `[1, num_ws, 512]` latent on CUDA; `synthesize(w)` returns `[H, W, 3]` uint8 CPU tensor at the configured resolution
- Phase 3 must implement gradient clipping (max_norm=1.0) and NaN detection — gradient explosion risk flagged in STATE.md
- All three Phase 2 requirements complete: GEN-01 (load/freeze), GEN-02 (GPU 512x1024 synthesis), GEN-03 (CPU 256x256 resize path)

---
*Phase: 02-stylegan-human-generator*
*Completed: 2026-03-06*
