# Phase 2: StyleGAN-Human Generator — Research

**Researched:** 2026-03-06
**Domain:** StyleGAN-Human v2 inference wrapper — pickle loading, W-space synthesis, GPU/CPU path switching
**Confidence:** HIGH (grounded in source code on disk: `StyleGAN-Human/legacy.py`, `generate.py`, `dnnlib/util.py`, `gpu_utils.py`)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GEN-01 | `stylegan_wrapper.py` loads `stylegan_human_v2_1024.pkl` using `legacy.load_network_pkl()` (not raw `pickle.load()`), exposes W-space synthesis via `G.synthesis(w, noise_mode='const')` | Loading pattern confirmed in `generate.py` lines 67-68; `legacy.load_network_pkl()` signature confirmed in `legacy.py` lines 22-73 |
| GEN-02 | Generator runs fp16 inference on CUDA-enabled GPU producing 1024x1024px output | `generate.py` line 102: `G.synthesis(w, noise_mode=noise_mode, force_fp32=True)` — synthesis stays float32 even on GPU; image post-processing confirmed in lines 108-109 |
| GEN-03 | Generator falls back to CPU with truncation=0.5 and 256x256px resolution when CUDA is unavailable | `gpu_utils.py` already returns `truncation_psi=0.5` and `resolution=256` on CPU path; synthesis output must be downsampled with `torch.nn.functional.interpolate` |
</phase_requirements>

---

## Summary

Phase 2 builds `generator/stylegan_wrapper.py`, a thin wrapper around the StyleGAN-Human v2 inference pipeline. The pretrained weights (`stylegan_human_v2_1024.pkl`) are already present at `StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl`. The complete inference pattern already exists in `StyleGAN-Human/generate.py` — the wrapper's job is to repackage that pattern as a class with clean GPU/CPU path selection, consume `gpu_utils.get_device_config()`, and produce a timestamped PNG in `outputs/`.

The single highest-risk area is the sys.path setup: `legacy.py` and `dnnlib/` must be importable from within `generator/stylegan_wrapper.py`. Since the wrapper lives at `generator/stylegan_wrapper.py` and `StyleGAN-Human/` is a sibling directory, the wrapper must insert `StyleGAN-Human/` onto `sys.path` at import time. This is the standard pattern used by every upstream StyleGAN-Human script.

Key constraint confirmed from source: `G.synthesis()` always receives `force_fp32=True` regardless of device. The `dtype` from `gpu_utils` controls W-space latent dtype and Adam optimizer dtype only — not the synthesis call. GEN-02's "fp16 inference" refers to this dtype contract for the latent tensor, not a force-fp16 synthesis call.

**Primary recommendation:** Wrap `generate.py` line 67-108 into a `StyleGANWrapper` class that accepts a `cfg` dict from `gpu_utils.get_device_config()`, calls `legacy.load_network_pkl()` with `dnnlib.util.open_url()` for local files, and exposes `synthesize(w)` and `sample_w(truncation_psi)` methods.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `StyleGAN-Human/legacy.py` | On disk (cloned) | `load_network_pkl()` — safe pkl loading | Required by GEN-01; avoids raw pickle failure on dnnlib custom classes |
| `StyleGAN-Human/dnnlib/util.py` | On disk | `open_url()` — local file open for pkl | Accepts plain path string as local file (no `://` scheme detection); returns `open(path, "rb")` |
| `torch` | 1.9.1 (conda pinned) | Tensor operations, device management | Already in base env |
| `PIL.Image` | 8.3.1 (conda pinned) | PNG save | Already in base env |
| `numpy` | 1.23.5 (pinned to <1.24) | Array manipulation for image post-processing | Already pinned in Phase 1 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.nn.functional.interpolate` | Built-in | Resize 1024px output to 256px on CPU path | CPU fallback only (GEN-03) |
| `gpu_utils.get_device_config` | Phase 1 deliverable | Single source of truth for device/dtype/resolution/truncation_psi | Always — imported at wrapper init |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `legacy.load_network_pkl()` | `pickle.load()` | Raw pickle fails with `AttributeError` on dnnlib custom class references — locked decision |
| `dnnlib.util.open_url(path)` | `open(path, 'rb')` directly | Both work for local paths; `open_url` is the idiom used in `generate.py` and handles potential future URL paths consistently |
| `G.synthesis(w, force_fp32=True)` | `G.synthesis(w)` without flag | Without `force_fp32=True`, synthesis may produce NaN on some CUDA configs; this flag is what `generate.py` uses |

---

## Architecture Patterns

### Recommended Project Structure

```
EchoFace/
├── generator/
│   ├── __init__.py          # empty
│   └── stylegan_wrapper.py  # StyleGANWrapper class
├── outputs/                 # PNG files written here
├── tests/
│   └── test_stylegan_wrapper.py  # Phase 2 pytest suite
├── gpu_utils.py             # Phase 1 deliverable — consumed here
└── StyleGAN-Human/          # DO NOT MODIFY — read-only
    ├── legacy.py
    ├── dnnlib/
    └── pretrained_models/
        └── stylegan_human_v2_1024.pkl
```

### Pattern 1: sys.path Injection for Legacy Module Access

**What:** Insert the `StyleGAN-Human/` directory onto `sys.path` before any `import legacy` or `import dnnlib` call.

**When to use:** At the top of `stylegan_wrapper.py`, before module-level imports of legacy/dnnlib.

**Why:** `legacy.py` and `dnnlib/` live inside `StyleGAN-Human/`, which is not on the Python path by default. Every upstream script (generate.py, edit.py, etc.) implicitly relies on being run from within `StyleGAN-Human/`. The wrapper must reproduce this behavior without being run from that directory.

```python
# Source: StyleGAN-Human/generate.py (implicit — run from StyleGAN-Human/ dir)
# Pattern: insert parent at import time
import sys, os
_SG_ROOT = os.path.join(os.path.dirname(__file__), '..', 'StyleGAN-Human')
_SG_ROOT = os.path.abspath(_SG_ROOT)
if _SG_ROOT not in sys.path:
    sys.path.insert(0, _SG_ROOT)

import dnnlib
import legacy
```

### Pattern 2: Correct PKL Loading Sequence

**What:** Use `dnnlib.util.open_url()` as the context manager, pass the file handle to `legacy.load_network_pkl()`, extract `G_ema`.

**When to use:** Always — this is the only safe loading path.

**Critical insight from `dnnlib/util.py` line 390-391:** `open_url()` checks if the argument matches `'^[a-z]+://'`. A plain Windows/Unix path like `StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl` does NOT match this pattern, so `open_url()` calls `open(url, "rb")` directly and returns the file object. The context manager pattern in `generate.py` therefore works unchanged for local paths.

```python
# Source: StyleGAN-Human/generate.py lines 67-68
import dnnlib
import legacy
import torch

def _load_generator(pkl_path: str, device: torch.device) -> torch.nn.Module:
    with dnnlib.util.open_url(pkl_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G = G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G
```

### Pattern 3: W-Space Sampling via G.mapping()

**What:** Generate a random Z vector, pass through `G.mapping()` to get W-space latent. Never optimize in Z-space.

**When to use:** When `sample_w()` is called to produce an initial latent for the optimizer (Phase 3) or for the standalone smoke test.

**Key dimensions confirmed from `networks.py`:** `G.z_dim` (typically 512), `G.c_dim` (0 for unconditional), `G.w_dim` (512), `G.num_ws` (number of synthesis layers — 18 for 1024px model).

```python
# Source: StyleGAN-Human/generate.py lines 94-101
def sample_w(self, truncation_psi: float = 1.0) -> torch.Tensor:
    """Returns w: Tensor [1, num_ws, w_dim] — the full W+ space latent."""
    z = torch.randn(1, self.G.z_dim, device=self.device)
    label = torch.zeros([1, self.G.c_dim], device=self.device)
    w = self.G.mapping(z, label, truncation_psi=truncation_psi)
    return w  # shape: [1, num_ws, 512]
```

### Pattern 4: Synthesis Call and Image Post-Processing

**What:** Call `G.synthesis(w, noise_mode='const', force_fp32=True)`, then convert output tensor to uint8 RGB, then save as PNG.

**When to use:** In `synthesize()` method. The `noise_mode='const'` ensures deterministic output for the same `w`.

**force_fp32 is mandatory:** Confirmed in `generate.py` line 102. Omitting it can cause synthesis to run in fp16 internally, which may produce NaN on some GPU configurations.

```python
# Source: StyleGAN-Human/generate.py lines 102-109
def synthesize(self, w: torch.Tensor) -> torch.Tensor:
    """
    Args:
        w: Tensor [1, num_ws, 512] — W+ space latent
    Returns:
        img_uint8: Tensor [H, W, 3] uint8 on CPU
    """
    img = self.G.synthesis(w, noise_mode='const', force_fp32=True)
    # Output range is [-1, 1]; convert to [0, 255] uint8
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img[0].cpu()  # [H, W, 3]
```

### Pattern 5: CPU Fallback — Resize to 256x256

**What:** On CPU, the model still synthesizes at native resolution (1024px). Resize to 256x256 using `torch.nn.functional.interpolate` before saving.

**When to use:** When `cfg['resolution'] == 256` (CPU path from `gpu_utils`).

**Why resize rather than a different model:** There is no separate 256px pkl. The 1024px model runs on CPU too, just slowly. Truncation=0.5 (already in `gpu_utils`) constrains the output to more realistic samples.

```python
import torch.nn.functional as F

def _maybe_resize(img_chw: torch.Tensor, target_res: int) -> torch.Tensor:
    """img_chw: [3, H, W] float or uint8. Returns [3, target_res, target_res]."""
    if img_chw.shape[-1] == target_res:
        return img_chw
    img_float = img_chw.float().unsqueeze(0)  # [1, 3, H, W]
    resized = F.interpolate(img_float, size=(target_res, target_res), mode='bilinear', align_corners=False)
    return resized.squeeze(0).to(img_chw.dtype)
```

### Pattern 6: PNG Save with Timestamp

**What:** Convert the [H, W, 3] uint8 numpy array to PIL.Image and save to `outputs/` with a timestamp filename.

**When to use:** In the `save()` or `generate_and_save()` method.

```python
import PIL.Image
from datetime import datetime

def save_png(img_hwc: torch.Tensor, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(out_dir, f"gen_{timestamp}.png")
    PIL.Image.fromarray(img_hwc.numpy(), 'RGB').save(path)
    return path
```

### Recommended StyleGANWrapper Class Interface

```python
class StyleGANWrapper:
    def __init__(self, pkl_path: str, cfg: dict):
        """
        Args:
            pkl_path: absolute path to .pkl file
            cfg: dict from gpu_utils.get_device_config()
        """

    def sample_w(self) -> torch.Tensor:
        """Returns [1, num_ws, 512] W+ latent using cfg['truncation_psi']."""

    def synthesize(self, w: torch.Tensor) -> torch.Tensor:
        """Returns [H, W, 3] uint8 tensor (H=W=cfg['resolution'])."""

    def generate_and_save(self, out_dir: str = 'outputs') -> str:
        """Sample random w, synthesize, resize if CPU, save PNG. Returns path."""
```

### Anti-Patterns to Avoid

- **Raw `pickle.load()`:** Fails with `AttributeError` on dnnlib custom class references. Always use `legacy.load_network_pkl()`.
- **Skipping `requires_grad_(False)` on G parameters:** Causes the full synthesis graph to accumulate gradients during Phase 3 optimization, OOM on consumer GPUs.
- **Hard-coding `device='cuda'`:** Crashes on CPU-only machines. Always consume `cfg['device']` from `gpu_utils`.
- **Importing `legacy` or `dnnlib` before sys.path injection:** `ModuleNotFoundError` at import time. The sys.path insert must happen first.
- **Passing `w` in float16 to `G.synthesis()`:** The `force_fp32=True` flag handles dtype coercion inside synthesis. The wrapper's `w` tensor may be float32 or float16; `force_fp32=True` ensures synthesis always runs float32 internally.
- **Not calling `.eval()` on G after load:** `load_network_pkl()` does not guarantee eval mode. Always call `G.eval()` after loading.
- **Saving to CWD instead of `outputs/`:** All generated images must go to `outputs/` (created with `os.makedirs(..., exist_ok=True)`).
- **Modifying any file under `StyleGAN-Human/`:** Read-only; upstream merges must remain possible.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Safe pkl deserialization | Custom unpickler | `legacy.load_network_pkl()` | Already handles dnnlib class stubs, TF conversion, FP16 conversion, validation |
| Local file opening for pkl | `open(path, 'rb')` directly | `dnnlib.util.open_url(path)` | Idiom used by all upstream scripts; handles both local paths and URLs transparently |
| Z→W mapping | Custom mapping network | `G.mapping(z, label, truncation_psi=...)` | Mapping network is built into the loaded G; reproducing it manually is unnecessary and error-prone |
| Image tensor→PNG conversion | Custom byte conversion | `(img.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)` then `PIL.Image.fromarray` | Exact formula from `generate.py` line 108-109; alternative formulas produce color shift |

**Key insight:** The entire inference pattern already exists in `generate.py`. The wrapper's sole job is to repackage it as a class, not to invent new logic.

---

## Common Pitfalls

### Pitfall 1: ImportError on `import legacy` or `import dnnlib`

**What goes wrong:** `ModuleNotFoundError: No module named 'legacy'` when the wrapper is imported from the project root.

**Why it happens:** `legacy.py` lives inside `StyleGAN-Human/`. Python's module search doesn't include that directory unless explicitly added.

**How to avoid:** Insert `StyleGAN-Human/` onto `sys.path` at the very top of `stylegan_wrapper.py`, before any dnnlib/legacy imports. Use `os.path.abspath` to make the path absolute.

**Warning signs:** Import works when you `cd StyleGAN-Human && python` but fails when pytest runs from project root.

### Pitfall 2: `legacy.load_network_pkl()` Writes Log Files to CWD

**What goes wrong:** `load_network_pkl()` unconditionally opens `ori_model.txt` (or `ori_model_Gonly.txt`) in the current working directory for writing (see `legacy.py` lines 25-29). This creates an unexpected file in whatever directory the process is running from.

**Why it happens:** This is a debugging artifact in the StyleGAN-Human fork. It is not configurable without modifying the file (which is off-limits).

**How to avoid:** Accept this side-effect. The file is small and harmless. Document it in the wrapper docstring. Tests should not assert the absence of this file. If CWD is the project root, `ori_model.txt` will appear there.

**Warning signs:** Unexpected `ori_model.txt` appearing in test output directory.

### Pitfall 3: W-Space Shape Mismatch

**What goes wrong:** Passing `w` with shape `[1, 512]` (W-space, single vector) instead of `[1, num_ws, 512]` (W+ space, per-layer vectors). `G.synthesis()` expects the W+ form.

**Why it happens:** Confusing W-space and W+-space. `G.mapping()` returns `[1, num_ws, 512]` automatically when `num_ws` is set. Direct sampling of a single `[1, 512]` vector requires broadcasting: `w.unsqueeze(1).repeat(1, G.num_ws, 1)`.

**How to avoid:** Always use `G.mapping()` to produce `w`. Never construct `w` by hand unless broadcasting explicitly.

**Warning signs:** `RuntimeError: Expected input batch_size (X) to match target batch_size (Y)` inside synthesis.

### Pitfall 4: Synthesis Output Color Range Confusion

**What goes wrong:** Saving a PIL image with incorrect color range — appearing washed out (range [0,1]) or completely wrong (range [-1,1] saved raw).

**Why it happens:** StyleGAN synthesis outputs tensors in `[-1, 1]` range. The conversion formula must be `* 127.5 + 128`, not `* 255` or `/ 2 + 0.5`.

**How to avoid:** Use the exact formula from `generate.py` line 108: `(img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)`.

**Warning signs:** Generated images appear uniformly grey or have extreme saturation.

### Pitfall 5: CPU Synthesis is Slow — Tests Must Mock or Skip

**What goes wrong:** Running synthesis on CPU in pytest takes 2-15 minutes per call, making the test suite unusable.

**Why it happens:** StyleGAN 1024px synthesis involves millions of operations. CPU runs are 10-50x slower than GPU.

**How to avoid:** Test structure for GEN-03 (CPU path) must either:
  - Mock the synthesis call and only test the resize/save logic
  - Use `pytest.mark.slow` and skip by default in CI
  - Test with a tiny synthetic `w` tensor and mock `G.synthesis()` to return a known-shape tensor

**Warning signs:** `pytest tests/test_stylegan_wrapper.py` takes more than 2 minutes per test.

### Pitfall 6: `dnnlib.util.open_url` with Backslash Paths on Windows

**What goes wrong:** On Windows, `os.path.join` produces backslash paths (`D:\EchoFace\...`). The regex `'^[a-z]+://'` in `open_url` does NOT match these, so it falls through to `open(url, "rb")` correctly — but verify this assumption by using `os.path.abspath()` to normalize the path before passing.

**Why it happens:** Backslash paths contain no `://` substring, so the URL detection correctly identifies them as local paths. This is expected behavior per `dnnlib/util.py` line 390.

**How to avoid:** Always pass `os.path.abspath(pkl_path)` to ensure the path is normalized. This also ensures relative paths work correctly regardless of CWD.

---

## Code Examples

Verified patterns from on-disk source files:

### Loading the Generator (from generate.py lines 67-68)

```python
# Source: StyleGAN-Human/generate.py:67-68
import dnnlib
import legacy

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
```

### Z-to-W Mapping (from generate.py lines 94-101)

```python
# Source: StyleGAN-Human/generate.py:94-101
label = torch.zeros([1, G.c_dim], device=device)
z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
w = G.mapping(z, label, truncation_psi=truncation_psi)
```

### Synthesis and PNG Save (from generate.py lines 102-109)

```python
# Source: StyleGAN-Human/generate.py:102-109
img = G.synthesis(w, noise_mode=noise_mode, force_fp32=True)
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
```

### open_url Local Path Handling (from dnnlib/util.py lines 389-391)

```python
# Source: StyleGAN-Human/dnnlib/util.py:389-391
# Doesn't look like an URL scheme so interpret it as a local filename.
if not re.match('^[a-z]+://', url):
    return url if return_filename else open(url, "rb")
```

This confirms: passing a plain local path to `open_url()` opens it as a binary file. No URL encoding needed. Backslash Windows paths are safe (no `://` present).

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pickle.load()` for StyleGAN pkls | `legacy.load_network_pkl()` | StyleGAN-Human fork | Required — raw pickle fails on dnnlib custom classes |
| Z-space optimization | W-space optimization | StyleGAN2 paper (2020) | Better disentanglement, meaningful attribute directions |
| Force-fp16 synthesis | `force_fp32=True` always | StyleGAN-Human fork default | Prevents NaN on some CUDA configurations |
| Separate GPU/CPU model files | Single pkl, runtime device selection | N/A | Simplifies deployment; CPU path is slower but uses same weights |

**No deprecated APIs in scope for this phase.** The StyleGAN-Human v2 API is stable and the pkl is already on disk.

---

## Open Questions

1. **W-space dimension for the 1024px model**
   - What we know: `G.num_ws` is the number of synthesis layers; for a 1024px StyleGAN2 model this is typically 18 (log2(1024/4) * 2 = 18 layers). `G.w_dim` is 512.
   - What's unclear: Exact values without loading the pkl (it's 315MB; not loaded during research).
   - Recommendation: After loading G, assert `G.w_dim == 512` and log `G.num_ws` in the wrapper's `__init__`. The optimizer (Phase 3) will need `G.num_ws` to construct `w_init`.

2. **ori_model.txt Side-Effect**
   - What we know: `legacy.load_network_pkl()` unconditionally writes `ori_model.txt` to CWD (legacy.py lines 25-29). It is NOT configurable without modifying the file.
   - What's unclear: Whether this file causes issues in CI or test environments with read-only CWD.
   - Recommendation: Document in wrapper docstring. Gitignore `ori_model.txt`. Tests must be run from a writable CWD (project root is fine).

3. **CPU Synthesis Runtime**
   - What we know: 1024px StyleGAN synthesis on CPU is slow (estimated 5-15 minutes per image based on model size).
   - What's unclear: Whether the test environment machine has CUDA available for CI.
   - Recommendation: GEN-03 CPU path test must use mocking for the synthesis call. The integration test (actual CPU synthesis) should be marked `@pytest.mark.slow` and excluded from the default run.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already configured, Phase 1 suite exists) |
| Config file | `pytest.ini` or inline — no dedicated config found; tests run from project root |
| Quick run command | `pytest tests/test_stylegan_wrapper.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GEN-01 | `load_network_pkl()` called, not `pickle.load()`; `G_ema` extracted and on correct device | unit (mock pkl) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_load_uses_legacy -x` | Wave 0 |
| GEN-01 | `G.parameters()` all have `requires_grad=False` after load | unit (mock pkl) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_parameters_frozen -x` | Wave 0 |
| GEN-01 | `synthesize()` calls `G.synthesis(w, noise_mode='const', force_fp32=True)` | unit (mock G) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_synthesis_signature -x` | Wave 0 |
| GEN-02 | GPU path: `sample_w()` returns shape `[1, num_ws, 512]` on correct device | unit (CUDA required) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_gpu_sample_w_shape -x` | Wave 0 |
| GEN-02 | GPU path: `generate_and_save()` writes a valid PNG to `outputs/`; PIL can open it at 1024x1024 | integration (CUDA required) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_gpu_generates_1024_png -x` | Wave 0 |
| GEN-03 | CPU path: `synthesize()` output is resized to 256x256 | unit (mock G, CPU) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_cpu_resize_to_256 -x` | Wave 0 |
| GEN-03 | CPU path: `generate_and_save()` exits without error; PNG is 256x256 | unit (mock G, CPU) | `pytest tests/test_stylegan_wrapper.py::TestStyleGANWrapper::test_cpu_generates_256_png -x` | Wave 0 |

All tests require CUDA availability checks via `pytest.skip("No CUDA")` guard for GPU tests. CPU path tests must mock `G.synthesis()` to return a `[1, 3, 1024, 1024]` float tensor (to avoid 15-minute synthesis wait).

### Sampling Rate

- **Per task commit:** `pytest tests/test_stylegan_wrapper.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_stylegan_wrapper.py` — covers GEN-01, GEN-02, GEN-03
- [ ] `generator/__init__.py` — empty package marker
- [ ] `generator/stylegan_wrapper.py` — main deliverable (Wave 1)
- [ ] Confirm `ori_model.txt` is in `.gitignore` — side-effect of `load_network_pkl()`

---

## Sources

### Primary (HIGH confidence)

- `D:/EchoFace/StyleGAN-Human/legacy.py` (lines 22-73) — `load_network_pkl()` signature, `_LegacyUnpickler`, `G_ema` extraction
- `D:/EchoFace/StyleGAN-Human/generate.py` (lines 56-109) — complete inference pattern: device selection, loading, Z→W mapping, synthesis, post-processing, PNG save
- `D:/EchoFace/StyleGAN-Human/dnnlib/util.py` (lines 384-391) — `open_url()` local path handling: plain paths bypass URL detection, opened with `open(url, "rb")`
- `D:/EchoFace/gpu_utils.py` — `get_device_config()` return contract: `device`, `dtype`, `resolution`, `truncation_psi`
- `D:/EchoFace/StyleGAN-Human/training_scripts/sg2/training/networks.py` (lines 176-199) — `MappingNetwork` confirms `z_dim`, `c_dim`, `w_dim`, `num_ws` attributes on G
- `.planning/STATE.md` — locked decisions: `legacy.load_network_pkl()` required; `requires_grad_(False)`; `force_fp32=True`; numpy <1.24

### Secondary (MEDIUM confidence)

- `.planning/research/ARCHITECTURE.md` — StyleGANWrapper class design, anti-patterns, GPU/CPU table
- `.planning/research/STACK.md` — baseline environment constraints (Python 3.8.5, PyTorch 1.9.1, numpy pinned)

---

## Metadata

**Confidence breakdown:**
- Loading pattern (GEN-01): HIGH — source code on disk confirms exact API
- W-space sampling (GEN-01, GEN-02): HIGH — confirmed in generate.py and networks.py
- CPU resize approach (GEN-03): HIGH — standard PyTorch; no external dependency
- `ori_model.txt` side-effect: HIGH — confirmed in legacy.py lines 25-29
- Test mock strategy: MEDIUM — standard pytest mocking; specific mock shapes need verification after first G load

**Research date:** 2026-03-06
**Valid until:** 2026-06-06 (StyleGAN-Human v2 is stable; no active development expected)
