# Phase 5: Dataset Utilities and Evaluation - Research

**Researched:** 2026-03-07
**Domain:** PIL image loading, pytorch-fid programmatic API, Windows DataLoader pitfalls
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `load_dataset(path, max_images=None) -> List[PIL.Image]` — returns PIL Images at original sizes, filesystem order, no shuffle
- Missing dataset: print human-readable message then raise `FileNotFoundError` with exact wording: `"Dataset not found at {path}. Download the Kaggle 'Human Faces Dataset' by kaustubhdhote and place images there."`
- Invalid/unreadable images: skip silently, print `"Skipped N unreadable files"` at end
- `compute_fid(real_dir, gen_dir='outputs/') -> float` — pure function, no print inside function body
- CLI: `if __name__ == '__main__'` with argparse, two positional args: `python utils/metrics.py data/human_faces/ outputs/`
- CLI prints the returned float
- 2048-image minimum check lives in `metrics.py`, not `dataset_loader`
- pytorch-fid is the chosen library

### Claude's Discretion
- Exact error message wording for the 2048-image assertion
- How pytorch-fid is called internally (directory-based vs in-memory tensor path)
- Test mock strategy for PIL.Image.open

### Deferred Ideas (OUT OF SCOPE)
- ATTR-01: Latent attribute direction discovery (SVM/PCA on W-space embeddings) — v2 scope
- SVM vs PCA for latent direction discovery unresolved for StyleGAN-Human v2 — flag for Phase 5.1 or v2 planning
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | utils/dataset_loader.py loads and preprocesses images from data/human_faces/ (Kaggle Human Faces Dataset), handles missing dataset directory gracefully with a clear error message | PIL pathlib glob pattern, sorted() for determinism, PIL.Image.open with error handling |
| EVAL-01 | utils/metrics.py computes FID score between generated face outputs and the Kaggle Human Faces dataset real image distribution using pytorch-fid | calculate_fid_given_paths() API, directory-based invocation, Windows num_workers=0 requirement |
</phase_requirements>

---

## Summary

Phase 5 implements two standalone utilities: a dataset loader that produces PIL Images from the Kaggle Human Faces directory, and a metrics module that runs pytorch-fid to compute a FID score between real and generated images. Both modules follow the project's established pattern of pure/stateless functions that raise on error.

The key technical question for this phase was how to call pytorch-fid programmatically (not just via CLI). The library exposes `calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1)` directly from `pytorch_fid.fid_score`. It operates directory-to-directory — both `real_dir` and `gen_dir` must be filesystem paths containing image files. No in-memory PIL-list path exists in the library; the function reads images from disk itself. This means `compute_fid()` passes the directory strings directly rather than routing through `load_dataset()` for the real images.

The critical Windows pitfall: pytorch-fid's DataLoader defaults `num_workers=1`, which causes `BrokenPipeError` / `EOFError` on Windows with Python 3.8. Must pass `num_workers=0` explicitly. This is the single most likely silent failure on this platform.

**Primary recommendation:** Call `calculate_fid_given_paths([real_dir, gen_dir], batch_size=50, device=device, dims=2048, num_workers=0)`. Use `load_dataset()` only for the 2048-image count check before invoking FID — do not pass PIL images into pytorch-fid.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytorch-fid | >=0.2.1 (0.3.0 current) | FID score computation via InceptionV3 | Already pinned in requirements_extra.txt; official mseitzer port of TF implementation |
| Pillow (PIL) | already installed | Image loading into PIL.Image objects | Project already uses PIL throughout (CLIPEncoder, StyleGANWrapper output) |
| pathlib | stdlib | Directory traversal, image discovery | Deterministic sorted() ordering; cross-platform path handling |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| gpu_utils | project | `get_device_config()` for device/dtype | Pass `cfg['device']` to `calculate_fid_given_paths` for InceptionV3 placement |
| argparse | stdlib | CLI entry point for metrics.py | `if __name__ == '__main__'` block only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytorch-fid (directory-based) | torchmetrics FrechetInceptionDistance | torchmetrics requires tensor inputs and chunked update() calls — more code; pytorch-fid already installed |
| pytorch-fid (directory-based) | clean-fid | Cleaner resizing but adds dependency; not in requirements_extra.txt |

**Installation:** Already installed via `requirements_extra.txt` — no new installs needed for this phase.

---

## Architecture Patterns

### Recommended Project Structure
```
utils/
├── gpu_utils.py          # already done — get_device_config() used by metrics.py
├── dataset_loader.py     # DATA-01: load_dataset(path, max_images=None) -> List[PIL.Image]
├── metrics.py            # EVAL-01: compute_fid(real_dir, gen_dir='outputs/') -> float
└── dataset_loader.py     # (same file — no separate loader class needed)
tests/
├── test_dataset_loader.py  # Wave 0 gap
└── test_metrics.py         # Wave 0 gap
```

### Pattern 1: Sorted Pathlib Glob for Deterministic Loading

**What:** Use `sorted(Path(path).glob('*'))` to enumerate image files in a stable, reproducible order regardless of OS filesystem.
**When to use:** Always — `glob()` returns arbitrary order; `sorted()` gives alphabetical determinism matching CONTEXT.md requirement ("filesystem order for determinism and testability").

```python
# Source: Python stdlib pathlib docs + glob ordering behavior
from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def load_dataset(path, max_images=None):
    p = Path(path)
    if not p.exists() or not p.is_dir():
        msg = (
            f"Dataset not found at {path}. Download the Kaggle "
            "'Human Faces Dataset' by kaustubhdhote and place images there."
        )
        print(msg)
        raise FileNotFoundError(msg)

    files = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        msg = (
            f"Dataset not found at {path}. Download the Kaggle "
            "'Human Faces Dataset' by kaustubhdhote and place images there."
        )
        print(msg)
        raise FileNotFoundError(msg)

    images = []
    skip_count = 0
    for f in files:
        if max_images is not None and len(images) >= max_images:
            break
        try:
            img = Image.open(f).copy()   # .copy() detaches from file handle
            images.append(img)
        except Exception:
            skip_count += 1

    if skip_count:
        print(f"Skipped {skip_count} unreadable files")
    return images
```

**Important:** Call `.copy()` on each opened PIL Image before appending. On Windows, PIL keeps a lazy file handle open; if the Image object is later accessed after the file is gone or the iterator moves on, you get a `PermissionError`. `.copy()` forces decode and detaches the handle. This is the same pattern documented in Phase 2 decisions for PIL on Windows.

### Pattern 2: pytorch-fid Directory-Based Invocation

**What:** Pass two directory path strings to `calculate_fid_given_paths`. The function discovers images internally — do not preload PIL images for this call.
**When to use:** In `compute_fid()`. The 2048-image check uses `load_dataset()` or a quick count; the actual FID call does NOT use the loaded PIL list.

```python
# Source: mseitzer/pytorch-fid fid_score.py public API
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils.gpu_utils import get_device_config

def compute_fid(real_dir, gen_dir='outputs/'):
    cfg = get_device_config()
    device = str(cfg['device'])  # pytorch-fid accepts string device

    # 2048-image minimum check
    real_files = list(Path(real_dir).glob('*'))
    real_img_count = sum(
        1 for f in real_files
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    )
    if real_img_count < 2048:
        raise ValueError(
            f"FID requires at least 2048 real images; found {real_img_count} in {real_dir}. "
            "Download the full Kaggle Human Faces Dataset."
        )

    fid_value = calculate_fid_given_paths(
        [str(real_dir), str(gen_dir)],
        batch_size=50,
        device=device,
        dims=2048,
        num_workers=0,   # REQUIRED on Windows — see Pitfalls
    )
    return fid_value
```

### Pattern 3: CLI Entry Point

```python
# Source: project CONTEXT.md decision
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('real_dir', help='Path to real images directory')
    parser.add_argument('gen_dir', nargs='?', default='outputs/',
                        help='Path to generated images directory (default: outputs/)')
    args = parser.parse_args()
    score = compute_fid(args.real_dir, args.gen_dir)
    print(score)
```

### Anti-Patterns to Avoid
- **Passing PIL Image list to pytorch-fid:** `calculate_fid_given_paths` takes directory path strings, not image objects. There is no in-memory PIL path in the library.
- **num_workers > 0 on Windows:** Default is 1 — this causes BrokenPipeError/EOFError on Windows with PyTorch DataLoader multiprocessing. Always pass `num_workers=0`.
- **Double-resizing images:** Do not resize in `load_dataset()`. pytorch-fid's InceptionV3 resizes internally to 299x299. Pre-resizing degrades FID score.
- **Raw `open(f)` without copy():** PIL lazy-loads; file handle stays open. Call `.copy()` immediately after `Image.open()` to force decode and release the handle.
- **`glob()` without `sorted()`:** `glob()` returns OS-dependent arbitrary order. Always wrap with `sorted()` for reproducible results.
- **`dataset_loader` enforcing the 2048 minimum:** That check belongs in `metrics.py` only. `load_dataset()` is a general-purpose loader; its max_images parameter is for dev speed, not evaluation policy.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FID computation | Custom InceptionV3 feature extractor + Fréchet distance math | `pytorch_fid.fid_score.calculate_fid_given_paths` | Numerically sensitive matrix square root; scipy.linalg.sqrtm used internally; many edge cases |
| Image format discovery | Custom extension list or magic-byte sniffing | PIL.Image.open with try/except + known extension set | PIL handles EXIF, progressive JPEG, palette mode, corrupt headers gracefully via exception |
| FID statistics caching | Custom .npz save/load | `pytorch_fid.fid_score.save_fid_stats` (available but not required for v1) | Same library handles the format — no custom serialization needed |

**Key insight:** FID math (Fréchet distance between multivariate Gaussians, matrix square root via scipy) is deceptively hard to implement correctly. pytorch-fid is the established reference implementation — use it.

---

## Common Pitfalls

### Pitfall 1: Windows DataLoader Multiprocessing Crash
**What goes wrong:** `calculate_fid_given_paths` with default `num_workers=1` raises `BrokenPipeError`, `EOFError`, or silently hangs on Windows with PyTorch 1.9.1 / Python 3.8.
**Why it happens:** Windows uses `spawn` for multiprocessing (not `fork`); PyTorch DataLoader with num_workers > 0 requires pickling that fails on Windows for certain objects.
**How to avoid:** Always pass `num_workers=0` explicitly — this forces single-process data loading, which is slower but reliable.
**Warning signs:** Process hangs indefinitely or raises BrokenPipeError when calling compute_fid().

### Pitfall 2: PIL File Handle Leak on Windows
**What goes wrong:** `Image.open(f)` without `.copy()` keeps the file descriptor open. On Windows, trying to move/delete/re-open the same file while the Image object exists raises `PermissionError`.
**Why it happens:** PIL uses lazy decoding — it holds the file open until the Image is closed or GC'd. Windows file locking is strict.
**How to avoid:** Call `img = Image.open(f).copy()` — force immediate decode, detach file handle.
**Warning signs:** `PermissionError: [WinError 32] The process cannot access the file`. Already documented as a known pattern from Phase 2.

### Pitfall 3: Empty Generated Outputs Directory
**What goes wrong:** `compute_fid()` called before any images are generated — `outputs/` exists (scaffolded in Phase 1) but is empty. pytorch-fid will raise an error internally.
**Why it happens:** The `outputs/` directory is always present (`.gitkeep`), so a path existence check passes. But 0 images cause InceptionV3 activation matrix to be empty.
**How to avoid:** Add a gen_dir image count check (same pattern as real_dir) before calling `calculate_fid_given_paths`. Clear error message: `"No generated images found in {gen_dir}. Run optimization first."`.
**Warning signs:** `RuntimeError` or `ValueError` from deep inside pytorch_fid internals.

### Pitfall 4: Kaggle Dataset Images with Mode 'P' (Palette/Indexed)
**What goes wrong:** Some Kaggle dataset images are indexed palette mode ('P') rather than RGB. CLIP and InceptionV3 both expect 3-channel RGB; mode 'P' images cause shape mismatches downstream.
**Why it happens:** Kaggle datasets often contain PNG files saved with palette mode for compression.
**How to avoid:** Convert on load: `img = Image.open(f).convert('RGB').copy()`. This handles P, L (grayscale), RGBA, and any other mode transparently.
**Warning signs:** `RuntimeError: input channels must be 3` from InceptionV3 inside pytorch-fid.

### Pitfall 5: max_images=None Loads 13k+ Images into RAM
**What goes wrong:** The full Kaggle Human Faces Dataset has ~13,000 images. Loading all into RAM as PIL Images can exhaust memory on systems with <8GB RAM.
**Why it happens:** `load_dataset()` returns List[PIL.Image] — all in memory simultaneously.
**How to avoid:** Document that `max_images=2048` is the recommended call from `compute_fid()`. The function does NOT need to load all real images — it only needs the count check. The actual FID computation reads files from disk via pytorch-fid's own DataLoader. Use a fast count (directory glob) not `load_dataset()` for the 2048 check.
**Warning signs:** `MemoryError` or system swap thrashing during `load_dataset()` call.

---

## Code Examples

Verified patterns from API research and official pytorch-fid source:

### calculate_fid_given_paths Full Signature
```python
# Source: mseitzer/pytorch-fid fid_score.py (verified via WebFetch 2026-03-07)
from pytorch_fid.fid_score import calculate_fid_given_paths

fid_value = calculate_fid_given_paths(
    paths=[str(real_dir), str(gen_dir)],  # list of exactly 2 directory paths
    batch_size=50,                          # InceptionV3 batch size
    device='cpu',                           # or 'cuda:0' — string form
    dims=2048,                              # InceptionV3 pool_3 layer (standard)
    num_workers=0,                          # MUST be 0 on Windows
)
# Returns: float
```

### Mock Strategy for Tests
Following the established project pattern (`utils.dataset_loader.PIL` not top-level `PIL`):

```python
# test_dataset_loader.py pattern
from unittest.mock import patch, MagicMock, call
from utils.dataset_loader import load_dataset

class TestLoadDataset(unittest.TestCase):

    @patch('utils.dataset_loader.Path')
    def test_missing_dir_raises(self, mock_path_cls):
        mock_p = MagicMock()
        mock_p.exists.return_value = False
        mock_path_cls.return_value = mock_p
        with self.assertRaises(FileNotFoundError):
            load_dataset('/nonexistent/path')

    @patch('utils.dataset_loader.Image')
    @patch('utils.dataset_loader.Path')
    def test_skips_unreadable(self, mock_path_cls, mock_image):
        # ... mock iterdir() to return 3 paths, Image.open raises on 1
        pass
```

```python
# test_metrics.py pattern — patch at module namespace level
from unittest.mock import patch
from utils.metrics import compute_fid

class TestComputeFID(unittest.TestCase):

    @patch('utils.metrics.calculate_fid_given_paths')
    @patch('utils.metrics.Path')
    def test_returns_float(self, mock_path_cls, mock_fid_fn):
        # mock Path glob to return 2048+ fake image paths
        # mock calculate_fid_given_paths to return 42.0
        mock_fid_fn.return_value = 42.0
        result = compute_fid('data/human_faces/', 'outputs/')
        self.assertIsInstance(result, float)

    @patch('utils.metrics.Path')
    def test_raises_below_2048(self, mock_path_cls):
        # mock glob to return only 100 image files
        with self.assertRaises(ValueError):
            compute_fid('data/human_faces/', 'outputs/')
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TensorFlow FID (official) | pytorch-fid (PyTorch port) | 2018-2020 | No TF dependency needed |
| `os.listdir()` for image enumeration | `pathlib.Path.iterdir()` + `sorted()` | Python 3.4+ | Deterministic cross-platform ordering |
| `Image.open(f)` file handle leak | `Image.open(f).copy()` | Documented Windows issue | Prevents PermissionError on Windows |

**Deprecated/outdated:**
- `os.path.join` + `os.listdir`: Replaced by pathlib for cross-platform safety — use `Path.iterdir()` instead.
- FID computation with PIL-bicubic pre-resize: clean-fid addressed this, but for our use case pytorch-fid's internal resize is sufficient and already installed.

---

## Open Questions

1. **Should compute_fid() also guard against empty gen_dir?**
   - What we know: CONTEXT.md specifies the 2048 real-image check but is silent on gen_dir guard
   - What's unclear: Whether Phase 6 UI will always ensure outputs/ is non-empty before calling compute_fid()
   - Recommendation: Add a gen_dir image count guard in compute_fid() with a clear error message — cheap to add, prevents confusing pytorch-fid internal errors

2. **Should load_dataset() convert all images to RGB?**
   - What we know: Kaggle datasets can contain palette-mode ('P') images; CONTEXT.md says "no resize" but is silent on mode
   - What's unclear: Whether the Kaggle Human Faces Dataset specifically contains non-RGB images
   - Recommendation: Always `.convert('RGB')` on load — zero cost for already-RGB images, prevents downstream errors; consistent with how CLIPEncoder uses images

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | unittest (Python stdlib, Python 3.8.5 compatible) |
| Config file | none — run directly via pytest discovery or python -m unittest |
| Quick run command | `conda run -n stylehuman python -m pytest tests/test_dataset_loader.py tests/test_metrics.py -v` |
| Full suite command | `conda run -n stylehuman python -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | load_dataset returns list of PIL Images | unit | `python -m pytest tests/test_dataset_loader.py -v` | Wave 0 |
| DATA-01 | load_dataset raises FileNotFoundError on missing dir | unit | `python -m pytest tests/test_dataset_loader.py::TestLoadDataset::test_missing_dir_raises -v` | Wave 0 |
| DATA-01 | load_dataset prints and raises on empty dir | unit | `python -m pytest tests/test_dataset_loader.py::TestLoadDataset::test_empty_dir_raises -v` | Wave 0 |
| DATA-01 | load_dataset skips unreadable files, prints count | unit | `python -m pytest tests/test_dataset_loader.py::TestLoadDataset::test_skips_unreadable -v` | Wave 0 |
| DATA-01 | max_images caps the returned list length | unit | `python -m pytest tests/test_dataset_loader.py::TestLoadDataset::test_max_images -v` | Wave 0 |
| EVAL-01 | compute_fid raises ValueError below 2048 real images | unit | `python -m pytest tests/test_metrics.py::TestComputeFID::test_raises_below_2048 -v` | Wave 0 |
| EVAL-01 | compute_fid calls calculate_fid_given_paths with num_workers=0 | unit | `python -m pytest tests/test_metrics.py::TestComputeFID::test_num_workers_zero -v` | Wave 0 |
| EVAL-01 | compute_fid returns float | unit | `python -m pytest tests/test_metrics.py::TestComputeFID::test_returns_float -v` | Wave 0 |
| EVAL-01 | compute_fid raises on empty gen_dir | unit | `python -m pytest tests/test_metrics.py::TestComputeFID::test_raises_empty_gen_dir -v` | Wave 0 |

### Sampling Rate
- **Per task commit:** `conda run -n stylehuman python -m pytest tests/test_dataset_loader.py tests/test_metrics.py -v`
- **Per wave merge:** `conda run -n stylehuman python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_dataset_loader.py` — covers DATA-01 (5 tests)
- [ ] `tests/test_metrics.py` — covers EVAL-01 (4 tests)
- [ ] `utils/dataset_loader.py` — stub with `load_dataset` importing `PIL.Image` and `pathlib.Path` at module level (required for mock patching)
- [ ] `utils/metrics.py` — stub with `compute_fid` importing `calculate_fid_given_paths` and `Path` at module level (required for mock patching)

---

## Sources

### Primary (HIGH confidence)
- mseitzer/pytorch-fid fid_score.py — verified `calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1)` signature directly from source
- PyPI pytorch-fid — confirmed version 0.3.0, Python >=3.5 requirement; pinned >=0.2.1 in project requirements_extra.txt
- Python stdlib pathlib docs — `glob()` ordering behavior, `sorted()` requirement for determinism

### Secondary (MEDIUM confidence)
- PyTorch forums + GitHub issues on Windows DataLoader multiprocessing — `num_workers=0` fix for BrokenPipeError/EOFError, multiple corroborating sources
- Project CONTEXT.md / STATE.md — established patterns: `.copy()` after `Image.open()` on Windows (Phase 2 decision), namespace patching for mocks

### Tertiary (LOW confidence)
- Kaggle Human Faces Dataset palette-mode image claim — inferred from general Kaggle dataset behavior; not verified against the specific dataset

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pytorch-fid already in requirements_extra.txt; API verified from source
- Architecture: HIGH — function signatures locked by CONTEXT.md decisions; directory-based approach confirmed by pytorch-fid source
- Pitfalls: HIGH (Windows num_workers, PIL copy) / MEDIUM (palette mode) — Windows multiprocessing issue is well-documented across multiple PyTorch issues

**Research date:** 2026-03-07
**Valid until:** 2026-09-07 (pytorch-fid is stable; last release Jan 2023)
