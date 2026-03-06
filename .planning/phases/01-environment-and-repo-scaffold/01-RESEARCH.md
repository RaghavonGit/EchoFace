# Phase 1: Environment and Repo Scaffold - Research

**Researched:** 2026-03-06
**Domain:** Python environment setup, conda/pip dependency management, project scaffolding
**Confidence:** HIGH — base environment pinned from file on disk; extra dep pins derived from confirmed version conflict analysis and actual StyleGAN-Human source code on disk.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | Developer can run a validation script that confirms all extra dependencies install cleanly on Python 3.8.5 / PyTorch 1.9.1 / CUDA 11.1 base environment | Import smoke test pattern documented; exact imports and CUDA check code provided |
| ENV-02 | Repository contains full directory scaffold with .gitkeep placeholders | Full directory tree with rationale documented; .gitkeep convention confirmed |
| ENV-03 | requirements_extra.txt with pinned versions installable on top of StyleGAN-Human base conda env without conflicts | Exact content derived from environment.yml (on disk) + conflict map; every pin justified |
| ENV-04 | gpu_utils.py detects CUDA availability, selects fp16 vs fp32 mode, returns correct resolution and device string | Implementation pattern fully documented with actual API calls from generate.py source (on disk) |
</phase_requirements>

---

## Summary

Phase 1 establishes the foundation every subsequent phase depends on. The work is concrete and low-ambiguity: the base conda environment already exists in `StyleGAN-Human/environment.yml` (confirmed on disk, Python 3.8.5 / PyTorch 1.9.1 / CUDA 11.1), the pretrained weights are already present at `StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl` (confirmed on disk), and the StyleGAN-Human source is fully cloned. Phase 1 does not re-clone or re-download the base — it adds extra pip dependencies on top, creates the EchoFace project structure, writes the validation script, and implements `gpu_utils.py`.

The primary risk in this phase is dependency conflict between the extra pip packages and the pinned conda base. Three traps are well-documented: Gradio 4.x breaks on Python 3.8 (use `gradio>=3.40,<4.0`), librosa 0.10+ bumps scipy above the pinned 1.7.1 (pin `librosa==0.9.2`), and tiktoken 0.6+ may lack Python 3.8 wheels (pin `tiktoken==0.5.2`). A fourth trap — numpy 1.24 deprecating `np.bool`/`np.int` aliases used in StyleGAN-Human internals — requires capping `numpy<1.24`. All four pins must appear in `requirements_extra.txt` before any other phase begins.

`gpu_utils.py` is a pure utility with no ML logic — it wraps `torch.cuda.is_available()`, selects device and dtype, and returns a structured result used by every downstream module. Its implementation pattern is directly verifiable from `StyleGAN-Human/generate.py` on disk, which shows the exact device and dtype usage the generator expects.

**Primary recommendation:** Install extra deps in the order documented below (openai-whisper first to let it resolve tiktoken, then pin tiktoken down, then CLIP, then librosa, then gradio), run `pip check` after all installs, and run the validation script before declaring Phase 1 done.

---

## Standard Stack

### Core — Base Environment (Non-Negotiable)

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| python | 3.8.5 | Runtime | StyleGAN-Human/environment.yml (on disk) |
| pytorch | 1.9.1 | Tensor ops, GPU acceleration | StyleGAN-Human/environment.yml (on disk) |
| cudatoolkit | 11.1 | GPU kernel runtime | StyleGAN-Human/environment.yml (on disk) |
| numpy | >=1.20 (pinned in yml) | Array operations | StyleGAN-Human/environment.yml (on disk) |
| pillow | 8.3.1 | Image I/O | StyleGAN-Human/environment.yml (on disk) |
| scipy | 1.7.1 | Signal processing deps | StyleGAN-Human/environment.yml (on disk) |
| lpips | 0.1.4 | Already installed by StyleGAN-Human pip block | StyleGAN-Human/environment.yml (on disk) |

**Do NOT upgrade any of the above.** StyleGAN-Human's `dnnlib`, `torch_utils`, and LPIPS are tested only against this triplet. Any upgrade breaks the generator.

### Extra Dependencies (requirements_extra.txt)

| Library | Pin | Purpose | Why This Pin |
|---------|-----|---------|--------------|
| openai-whisper | latest (no pin) | Local ASR transcription | torch>=1.7.1 requirement satisfied; no version floor issues |
| tiktoken | ==0.5.2 | Whisper's tokenizer dep | 0.6+ lacks Python 3.8 wheels in some builds; pin as guard |
| clip (from GitHub) | HEAD | Text + image embeddings | No PyPI release; GitHub URL is the only correct install |
| librosa | ==0.9.2 | Audio preprocessing, resampling | 0.10+ requires scipy>=1.7.3; base env pins scipy=1.7.1 |
| sounddevice | >=0.4.4 | Microphone capture | No Python version floor; stable |
| soundfile | >=0.10.3 | WAV/MP3 file I/O | libsndfile backend; stable |
| gradio | >=3.40,<4.0 | Localhost UI | 4.x uses Python 3.10+ syntax; breaks on Python 3.8.5 |
| pytorch-fid | >=0.2.1 | FID scoring utility | 0.2.x compatible with PyTorch 1.9.x |
| numpy | <1.24 | Cap to avoid deprecated alias hits | numpy 1.24 removed np.bool/np.int used in StyleGAN internals |

### Already Installed by Base Environment (do NOT reinstall)

| Library | Why it's there |
|---------|----------------|
| opencv-python | StyleGAN-Human pip block |
| pandas | StyleGAN-Human pip block |
| matplotlib | StyleGAN-Human conda block |
| lpips | StyleGAN-Human pip block — do not install again |
| imageio | StyleGAN-Human conda block |

### Installation Sequence

Order matters. Run these commands inside the activated `stylehuman` conda environment:

```bash
# Activate the base env first
conda activate stylehuman

# Step 1: Verify torchvision (CLIP dependency — not explicit in environment.yml)
python -c "import torchvision; print(torchvision.__version__)"
# If missing:
pip install torchvision==0.10.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111

# Step 2: Whisper first (lets it resolve tiktoken freely, then we pin down)
pip install openai-whisper

# Step 3: Immediately pin tiktoken to avoid 0.6+ wheel gap
pip install "tiktoken==0.5.2"

# Step 4: CLIP from GitHub (ONLY this URL — not PyPI)
pip install git+https://github.com/openai/CLIP.git

# Step 5: Audio libs
pip install "librosa==0.9.2" sounddevice soundfile

# Step 6: Gradio — 3.x only
pip install "gradio>=3.40,<4.0"

# Step 7: FID utility
pip install pytorch-fid

# Step 8: Numpy cap (must come after all other installs)
pip install "numpy<1.24"

# Step 9: Verify no conflicts
pip check
```

**requirements_extra.txt content** (for reproducibility — used alongside the conda env, not standalone):

```
openai-whisper
tiktoken==0.5.2
clip @ git+https://github.com/openai/CLIP.git
librosa==0.9.2
sounddevice>=0.4.4
soundfile>=0.10.3
gradio>=3.40,<4.0
pytorch-fid>=0.2.1
numpy<1.24
```

---

## Architecture Patterns

### Recommended Project Structure

The EchoFace repo structure must be created from scratch alongside the existing `StyleGAN-Human/` clone. Required directories and their rationale:

```
D:/EchoFace/
├── StyleGAN-Human/          # Cloned, DO NOT modify — contains environment.yml, legacy.py, generate.py
│   └── pretrained_models/   # stylegan_human_v2_1024.pkl already present
├── asr/                     # Phase 4: Whisper transcriber, audio preprocessor
│   └── .gitkeep
├── encoder/                 # Phase 3: CLIP text/image encoder
│   └── .gitkeep
├── generator/               # Phase 2: StyleGAN-Human adapter wrapper
│   └── .gitkeep
├── optimizer/               # Phase 3: CLIP-guided W-space optimizer
│   └── .gitkeep
├── ui/                      # Phase 6: Gradio app
│   └── .gitkeep
├── utils/                   # Phase 5: dataset_loader, metrics, audio_preprocess
│   └── .gitkeep
├── outputs/                 # Runtime: generated PNG images land here
│   └── .gitkeep
├── data/
│   └── human_faces/         # User-placed Kaggle dataset — must exist, may be empty
│       └── .gitkeep
├── gpu_utils.py             # ENV-04: Device/dtype/resolution selector — used by all modules
├── validate_env.py          # ENV-01: Import smoke test + CUDA check
└── requirements_extra.txt   # ENV-03: Pinned extra deps
```

**Why these directories now:** Every module in phases 2-6 imports from its own package directory. Creating the scaffold in Phase 1 ensures that `import generator.stylegan_wrapper` works from the project root without `sys.path` hacks inside application code. The `outputs/` directory must exist before any generation runs or PIL's `save()` will raise `FileNotFoundError`. `data/human_faces/` must exist so `dataset_loader.py` can distinguish "directory missing" from "directory empty".

### Pattern 1: gpu_utils.py — Device and Dtype Selection

This module is the single source of truth for device, dtype, and resolution across all modules. Every downstream module calls `get_device_config()` rather than calling `torch.cuda.is_available()` inline.

**What:** Returns a named dict (or dataclass) with `device`, `dtype`, `use_fp16`, and `resolution`.
**When to use:** At module load time in any file that needs to know GPU/CPU context. Called once per process, result passed as argument.

```python
# gpu_utils.py
# Source: pattern derived from StyleGAN-Human/generate.py (on disk, lines 66-68, 101-108)

import torch

def get_device_config() -> dict:
    """
    Detects CUDA availability and returns device configuration used by all
    downstream modules. Returns consistent settings regardless of call order.

    Returns:
        dict with keys:
            device   (torch.device) — 'cuda' or 'cpu'
            dtype    (torch.dtype)  — torch.float16 on CUDA, torch.float32 on CPU
            use_fp16 (bool)         — True on CUDA only; Whisper transcribe flag
            resolution (int)        — 1024 on CUDA, 256 on CPU
            truncation_psi (float)  — 1.0 on CUDA, 0.5 on CPU (per GEN-03)
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    return {
        "device": device,
        "dtype": torch.float16 if cuda_available else torch.float32,
        "use_fp16": cuda_available,
        "resolution": 1024 if cuda_available else 256,
        "truncation_psi": 1.0 if cuda_available else 0.5,
    }
```

Key facts from generate.py source (verified on disk):
- generate.py uses `device = torch.device('cuda')` — no CPU fallback in that script
- The synthesis call is: `G.synthesis(w, noise_mode=noise_mode, force_fp32=True)` — note `force_fp32=True` is passed explicitly in generate.py for safe inference
- W-space latent `w` is produced via: `w = G.mapping(z, label, truncation_psi=truncation_psi)`
- Image conversion from StyleGAN output: `(img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)`

**IMPORTANT:** generate.py passes `force_fp32=True` to `G.synthesis()`. This means even on CUDA, synthesis runs in float32 by default in the reference implementation. For the optimization loop, the `dtype` in `gpu_utils.py` controls the W-space latent dtype and Adam optimizer dtype — NOT the synthesis network itself. The generator network can be kept at float32 throughout and will still benefit from CUDA speed.

### Pattern 2: validate_env.py — Import Smoke Test

**What:** Runs at end of Phase 1 to confirm every extra dependency is importable and CUDA is detected as expected. Produces clear PASS/FAIL output.
**When to use:** Once after `pip install -r requirements_extra.txt` completes. Re-run after any `pip install` that touches the env.

```python
# validate_env.py — ENV-01

import sys

def check(label: str, fn):
    try:
        result = fn()
        print(f"  [PASS] {label}" + (f": {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

def main():
    print("=== EchoFace Environment Validation ===\n")
    failures = 0

    # Core torch
    failures += 0 if check("torch", lambda: __import__("torch").__version__) else 1
    failures += 0 if check("torch.cuda.is_available", lambda: __import__("torch").cuda.is_available()) else 1

    # Extra dependencies
    failures += 0 if check("gradio", lambda: __import__("gradio").__version__) else 1
    failures += 0 if check("whisper", lambda: __import__("whisper") and "loaded") else 1
    failures += 0 if check("clip", lambda: __import__("clip") and "loaded") else 1
    failures += 0 if check("librosa", lambda: __import__("librosa").__version__) else 1
    failures += 0 if check("soundfile", lambda: __import__("soundfile").__version__) else 1
    failures += 0 if check("numpy (version cap)", lambda: (
        lambda np: np.__version__ if tuple(int(x) for x in np.__version__.split(".")[:2]) < (1, 24)
        else (_ for _ in ()).throw(RuntimeError(f"numpy {np.__version__} >= 1.24 — must cap"))
    )(__import__("numpy"))) else 1

    # CUDA detection
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"\n  CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # gpu_utils smoke test
    try:
        import gpu_utils
        cfg = gpu_utils.get_device_config()
        print(f"\n  gpu_utils.get_device_config() returned:")
        for k, v in cfg.items():
            print(f"    {k}: {v}")
        print(f"  [PASS] gpu_utils")
    except Exception as e:
        print(f"  [FAIL] gpu_utils: {e}")
        failures += 1

    print(f"\n{'='*40}")
    if failures == 0:
        print("RESULT: ALL CHECKS PASSED — environment is ready")
    else:
        print(f"RESULT: {failures} CHECK(S) FAILED — resolve before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Anti-Patterns to Avoid

- **Installing CLIP from PyPI:** `pip install clip` installs an unrelated package. `pip install openai` installs the API SDK. GitHub URL only.
- **Installing Gradio 4.x:** `pip install gradio` without a version pin will pull Gradio 5.x or 4.x which uses Python 3.10+ syntax. Always pin `gradio>=3.40,<4.0`.
- **Reinstalling lpips:** The base conda env already installs `lpips==0.1.4`. Reinstalling risks version drift and breaks StyleGAN training utilities.
- **Skipping `pip check`:** Silent conflicts between conda-installed and pip-installed packages are the primary failure mode in this phase. `pip check` must be a required step.
- **Calling `torch.cuda.is_available()` in multiple modules:** Centralizing in `gpu_utils.py` prevents inconsistency if the check is ever conditional on environment variables.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CUDA device detection | Custom `nvidia-smi` subprocess call | `torch.cuda.is_available()` + `torch.cuda.get_device_name()` | PyTorch already wraps CUDA device query cleanly |
| Dependency conflict detection | Custom version comparison script | `pip check` | pip's resolver has full dependency graph; custom scripts miss transitive conflicts |
| Import validation | Try/except per file in production code | `validate_env.py` as a standalone script | Validation belongs in setup, not in application import chain |
| .gitkeep creation | Shell script loops | Explicit file creation in setup notes | One-time action; no abstraction needed |

---

## Common Pitfalls

### Pitfall 1: Gradio 4.x / 5.x installed by pip's default resolution
**What goes wrong:** `pip install gradio` without version bounds pulls the latest (5.x or 4.29+), which uses Python 3.10+ syntax (`match` statements, newer type hints). The install may succeed (pip installs the wheel), but `import gradio` crashes with `SyntaxError` at runtime.
**Why it happens:** pip does not check Python syntax compatibility in wheel contents — it only checks Python version tags in wheel metadata, and some Gradio 4.x wheels are tagged as `py3` (any Python 3.x).
**How to avoid:** Pin `gradio>=3.40,<4.0` in requirements_extra.txt. Verify with `python -c "import gradio; print(gradio.__version__)"` immediately after install.
**Warning signs:** `SyntaxError: invalid syntax` on `import gradio`, or a version number >=4.0 printed by the verify command.

### Pitfall 2: tiktoken 0.6+ lacks Python 3.8 wheel
**What goes wrong:** `pip install openai-whisper` pulls the latest tiktoken as a transitive dependency. If tiktoken 0.6+ is selected and no Python 3.8 wheel exists for that version, pip falls back to building from source, which fails silently or produces a broken install.
**Why it happens:** tiktoken uses Rust extensions. Some release versions only publish wheels for Python 3.9+.
**How to avoid:** Immediately after installing openai-whisper, run `pip install "tiktoken==0.5.2"` to force the known-good version. This pin overrides whatever openai-whisper resolved.
**Warning signs:** `import tiktoken` raises `ModuleNotFoundError` or `ImportError` even though pip reports it as installed.

### Pitfall 3: librosa 0.10+ scipy conflict
**What goes wrong:** `pip install librosa` without a pin installs 0.10.x, which internally imports `scipy.signal` functions that require scipy >= 1.7.3. The base env has scipy == 1.7.1. The error only appears when calling librosa functions, not at import time.
**Why it happens:** The scipy incompatibility is in librosa's runtime code, not its setup.py requirements, so pip does not flag it during install.
**How to avoid:** Pin `librosa==0.9.2` explicitly. No other librosa version should be used.
**Warning signs:** `AttributeError` or `ImportError` inside librosa functions at runtime, not at `import librosa`.

### Pitfall 4: numpy 1.24+ breaks StyleGAN-Human internals
**What goes wrong:** numpy 1.24 removed the deprecated `np.bool`, `np.int`, `np.float`, `np.complex` aliases. StyleGAN-Human's utility scripts may use these. The failure surfaces as `AttributeError: module 'numpy' has no attribute 'bool'` when running any StyleGAN-Human code path.
**Why it happens:** numpy was installed without an upper bound; pip selected the latest.
**How to avoid:** Add `numpy<1.24` to requirements_extra.txt. Install it as the last step after all other deps, so any dep that wants a newer numpy is already satisfied with 1.23.x.
**Warning signs:** `AttributeError: module 'numpy' has no attribute 'bool'` anywhere in the stack.

### Pitfall 5: torchvision missing from conda env
**What goes wrong:** StyleGAN-Human's environment.yml does not explicitly list torchvision. The conda pytorch channel sometimes installs it as a companion package, sometimes does not. CLIP requires torchvision at import time.
**Why it happens:** torchvision's inclusion in the pytorch conda channel is version-dependent.
**How to avoid:** Run `python -c "import torchvision"` after conda env setup. If it fails, install `torchvision==0.10.1+cu111` from the PyTorch wheel index.
**Warning signs:** `ModuleNotFoundError: No module named 'torchvision'` when importing CLIP.

### Pitfall 6: Wrong working directory when validate_env.py runs
**What goes wrong:** `gpu_utils.py` is at the project root (`D:/EchoFace/gpu_utils.py`). If `validate_env.py` is run from a different directory, `import gpu_utils` fails.
**Why it happens:** `gpu_utils.py` is a top-level module, not in a package, so it is only importable when the project root is on `sys.path` or is the working directory.
**How to avoid:** Document in setup notes that `validate_env.py` must be run from `D:/EchoFace/` (the project root). Alternatively, add `sys.path.insert(0, os.path.dirname(__file__))` at the top of `validate_env.py`.
**Warning signs:** `ModuleNotFoundError: No module named 'gpu_utils'` when running the validation script.

---

## Code Examples

Verified from `StyleGAN-Human/generate.py` source (on disk):

### Official G.mapping + G.synthesis Call Pattern
```python
# Source: D:/EchoFace/StyleGAN-Human/generate.py lines 94-108 (on disk, verified)
import torch
import dnnlib
import legacy

device = torch.device('cuda')

# Load network — MUST use dnnlib.util.open_url + legacy.load_network_pkl
with dnnlib.util.open_url("StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# Z → W mapping (run ONCE at init, not in optimization loop)
label = torch.zeros([1, G.c_dim], device=device)
z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
w = G.mapping(z, label, truncation_psi=1.0)   # shape: [1, 18, 512]

# Synthesis from W (this is what the optimizer calls every step)
img = G.synthesis(w, noise_mode='const', force_fp32=True)

# Tensor → PIL image conversion (verified from generate.py line 108)
img_uint8 = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
from PIL import Image
pil_img = Image.fromarray(img_uint8[0].cpu().numpy(), 'RGB')
```

### gpu_utils.py Implementation
```python
# gpu_utils.py — ENV-04
import torch

def get_device_config() -> dict:
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    return {
        "device": device,
        "dtype": torch.float16 if cuda_available else torch.float32,
        "use_fp16": cuda_available,
        "resolution": 1024 if cuda_available else 256,
        "truncation_psi": 1.0 if cuda_available else 0.5,
    }
```

### Whisper fp16 Flag Pattern (from PITFALLS — M1)
```python
# Always compute fp16 from device, never hardcode True
import whisper
import gpu_utils

cfg = gpu_utils.get_device_config()
model = whisper.load_model("medium")
result = model.transcribe(audio_array, fp16=cfg["use_fp16"])
```

### legacy.load_network_pkl — Actual Source Signature
```python
# Source: D:/EchoFace/StyleGAN-Human/legacy.py lines 22-73 (on disk, verified)
# The function signature:
def load_network_pkl(f, force_fp16=False, G_only=False):
    # Uses _LegacyUnpickler (custom unpickler) not raw pickle.Unpickler
    # Returns dict with keys: 'G', 'G_ema', 'D', 'training_set_kwargs', 'augment_pipe'
    # For inference, always use: data['G_ema']
    ...
```

Note: `load_network_pkl` has a `force_fp16` parameter. Do NOT use `force_fp16=True` for Phase 1 or the optimization loop — it converts the full generator to fp16, which can cause numerical instability during gradient computation. Use the default `force_fp16=False` and let the optimizer manage latent dtype separately.

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `pickle.load(open(pkl_path, 'rb'))` | `dnnlib.util.open_url(pkl_path)` + `legacy.load_network_pkl(f)` | Avoids UnpicklingError from custom dnnlib class registration |
| Gradio 4.x for Python 3.8 | Gradio >=3.40,<4.0 | Prevents SyntaxError from match statements |
| `librosa` latest | `librosa==0.9.2` | Prevents scipy 1.7.1 incompatibility |
| Global `sys.path.insert` everywhere | Single adapter module does all sys.path manipulation | Prevents import coupling to StyleGAN-Human internals |

**Deprecated / do not use:**
- `pickle.load` for .pkl files — always use legacy loader
- `pip install clip` from PyPI — wrong package; use GitHub URL
- `pip install openai` for CLIP — that is the API SDK requiring OPENAI_API_KEY
- Gradio 4.x or 5.x — Python 3.10+ requirement

---

## Open Questions

1. **torchvision presence in stylehuman conda env on this machine**
   - What we know: environment.yml does not list torchvision explicitly
   - What's unclear: Whether the conda pytorch channel auto-installed it alongside pytorch 1.9.1
   - Recommendation: First task in Phase 1 plan should verify `python -c "import torchvision"` and install if missing

2. **Gradio 3.x exact version floor for Python 3.8 compatibility**
   - What we know: PITFALLS.md specifies `gradio>=3.40,<4.0`; STACK.md had an inconsistent recommendation of `gradio==4.19.2` which is now overridden
   - What's unclear: The exact floor within 3.x (3.35? 3.40?) at which mic capture components work correctly
   - Recommendation: Use `gradio>=3.40,<4.0` as the pin; `pip check` + smoke import is the verification gate

3. **ffmpeg system installation on Windows**
   - What we know: librosa delegates MP3 decode to system ffmpeg; it must be on PATH
   - What's unclear: Whether ffmpeg is already installed on the developer's Windows machine
   - Recommendation: Phase 1 plan should include a step to verify `ffmpeg -version` and document installation via `winget install ffmpeg` or conda-forge

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None (validate_env.py is a standalone script, not pytest) |
| Config file | None — Wave 0 creates validate_env.py |
| Quick run command | `python validate_env.py` (from project root) |
| Full suite command | `python validate_env.py` (same — single script for this phase) |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| ENV-01 | All imports succeed + CUDA detected | smoke | `python validate_env.py` | ❌ Wave 0 |
| ENV-02 | All scaffold directories exist with .gitkeep | smoke | `python validate_env.py` (add dir checks) | ❌ Wave 0 |
| ENV-03 | `pip check` exits 0 after requirements_extra.txt install | manual | `pip check` | N/A — manual |
| ENV-04 | `get_device_config()` returns correct keys and types | unit | `python -c "import gpu_utils; cfg=gpu_utils.get_device_config(); assert 'device' in cfg"` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python validate_env.py`
- **Per wave merge:** `python validate_env.py` + `pip check`
- **Phase gate:** `validate_env.py` exits 0 before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `validate_env.py` — covers ENV-01, ENV-02, ENV-04
- [ ] `gpu_utils.py` — covers ENV-04
- [ ] `requirements_extra.txt` — covers ENV-03

None of these exist yet — all are created in Phase 1.

---

## Sources

### Primary (HIGH confidence)
- `D:/EchoFace/StyleGAN-Human/environment.yml` — verified on disk; exact version pins for base env
- `D:/EchoFace/StyleGAN-Human/generate.py` — verified on disk; exact G.mapping/G.synthesis API, image conversion formula, dnnlib.util.open_url usage
- `D:/EchoFace/StyleGAN-Human/legacy.py` — verified on disk; load_network_pkl signature, force_fp16 parameter, G_ema key, _LegacyUnpickler usage
- `D:/EchoFace/StyleGAN-Human/pretrained_models/` — verified on disk; stylegan_human_v2_1024.pkl present

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` — version conflict map; Gradio/tiktoken/librosa/numpy pins with rationale
- `.planning/research/PITFALLS.md` — C1-C13 and M1-Mi3 pitfall catalog; Whisper fp16, numpy alias, CLIP package confusion
- `.planning/research/SUMMARY.md` — ecosystem overview, phase ordering rationale

### Tertiary (LOW confidence — verify at install time)
- tiktoken 0.6+ Python 3.8 wheel availability — may have been resolved in later tiktoken releases; pin 0.5.2 is conservative
- pytorch-fid PyTorch 1.9.x compatibility — no direct verification; 0.2.x is reported compatible in training data
- Gradio 3.x exact version floor — 3.40 is documented recommendation; actual minimum for mic capture may differ slightly

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — base env from file on disk; extra pins derived from documented conflict map + verified source code
- Architecture: HIGH — directory structure derived directly from module list in REQUIREMENTS.md; generate.py API verified on disk
- Pitfalls: HIGH for C2 (legacy loader), C8 (Gradio version), M1 (Whisper fp16); MEDIUM for tiktoken, torchvision presence

**Research date:** 2026-03-06
**Valid until:** 2026-06-06 (stable ecosystem; 90 days reasonable given Python 3.8 EOL status meaning no new breaking releases expected)
