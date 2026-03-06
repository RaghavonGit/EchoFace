# Technology Stack

**Project:** EchoFace
**Researched:** 2026-03-06
**Confidence:** MEDIUM — web search unavailable; findings based on training data (cutoff Aug 2025) cross-referenced against the concrete environment.yml from StyleGAN-Human/environment.yml. Version-specific compatibility claims are flagged individually.

---

## Baseline Environment (Non-Negotiable)

The StyleGAN-Human conda environment is the immovable foundation. Every extra dependency must fit inside it.

```
conda env: stylehuman
python:       3.8 (pinned)
pytorch:      1.9.1
cudatoolkit:  11.1
numpy:        >=1.20
pillow:       8.3.1
scipy:        1.7.1
```

Source: `D:/EchoFace/StyleGAN-Human/environment.yml` — HIGH confidence (file on disk).

Do NOT upgrade Python, PyTorch, or CUDA. StyleGAN-Human's dnnlib, torch_utils, and LPIPS are tested only against this triplet. Upgrading breaks the generator.

---

## Recommended Stack

### ASR — Speech to Text

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| openai-whisper | 20231117 (latest stable as of mid-2025) | Transcribe mic/file audio to text | Fully local, no API key, ships model weights to ~/.cache/whisper/, medium model ~1.5 GB VRAM, tested on Python 3.8 + PyTorch 1.x |

**Install:**
```
pip install openai-whisper
```

**Confidence:** MEDIUM. openai-whisper was designed for PyTorch 1.x compatibility and is routinely used in offline setups. The library's setup.py specifies `torch>=1.7.1`. PyTorch 1.9.1 satisfies this. The pinned version string `20231117` is the wheel tag for the last stable release before the project adopted newer features; if pip resolves a newer minor, that is also acceptable as long as it does not pull in PyTorch >= 2.0 as a hard dependency (it does not — torch is listed as `>=`, not `==`).

**API key requirement:** None. Model weights are downloaded from OpenAI's CDN on first call, then cached locally. Zero network calls at inference time.

**Dependency to watch:** openai-whisper pulls in `tiktoken>=0.3.1`. tiktoken 0.3.x–0.5.x builds cleanly on Python 3.8 with no conflicts against the base env. Versions 0.6+ require Python 3.9+ in some wheels — if pip selects tiktoken >=0.6, pin to `tiktoken==0.5.2`.

---

### CLIP Encoding

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| openai/CLIP (GitHub HEAD) | clip==1.0 (internal version string) | Text and image embeddings for optimization loop | The ONLY official CLIP library; runs locally on ViT-B/32; no API key; weights downloaded once to torch hub cache |

**Install:**
```
pip install git+https://github.com/openai/CLIP.git
```

**Do NOT install `clip` from PyPI** — that is an unrelated package. Do NOT install `openai` from PyPI — that is the SDK that requires `OPENAI_API_KEY`. The GitHub install is the correct one, as specified in PROJECT.md.

**Confidence:** HIGH. The openai/CLIP GitHub library has no API key dependency; it downloads ViT weights directly via `urllib` to torch hub cache (~/.cache/clip/). Compatible with Python 3.8 and PyTorch 1.9.1 (CLIP's own requirements: `torch`, `torchvision`, `ftfy`, `regex`, `tqdm` — all compatible).

**Dependency note:** CLIP requires `torchvision`. StyleGAN-Human's environment.yml does NOT explicitly list torchvision, but the `pytorch=1.9.1` conda channel typically installs `torchvision=0.10.1` alongside it. Verify with `python -c "import torchvision; print(torchvision.__version__)"` after env setup. If missing, install `torchvision==0.10.1+cu111` from the PyTorch wheel index.

---

### Audio I/O and Preprocessing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| sounddevice | >=0.4.4 | Real-time mic capture | Minimal, cross-platform, no audio driver abstraction overhead; returns numpy arrays directly |
| soundfile | >=0.10.3 | Read/write .wav/.mp3/.flac files | libsndfile-backed, handles 16kHz resampling, pairs with sounddevice |
| librosa | >=0.8.1, <0.10 | Noise filtering, amplitude normalization, silence trimming | Industry standard for audio DSP; 0.8.x branch is the Python 3.8 / numpy 1.20 sweet spot |
| ffmpeg-python | >=0.2.0 | MP3 decode (librosa delegates to ffmpeg for compressed formats) | Required when accepting .mp3 input; wrap system ffmpeg binary |

**Confidence:** HIGH for sounddevice/soundfile/librosa — these are stable, widely used, have no Python 3.9+ hard dependencies in these version ranges. MEDIUM for ffmpeg-python — requires system ffmpeg installed separately (not a pip dependency); user must have ffmpeg on PATH.

**Version trap:** librosa 0.10+ dropped Python 3.8 support in some sub-releases and bumped scipy requirements above what the base env pins (scipy=1.7.1). Pin `librosa<0.10` to stay within the conda-pinned scipy. Specifically `librosa==0.9.2` is the recommended pin.

**API key requirement:** None.

---

### Gradio UI

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| gradio | >=4.0, <5.0 | Localhost web UI with mic input, live preview, sliders | Gradio 4.x introduced real-time streaming components and mic capture; runs on localhost:7860 with no external service |

**Confidence:** LOW — CRITICAL COMPATIBILITY WARNING.

Gradio 4.x (released late 2023) raised its minimum Python requirement to **Python 3.8 is still supported in Gradio 4.x early releases but Gradio 4.29+ may require Python >=3.9 in some builds**. This is the single highest-risk dependency in the stack.

**Recommended mitigation:** Pin to `gradio==4.19.2`. This is a known-good Gradio 4.x release that explicitly lists Python >=3.8 in its wheel metadata and was the last widely-tested release on Python 3.8 before newer builds began targeting 3.9+.

**Verification required (Phase 1):** After installing the base conda env, run:
```bash
pip install gradio==4.19.2
python -c "import gradio as gr; print(gr.__version__)"
```
If this fails, fall back to `gradio==3.50.2` (Gradio 3.x, stable Python 3.8 support, slightly different component API but fully functional for this use case).

**Gradio 5.x:** Do NOT use. Gradio 5 requires Python >=3.10 and restructures the component API entirely.

**API key requirement:** None. Gradio runs entirely locally. The `share=False` flag (default) prevents any Cloudflare tunnel from being created.

---

### Image Processing (Optimization Loop Output)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Pillow | 8.3.1 (pinned by base env) | Save generated images as PNG | Already in base env; do not reinstall |
| numpy | >=1.20 (pinned by base env) | Tensor/array bridge between PyTorch and image display | Already in base env |
| opencv-python | >=4.5 (already in base env via pip) | Frame processing, resize operations | Already installed by StyleGAN-Human's pip block |

These are already present. No additional installation needed.

---

### FID Scoring Utility

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pytorch-fid | >=0.2.1 | Compute FID between generated samples and Kaggle Human Faces dataset | Standard reference implementation; no training required; wraps InceptionV3 from torchvision |

**Install:**
```
pip install pytorch-fid
```

**Confidence:** MEDIUM. pytorch-fid 0.2.x is compatible with PyTorch 1.9.x. It does not require any API key. The InceptionV3 weights it downloads are cached locally to torch hub.

**Alternative:** The `clean-fid` library offers improved numerical consistency but requires Python >=3.8 and torchvision >=0.11, which may conflict with the base env's pinned torchvision. Avoid clean-fid unless pytorch-fid fails.

---

### Dataset Loading

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pandas | Already in base env | CSV/metadata parsing for Kaggle Human Faces dataset | Already installed |
| torch.utils.data.Dataset | Built into PyTorch 1.9.1 | Custom dataset loader for latent direction discovery | No extra install; write utils/dataset_loader.py as thin wrapper |

---

## Alternatives Considered and Rejected

| Category | Recommended | Alternative | Why Rejected |
|----------|-------------|-------------|--------------|
| ASR | openai-whisper (local) | OpenAI Whisper API / Deepgram / AssemblyAI | Require API keys — hard constraint violation |
| ASR | openai-whisper (local) | faster-whisper (CTranslate2) | faster-whisper uses CTranslate2 backend which has separate CUDA kernel requirements; compatibility with CUDA 11.1 + PyTorch 1.9 is uncertain; not worth the risk for this environment |
| CLIP | openai/CLIP from GitHub | openai Python SDK (pip install openai) | The openai SDK is the API client — requires OPENAI_API_KEY; does NOT provide local CLIP inference |
| CLIP | openai/CLIP from GitHub | open_clip (mlfoundations) | open_clip is a viable alternative but introduces an extra dependency; the original openai/CLIP is sufficient for ViT-B/32 and has simpler API; prefer the known quantity |
| UI | Gradio | Streamlit | Streamlit lacks native mic capture widget; would require additional JS/audio libraries; Gradio's gr.Audio(source="microphone") handles this natively |
| UI | Gradio | FastAPI + React | Massively overbuilt for a dev-tool localhost UI; setup complexity is hostile to the "clone and run" user experience |
| FID | pytorch-fid | torchmetrics FID | torchmetrics FID requires torchmetrics >=0.11 which bumps minimum torch to 1.10; incompatible with base env |
| Audio preprocessing | librosa | torchaudio | torchaudio 0.9.x (the version matching torch 1.9.1) has limited audio augmentation; librosa is more feature-complete for DSP operations |
| Latent optimization | custom Adam loop | diffusers / Hugging Face pipelines | Diffusers targets diffusion models, not StyleGAN latent optimization; wrong abstraction entirely |

---

## Version Conflict Map

This is the most important section for implementation. These are the known traps:

### Trap 1: tiktoken and Python 3.8
- openai-whisper depends on tiktoken
- tiktoken >=0.6.0 dropped Python 3.8 wheel builds in some releases
- **Pin:** `tiktoken==0.5.2` in requirements_extra.txt as a precaution

### Trap 2: librosa vs scipy pinned at 1.7.1
- librosa 0.10+ requires scipy >=1.7.3 in some internal calls
- Base env pins `scipy=1.7.1`
- **Pin:** `librosa==0.9.2` to stay within the scipy pin

### Trap 3: Gradio 4.x Python floor
- Gradio 4.29+ may require Python >=3.9
- **Pin:** `gradio==4.19.2` and verify at env setup time
- **Fallback:** `gradio==3.50.2` if 4.19.2 fails

### Trap 4: torchvision not explicitly in environment.yml
- StyleGAN-Human's environment.yml installs pytorch but not torchvision explicitly
- CLIP requires torchvision
- **Explicit install:** `torchvision==0.10.1+cu111` from PyTorch wheel index if not auto-installed

### Trap 5: LPIPS already installed by base env
- StyleGAN-Human pip-installs `lpips==0.1.4`
- Do not reinstall LPIPS; it is used internally by StyleGAN-Human's PTI training scripts
- Any CLIP-guided optimization loop should call `clip.encode_image()` for similarity, not LPIPS

### Trap 6: numpy API changes between 1.20 and 1.24
- Base env allows `numpy>=1.20` (no upper bound)
- numpy 1.24 deprecated `np.bool`, `np.int`, `np.float` aliases
- Some older StyleGAN-Human utility code may use these deprecated aliases
- **Do not upgrade numpy above 1.23.x** unless you have verified StyleGAN-Human internals do not use deprecated aliases
- Add `numpy<1.24` to requirements_extra.txt

---

## Installation

The install sequence matters. Base env first, then extras on top:

```bash
# 1. Clone StyleGAN-Human and create base env
git clone https://github.com/stylegan-human/StyleGAN-Human.git
cd StyleGAN-Human
conda env create -f environment.yml
conda activate stylehuman

# 2. Verify torchvision is present (CLIP dependency)
python -c "import torchvision; print(torchvision.__version__)"
# If missing:
pip install torchvision==0.10.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111

# 3. Install extra dependencies (create requirements_extra.txt with pinned versions)
pip install openai-whisper
pip install tiktoken==0.5.2         # pin to avoid Python 3.8 wheel gap
pip install git+https://github.com/openai/CLIP.git
pip install librosa==0.9.2          # pin below scipy 1.7.3 requirement
pip install sounddevice soundfile
pip install ffmpeg-python
pip install gradio==4.19.2          # verify; fallback: gradio==3.50.2
pip install pytorch-fid
pip install "numpy<1.24"            # prevent numpy API deprecation hits

# 4. System dependency (not pip)
# Install ffmpeg and ensure it is on PATH (required for .mp3 input via librosa)
# Windows: winget install ffmpeg  OR  conda install -c conda-forge ffmpeg
```

---

## What Requires a Network Connection

| Action | Network Required | Cached After First Run |
|--------|-----------------|----------------------|
| `whisper.load_model("medium")` | Yes (first run only) | `~/.cache/whisper/medium.pt` |
| `clip.load("ViT-B/32")` | Yes (first run only) | `~/.cache/clip/ViT-B-32.pt` |
| pytorch-fid InceptionV3 weights | Yes (first run only) | torch hub cache |
| All pipeline inference | No | — |

Zero network calls during actual pipeline execution after the one-time model download.

---

## What Requires an API Key

Nothing in this stack requires an API key. Any library that would require one is explicitly excluded. Confirmed API-key-free:

- openai-whisper — downloads weights directly, no API auth
- openai/CLIP (GitHub) — downloads weights directly, no API auth
- Gradio — localhost only, `share=False`
- pytorch-fid — downloads InceptionV3 via torch hub, no auth
- librosa, sounddevice, soundfile, ffmpeg-python — pure local processing

---

## Sources

| Claim | Source | Confidence |
|-------|--------|------------|
| Base env versions (Python 3.8, PyTorch 1.9.1, CUDA 11.1, scipy 1.7.1, pillow 8.3.1) | `D:/EchoFace/StyleGAN-Human/environment.yml` (file on disk) | HIGH |
| openai-whisper torch>=1.7.1 requirement | Training data: openai/whisper setup.py | MEDIUM |
| CLIP torchvision dependency | Training data: openai/CLIP README | HIGH |
| Gradio 4.x Python >=3.8 support in early 4.x | Training data: Gradio changelog | LOW — verify at setup time |
| librosa 0.9.2 scipy compatibility | Training data: librosa changelog | MEDIUM |
| tiktoken 0.6+ Python 3.9+ wheel gap | Training data: tiktoken release notes | MEDIUM |
| numpy 1.24 deprecated aliases | numpy 1.24 migration guide (training data) | HIGH |
| pytorch-fid PyTorch 1.9 compatibility | Training data | MEDIUM |
