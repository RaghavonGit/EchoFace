# EchoFace — Project Context for Claude Code
<!-- Last updated: 2026-03-08 — FaceGAN v2: 128×128 output, no dropout, L1=10 -->

## Project Overview
EchoFace is a multimodal AI framework that generates photorealistic human faces from spoken descriptions.

**Primary Pipeline (active):** Speech Input → Whisper ASR → CLIP Text Embed → FaceGAN → 256×256 Face Image → Gradio UI
**Legacy Pipeline (fallback):** Speech Input → Whisper ASR → CLIP Embedding → StyleGAN-Human Latent Optimization → Gradio UI

## CRITICAL CONSTRAINT — NO PAID APIs
Every component runs 100% locally. No API keys. No cloud inference.

### Explicitly Banned:
- openai Python SDK
- `os.environ["OPENAI_API_KEY"]` or any API key usage
- replicate, huggingface_hub inference client
- Any network calls during inference (after initial model download)

---

## Environment
- **Conda env:** `stylehuman` (Python 3.8.5)
- **Activate:** `conda activate stylehuman`
- **Project root:** `D:\EchoFace`
- **StyleGAN-Human repo:** `D:\EchoFace\StyleGAN-Human`
- **Pretrained weights:** `D:\EchoFace\StyleGAN-Human\pretrained_models\stylegan_human_v2_1024.pkl`
- **FaceGAN checkpoint:** `D:\EchoFace\checkpoints\face_gan\gen_best.pth` (trained, 19.7MB)
- **FaceGAN dataset:** `D:\EchoFace\datasets\faces_clip.hdf5` (9,630 images, CLIP-embedded)

## Installed Dependencies (all working)
- PyTorch 1.9.1 + CUDA (conda env)
- openai-whisper (local, no API key — uses `whisper.load_model("medium")`)
- CLIP from GitHub (local — uses `clip.load("ViT-B/32", device=device)`)
- Gradio >=3.40,<4.0 (resolved 3.50.2)
- librosa, soundfile, numpy, Pillow, scipy, tqdm, h5py

---

## Project Structure
```
D:\EchoFace\
├── StyleGAN-Human/          ← cloned repo (WORKING — generates full-body images, legacy fallback)
│   └── pretrained_models/
│       └── stylegan_human_v2_1024.pkl
├── Speech 2 Face/           ← reference project (not imported, used for training data)
│   └── Speech 2 Face/
│       ├── Real Images/     ← 5,000 real face images (training data)
│       └── AI-Generated Images/ ← 4,630 AI face images (training data)
├── datasets/
│   └── faces_clip.hdf5      ← 9,630 CLIP-embedded faces (train/val splits)
├── checkpoints/
│   └── face_gan/
│       └── gen_best.pth     ← best FaceGAN generator checkpoint
├── data/
│   └── human_faces/         ← Kaggle dataset (currently empty)
├── asr/
│   ├── audio_processor.py
│   └── transcriber.py
├── encoder/
│   └── clip_encoder.py      ← CLIPEncoder class
├── generator/
│   ├── stylegan_wrapper.py  ← StyleGANWrapper (legacy)
│   ├── face_gan.py          ← FaceGenerator + FaceDiscriminator architecture
│   ├── face_gan_wrapper.py  ← FaceGANWrapper inference class (PRIMARY)
│   ├── dataset_builder.py   ← builds faces_clip.hdf5 from image directories
│   └── train_face_gan.py    ← FaceGAN training script
├── optimizer/
│   ├── clip_optimizer.py    ← optimize() — legacy fallback only
│   └── attribute_directions.py
├── ui/
│   └── app.py               ← Gradio Blocks app (auto-selects GAN or optimizer)
├── utils/
│   ├── gpu_utils.py         ← get_device_config() single source of truth
│   ├── dataset_loader.py
│   └── metrics.py
├── .planning/               ← GSD internal planning files (do not edit manually)
├── outputs/
├── requirements_extra.txt
└── README.md
```

---

## Architecture — Modules

### Module 1: Audio Preprocessing & ASR (`/asr`)
- Accept mic input OR pre-recorded .wav/.mp3 (16kHz)
- Preprocessing: librosa + soundfile (noise filter, normalize, trim)
- ASR: `whisper.load_model("medium")` — fully local, weights at `~/.cache/whisper/`
- Output: clean text describing face (gender, age, hair, skin, accessories)
- Manual text override option for low-quality audio

### Module 2: CLIP Semantic Encoder (`/encoder`)
- `clip.load("ViT-B/32", device=device, jit=False)` — fully local
- **CLIPEncoder class** with `encode_text()` and `encode_image()` sharing one loaded model
- Always apply `F.normalize(feat.float(), dim=-1)` — CLIP returns unnormalized embeddings
- FaceGANWrapper also loads its own CLIP instance (no shared singleton)

### Module 3a: FaceGAN Generator (`/generator`) — PRIMARY
- `FaceGenerator` + `FaceDiscriminator` in `generator/face_gan.py` (DCGAN-style, spectral norm)
- Input: 512-dim CLIP text embedding + 128-dim noise → Output: 64×64 face in [-1, 1]
- `FaceGANWrapper` in `generator/face_gan_wrapper.py`:
  - `__init__(checkpoint_path, cfg)` — loads CLIP + FaceGenerator
  - `generate(text_prompt) -> (PIL.Image, float)` — instant single forward pass, returns 256×256 upscaled image + CLIP sim score
- Trained on 9,630 face images (5,000 real + 4,630 AI-generated) with CLIP image embeddings
- **Training:** `python generator/train_face_gan.py --dataset datasets/faces_clip.hdf5 --epochs 150`
- **Rebuild dataset:** `python generator/dataset_builder.py --images_dirs "Real Images dir" "AI-gen dir"`

### Module 3b: StyleGAN-Human Generator (`/generator`) — LEGACY FALLBACK
- Only used when `checkpoints/face_gan/gen_best.pth` does not exist
- `StyleGANWrapper` class in `generator/stylegan_wrapper.py`
- Generates 512×1024 full-body portraits — **known issue: poor face alignment with CLIP text**
- Generator frozen via `requires_grad_(False)` throughout optimization

### Module 4: CLIP-Guided Latent Optimizer (`/optimizer`) — LEGACY FALLBACK
- Only called when FaceGAN is not available
- **API:** `optimize(text_prompt, wrapper, cfg, callback=None) -> (PIL.Image, w_tensor, final_sim_score)`
- Loss: `L_total = -Sim(E_text, E_image) + λ||w - w_init||²` (λ=0.1, lr=0.01, 150 steps)
- **Known problem:** CLIP score ~0.23, wrong gender/attributes — because StyleGAN-Human outputs full body not face

### Module 5: Gradio Interactive UI (`/ui/app.py`)
- Localhost only, `python ui/app.py`
- Auto-detects `checkpoints/face_gan/gen_best.pth`:
  - If exists → loads `FaceGANWrapper`, instant generation
  - If missing → falls back to `StyleGANWrapper` + `optimize()`
- Input: mic recording + file upload + manual text override
- Two-stage ASR UX: audio-only → fills text box; second click runs generation
- Live CLIP similarity score display
- Export PNG button → timestamped file in `outputs/`

---

## Verified Working
- StyleGAN-Human generates images ✅ (full-body, legacy)
- FaceGAN training: ~14s/epoch on CUDA, 150 epochs = ~35min ✅
- FaceGAN dataset built: 9,630 images CLIP-embedded ✅
- All 84 tests GREEN ✅
- CUDA kernels fall back to CPU on Windows — this is fine and expected

## Known Issues / Notes
- StyleGAN-Human CUDA custom kernels (bias_act, upfirdn2d) fail to compile on Windows — fallback is fine
- System CUDA is v13 but conda env uses CUDA 11.x internally — do not try to "fix" this
- Always run from within `stylehuman` conda env
- Test pattern: `unittest.mock.patch` on module-level namespace bindings, `MagicMock` for GPU objects
- FaceGAN at epoch 0-10 produces blurry faces — quality improves significantly after epoch 50+
- h5py must be installed: `pip install h5py` (added to stylehuman env 2026-03-08)
- FaceClipDataset loads full HDF5 split into RAM at init (~816MB for train) — needed for fast training
- CLIP weights in FaceGANWrapper cast to float32 immediately after load — FP16 causes NaN in encode_image

## Change Log (session tracking)
| Date | Change |
|------|--------|
| 2026-03-08 | Identified root cause: StyleGAN-Human outputs 512×1024 full body → CLIP score 0.23, wrong gender |
| 2026-03-08 | Added `generator/face_gan.py` — CLIP-conditioned DCGAN architecture (FaceGenerator + FaceDiscriminator) |
| 2026-03-08 | Added `generator/dataset_builder.py` — builds HDF5 from multi-dir image collections + CLIP image embeddings |
| 2026-03-08 | Added `generator/train_face_gan.py` — training loop, RAM-loaded dataset, best-checkpoint tracking |
| 2026-03-08 | Added `generator/face_gan_wrapper.py` — FaceGANWrapper inference: text → CLIP → GAN → 256×256 face |
| 2026-03-08 | Updated `ui/app.py` — auto-selects FaceGAN (fast path) or optimizer (legacy fallback) |
| 2026-03-08 | Installed h5py into stylehuman env |
| 2026-03-08 | Built datasets/faces_clip.hdf5 from 5,000 real + 4,630 AI-gen faces |
| 2026-03-08 | Started full 150-epoch FaceGAN training (~35min on CUDA) |
| 2026-03-08 | FaceGAN v1 output was blurry (64×64, L1=50, dropout) — upgraded to v2 |
| 2026-03-08 | FaceGAN v2: IMAGE_SIZE=128, no dropout, l1_coef=10; old checkpoints deleted; dataset rebuild + retrain required |

---

## Build Milestones
- [x] Phase 1: Project setup, gpu_utils, dataset_loader
- [x] Phase 2: StyleGANWrapper — generate images working
- [x] Phase 3: CLIPEncoder + clip_optimizer
- [x] Phase 4: ASR pipeline — mic → local Whisper → text
- [x] Phase 5: Dataset utilities + metrics (FID)
- [x] Phase 6: Gradio UI — localhost, sliders, live preview
- [x] Phase 7 (unplanned): FaceGAN — replace CLIP optimizer with CLIP-conditioned face GAN

---

## GSD Workflow Notes
- Always run `/gsd:discuss-phase N` before `/gsd:plan-phase N`
- Run `/clear` between phases for a fresh context window
- GSD planning files live in `.planning/` — do not edit manually
