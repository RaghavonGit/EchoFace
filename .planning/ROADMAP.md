# Roadmap: EchoFace

## Overview

EchoFace is built in strict dependency order: environment first (the single highest-risk element), then the StyleGAN generator (the foundation all other ML components depend on), then CLIP encoding and latent optimization together (tightly coupled, debugged in isolation), then Whisper ASR (independent of the optimizer, manages its own VRAM), then dataset utilities and FID evaluation (require user-placed data, do not block core pipeline), and finally the Gradio UI that wires every proven component into a runnable end-to-end experience. Each phase delivers one independently verifiable capability before the next begins.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Environment and Repo Scaffold** - Verified conda environment and directory structure ready for ML code (completed 2026-03-06)
- [x] **Phase 2: StyleGAN-Human Generator** - Generator loads, synthesizes, and saves a face from a random W-space sample (completed 2026-03-06)
- [x] **Phase 3: CLIP Encoder and Latent Optimizer** - CLIP-guided 150-step W-space optimization loop producing semantically aligned faces (completed 2026-03-06)
- [ ] **Phase 4: ASR and Audio Preprocessing** - Whisper transcribes mic and file input to text with VRAM-safe sequencing
- [ ] **Phase 5: Dataset Utilities and Evaluation** - Dataset loader and FID scorer operational with Kaggle Human Faces dataset
- [ ] **Phase 6: Gradio UI Integration** - Full localhost pipeline: speak a description, receive a face, export PNG

## Phase Details

### Phase 1: Environment and Repo Scaffold
**Goal**: Developer can clone the repo, run one setup command, and confirm every dependency installs cleanly on Python 3.8.5 / PyTorch 1.9.1 / CUDA 11.1 with no conflicts
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04
**Success Criteria** (what must be TRUE):
  1. Running `python -c "import gradio, whisper, clip, torch; print('OK')"` in the conda env prints OK with no import errors
  2. The validation script reports whether CUDA is available, selects fp16 vs fp32 mode, and prints the correct resolution (1024px GPU / 256px CPU)
  3. All directories listed in the repo scaffold exist with .gitkeep placeholders (asr/, encoder/, generator/, optimizer/, ui/, utils/, outputs/, StyleGAN-Human/pretrained_models/, data/human_faces/)
  4. `pip check` reports no dependency conflicts after installing requirements_extra.txt on top of the StyleGAN-Human base conda env
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Write requirements_extra.txt, create repo scaffold, install deps into stylehuman env
- [ ] 01-02-PLAN.md — Implement gpu_utils.py, validate_env.py, and pytest test suite

### Phase 2: StyleGAN-Human Generator
**Goal**: Developer can call the generator wrapper with a W-space latent and receive a valid face image, on both GPU and CPU paths
**Depends on**: Phase 1
**Requirements**: GEN-01, GEN-02, GEN-03
**Success Criteria** (what must be TRUE):
  1. `stylegan_wrapper.py` loads the .pkl using `legacy.load_network_pkl()` without errors (no raw pickle.load)
  2. Calling the wrapper with a random W-space sample on a CUDA GPU produces a 1024x1024px PNG saved to outputs/
  3. Calling the wrapper with CUDA unavailable falls back to CPU, produces a 256x256px PNG with truncation=0.5, and exits without error
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — Create test scaffold, generator package, and implement StyleGANWrapper loading/freezing (GEN-01)
- [ ] 02-02-PLAN.md — Implement GPU and CPU synthesis paths, integration test, visual checkpoint (GEN-02, GEN-03)

### Phase 3: CLIP Encoder and Latent Optimizer
**Goal**: A hardcoded text prompt drives a 150-step CLIP-guided W-space optimization that produces a face visually aligned with the prompt, with stable loss and no NaN failures
**Depends on**: Phase 2
**Requirements**: CLIP-01, CLIP-02, OPT-01, OPT-02, OPT-03
**Success Criteria** (what must be TRUE):
  1. `clip_encoder.py` encodes a text string and a PIL image into normalized embeddings using the local CLIP ViT-B/32 model (no API key, no openai Python SDK)
  2. The optimizer runs 150 gradient descent steps on the W-space latent with Adam (lr=0.01, λ=0.1 regularization) and StyleGAN generator parameters frozen (requires_grad=False throughout)
  3. CLIP cosine similarity between the text prompt and the generated image increases monotonically (or is stable) over the 150 steps — loss does not diverge
  4. The optimizer emits a progress callback every 10 steps without raising a NaN error on any of the 15 checkpoints
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — Test scaffolds + package stubs (RED state for encoder and optimizer)
- [ ] 03-02-PLAN.md — Implement CLIPEncoder class (CLIP-01, CLIP-02)
- [ ] 03-03-PLAN.md — Implement optimize() function + integration smoke test (OPT-01, OPT-02, OPT-03)

### Phase 4: ASR and Audio Preprocessing
**Goal**: Developer can transcribe a spoken description from a microphone recording or WAV/MP3 file, with VRAM safely released before the optimization loop begins
**Depends on**: Phase 2
**Requirements**: ASR-01, ASR-02, ASR-03, ASR-04
**Success Criteria** (what must be TRUE):
  1. A 44.1kHz stereo WAV file fed to `audio_processor.py` exits as a 16kHz mono float32 array with noise filtered and silence trimmed
  2. `transcriber.py` transcribes a known spoken phrase from the preprocessed array using the local Whisper medium model with no API key
  3. After transcription completes, Whisper model weights are unloaded from VRAM before the optimizer is invoked (verified by inspecting GPU memory between the two calls)
  4. A user can bypass ASR entirely by typing a text description directly, and the pipeline proceeds identically from that text
**Plans**: TBD

### Phase 5: Dataset Utilities and Evaluation
**Goal**: Developer can load the Kaggle Human Faces dataset and compute a FID score against a set of generated outputs, with graceful error messages when the dataset is absent
**Depends on**: Phase 3
**Requirements**: DATA-01, EVAL-01
**Success Criteria** (what must be TRUE):
  1. `utils/dataset_loader.py` loads images from `data/human_faces/` when the directory is populated, and prints a clear error message (not a traceback) when the directory is missing or empty
  2. `utils/metrics.py` computes and prints a FID score between the generated outputs and the real dataset images using pytorch-fid, asserting at least 2048 real images before running
**Plans**: TBD

### Phase 6: Gradio UI Integration
**Goal**: A developer clones the repo, activates the conda env, runs `python ui/app.py`, and can speak a face description, watch the pipeline run, see a live CLIP similarity score, and export the result as PNG — all on localhost with no internet connection
**Depends on**: Phase 3, Phase 4
**Requirements**: UI-01, UI-02, CLIP-03
**Success Criteria** (what must be TRUE):
  1. The Gradio app launches on localhost with a mic record button, text description override box, and WAV/MP3 file upload — all wired to the ASR and generation pipeline — using gradio>=3.40,<4.0 (Python 3.8 compatible)
  2. After optimization completes, the UI displays the cosine similarity score between the typed/transcribed description and the generated face image
  3. User can click an export button and receive a timestamped PNG file saved to outputs/
  4. The full pipeline runs end-to-end — speech input to generated face image displayed in the UI — without any network call after model weights are cached
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment and Repo Scaffold | 2/2 | Complete    | 2026-03-06 |
| 2. StyleGAN-Human Generator | 2/2 | Complete   | 2026-03-06 |
| 3. CLIP Encoder and Latent Optimizer | 3/3 | Complete   | 2026-03-06 |
| 4. ASR and Audio Preprocessing | 0/? | Not started | - |
| 5. Dataset Utilities and Evaluation | 0/? | Not started | - |
| 6. Gradio UI Integration | 0/? | Not started | - |
