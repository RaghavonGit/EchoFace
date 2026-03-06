# Requirements: EchoFace

**Defined:** 2026-03-06
**Core Value:** A developer can speak a face description and receive a photorealistic generated image fully offline — no API keys, no cloud calls, everything runs locally after one-time model download.

## v1 Requirements

### Environment

- [ ] **ENV-01**: Developer can run a validation script that confirms all extra dependencies install cleanly on Python 3.8.5 / PyTorch 1.9.1 / CUDA 11.1 base environment
- [ ] **ENV-02**: Repository contains full directory scaffold (asr/, encoder/, generator/, optimizer/, ui/, utils/, outputs/, StyleGAN-Human/pretrained_models/, data/human_faces/) with .gitkeep placeholders
- [ ] **ENV-03**: requirements_extra.txt with pinned versions (gradio>=3.40,<4.0, openai-whisper, CLIP from GitHub, librosa==0.9.2, tiktoken==0.5.2, numpy<1.24, soundfile, pytorch-fid) is installable on top of StyleGAN-Human base conda env without conflicts
- [ ] **ENV-04**: gpu_utils.py detects CUDA availability, selects fp16 vs fp32 mode, and returns correct resolution (1024px GPU / 256px CPU) and device string used by all downstream modules

### ASR

- [ ] **ASR-01**: audio_processor.py applies noise filtering, amplitude normalization, and silence trimming to input audio using librosa + soundfile, outputting 16kHz mono float32 array
- [ ] **ASR-02**: transcriber.py loads Whisper medium model locally (no API key), transcribes preprocessed audio to text, and implements lazy load + unload to free VRAM before optimization begins
- [ ] **ASR-03**: User can provide audio via real-time microphone recording or .wav/.mp3 file upload; both paths are resampled to 16kHz mono before transcription
- [ ] **ASR-04**: User can type a text description directly to bypass ASR entirely when audio quality is insufficient

### CLIP Encoder

- [ ] **CLIP-01**: clip_encoder.py loads CLIP ViT-B/32 locally (no API key, no openai Python SDK), encodes text descriptions into normalized embeddings
- [ ] **CLIP-02**: clip_encoder.py encodes generated images into CLIP image embeddings for use in optimization loss computation
- [ ] **CLIP-03**: User can see a live cosine similarity score between their text description and the current generated image, updated after optimization completes

### Generator

- [ ] **GEN-01**: stylegan_wrapper.py loads stylegan_human_v2_1024.pkl using legacy.load_network_pkl() (not raw pickle.load()), exposes W-space synthesis via G.synthesis(w, noise_mode='const')
- [ ] **GEN-02**: Generator runs fp16 inference on CUDA-enabled GPU producing 1024x1024px output
- [ ] **GEN-03**: Generator falls back to CPU with truncation=0.5 and 256x256px resolution when CUDA is unavailable

### Optimizer

- [ ] **OPT-01**: Optimizer samples w_init from Gaussian distribution in W-space as the starting latent for each generation
- [ ] **OPT-02**: clip_optimizer.py runs 150-step gradient descent on W-space latent: L_total = -Sim(E_text, E_image) + λ||w - w_init||² (λ=0.1, lr=0.01), with StyleGAN generator parameters frozen (requires_grad=False)
- [ ] **OPT-03**: Optimizer applies gradient clipping and NaN detection/recovery to prevent optimization instability (CLIP cosine loss can explode within 5 steps without this)

### Dataset

- [ ] **DATA-01**: utils/dataset_loader.py loads and preprocesses images from data/human_faces/ (Kaggle Human Faces Dataset), handles missing dataset directory gracefully with a clear error message

### Evaluation

- [ ] **EVAL-01**: utils/metrics.py computes FID score between generated face outputs and the Kaggle Human Faces dataset real image distribution using pytorch-fid

### UI

- [ ] **UI-01**: gradio_app.py runs on localhost with an input panel containing a mic record button, text description override box, and .wav/.mp3 file upload — all wired to the ASR + generation pipeline
- [ ] **UI-02**: User can export the final generated image as a PNG file via an export button in the Gradio UI

## v2 Requirements

### UI Enhancements

- **UI-V2-01**: Live optimization preview showing intermediate generated image every 10 steps during the optimization loop
- **UI-V2-02**: Progress bar showing current optimization step (x/150) during generation

### Attribute Editing

- **ATTR-01**: attribute_directions.py discovers latent directions for age, smile, facial hair, glasses, and pose attributes from Kaggle dataset using SVM/PCA on W-space embeddings
- **ATTR-02**: UI exposes sliders for each attribute direction (age: -5 to +5, smile: 0 to 3, facial hair: 0 to 3, glasses: 0 to 2, pose: -2 to +2) applying w' = w + α*d
- **ATTR-03**: Attribute edits apply in real time without re-running the full optimization loop

### Optimization Tuning

- **OPT-V2-01**: Configurable hyperparameters (λ, lr, step count) exposed in UI or config file
- **OPT-V2-02**: Optimization step progress callbacks emitted every 10 steps for external consumers

## Out of Scope

| Feature | Reason |
|---------|--------|
| Any paid or cloud API (OpenAI SDK, HuggingFace Inference API, Replicate, Runway) | Hard constraint — violates fully-local requirement |
| os.environ["OPENAI_API_KEY"] or any API key | Same reason — banned entirely |
| Gradio 4.x | Requires Python 3.10+; incompatible with StyleGAN-Human base environment (Python 3.8.5) |
| Real-time video / animation generation | Single PNG output per description is the scope |
| Mobile or web-hosted deployment | Localhost Gradio only — no hosting |
| Training new models from scratch | Inference + optimization only; pretrained weights only |
| Z-space optimization | Entangled — breaks attribute editing; W-space only |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
| ENV-04 | Phase 1 | Pending |
| GEN-01 | Phase 2 | Pending |
| GEN-02 | Phase 2 | Pending |
| GEN-03 | Phase 2 | Pending |
| OPT-01 | Phase 3 | Pending |
| OPT-02 | Phase 3 | Pending |
| OPT-03 | Phase 3 | Pending |
| CLIP-01 | Phase 3 | Pending |
| CLIP-02 | Phase 3 | Pending |
| ASR-01 | Phase 4 | Pending |
| ASR-02 | Phase 4 | Pending |
| ASR-03 | Phase 4 | Pending |
| ASR-04 | Phase 4 | Pending |
| DATA-01 | Phase 5 | Pending |
| EVAL-01 | Phase 5 | Pending |
| UI-01 | Phase 6 | Pending |
| UI-02 | Phase 6 | Pending |
| CLIP-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-06 after initial definition*
