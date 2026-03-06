# Project Research Summary

**Project:** EchoFace
**Domain:** Speech-to-face generative AI — CLIP-guided StyleGAN latent optimization, fully local
**Researched:** 2026-03-06
**Confidence:** MEDIUM — No live web search available; findings grounded in PROJECT.md (on disk), StyleGAN-Human environment.yml (on disk), and training knowledge through August 2025.

## Executive Summary

EchoFace is a single-user, fully-local developer tool that converts spoken descriptions into generated human faces. The architecture is a sequential pipeline: microphone or audio file → Whisper ASR transcription → CLIP text embedding → W-space latent optimization loop against StyleGAN-Human → PNG export, surfaced through a Gradio localhost UI. The defining characteristic is that every computation happens on the user's machine — no API keys, no cloud calls, no data leaving the device after a one-time model download. This puts the project firmly in the CLIP-guided GAN optimization tradition established by StyleCLIP (2021), adapted to the StyleGAN-Human full-body/face model and augmented with a speech input stage.

The recommended approach is to build in dependency order: environment first, then the StyleGAN generator wrapper, then CLIP encoding, then the optimization loop in isolation, then Whisper ASR, then the Gradio UI, and finally dataset-dependent features (attribute sliders, FID evaluation) as a second tier. The biggest architectural risk is the Python 3.8.5 / PyTorch 1.9.1 environment imposed by StyleGAN-Human. Most extra dependencies (Whisper, CLIP, librosa) are compatible with these constraints, but Gradio 4.x is not — the safe choice is Gradio 3.x (pinned `gradio>=3.40,<4.0`). Every additional dependency must be pinned explicitly against this environment.

The top implementation risks are: incorrect StyleGAN pkl loading (must use `legacy.load_network_pkl`, not `pickle.load`); optimizing in Z-space instead of W-space (produces entangled attributes); gradient explosion in the CLIP cosine loss (requires gradient clipping and NaN assertions); and VRAM budget pressure when Whisper medium and StyleGAN-Human coexist on smaller GPUs. These are all well-understood problems with documented mitigations. Following the adapter-module boundary for StyleGAN-Human internals and the generator-yield pattern for Gradio live preview are the two structural decisions that prevent the most expensive rewrites.

---

## Key Findings

### Recommended Stack

The baseline environment is non-negotiable: Python 3.8.5, PyTorch 1.9.1, CUDA 11.1, numpy>=1.20, Pillow 8.3.1, scipy 1.7.1 — all pinned by `StyleGAN-Human/environment.yml` on disk. Extra dependencies install on top via `requirements_extra.txt`. The critical version traps are librosa (must pin to 0.9.2 to stay within scipy 1.7.1), tiktoken (pin to 0.5.2 to avoid Python 3.8 wheel gaps), numpy (cap at <1.24 to avoid deprecated alias hits in StyleGAN internals), and Gradio (use 3.x, not 4.x, due to Python 3.8 incompatibility). See `STACK.md` for full version conflict map.

**Core technologies:**
- **openai-whisper (local)**: ASR transcription — fully offline after model download, compatible with PyTorch 1.9.x, no API key
- **openai/CLIP from GitHub** (not PyPI): text and image embedding — ViT-B/32, local weights, dual-mode (text + image) service pattern
- **StyleGAN-Human** (cloned repo, never modified): face synthesis — accessed only via `G.synthesis(w)` through an adapter module
- **librosa 0.9.2 + sounddevice + soundfile**: audio I/O and preprocessing — resample to 16kHz mono before Whisper
- **gradio>=3.40,<4.0**: localhost UI — mic input, text override, live preview via generator yield pattern
- **pytorch-fid**: FID evaluation utility — developer/research feature, dataset-dependent
- **ffmpeg** (system binary): MP3 decode — required on PATH for compressed audio input

### Expected Features

**Must have (table stakes) — MVP, no dataset required:**
- Local ASR transcription from mic and WAV/MP3 files — core thesis
- Manual text override — unblocks pipeline when audio quality fails
- Audio preprocessing (noise filter, 16kHz resample, normalize, silence trim) — Whisper accuracy depends on it
- CLIP-guided W-space latent optimization loop (150 steps, Adam lr=0.01, regularization λ=0.1)
- StyleGAN-Human face generation with GPU fp16 + CPU fallback (256px, truncation=0.5)
- Progress indicator every 10 steps + live CLIP similarity score
- PNG export with timestamp filename
- Gradio UI on localhost only (share=False)

**Should have (differentiators) — require Kaggle Human Faces dataset:**
- Latent attribute sliders (age, smile, facial hair, glasses, pose) via pre-discovered W-space directions
- Latent direction discovery utility (SVM or PCA on dataset labels)
- FID scoring utility (requires >= 2048 dataset images)
- Real-time intermediate face preview during optimization (strong UX differentiator)

**Defer entirely (v2+):**
- Real-time streaming face generation / webcam input
- Automated model weight download scripts
- Multi-user or public deployment
- Any cloud API integration — hard constraint, never build

### Architecture Approach

EchoFace is a sequential pipeline with one non-linear section (the optimization loop). All inter-component communication uses plain Python objects — tensors, strings, and callbacks. No message queues, no shared state beyond the Gradio session object. Models are loaded once at startup as singletons; loading per-request is an anti-pattern that wastes 5-30 seconds and risks OOM. The Gradio UI orchestrates the pipeline via a generator function that yields intermediate results every 10 steps, avoiding UI freeze during the 30-60 second optimization.

**Major components:**
1. `asr/transcriber.py` — WhisperTranscriber; accepts 16kHz PCM array, returns UTF-8 text
2. `encoder/clip_encoder.py` — CLIPEncoder (dual-mode: text + image); loaded once, shared across pre-loop encoding and per-step image encoding
3. `generator/stylegan_wrapper.py` — StyleGANWrapper; wraps `legacy.load_network_pkl` + `G.synthesis(w)`; stateless callable; generator weights frozen at load
4. `optimizer/latent_optimizer.py` — `optimize()` function; Adam on W-space latent only; emits progress callbacks every 10 steps
5. `ui/app.py` — Gradio Blocks app; orchestrates pipeline; uses generator yield for streaming updates
6. `utils/` — audio preprocessing, dataset loader, latent direction discovery, FID scorer

### Critical Pitfalls

1. **Wrong StyleGAN pkl loader (C2)** — Use `legacy.load_network_pkl()` exclusively. Raw `pickle.load()` fails with cryptic `AttributeError` because the pkl embeds custom class references from `dnnlib`.

2. **Z-space instead of W-space optimization (C3)** — Map Z→W once with `G.mapping()` at initialization; optimize `w` directly in the loop. Z-space optimization produces entangled attribute edits that break sliders and reduce CLIP semantic fidelity.

3. **Gradio 4.x on Python 3.8.5 (C8)** — Gradio 4.x uses Python 3.10+ syntax. Pin `gradio>=3.40,<4.0`. Validate with `pip check` after all installs. This must be resolved before any code is written.

4. **CLIP gradient explosion / NaN latent (C4)** — Add `torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)` before every optimizer step. Assert `if torch.isnan(w).any(): raise RuntimeError(...)` to catch silent NaN propagation early.

5. **VRAM budget: Whisper + StyleGAN on small GPUs (C6)** — Load Whisper lazily (first transcription call only); move Whisper to CPU before running StyleGAN optimization. Document 6 GB minimum VRAM, 8 GB recommended.

6. **StyleGAN-Human adapter boundary (C1)** — All `sys.path` manipulation and imports from `StyleGAN-Human/` internals must live exclusively in `generator/stylegan_wrapper.py`. Any cross-file import from StyleGAN internals is a coupling violation.

7. **Audio sample rate mismatch (C7, Mi1)** — Always resample to 16kHz mono with librosa before passing to Whisper. Gradio's mic component captures at browser native rate (often 48kHz); always resample in the Gradio handler.

---

## Implications for Roadmap

Based on the dependency graph from ARCHITECTURE.md, feature prioritization from FEATURES.md, and phase-specific warnings from PITFALLS.md, the recommended phase structure is:

### Phase 0: Environment and Repo Scaffold
**Rationale:** The conda environment is the single highest-risk element in the project. Three dependency conflicts (Gradio version, tiktoken, numpy cap) can silently corrupt the environment. Phase 0 must resolve all of them before any ML code is written. This is also the phase where the project directory structure, adapter module boundaries, and `requirements_extra.txt` are established.
**Delivers:** A verified conda environment where `python -c "import gradio, whisper, clip, torch; print('OK')"` passes. The `generator/stylegan_wrapper.py` adapter stub. `requirements_extra.txt` with all pins.
**Addresses:** Table stakes: one-command environment setup; graceful missing dataset handling
**Avoids:** C8 (Gradio Python 3.8 incompatibility), C9 (wrong CLIP package), C10 (numpy/Pillow conflicts), C1 (StyleGAN internals boundary)

### Phase 1: StyleGAN-Human Integration
**Rationale:** The generator is the dependency foundation for every other ML component. It must be proven callable before the optimizer or encoder depend on its contract. This phase also flushes out the pkl loading issue (C2) and the in-process API vs subprocess issue (C13) immediately.
**Delivers:** `generator/stylegan_wrapper.py` fully functional: loads `.pkl` via `legacy.load_network_pkl`, synthesizes a face from a random W-space sample, saves PNG. Verified on both GPU and CPU device paths.
**Uses:** StyleGAN-Human clone, PyTorch 1.9.1, GPU/CPU device detection utility
**Avoids:** C2 (raw pickle.load), C13 (generate.py subprocess), C3 (Z-space — establish W-space contract here), Mi2 (float32→PIL conversion pipeline)

### Phase 2: CLIP Encoder + Latent Optimizer
**Rationale:** The encoder and optimizer are tightly coupled and must be built and validated together, isolated from UI and ASR complexity. Debugging gradient explosion and semantic drift is much harder when the UI is also in the loop. Build and test this phase as a standalone script with a hardcoded text prompt.
**Delivers:** `encoder/clip_encoder.py` (text + image, single model instance) and `optimizer/latent_optimizer.py` (150-step Adam loop, callback every 10 steps). Verified: loss decreases monotonically; output face shifts toward prompt semantics; NaN guard active.
**Uses:** openai/CLIP GitHub install, torchvision 0.10.1, torch.optim.Adam
**Avoids:** C3 (W-space confirmed), C4 (gradient clipping + NaN assertion), C5 (regularization λ=0.1 enforced, augmentation before CLIP encoding), M5 (noise_mode="const" enforced), Anti-Pattern 3 (generator weights frozen)

### Phase 3: ASR Module + Audio Preprocessing
**Rationale:** Whisper is independent of the optimizer and can be built in parallel with Phase 2 or immediately after. It is isolated here because its VRAM interaction with StyleGAN (C6) needs explicit sequencing logic, which is easier to implement after the generator is proven.
**Delivers:** `asr/transcriber.py` (WhisperTranscriber) and `utils/audio_preprocess.py` (librosa resample + normalize + silence trim). Verified: known phrase transcribed correctly from a 44.1kHz WAV file after preprocessing; fp16 flag computed from device availability.
**Uses:** openai-whisper, librosa 0.9.2, sounddevice, soundfile, ffmpeg-python
**Avoids:** C6 (lazy Whisper load, CPU offload before StyleGAN), C7 (16kHz mono preprocessing mandatory), M1 (fp16=False on CPU), M2 (pre-warm at startup with console log)

### Phase 4: Gradio UI Integration
**Rationale:** All ML components are proven independently before the UI is introduced. This phase wires them into a Gradio Blocks app with the generator yield pattern for live preview. The Gradio version constraint (Python 3.8 / gradio 3.x) is already resolved in Phase 0.
**Delivers:** `ui/app.py` — full localhost pipeline: mic input, text override, optimization with live preview every 10 steps, similarity score display, PNG export. Runnable end-to-end by a non-expert user.
**Uses:** gradio>=3.40,<4.0, all Phase 1-3 components
**Avoids:** C8 (Gradio 3.x pinned), C11 (generator yield pattern for streaming, no threading), Mi1 (Gradio audio handler resamples to 16kHz), Anti-Pattern 2 (models loaded at startup, not per request)

### Phase 5: Dataset-Dependent Features (Sliders + FID)
**Rationale:** These features are independent of the core pipeline and require a separate setup step (Kaggle dataset placement). They are high-value differentiators but should not block MVP delivery. Build the core pipeline to completion before adding dataset-dependent features.
**Delivers:** `utils/latent_directions.py` (direction discovery via SVM/PCA), `utils/dataset_loader.py`, `utils/fid_scorer.py`, and latent attribute sliders wired into the Gradio UI.
**Avoids:** M3 (L2-normalize all directions before storage), M4 (assert >= 2048 images before FID), graceful no-op in UI when dataset is missing

### Phase Ordering Rationale

- Environment before ML code: The conda environment has three known dependency conflicts that can silently corrupt subsequent phases. Resolving them first prevents debugging time being wasted on environment issues disguised as code bugs.
- Generator before encoder/optimizer: The optimizer depends on the generator's W-space contract being stable. Building the generator first makes the optimizer's dependency surface concrete.
- Optimizer before UI: The 150-step loop is the hardest component to debug. Isolating it from UI complexity (Gradio threading, streaming) allows clean loss curve inspection before adding UI overhead.
- ASR independent of optimizer: Whisper and the CLIP optimization loop are parallel concerns. Either order works; placing ASR in Phase 3 allows it to be developed while optimizer validation is still in progress.
- Dataset features last: They require user setup (manual dataset placement) and add no value to the core speech-to-face pipeline. Deferring them to Phase 5 keeps the MVP achievable without dataset access.

### Research Flags

Phases that benefit from deeper research during planning:
- **Phase 2 (CLIP optimizer):** The interaction between CLIP augmentation strategies and StyleGAN-Human v2 W-space is not extensively documented for this exact model. Gradient clipping threshold (max_norm=1.0) and regularization weight (λ=0.1) should be treated as starting points requiring empirical validation, not fixed values.
- **Phase 5 (Latent directions):** The choice between SVM and PCA for direction discovery on the Kaggle Human Faces dataset is unresolved. Literature favors SVM for binary attributes (smile yes/no) and PCA for continuous attributes (age). This needs a small experiment before building the full slider pipeline.

Phases with well-documented patterns (can proceed without additional research):
- **Phase 0 (Environment):** All version pins are derived from files on disk (`environment.yml`) and well-documented compatibility requirements. Follow the conflict map in `STACK.md` exactly.
- **Phase 1 (StyleGAN wrapper):** The `legacy.load_network_pkl` + `G.synthesis(w, noise_mode="const")` pattern is the documented StyleGAN2/StyleGAN-Human inference API. No ambiguity.
- **Phase 3 (ASR):** Whisper transcription with librosa preprocessing is a standard pattern with clear documentation.
- **Phase 4 (Gradio UI):** The generator yield pattern for streaming updates is standard Gradio 3.x usage.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Base env versions are HIGH (file on disk). Extra deps (Gradio pin, tiktoken, librosa) are MEDIUM — version conflict logic is sound but exact pip resolver behavior needs verification at setup time. |
| Features | MEDIUM | Feature list is grounded in PROJECT.md (HIGH) and CLIP-StyleGAN literature (MEDIUM). Open questions on slider direction method and CPU fallback optimization quality remain. |
| Architecture | HIGH | Sequential pipeline structure is directly specified in PROJECT.md. Component boundaries, W-space optimization loop structure, and Gradio integration pattern are all derived from well-established, documented sources. |
| Pitfalls | MEDIUM | StyleGAN pkl loading (HIGH), W-space vs Z-space (HIGH), Whisper fp16 (HIGH), Gradio Python 3.8 compat (MEDIUM — verify exact 3.x cutoff), gradient explosion thresholds (MEDIUM — empirical). |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Gradio 3.x exact version floor for Python 3.8:** PITFALLS.md identifies the risk but STACK.md recommends `gradio==4.19.2` while PITFALLS.md recommends `gradio<4.0`. These conflict. Resolution: the PITFALLS research is more conservative and more specific about the Python 3.10+ syntax issue. **Use `gradio>=3.40,<4.0`.** Validate at Phase 0 with `pip check` and a smoke import.

- **Whisper VRAM sequencing:** Whether to load Whisper lazily and offload before StyleGAN synthesis, or to load both at startup and accept the combined VRAM footprint, depends on the target GPU. The conservative approach (lazy load + CPU offload) is recommended for broad hardware compatibility but needs a `--preload-whisper` flag for users with 8+ GB VRAM who prefer latency predictability.

- **Latent direction method for Phase 5:** SVM vs PCA not validated for StyleGAN-Human v2 specifically. Flag for a small exploratory experiment at the start of Phase 5 planning before committing to a direction.

- **Optimization step count:** 150 steps is specified in PROJECT.md. Whether this produces good semantic alignment for StyleGAN-Human v2 specifically (vs the StyleGAN2-FFHQ models used in the original CLIP+StyleGAN papers) is unverified. Plan an ablation at Phase 2 completion.

---

## Sources

### Primary (HIGH confidence)
- `D:/EchoFace/StyleGAN-Human/environment.yml` (file on disk) — base environment version pins
- `D:/EchoFace/.planning/PROJECT.md` (file on disk) — project specification, pipeline design, constraints
- StyleGAN2 paper (Karras et al., 2020) — W-space vs Z-space architecture
- StyleCLIP (Patashnik et al., 2021) — CLIP-guided GAN optimization loop structure
- OpenAI CLIP GitHub API (`encode_text`, `encode_image` signatures)

### Secondary (MEDIUM confidence)
- openai/whisper setup.py — torch>=1.7.1 requirement
- librosa changelog — 0.9.2 scipy compatibility boundary
- StyleGAN-Human GitHub README — pkl loading pattern, generate.py interface
- Gradio 3.x documentation — generator yield streaming pattern

### Tertiary (LOW confidence — verify before implementation)
- Gradio 4.x Python version floor — exact cutoff between 3.8-compatible and 3.9+-only builds
- tiktoken 0.6+ Python 3.8 wheel availability — may have been resolved in later releases
- pytorch-fid PyTorch 1.9.x compatibility — no direct verification available

---

*Research completed: 2026-03-06*
*Ready for roadmap: yes*
