# Feature Landscape

**Domain:** Speech-to-face generative AI — CLIP-guided StyleGAN pipeline, open-source, fully-local
**Researched:** 2026-03-06
**Confidence:** MEDIUM — WebSearch unavailable; findings from training knowledge of CLIP-StyleGAN literature, Whisper ASR tooling, and Gradio UI patterns (knowledge cutoff August 2025). Flag for validation where noted.

---

## Table Stakes

Features users expect. Missing = the tool feels broken or unusable on first run.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Local ASR transcription (no API key) | Core thesis of the project; devs clone this specifically to avoid API keys | Low | Whisper medium model; already specified in PROJECT.md |
| Accepts mic input AND pre-recorded audio | Developers test with WAV files before mic; mic-only creates friction | Low | librosa + soundfile for preprocessing; 16kHz requirement |
| Manual text override | Audio quality is unpredictable; without override, bad audio = silent failure | Low | Simple Gradio Textbox; unblocks the rest of the pipeline |
| Audio preprocessing (noise filter, normalize, trim silence) | Raw mic audio is noisy; Whisper accuracy degrades without normalization | Low-Medium | librosa effects chain; silence trimming prevents empty transcripts |
| CLIP text encoding of transcript | Without this, there is no guidance signal; the pipeline literally doesn't function | Medium | ViT-B/32, local weights from CLIP GitHub install |
| StyleGAN face generation (single image) | The deliverable. Users expect a face image as output | Medium | stylegan_human_v2_1024.pkl; interface via generate.py public API only |
| CLIP-guided latent optimization loop | Without optimization, generated face has no relationship to the text description | High | Adam on W-space; L_total = -Sim(E_text, E_image) + lambda * regularization; 150 steps |
| Progress indicator during optimization | 150 optimization steps takes seconds to minutes on CPU; silent waiting = assumed crash | Low | Callback every 10 steps; Gradio progress bar or streaming preview |
| Live CLIP similarity score display | Developers need feedback that the pipeline is working correctly | Low | Scalar value from cosine similarity; display in Gradio UI |
| Export PNG output | If there is no file output, the result is ephemeral and useless | Low | Single call to PIL.Image.save(); filename should include timestamp |
| Gradio UI on localhost | Removes need for terminal knowledge; lowers barrier for non-expert devs | Medium | Gradio 4.x; must not expose to public network by default |
| GPU (CUDA) inference with CPU fallback | Most devs have GPU; forcing CPU-only excludes GPU users; no fallback = broken on laptops | Medium | fp16 on CUDA; 256x256 truncation=0.5 on CPU; device detection at startup |
| W-space latent optimization (not Z-space) | Z-space produces entangled edits; W-space is the documented standard for attribute control | Medium | Disentanglement quality is directly visible to users trying sliders |
| Graceful handling of missing dataset | Dataset is manually placed; first-run will often not have it | Low | Check data/human_faces/ at startup; print actionable error, do not crash |
| One-command environment setup | Developer audience expects pip/conda install to just work | Low-Medium | requirements_extra.txt on top of StyleGAN-Human environment.yml |

---

## Differentiators

Features that set EchoFace apart from generic text-to-image tools. Not universally expected, but create significant value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Speech-driven input (mic to face) | The defining UX: speak a description and get a face. No other open-source local tool targets this exact flow | Medium | Whisper ASR + Gradio mic component; the combo is the differentiator |
| Latent attribute sliders (age, smile, facial hair, glasses, pose) | Lets users interactively edit the generated face without re-running the full pipeline | High | Requires pre-discovered latent direction vectors; SVM or PCA on dataset attributes |
| Latent direction discovery from dataset | Directions are learned from real data, not hard-coded offsets — more reliable, transferable | High | Dataset loader + latent direction training utility; depends on Kaggle dataset being present |
| FID scoring utility | Quantitative quality metric; useful for developers benchmarking changes | Medium | Requires dataset; uses torchmetrics or clean-fid; clearly a developer/research feature |
| Fully offline after one-time model download | Privacy guarantee: no speech or image data leaves the machine, ever | Low (policy) | Already enforced by architecture; must be prominently documented in README |
| CLIP image encoding feedback loop | The optimization loop re-encodes each generated frame and computes similarity — live convergence signal | Medium | Makes the tool explainable and debuggable; rare in end-user tools |
| Real-time latent space preview during optimization | Streaming intermediate images every N steps shows the face "materializing" — strong wow factor | Medium-High | Requires Gradio streaming or gr.Image update in loop callback |
| Structured audio pipeline (not raw Whisper pass-through) | Noise filter + normalization + silence trim before ASR improves transcript quality in real environments | Medium | Differentiates from naive Whisper wrappers; meaningful for noisy home setups |
| Zero network calls at inference time | Models cached; all computation local; works air-gapped | Low (policy) | Must be verified and documented; devs in regulated environments need this guarantee |
| Text override that feeds same CLIP pipeline | The override is not a fallback that bypasses optimization — it uses the identical encoding path | Low | Consistency matters: override and speech should produce identical results for same text |

---

## Anti-Features

Features to explicitly NOT build. These would violate core constraints or dilute focus.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Any cloud API integration (OpenAI, HuggingFace Inference API, Replicate) | Hard constraint from PROJECT.md; violates the fully-local thesis and introduces cost | All inference runs locally; document this explicitly |
| API key configuration (OPENAI_API_KEY, HF_TOKEN, etc.) | Presence of any API key path creates confusion about what is actually local | Remove any such code paths; fail loudly if a key-requiring library is accidentally imported |
| Real-time video generation | Requires fundamentally different architecture (latent interpolation, rendering loop, video codec); out of scope | Single PNG per description; label this clearly in UI |
| Training models from scratch | Requires dataset curation, weeks of GPU time, infrastructure; no value for end users of this tool | Inference + optimization only; document the pretrained weight sources |
| Web hosting / public deployment | Exposes user audio and generated faces to a server; privacy violation; outside the local-only thesis | Gradio localhost only; block server=True flag in production config |
| Mobile app | Different runtime, packaging, and model quantization story entirely | Out of scope; not mentioned in PROJECT.md |
| Multi-face or full-scene generation | StyleGAN-Human generates single humans; extending to scenes requires ControlNet or layout models | Scope creep; single face per run |
| Voice cloning or speaker identification | ASR is for transcription only; identifying who spoke crosses into biometric territory | Use transcribed text only; discard speaker embeddings |
| Fine-tuning or DreamBooth on user faces | Raises consent and privacy issues for a tool that generates faces | Document as out of scope; do not add dataset ingestion paths for faces |
| Automatic model weight download in CI/CD | Large binary downloads in automated pipelines are fragile and expensive | Document manual download locations; provide download script as utility only |
| Real-time streaming face generation (webcam input) | Requires <100ms latency loop; optimization takes seconds; architecturally incompatible | Emphasize single-shot workflow |

---

## Feature Dependencies

```
Mic input / WAV file input
  └── Audio preprocessing (noise filter, normalize, trim)
        └── Whisper ASR (local openai-whisper)
              └── CLIP text encoding (ViT-B/32)
                    └── CLIP-guided latent optimization loop
                          ├── StyleGAN-Human generation (generate.py)
                          ├── CLIP image encoding (each step)
                          ├── Progress callback (every 10 steps)
                          └── Live CLIP similarity score display
                                └── PNG export

Text override input
  └── [bypasses ASR, feeds directly into CLIP text encoding]
        └── (same path as above from CLIP text encoding onward)

Latent attribute sliders
  └── Requires: latent direction discovery (pre-computed)
        └── Requires: dataset loader (data/human_faces/)
              └── Requires: Kaggle dataset manually placed

FID scoring utility
  └── Requires: dataset loader
        └── Requires: Kaggle dataset manually placed

Gradio UI
  └── Wraps: all of the above as components
  └── Requires: localhost server (no public deployment)
```

**Critical path for MVP (no dataset required):**
Mic/WAV input -> Preprocessing -> ASR -> Text encoding -> Optimization loop -> StyleGAN generation -> PNG export -> Gradio display

**Optional path (dataset required):**
Dataset loader -> Latent direction discovery -> Sliders active
Dataset loader -> FID scoring utility

---

## MVP Recommendation

**Prioritize (no dataset dependency — works on day 1 of clone):**

1. Audio preprocessing + Whisper ASR — gate: can we get reliable text from mic?
2. CLIP text encoding (ViT-B/32 local) — gate: can we produce a guidance embedding?
3. StyleGAN-Human integration via generate.py — gate: can we produce any face?
4. CLIP-guided optimization loop (150 steps, W-space) — gate: does the face move toward the description?
5. Manual text override — gate: unblocks pipeline when audio fails
6. Progress callbacks + live CLIP score — gate: UI is usable during optimization
7. PNG export — gate: result is persistent
8. Gradio UI assembling all of the above — gate: tool is runnable without terminal interaction

**Second priority (dataset dependency — requires setup step):**

9. Dataset loader (graceful no-op if missing)
10. Latent direction discovery utility
11. Latent attribute sliders (age, smile, facial hair, glasses, pose)
12. FID scoring utility

**Defer entirely:**

- Real-time streaming preview during optimization (High complexity, low MVP value — progress bar covers it)
- Automated model download scripts (Low priority; one-time manual step acceptable)

---

## Module-to-Feature Mapping

| Module | Table Stakes Features | Differentiating Features |
|--------|----------------------|--------------------------|
| ASR preprocessing | Audio preprocessing, mic + WAV input, graceful error on bad audio | Structured noise/normalize/trim pipeline above raw Whisper |
| CLIP encoding | Text encoding of transcript, image encoding for optimization feedback | Identical encoding path for text override and speech |
| StyleGAN generation | Single face image output, GPU fp16 + CPU fallback, W-space optimization | Real-time intermediate preview during optimization steps |
| Latent optimization | 150-step CLIP-guided loop, regularization term, progress callbacks | Convergence visualization, live similarity score display |
| Gradio UI + sliders | Localhost UI, mic input component, text override, PNG export | Latent attribute sliders (requires direction discovery), live score display |

---

## Open Questions (LOW confidence — flag for phase research)

- **Slider direction quality:** Pre-computing latent directions via SVM on labeled dataset vs. PCA on unlabeled — which produces more reliable directions for StyleGAN-Human v2 W-space? Literature favors SVM for binary attributes, PCA for continuous. Needs validation.
- **Optimization step count:** 150 steps is specified in PROJECT.md. Is this empirically validated for the StyleGAN-Human v2 model specifically, or borrowed from CLIP+StyleGAN2 face papers? May need ablation.
- **CPU fallback quality:** 256x256 at truncation=0.5 on CPU — does the optimization loop still converge meaningfully at this resolution, or does reduced quality make CLIP similarity scores unreliable? Needs testing.
- **Gradio 4.x + Python 3.8.5 compatibility:** Gradio 4.x may have dropped Python 3.8 support. The base environment is pinned at 3.8.5. This is a potential dependency conflict requiring explicit version pinning. HIGH priority to verify before implementation.
- **Whisper medium on 4GB VRAM:** Whisper medium requires ~5GB VRAM. On machines with 4GB GPU, simultaneous Whisper + StyleGAN inference may OOM. May need sequential (not parallel) model loading.

---

## Sources

- Project context: D:/EchoFace/.planning/PROJECT.md (HIGH confidence — authoritative spec)
- CLIP-StyleGAN literature (CLIP+StyleGAN2 by Patashnik et al., StyleCLIP): training knowledge (MEDIUM confidence)
- Whisper ASR capabilities and hardware requirements: training knowledge (MEDIUM confidence)
- Gradio 4.x Python version compatibility: training knowledge flagged LOW — verify before implementation
- StyleGAN-Human W-space optimization patterns: training knowledge from StyleGAN-Human paper and adjacent work (MEDIUM confidence)
- WebSearch: unavailable in this session — all non-PROJECT.md findings are from training knowledge and should be spot-checked against current docs
