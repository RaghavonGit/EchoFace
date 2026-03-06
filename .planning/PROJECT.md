# EchoFace

## What This Is

EchoFace is an open-source multimodal AI framework that generates photorealistic human faces from spoken descriptions. Developers clone the repo, set up the conda environment, and run the entire pipeline — speech input to generated face — fully offline on their own hardware. No API keys, no cloud services, no paid dependencies.

## Core Value

A developer can speak a face description and receive a photorealistic generated image within seconds, with zero external API calls — everything runs on their local machine after a one-time model download.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Accept real-time mic input and pre-recorded .wav/.mp3 at 16kHz
- [ ] Preprocess audio: noise filtering, amplitude normalization, silence trimming (librosa + soundfile)
- [ ] ASR via local openai-whisper (whisper.load_model("medium")) — no API key
- [ ] Manual text override input for low-quality audio
- [ ] CLIP text encoding via local CLIP library (ViT-B/32) — no API key
- [ ] CLIP image encoding for optimization loop feedback
- [ ] StyleGAN-Human generator integration (stylegan_human_v2_1024.pkl)
- [ ] CLIP-guided latent optimization: L_total = -Sim(E_text, E_image) + λ||w - w_init||² (λ=0.1, lr=0.01, 150 steps)
- [ ] GPU fp16 inference with CUDA; CPU fallback at 256x256 with truncation=0.5
- [ ] Progress callbacks every 10 optimization steps for UI updates
- [ ] Gradio UI running on localhost with mic input, text override, live preview
- [ ] Latent direction sliders: age, smile intensity, facial hair, glasses, pose
- [ ] Live CLIP similarity score display
- [ ] Export final image as PNG
- [ ] FID scoring utility using Kaggle Human Faces dataset
- [ ] Latent attribute direction discovery from dataset
- [ ] Dataset loader (utils/dataset_loader.py) for Kaggle Human Faces dataset
- [ ] End-to-end pipeline: speech → Whisper → CLIP → StyleGAN latent → Gradio output

### Out of Scope

- Any paid or cloud API (OpenAI SDK, HuggingFace Inference API, Replicate, Runway) — hard constraint, violates the fully-local requirement
- os.environ["OPENAI_API_KEY"] or any API key usage — same reason
- Mobile or web-hosted deployment — localhost Gradio only
- Real-time video generation — single image output per description
- Training new models from scratch — inference + optimization only

## Context

- **StyleGAN-Human repo:** https://github.com/stylegan-human/StyleGAN-Human — clone into project root; do not modify internals
- **Pretrained weights:** stylegan_human_v2_1024.pkl under StyleGAN-Human/pretrained_models/
- **Dataset:** Kaggle "Human Faces Dataset" by kaustubhdhote — manually placed under data/human_faces/
- **Base environment:** conda env from StyleGAN-Human's environment.yml (Python 3.8.5, PyTorch 1.9.1 + CUDA 11.1)
- **Extra deps:** requirements_extra.txt adds openai-whisper, CLIP (from GitHub), Gradio 4.x on top of base env
- **Audience:** Other developers cloning and running locally — setup experience and README clarity matter
- **CLIP library note:** Install via `pip install git+https://github.com/openai/CLIP.git` — NOT the openai Python SDK

## Constraints

- **Tech Stack**: Python 3.8.5, PyTorch 1.9.1 + CUDA 11.1 — matches StyleGAN-Human tested environment; do not upgrade
- **No paid APIs**: Every inference call must run locally; any library requiring an API key must be replaced
- **No network at inference time**: Models download once to local cache (~/.cache/whisper/, CLIP cache); zero network calls during actual pipeline execution
- **StyleGAN-Human internals**: Do not modify the cloned repo; interface only through generate.py and its public API
- **Dataset placement**: data/human_faces/ is populated manually by the user; code must handle missing dataset gracefully

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| openai-whisper (local) not OpenAI ASR API | Zero cost, fully offline, same model weights | — Pending |
| CLIP from GitHub not openai Python SDK | Avoids API key requirement entirely | — Pending |
| StyleGAN-Human v2 1024px weights | State-of-art quality for full-body/face generation | — Pending |
| W-space optimization not Z-space | Better disentanglement for attribute editing | — Pending |
| Gradio on localhost only | No hosting cost, no data privacy concerns for users | — Pending |
| conda base env from StyleGAN-Human's environment.yml | Ensures compatibility with tested PyTorch/CUDA versions | — Pending |

---
*Last updated: 2026-03-06 after initialization*
