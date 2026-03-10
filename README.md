# EchoFace

**Speak a face into existence.**

EchoFace is a fully local AI pipeline that turns a spoken (or typed) face description into a photorealistic portrait. No cloud, no API keys — everything runs on your machine.

---

## How it works

```
Your voice  →  Whisper ASR  →  CLIP embedding  →  StyleGAN2 optimization  →  Face image
```

1. You describe a face out loud (or type it)
2. Whisper transcribes your speech to text locally
3. CLIP encodes the text into a semantic embedding
4. StyleGAN2-ADA (FFHQ) optimizes a latent vector toward that embedding over ~60 steps
5. A 1024×1024 face portrait is rendered in your browser

---

## Requirements

- Python 3.8
- CUDA-capable GPU strongly recommended (CPU works but is very slow — ~3-5 min/image)
- ~5 GB free disk space (model weights)
- Conda (Miniconda or Anaconda)

---

## Setup

### 1. Clone the repo (with submodules)

```bash
git clone https://github.com/RaghavonGit/EchoFace.git
cd EchoFace
git submodule update --init --recursive
```

> **This step is required.** The `git submodule update` command pulls the StyleGAN-Human repo which contains `dnnlib` and `legacy` — modules the app depends on. Skipping it will cause a `ModuleNotFoundError: No module named 'dnnlib'` error.
>
> If you cloned via a GUI (VS Code, GitHub Desktop, etc.), the submodule folder will appear empty. Just run the `git submodule update` command above inside the project folder to fix it.

---

### 2. Create the conda environment

```bash
conda create -n stylehuman python=3.8 -y
conda activate stylehuman
```

Install PyTorch with CUDA (adjust the CUDA version to match yours):

```bash
# CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Or for CPU-only
conda install pytorch torchvision cpuonly -c pytorch -y
```

Install remaining dependencies:

```bash
pip install -r requirements_extra.txt
```

---

### 3. Add the FFHQ model weights

The FFHQ StyleGAN2-ADA checkpoint (~364 MB) is too large for GitHub and is **not included** in the repo. You must add it manually before the app will start.

First, create the folder:

```bash
# Windows
mkdir checkpoints\ffhq

# macOS / Linux
mkdir -p checkpoints/ffhq
```

Then get `ffhq.pkl` using one of these options:

**Option A — Download from Google Drive**
1. Go to the [official FFHQ StyleGAN2-ADA weights](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IchBJ4S)
2. Download `ffhq.pkl`
3. Move it to `checkpoints/ffhq/ffhq.pkl`

**Option B — Copy from an existing install**

If you already have EchoFace set up elsewhere on the same machine:

```bash
# Windows (adjust source path to match your existing install)
copy "C:\path\to\old\EchoFace\checkpoints\ffhq\ffhq.pkl" "checkpoints\ffhq\ffhq.pkl"
```

Either way, the final layout must look like this:

```
EchoFace/
└── checkpoints/
    └── ffhq/
        └── ffhq.pkl   ← required
```

> Skipping this step causes: `FileNotFoundError: No such file or directory: '...\checkpoints\ffhq\ffhq.pkl'`

> Whisper model weights (~1.5 GB for `medium`) download automatically on first run and are cached at `~/.cache/whisper/` — no action needed.

---

### 4. Run the app

```bash
conda activate stylehuman
python ui/app.py
```

Then open your browser at **http://127.0.0.1:7860**

---

## Using the app

The UI has two panels — **Input** on the left, **Output** on the right.

### Option A — Type a description

1. Type a face description in the **Face Description** box
2. Click **Generate**
3. Watch the face take shape over ~60 optimization steps
4. When done, click **Export PNG** to save to the `outputs/` folder

### Option B — Record or upload audio

**Two-stage flow:**

1. Record via microphone or upload a `.wav` / `.mp3` file
2. Click **Generate** — EchoFace transcribes your audio using Whisper
3. The transcription appears in the text box — review and edit if needed
4. Click **Generate** again to run face generation

> The two-stage flow lets you correct any transcription mistakes before spending time on generation.

---

## Example prompts to try

```
a young woman with dark curly hair, green eyes, and olive skin
an elderly man with a white beard, kind eyes, and deep wrinkles
a teenage boy with freckles, red hair, and a wide smile
a middle-aged woman with sharp cheekbones and short silver hair
a young man with a fade haircut, brown eyes, and strong jaw
```

**Tips for better results:**
- Be specific — mention hair color, eye color, age, and skin tone
- Keep it under ~20 words (CLIP has a 77-token limit)
- CLIP Score above 0.28 = good match; above 0.32 = excellent

---

## Project structure

```
EchoFace/
├── asr/                    # Audio preprocessing + Whisper transcription
│   ├── audio_processor.py
│   └── transcriber.py
├── encoder/                # CLIP ViT-B/32 text & image encoder
│   └── clip_encoder.py
├── generator/              # StyleGAN2 + CLIP optimization
│   └── styleclip_wrapper.py
├── ui/                     # Gradio web interface
│   └── app.py
├── checkpoints/ffhq/       # Place ffhq.pkl here (not in repo)
├── outputs/                # Exported PNGs saved here
├── StyleGAN-Human/         # Git submodule (pulled automatically)
├── gpu_utils.py            # Device config (CUDA / CPU auto-detect)
└── requirements_extra.txt  # Python dependencies
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dnnlib'` | StyleGAN-Human submodule is empty — run `git submodule update --init --recursive` inside the project folder |
| `FileNotFoundError: ffhq.pkl` | Download the checkpoint and place it at `checkpoints/ffhq/ffhq.pkl` |
| Generation is very slow | No GPU detected — CPU fallback is ~3-5 min/image, this is expected |
| CUDA kernel build warnings | Safe to ignore — Python fallback is used automatically on Windows |
| Audio not transcribing | Check that `sounddevice` and `soundfile` are installed; try uploading a file instead |

---

## Tech stack

| Component | Library |
|---|---|
| Speech recognition | [OpenAI Whisper](https://github.com/openai/whisper) (local, `medium` model) |
| Semantic encoding | [CLIP ViT-B/32](https://github.com/openai/CLIP) (local) |
| Face generation | [StyleGAN2-ADA FFHQ](https://github.com/stylegan-human/StyleGAN-Human) |
| Web UI | [Gradio 3.x](https://gradio.app) |
| Audio processing | librosa + soundfile |

---

## Notes

- Everything runs **100% locally** — no internet connection needed after setup
- The optimizer runs for up to 60 steps with early stopping (exits earlier if the image converges)
- Generated images are 1024×1024 PNG
- Exported files are saved to `outputs/` with a `YYYYMMDD_HHMMSS.png` timestamp

---

*EchoFace — FFHQ StyleGAN2-ADA + CLIP ViT-B/32 + Whisper — Fully Local*
