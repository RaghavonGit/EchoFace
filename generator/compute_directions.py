"""
compute_directions.py — Pre-compute W-space attribute directions.

Run once (offline):
    conda activate stylehuman
    python generator/compute_directions.py

Saves: checkpoints/directions.npz
  Keys: one per attribute name (e.g. "glasses"), value shape [18, 512] float32.

Runtime: ~3-5 min on CPU, ~30s if CUDA custom ops compile.
"""

import sys
import os
import contextlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "StyleGAN-Human"))

import torch
import numpy as np
import clip

import dnnlib
import legacy

PKL_PATH   = os.path.join(PROJECT_ROOT, "checkpoints", "ffhq", "ffhq.pkl")
OUT_PATH   = os.path.join(PROJECT_ROOT, "checkpoints", "directions.npz")
N_SAMPLES  = 200
BATCH_SIZE = 4
SYNTH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRIBUTES = {
    "glasses":        "a photo of a face with glasses",
    "beard":          "a photo of a face with a beard",
    "older":          "a photo of an older face",
    "younger":        "a photo of a younger face",
    "smile":          "a photo of a face with a wide smile",
    "blonde hair":    "a photo of a face with blonde hair",
    "dark hair":      "a photo of a face with dark hair",
    "curly hair":     "a photo of a face with curly hair",
    "straight hair":  "a photo of a face with straight hair",
    "more masculine": "a photo of a masculine face",
}

_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def _clip_encode_images(clip_model, imgs_01: torch.Tensor) -> torch.Tensor:
    imgs_224 = torch.nn.functional.interpolate(
        imgs_01, size=(224, 224), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(_CLIP_MEAN, device=imgs_224.device).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=imgs_224.device).view(1, 3, 1, 1)
    imgs_norm = ((imgs_224 - mean) / std).float()
    with torch.no_grad():
        feat = clip_model.encode_image(imgs_norm)
    return (feat / feat.norm(dim=-1, keepdim=True)).float()


def main():
    print(f"[compute_directions] Device: {DEVICE}")
    print(f"[compute_directions] N_SAMPLES={N_SAMPLES}, BATCH_SIZE={BATCH_SIZE}")

    print(f"[compute_directions] Loading StyleGAN from {PKL_PATH} ...")
    with _quiet(), dnnlib.util.open_url(PKL_PATH) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(DEVICE)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    num_ws = G.mapping.num_ws
    w_avg  = G.mapping.w_avg.to(DEVICE)
    print(f"[compute_directions] num_ws={num_ws}")

    print("[compute_directions] Loading CLIP ViT-B/32 ...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model = clip_model.float().eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    print("[compute_directions] Encoding attribute text prompts ...")
    text_feats = {}
    for attr_name, prompt in ATTRIBUTES.items():
        tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_text(tokens)
        text_feats[attr_name] = (feat / feat.norm(dim=-1, keepdim=True)).float()
        print(f"  '{attr_name}'")

    print(f"\n[compute_directions] Sampling {N_SAMPLES} random W+ latents ...")
    all_ws   = []
    all_imgs = []

    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(n_batches):
        n_this = min(BATCH_SIZE, N_SAMPLES - batch_idx * BATCH_SIZE)
        z = torch.randn(n_this, G.z_dim, device=DEVICE)

        with torch.no_grad():
            ws = G.mapping(z, None)

        with torch.no_grad():
            with _quiet():
                imgs_raw = G.synthesis(ws, noise_mode="const", force_fp32=True)
            imgs_01 = (imgs_raw + 1.0) * 0.5
            if SYNTH_SIZE != 1024:
                imgs_01 = torch.nn.functional.interpolate(
                    imgs_01, size=(SYNTH_SIZE, SYNTH_SIZE),
                    mode="bilinear", align_corners=False
                )
            img_feats = _clip_encode_images(clip_model, imgs_01)

        all_ws.append(ws.cpu())
        all_imgs.append(img_feats.cpu())

        done = min((batch_idx + 1) * BATCH_SIZE, N_SAMPLES)
        if done % 20 == 0 or done == N_SAMPLES:
            print(f"  {done}/{N_SAMPLES} samples ...")

    all_ws   = torch.cat(all_ws,   dim=0)   # [N, 18, 512]
    all_imgs = torch.cat(all_imgs, dim=0)   # [N, 512]

    w_avg_cpu  = w_avg.cpu()
    w_centered = all_ws - w_avg_cpu.unsqueeze(0).unsqueeze(0)  # [N, 18, 512]

    print("\n[compute_directions] Computing attribute directions ...")
    directions = {}
    for attr_name, text_feat in text_feats.items():
        text_feat_cpu = text_feat.cpu()
        sims = (all_imgs * text_feat_cpu).sum(dim=-1)   # [N]
        weighted  = (sims.view(N_SAMPLES, 1, 1) * w_centered).mean(dim=0)  # [18, 512]
        direction = weighted / (weighted.norm() + 1e-8)
        directions[attr_name] = direction.numpy().astype(np.float32)
        print(f"  '{attr_name}': mean_sim={sims.mean().item():.4f}  "
              f"max_sim={sims.max().item():.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, **directions)
    print(f"\n[compute_directions] Saved {len(directions)} directions -> {OUT_PATH}")
    print("[compute_directions] Keys:", list(directions.keys()))


if __name__ == "__main__":
    main()
