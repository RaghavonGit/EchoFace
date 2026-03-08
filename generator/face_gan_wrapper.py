"""
FaceGANWrapper — CLIP-conditioned face GAN inference.

Replaces the 150-step CLIP optimizer for text-to-face generation.
Single forward pass: CLIP text embed → FaceGenerator → 64x64 face image.

Usage:
    from generator.face_gan_wrapper import FaceGANWrapper
    gan = FaceGANWrapper("checkpoints/face_gan/gen_best.pth", cfg)
    pil_image, sim_score = gan.generate("a young woman with dark curly hair")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import clip
from generator.face_gan import FaceGenerator, NOISE_DIM


class FaceGANWrapper:
    """Wraps FaceGenerator + CLIP for text-to-face generation.

    Args:
        checkpoint_path: path to generator .pth checkpoint (gen_best.pth)
        cfg: dict from gpu_utils.get_device_config()
        upscale_size: output PIL image size (default 256 — upscaled from 64)
    """

    def __init__(self, checkpoint_path: str, cfg: dict, upscale_size: int = 256):
        self.device = cfg["device"]
        self.upscale_size = upscale_size

        # ── Load CLIP ─────────────────────────────────────────────────────────
        print("[FaceGANWrapper] Loading CLIP ViT-B/32 ...")
        self._clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        self._clip_model.eval().requires_grad_(False)
        # Cast weights to float32 to avoid NaN from FP16
        self._clip_model = self._clip_model.float()

        # ── Load Generator ────────────────────────────────────────────────────
        self.netG = FaceGenerator().to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.netG.load_state_dict(state)
            print(f"[FaceGANWrapper] Generator loaded from {checkpoint_path}")
        else:
            print("[FaceGANWrapper] No checkpoint — using random weights (untrained).")
        self.netG.eval()

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """Return L2-normalised CLIP text embedding (1, 512)."""
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        feat = self._clip_model.encode_text(tokens)
        return F.normalize(feat.float(), dim=-1)  # (1, 512)

    @torch.no_grad()
    def _encode_image_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """img_tensor: (1, 3, 64, 64) in [-1, 1].  Returns (1, 512) normalised."""
        # CLIP expects (1, 3, 224, 224) in range [-1,1] after its own normalise.
        # We resize and apply CLIP's normalise manually.
        resized = F.interpolate(img_tensor, size=(224, 224), mode="bilinear",
                                align_corners=False)
        # CLIP normalise: mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=self.device).view(1, 3, 1, 1)
        # Our image is in [-1, 1]; convert to [0, 1] first
        x = (resized + 1.0) / 2.0
        x = (x - mean) / std
        feat = self._clip_model.encode_image(x)
        return F.normalize(feat.float(), dim=-1)  # (1, 512)

    @torch.no_grad()
    def generate(self, text_prompt: str) -> tuple:
        """Generate a face image from a text description.

        Args:
            text_prompt: natural language face description

        Returns:
            (PIL.Image, float) — upscaled face image, CLIP cosine similarity score
        """
        text_embed = self._encode_text(text_prompt)  # (1, 512)

        # Sample noise and generate
        noise = torch.randn(1, NOISE_DIM, device=self.device)
        fake = self.netG(text_embed, noise)  # (1, 3, 64, 64) in [-1, 1]

        # Compute CLIP similarity: text embed vs generated image embed
        img_embed = self._encode_image_tensor(fake)  # (1, 512)
        sim_score = float((text_embed * img_embed).sum().item())

        # Convert to PIL Image
        img_np = fake.squeeze(0).cpu().float().numpy()        # (3, 64, 64)
        img_np = (img_np * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        img_np = img_np.transpose(1, 2, 0)                    # CHW → HWC
        pil = Image.fromarray(img_np, "RGB")

        if self.upscale_size != 64:
            pil = pil.resize((self.upscale_size, self.upscale_size), Image.LANCZOS)

        return pil, sim_score
