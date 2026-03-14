"""
StyleCLIPWrapper — Text-Driven Manipulation of StyleGAN Imagery.

This module drives face generation by optimizing the StyleGAN W+ latent space
toward a user-supplied text prompt using CLIP as the loss signal.

Key design decisions:
  - We use CLIP directly (no CLIPLoss wrapper) to avoid the unclear upsample/pool
    tensor dimension assumptions in the original StyleCLIP codebase.
  - We clamp the StyleGAN output to 256x256 during the optimization loop for speed
    (full 1024x1024 inference only happens for the final output image).
  - We use torch.float32 throughout for CPU compatibility.
"""

import sys
import os
import io
import traceback
import contextlib

import torch
import clip
import torchvision.transforms.functional as TVF
from PIL import Image

# ---------------------------------------------------------------------------
# Extend Python path so legacy StyleGAN-Human imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "StyleGAN-Human"))

import dnnlib
import legacy

# ---------------------------------------------------------------------------
# Suppress the repeated "Setting up PyTorch plugin ... Failed!" spam from
# dnnlib's custom CUDA ops — they fall back to Python automatically and the
# messages are printed to stdout on every synthesis call on Windows.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


# CLIP normalisation constants (ViT-B/32)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class StyleCLIPWrapper:
    """
    Wraps a pre-trained StyleGAN-Human generator and optimizes its W+ latent
    vector so that the synthesised face maximises cosine similarity to the
    given text prompt as scored by CLIP ViT-B/32.
    """

    def __init__(self, pkl_path: str, cfg: dict) -> None:
        self.device = cfg["device"]
        print(f"[StyleCLIPWrapper] Device: {self.device}")
        print(f"[StyleCLIPWrapper] Loading FFHQ StyleGAN2 from:\n  {pkl_path}")

        # ── 1. StyleGAN generator ────────────────────────────────────────────
        with _quiet(), dnnlib.util.open_url(pkl_path) as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].to(self.device)
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad_(False)

        print(f"[StyleCLIPWrapper] StyleGAN loaded. "
              f"num_ws={self.G.mapping.num_ws}  "
              f"w_avg shape={self.G.mapping.w_avg.shape}")

        # ── 2. CLIP model (frozen, float32) ─────────────────────────────────
        print("[StyleCLIPWrapper] Loading CLIP ViT-B/32 ...")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        # Cast to float32 immediately — CLIP loads in float16 on CUDA by default.
        # float16 activations during backward (attention softmax, matmul) produce
        # NaN gradients when used as a differentiable loss signal.
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # Compute & cache the starting w-latent (mean face)
        self._w_avg = (
            self.G.mapping.w_avg                        # [512]
            .detach()
            .clone()
            .to(self.device)
        )

        # ── Edit state ────────────────────────────────────────────────────────
        self._current_w:    "torch.Tensor | None" = None
        self._original_w:   "torch.Tensor | None" = None
        self._original_prompt: "str | None" = None
        self._original_sim: "float | None" = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, prompt: str) -> torch.Tensor:
        """Tokenise and embed a text prompt via CLIP. Returns [1, 512]."""
        tokens = clip.tokenize([prompt], truncate=True).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(tokens)
        return feat / feat.norm(dim=-1, keepdim=True)   # normalised

    def _synthesise_small(self, w: torch.Tensor) -> torch.Tensor:
        """
        Synthesise full-body portrait and resize to 256×256 for CLIP scoring.
        w shape: [1, num_ws, 512].
        Returns a float32 image tensor in [0, 1], shape [1, 3, 256, 256],
        WITH gradients flowing back through w.
        Full body is kept so CLIP can score both face attributes AND clothing.
        """
        with _quiet():
            img = self.G.synthesis(w, noise_mode="const", force_fp32=True)  # [-1, 1]
        img = (img + 1.0) * 0.5                                          # [0, 1]
        # FFHQ output is square [B, 3, 1024, 1024] — no face crop needed
        img = torch.nn.functional.interpolate(
            img, size=(256, 256), mode="bilinear", align_corners=False
        )
        return img

    def _clip_encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resize to 224x224, normalise, and encode through CLIP.
        img: [B, 3, H, W] float32 in [0, 1].
        Returns normalised feature vector [B, 512].
        NOTE: Always keep float32 — converting to float16 on CUDA 
        causes gradient overflow (NaN) during backpropagation.
        """
        img224 = torch.nn.functional.interpolate(
            img, size=(224, 224), mode="bilinear", align_corners=False
        )
        # Normalise channel-wise
        mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor(_CLIP_STD,  device=self.device).view(1, 3, 1, 1)
        img224_norm = ((img224 - mean) / std).float()  # always float32
        feat = self.clip_model.encode_image(img224_norm)
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    # ------------------------------------------------------------------
    # Public API (matches DiffusionWrapper signature expected by app.py)
    # ------------------------------------------------------------------

    def generate(
        self,
        text_prompt: str,
        steps: int = 60,
        lr: float = 0.05,
        callback=None,
    ) -> tuple:
        """
        Optimise the W+ latent to match text_prompt.

        Args:
            text_prompt: Natural language description of the face.
            steps:       Number of Adam optimisation steps.
            lr:          Learning rate for Adam.
            callback:    Optional fn(step, loss, PIL.Image) called every 10 steps.

        Returns:
            (PIL.Image, float) — final face image and cosine similarity score.
        """
        # ── Prompt engineering: prepend photo context for better CLIP alignment
        clipped_prompt = f"a photo of a face, {text_prompt}"
        print(f"[StyleCLIP] Prompt: '{clipped_prompt}'")

        # ── Build initial W+ latent from mean face ───────────────────────────
        num_ws = self.G.mapping.num_ws
        w_opt = (
            self._w_avg
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_ws, -1)
            .clone()
            .requires_grad_(True)
        )

        optimizer = torch.optim.Adam([w_opt], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        # ── Encode the text prompt once (it doesn't change) ─────────────────
        text_feat = self._encode_text(clipped_prompt)   # [1, 512], no grad

        # ── W+ anchor for L2 regulariser — must stay fixed at w_avg ─────────
        # Anchoring to anything else lets the optimizer drift into artifact space.
        w_avg_expand = (
            self._w_avg
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_ws, -1)
            .detach()
        )

        best_loss = float("inf")
        no_improve = 0

        try:
            for step in range(steps):

                optimizer.zero_grad()

                # Forward: synthesise small image WITH gradient
                img_small = self._synthesise_small(w_opt)   # [1,3,256,256]

                # Image CLIP embed (grads flow back through img → w_opt)
                img_feat = self._clip_encode_image(img_small.float())

                # Maximise cosine similarity → minimise negative cosine
                cos_loss = 1.0 - (img_feat * text_feat).sum(dim=-1).mean()

                # L2 regulariser: stay close to the mean face
                # epsilon inside sqrt prevents NaN gradient at step 0 when w_opt == w_avg
                l2_loss = (w_opt - w_avg_expand).pow(2).sum().add(1e-8).sqrt()

                loss = cos_loss + 0.02 * l2_loss

                # Guard: skip step if loss is NaN (can happen on first few CUDA steps)
                if torch.isnan(loss):
                    print(f"  ⚠ Step {step}: NaN loss detected, skipping update")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_([w_opt], max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Early stopping: quit if loss hasn't improved by >0.001 in 15 steps
                if loss.item() < best_loss - 0.001:
                    best_loss = loss.item()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 15:
                    print(f"  → Early stop at step {step} (no improvement for 15 steps)")
                    break

                if step % 10 == 0:
                    print(
                        f"  → Step {step:3d}/{steps} "
                        f"| CLIP loss: {cos_loss.item():.4f} "
                        f"| L2: {l2_loss.item():.2f}"
                    )
                    if callback is not None:
                        with torch.no_grad():
                            preview = self._synthesise_small(w_opt.detach())
                            pil_preview = Image.fromarray(
                                (preview[0]
                                 .permute(1, 2, 0)
                                 .clamp(0, 1)
                                 .mul(255)
                                 .to(torch.uint8)
                                 .cpu()
                                 .numpy()),
                                "RGB",
                            )
                        callback(step, loss.item(), pil_preview)

            # ── Final render: FFHQ face → 1024×1024 ─────────────────────────
            print("[StyleCLIP] Rendering final face (1024×1024) ...")
            with torch.no_grad():
                with _quiet():
                    final_raw = self.G.synthesis(w_opt.detach(), noise_mode="const", force_fp32=True)
                final_raw = (final_raw + 1.0) * 0.5
                final_raw = torch.nn.functional.interpolate(
                    final_raw, size=(1024, 1024), mode="bilinear", align_corners=False
                )
                final_arr = (
                    final_raw[0]
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                    .mul(255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )

            final_pil = Image.fromarray(final_arr, "RGB")
            final_sim  = 1.0 - cos_loss.item()   # re-use last step's value

            if callback is not None:
                callback(steps, final_sim, final_pil)

            # Store state for edit / reset
            self._original_prompt = text_prompt          # raw, pre-prefix
            self._current_w  = w_opt.detach().clone()
            self._original_w = self._current_w.clone()
            self._original_sim = final_sim

            return final_pil, final_sim

        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            traceback.print_exc()
            raise

    def edit(
        self,
        edit_instruction: str,
        steps: int = 80,
        lr: float = 0.08,
        callback=None,
    ) -> tuple:
        """
        Apply a targeted edit to the current face by optimizing from _current_w.

        Args:
            edit_instruction: Natural language edit (e.g. "change eyes to blue").
            steps:            Number of Adam optimisation steps.
            lr:               Learning rate for Adam.
            callback:         Optional fn(step, loss, PIL.Image) called every 10 steps.

        Returns:
            (PIL.Image, float) — updated face image and cosine similarity score.

        Raises:
            RuntimeError: If generate() has not been called yet.
        """
        if self._current_w is None:
            raise RuntimeError("Generate a face first before editing.")

        # Use the edit instruction as a standalone CLIP target.
        # "a photo of a face, original + change eyes to red" confuses CLIP — it
        # can't subtract the original description.  A clean positive target
        # ("a photo of a face with red eyes") gives a much stronger gradient signal.
        combined_prompt = f"a photo of a face, {edit_instruction}"
        print(f"[StyleCLIP.edit] Prompt: '{combined_prompt}'")

        # Start from the current (possibly already edited) W+ latent
        w_start = self._current_w.clone()          # snapshot for diagnostics
        w_opt   = self._current_w.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([w_opt], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        text_feat = self._encode_text(combined_prompt)  # [1, 512], no grad

        # L2 anchor: current face, not w_avg — keeps identity stable during edit
        w_anchor = self._current_w.detach()

        best_cos = float("inf")
        no_improve = 0

        try:
            for step in range(steps):

                optimizer.zero_grad()

                img_small = self._synthesise_small(w_opt)
                img_feat  = self._clip_encode_image(img_small.float())

                cos_loss = 1.0 - (img_feat * text_feat).sum(dim=-1).mean()
                l2_loss  = (w_opt - w_anchor).pow(2).sum().add(1e-8).sqrt()

                # L2 anchors to current face to prevent identity drift.
                # 0.04 keeps changes meaningful but stops the whole face from changing.
                loss = cos_loss + 0.04 * l2_loss

                if torch.isnan(loss):
                    print(f"  ⚠ Edit step {step}: NaN loss, skipping")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_([w_opt], max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Track only CLIP loss for early stopping — the L2 penalty grows
                # as w_opt drifts and would otherwise dominate and kill the edit
                # before the attribute change (e.g. glasses) has time to appear.
                if cos_loss.item() < best_cos - 0.001:
                    best_cos = cos_loss.item()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 40:
                    print(f"  → Early stop at step {step} (CLIP plateau)")
                    break

                if step % 10 == 0:
                    print(
                        f"  → Edit step {step:3d}/{steps} "
                        f"| CLIP loss: {cos_loss.item():.4f} "
                        f"| L2: {l2_loss.item():.2f}"
                    )
                    if callback is not None:
                        with torch.no_grad():
                            preview = self._synthesise_small(w_opt.detach())
                            pil_preview = Image.fromarray(
                                (preview[0]
                                 .permute(1, 2, 0)
                                 .clamp(0, 1)
                                 .mul(255)
                                 .to(torch.uint8)
                                 .cpu()
                                 .numpy()),
                                "RGB",
                            )
                        callback(step, loss.item(), pil_preview)

            # Diagnostics — tells us definitively whether the optimizer moved
            w_delta = (w_opt.detach() - w_start).abs()
            print(
                f"[StyleCLIP.edit] w_opt change — "
                f"max={w_delta.max().item():.6f}  "
                f"mean={w_delta.mean().item():.6f}  "
                f"(zero means grads not flowing)"
            )

            # Final render at 1024×1024
            print("[StyleCLIP.edit] Rendering final face (1024×1024) ...")
            with torch.no_grad():
                with _quiet():
                    final_raw = self.G.synthesis(
                        w_opt.detach(), noise_mode="const", force_fp32=True
                    )
                final_raw = (final_raw + 1.0) * 0.5
                final_raw = torch.nn.functional.interpolate(
                    final_raw, size=(1024, 1024), mode="bilinear", align_corners=False
                )
                final_arr = (
                    final_raw[0]
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                    .mul(255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )

            final_pil = Image.fromarray(final_arr, "RGB")
            final_sim  = 1.0 - cos_loss.item()

            if callback is not None:
                callback(steps, final_sim, final_pil)

            # Advance _current_w for cumulative edits
            self._current_w = w_opt.detach().clone()

            return final_pil, final_sim

        except Exception as e:
            print(f"\n[CRITICAL ERROR in edit] {e}")
            traceback.print_exc()
            raise

    def reset(self) -> tuple:
        """
        Restore the face to the state produced by the most recent generate() call.

        Returns:
            (PIL.Image, float) — original generated face and its CLIP similarity score.

        Raises:
            RuntimeError: If generate() has not been called yet.
        """
        if self._original_w is None:
            raise RuntimeError("Generate a face first.")

        # Roll back latent to original
        self._current_w = self._original_w.clone()

        # Single synthesis forward pass — no optimisation
        print("[StyleCLIP.reset] Rendering original face ...")
        with torch.no_grad():
            with _quiet():
                raw = self.G.synthesis(
                    self._current_w, noise_mode="const", force_fp32=True
                )
            raw = (raw + 1.0) * 0.5
            raw = torch.nn.functional.interpolate(
                raw, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            arr = (
                raw[0]
                .permute(1, 2, 0)
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

        pil_image = Image.fromarray(arr, "RGB")
        return pil_image, self._original_sim
