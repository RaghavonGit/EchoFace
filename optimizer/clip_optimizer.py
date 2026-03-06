"""
CLIP-guided W-space latent optimizer for EchoFace.

Entry point: optimize(text_prompt, wrapper, cfg, callback=None)

Gradient flow:
    w (requires_grad=True)
      -> G.synthesis(w, noise_mode='const', force_fp32=True)   [float32, grad flows]
      -> manual CLIP normalization + F.interpolate to 224x224  [differentiable]
      -> encoder.model.encode_image()                          [CLIP frozen, grad flows fwd]
      -> F.normalize + cosine sim + L2 reg                     [loss scalar]
      -> loss.backward()                                        [grad accumulates on w]

wrapper.synthesize(w) is used ONLY for the final image render (uint8, no grad).
"""

import logging

import torch
import torch.nn.functional as F
from PIL import Image

from encoder.clip_encoder import CLIPEncoder  # noqa: F401 — imported for namespace patching in tests

logger = logging.getLogger(__name__)

# CLIP ViT-B/32 training normalization constants (canonical values from openai/CLIP).
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

_N_STEPS       = 150
_LR            = 0.01
_LAMBDA        = 0.1
_MAX_NORM      = 1.0
_CALLBACK_EVERY = 10


def _synth_for_clip(w: torch.Tensor, wrapper, device: torch.device) -> torch.Tensor:
    """Differentiable synthesis -> CLIP-ready [1, 3, 224, 224] tensor.

    Gradient flows from this tensor back through w.
    Must NOT call wrapper.synthesize() here — uint8 conversion breaks the graph.
    """
    mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=device).view(1, 3, 1, 1)

    # Raw float synthesis: [1, 3, H, W], range approx [-1, 1]
    img = wrapper.G.synthesis(w, noise_mode='const', force_fp32=True)
    img = (img * 0.5 + 0.5).clamp(0, 1)   # [-1, 1] -> [0, 1]
    img = (img - mean) / std               # CLIP normalization
    img = F.interpolate(img, size=(224, 224), mode='bicubic', align_corners=False)
    return img  # [1, 3, 224, 224], gradient-safe


def optimize(text_prompt: str, wrapper, cfg: dict, callback=None):
    """Run CLIP-guided W-space latent optimization.

    Args:
        text_prompt: Face description (keep under 77 tokens).
        wrapper: StyleGANWrapper instance (generator parameters must be frozen).
        cfg: Config dict from gpu_utils.get_device_config().
        callback: Optional callable. Signature: callback(step, loss, sim_score).
                  Called every 10 steps (at steps 10, 20, ..., 150).

    Returns:
        Tuple of (PIL.Image, w_tensor, final_sim_score):
            PIL.Image       -- final rendered face image
            w_tensor        -- best W-space latent (torch.Tensor, detached)
            final_sim_score -- cosine similarity at best step (Python float)
    """
    device = cfg['device']

    # Build encoder (loads CLIP once; CLIP weights stay frozen)
    encoder = CLIPEncoder(cfg)

    # Pre-compute text embedding — constant throughout optimization
    text_feat = encoder.encode_text(text_prompt)  # [1, 512] normalized, no grad

    # OPT-01: sample w_init from W-space via G.mapping
    w_init = wrapper.sample_w().detach().clone()  # detach + clone: w_init must NOT carry grad
    w = w_init.clone().requires_grad_(True)        # optimization variable

    optimizer = torch.optim.Adam([w], lr=_LR)

    best_w   = w_init.clone()
    best_sim = -float('inf')

    for step in range(_N_STEPS):
        optimizer.zero_grad()

        # Differentiable path: gradient flows through w -> G.synthesis -> CLIP
        img_clip = _synth_for_clip(w, wrapper, device)
        img_feat_raw = encoder.model.encode_image(img_clip)
        img_feat = F.normalize(img_feat_raw.float(), dim=-1)

        # Cosine similarity (dot product of unit vectors)
        sim = (text_feat * img_feat).sum(dim=-1)  # scalar tensor

        # Regularization: L2 distance from w_init prevents latent collapse
        reg = ((w - w_init) ** 2).sum()

        loss = -sim + _LAMBDA * reg

        # OPT-03: NaN detection — check BEFORE backward (backward on NaN raises / corrupts grads)
        if torch.isnan(loss):
            logger.warning(
                "NaN loss detected at step %d; stopping early. "
                "Returning best latent (sim=%.4f).",
                step,
                best_sim,
            )
            break

        loss.backward()
        # OPT-03: gradient clipping — AFTER backward, BEFORE optimizer.step()
        torch.nn.utils.clip_grad_norm_([w], max_norm=_MAX_NORM)
        optimizer.step()

        sim_val = sim.item()
        if sim_val > best_sim:
            best_sim = sim_val
            best_w = w.detach().clone()

        if callback is not None and (step + 1) % _CALLBACK_EVERY == 0:
            callback(step=step + 1, loss=loss.item(), sim_score=sim_val)

    # Render final image using the uint8 path (outside gradient context)
    with torch.no_grad():
        img_hwc = wrapper.synthesize(best_w)
    final_image = Image.fromarray(img_hwc.numpy(), 'RGB')

    return final_image, best_w, float(best_sim)
