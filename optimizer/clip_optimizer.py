from PIL import Image
import torch

from encoder.clip_encoder import CLIPEncoder  # noqa: F401 — imported for namespace patching in tests


def optimize(text_prompt, wrapper, cfg, callback=None):
    """
    Run CLIP-guided W-space latent optimization.

    Returns:
        (PIL.Image, w_tensor: torch.Tensor, final_sim_score: float)
    """
    raise NotImplementedError
