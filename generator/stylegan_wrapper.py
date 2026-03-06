"""
StyleGANWrapper — thin wrapper around the StyleGAN-Human generator network.

Side-effect note: legacy.load_network_pkl() unconditionally writes ori_model.txt
to the current working directory. This cannot be disabled without modifying
StyleGAN-Human/ (off-limits). The file is gitignored via .gitignore.
"""

import sys
import os
import datetime

# ---------------------------------------------------------------------------
# sys.path injection — MUST precede any dnnlib / legacy import
# ---------------------------------------------------------------------------
_SG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'StyleGAN-Human'))
if _SG_ROOT not in sys.path:
    sys.path.insert(0, _SG_ROOT)

import dnnlib
import legacy

# ---------------------------------------------------------------------------
# Standard imports (after StyleGAN-Human is on sys.path)
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# gpu_utils lives at the project root
import gpu_utils  # noqa: E402  (imported after sys.path mutation)


class StyleGANWrapper:
    """Wraps a StyleGAN-Human EMA generator for W-space sampling and synthesis.

    Args:
        pkl_path: Path to the .pkl file containing the trained network.
        cfg: Config dict as returned by gpu_utils.get_device_config(). Expected keys:
            device        (torch.device)
            dtype         (torch.dtype)
            use_fp16      (bool)
            resolution    (int)
            truncation_psi (float)

    Side-effect: legacy.load_network_pkl() writes ori_model.txt to CWD
    unconditionally. This cannot be disabled without modifying StyleGAN-Human/
    (off-limits). The file is gitignored.
    """

    def __init__(self, pkl_path: str, cfg: dict) -> None:
        self.device = cfg['device']
        self.dtype = cfg['dtype']
        self.resolution = cfg['resolution']
        self.truncation_psi = cfg['truncation_psi']

        # Load EMA generator via legacy helper (safe deserialization)
        with dnnlib.util.open_url(os.path.abspath(pkl_path)) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        # Set eval mode (disables dropout / running-stat updates)
        self.G = self.G.eval()

        # Freeze ALL parameters — only the W-space latent carries gradients
        # during downstream optimization (Phase 3).
        for p in self.G.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_w(self) -> torch.Tensor:
        """Sample a random W+ latent code.

        Returns:
            Tensor of shape [1, G.num_ws, 512] on self.device.
        """
        z = torch.randn(1, self.G.z_dim, device=self.device)
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        return self.G.mapping(z, label, truncation_psi=self.truncation_psi)

    def synthesize(self, w: torch.Tensor) -> torch.Tensor:
        """Render a W+ latent code to a uint8 HWC image tensor.

        Args:
            w: Latent tensor of shape [1, num_ws, 512].

        Returns:
            Tensor of shape [H, W, 3] uint8 on CPU, where H=W=self.resolution.
        """
        # force_fp32=True is MANDATORY — matches generate.py reference impl.
        img = self.G.synthesis(w, noise_mode='const', force_fp32=True)

        # Exact normalisation formula from StyleGAN-Human/generate.py
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_hwc = img[0].cpu()  # shape [H_native, W_native, 3]

        # Resize to target resolution when cfg differs from native resolution
        native_h = img_hwc.shape[0]
        if self.resolution != native_h:
            # F.interpolate expects BCHW float; permute HWC -> CHW -> BCHW
            img_f = img_hwc.permute(2, 0, 1).unsqueeze(0).float()
            img_f = F.interpolate(
                img_f,
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False,
            )
            img_hwc = img_f[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8)

        return img_hwc

    def generate_and_save(self, out_dir: str = 'outputs') -> str:
        """Sample a latent, synthesise an image, and save it as a PNG.

        Args:
            out_dir: Directory to write the PNG into (created if absent).

        Returns:
            Absolute path to the saved PNG file.
        """
        w = self.sample_w()
        img_hwc = self.synthesize(w)

        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(out_dir, f"gen_{timestamp}.png")
        Image.fromarray(img_hwc.numpy(), 'RGB').save(path)
        return path
