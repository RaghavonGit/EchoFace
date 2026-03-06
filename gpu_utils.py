# gpu_utils.py
# Single source of truth for device, dtype, and resolution across all EchoFace modules.
# Pattern derived from StyleGAN-Human/generate.py (on disk, lines 66-68, 101-108).
#
# NOTE: The 'dtype' field controls W-space latent dtype and Adam optimizer dtype.
# It does NOT control the StyleGAN synthesis network, which runs float32 throughout
# (generate.py passes force_fp32=True to G.synthesis() on CUDA as well).

import torch


def get_device_config() -> dict:
    """
    Detects CUDA availability and returns device configuration used by all
    downstream modules. Call once at module load time; pass result as argument.

    Returns:
        dict with keys:
            device        (torch.device) -- 'cuda' or 'cpu'
            dtype         (torch.dtype)  -- torch.float16 on CUDA, torch.float32 on CPU
            use_fp16      (bool)         -- True on CUDA only; pass to whisper.transcribe(fp16=...)
            resolution    (int)          -- 1024 on CUDA, 256 on CPU
            truncation_psi (float)       -- 1.0 on CUDA, 0.5 on CPU (per GEN-03 requirement)
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    return {
        "device": device,
        "dtype": torch.float16 if cuda_available else torch.float32,
        "use_fp16": cuda_available,
        "resolution": 1024 if cuda_available else 256,
        "truncation_psi": 1.0 if cuda_available else 0.5,
    }
