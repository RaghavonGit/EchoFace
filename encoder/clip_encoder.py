import torch
import clip  # noqa: F401 — imported so tests can patch encoder.clip_encoder.clip
from PIL import Image


class CLIPEncoder:
    def __init__(self, cfg: dict) -> None:
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        raise NotImplementedError
