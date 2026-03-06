import clip
import torch
import torch.nn.functional as F
from PIL import Image


class CLIPEncoder:
    """Wraps CLIP ViT-B/32 for text and image embedding.

    Args:
        cfg: Config dict from gpu_utils.get_device_config(). Uses 'device' key.

    Attributes:
        model: Frozen CLIP model (requires_grad=False on all parameters).
        preprocess: CLIP built-in image transform (resize, center-crop, normalize).
        device: torch.device from cfg.
    """

    def __init__(self, cfg: dict) -> None:
        self.device = cfg['device']
        # jit=False required: allows weight access and casting to float32
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # Freeze CLIP — FP16 weights produce NaN under gradient updates
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text description into a normalized embedding.

        Args:
            text: Face description string (keep under 77 tokens).

        Returns:
            Tensor of shape [1, 512], float32, L2-normalized.
        """
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
        return F.normalize(feat.float(), dim=-1)

    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into a normalized embedding.

        Args:
            pil_image: PIL Image (any size; preprocess handles resize/crop).

        Returns:
            Tensor of shape [1, 512], float32, L2-normalized.
        """
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img_tensor)
        return F.normalize(feat.float(), dim=-1)
