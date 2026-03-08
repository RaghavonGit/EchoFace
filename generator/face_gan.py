"""
CLIP-conditioned face GAN architecture.

Generator:  CLIP embedding (512-dim) → project (128-dim) + noise (128-dim)
            → DCGAN deconv upsample → 3 x 64 x 64 face image in [-1, 1]

Discriminator: 3 x 64 x 64 face + CLIP embed → real/fake logit
               Spectral norm on all conv layers for training stability.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# ── Dimensions ────────────────────────────────────────────────────────────────
EMBED_DIM = 512   # CLIP ViT-B/32 output dimension
PROJ_DIM  = 128   # projected embedding dimension
NOISE_DIM = 128   # noise vector dimension
LATENT_DIM = NOISE_DIM + PROJ_DIM  # 256 — generator seed
NGF = 64          # generator base filter count
NDF = 64          # discriminator base filter count
IMAGE_SIZE = 64   # output face resolution
# ─────────────────────────────────────────────────────────────────────────────


class FaceGenerator(nn.Module):
    """DCGAN-style generator conditioned on a CLIP text/image embedding.

    Args:
        embed_dim: dimensionality of input embedding (default 512 for CLIP ViT-B/32)
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Project CLIP embedding to a compact conditioning vector
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, PROJ_DIM),
            nn.BatchNorm1d(PROJ_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # DCGAN deconvolution backbone: (LATENT_DIM, 1, 1) → (3, 64, 64)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.Dropout2d(0.3),                                        # anti-overfit
            # → (NGF*8, 4, 4)
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            # → (NGF*4, 8, 8)
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # → (NGF*2, 16, 16)
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # → (NGF, 32, 32)
            nn.ConvTranspose2d(NGF, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # → (3, 64, 64)
        )

    def forward(self, embed: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embed: (B, embed_dim) CLIP embedding, L2-normalised
            noise: (B, NOISE_DIM) standard normal noise

        Returns:
            (B, 3, 64, 64) tensor in [-1, 1]
        """
        proj = self.projection(embed)                          # (B, PROJ_DIM)
        latent = torch.cat([proj, noise], dim=1)               # (B, LATENT_DIM)
        latent = latent.unsqueeze(2).unsqueeze(3)              # (B, LATENT_DIM, 1, 1)
        return self.net(latent)


class FaceDiscriminator(nn.Module):
    """Spectral-norm DCGAN discriminator conditioned on a CLIP embedding.

    Uses spectral normalisation on all conv layers to stabilise training
    and reduce mode collapse.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Image encoder: (3, 64, 64) → (NDF*8, 4, 4)
        self.img_encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(3, NDF, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (NDF, 32, 32)
            spectral_norm(nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (NDF*2, 16, 16)
            spectral_norm(nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (NDF*4, 8, 8)
            spectral_norm(nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (NDF*8, 4, 4)
        )

        # Project CLIP embed to a spatial conditioning tensor
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, PROJ_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final real/fake classifier: concat image features + embed projection
        # (NDF*8 + PROJ_DIM) x 4 x 4 → scalar logit
        self.classifier = spectral_norm(
            nn.Conv2d(NDF * 8 + PROJ_DIM, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:   (B, 3, 64, 64) face image in [-1, 1]
            embed: (B, embed_dim) CLIP embedding, L2-normalised

        Returns:
            (B,) real/fake logits (no sigmoid — use BCEWithLogitsLoss)
        """
        feat = self.img_encoder(img)                              # (B, NDF*8, 4, 4)
        proj = self.embed_proj(embed)                             # (B, PROJ_DIM)
        proj_spatial = proj.unsqueeze(2).unsqueeze(3).expand(-1, -1, 4, 4)
        combined = torch.cat([feat, proj_spatial], dim=1)         # (B, NDF*8+PROJ_DIM, 4, 4)
        out = self.classifier(combined)                           # (B, 1, 1, 1)
        return out.view(-1)                                       # (B,)


def weights_init(module: nn.Module) -> None:
    """Xavier init for Conv/Linear, constant 1 for BatchNorm."""
    classname = module.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname and hasattr(module, "weight"):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
