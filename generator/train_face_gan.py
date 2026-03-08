"""
Train the CLIP-conditioned FaceGAN on the pre-built HDF5 dataset.

Usage:
    conda run -n stylehuman python generator/train_face_gan.py ^
        --dataset  datasets/faces_clip.hdf5 ^
        --epochs   150 ^
        --batch_size 64 ^
        --output   checkpoints/face_gan

Resume from a checkpoint:
    ... --gen_ckpt  checkpoints/face_gan/gen_epoch_50.pth ^
        --disc_ckpt checkpoints/face_gan/disc_epoch_50.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from generator.face_gan import FaceGenerator, FaceDiscriminator, weights_init, NOISE_DIM


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceClipDataset(Dataset):
    """Loads the HDF5 split into RAM at init for fast iteration."""

    def __init__(self, h5_path: str, split: str = "train"):
        print(f"  Loading {split} split into RAM ...", flush=True)
        with h5py.File(h5_path, "r") as f:
            self._faces  = torch.from_numpy(f[split]["faces"][:])        # (N,3,64,64)
            self._embeds = torch.from_numpy(f[split]["embeds"][:])       # (N,512)
            self._wrong  = torch.from_numpy(f[split]["wrong_faces"][:])  # (N,3,64,64)
        print(f"  {split}: {len(self._faces)} samples loaded.", flush=True)

    def __len__(self):
        return len(self._faces)

    def __getitem__(self, idx):
        return self._faces[idx], self._embeds[idx], self._wrong[idx]


# ── Training ──────────────────────────────────────────────────────────────────

def _smooth(labels: torch.Tensor, amount: float = 0.1) -> torch.Tensor:
    """One-sided label smoothing: real labels → 1-amount."""
    return labels - amount


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = FaceClipDataset(args.dataset, "train")
    val_ds   = FaceClipDataset(args.dataset, "val")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Models ────────────────────────────────────────────────────────────────
    netG = FaceGenerator().to(device)
    netD = FaceDiscriminator().to(device)

    start_epoch = 0
    if args.gen_ckpt and os.path.exists(args.gen_ckpt):
        netG.load_state_dict(torch.load(args.gen_ckpt, map_location=device))
        print(f"Resumed G from {args.gen_ckpt}")
        # Try to infer epoch from filename gen_epoch_N.pth
        try:
            start_epoch = int(os.path.basename(args.gen_ckpt).split("_")[-1].replace(".pth", "")) + 1
        except Exception:
            pass
    else:
        netG.apply(weights_init)

    if args.disc_ckpt and os.path.exists(args.disc_ckpt):
        netD.load_state_dict(torch.load(args.disc_ckpt, map_location=device))
        print(f"Resumed D from {args.disc_ckpt}")
    else:
        netD.apply(weights_init)

    # ── Optimisers & losses ───────────────────────────────────────────────────
    optimG = torch.optim.Adam(netG.parameters(), lr=args.lr,        betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=args.lr * 2.0,  betas=(0.5, 0.999))
    bce    = nn.BCEWithLogitsLoss()
    l1     = nn.L1Loss()

    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        netG.train()
        netD.train()
        ep_d = ep_g = 0.0

        for face, embed, wrong in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            face  = face.to(device)
            embed = embed.to(device)
            wrong = wrong.to(device)
            bs    = face.size(0)

            noise = torch.randn(bs, NOISE_DIM, device=device)
            real_lbl  = _smooth(torch.ones(bs,  device=device))
            fake_lbl  =         torch.zeros(bs, device=device)

            # ── Discriminator step ────────────────────────────────────────────
            netD.zero_grad()
            fake           = netG(embed, noise).detach()
            loss_d_real    = bce(netD(face,  embed), real_lbl)
            loss_d_fake    = bce(netD(fake,  embed), fake_lbl)
            loss_d_wrong   = bce(netD(wrong, embed), fake_lbl)
            loss_d = loss_d_real + loss_d_fake + loss_d_wrong
            loss_d.backward()
            optimD.step()

            # ── Generator step ────────────────────────────────────────────────
            netG.zero_grad()
            noise2         = torch.randn(bs, NOISE_DIM, device=device)
            fake2          = netG(embed, noise2)
            loss_g_adv     = bce(netD(fake2, embed), real_lbl)
            loss_g_l1      = l1(fake2, face) * args.l1_coef
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            optimG.step()

            ep_d += loss_d.item()
            ep_g += loss_g.item()

        n_batches = len(train_dl)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"D={ep_d/n_batches:.4f}  G={ep_g/n_batches:.4f}")

        # ── Validation loss ───────────────────────────────────────────────────
        netG.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for face, embed, _ in val_dl:
                face  = face.to(device)
                embed = embed.to(device)
                noise = torch.randn(face.size(0), NOISE_DIM, device=device)
                fake  = netG(embed, noise)
                val_l1 += l1(fake, face).item()
        val_l1 /= max(len(val_dl), 1)
        print(f"           val_L1={val_l1:.4f}")

        # ── Checkpoints ───────────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            gen_path  = os.path.join(args.output, f"gen_epoch_{epoch}.pth")
            disc_path = os.path.join(args.output, f"disc_epoch_{epoch}.pth")
            torch.save(netG.state_dict(), gen_path)
            torch.save(netD.state_dict(), disc_path)
            print(f"  Saved: {gen_path}")

        if val_l1 < best_val_loss:
            best_val_loss = val_l1
            best_path = os.path.join(args.output, "gen_best.pth")
            torch.save(netG.state_dict(), best_path)
            print(f"  New best val_L1={val_l1:.4f}  -> {best_path}")

    print(f"\nTraining complete. Best val L1: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.output, 'gen_best.pth')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP-conditioned FaceGAN")
    parser.add_argument("--dataset",    default="D:/EchoFace/datasets/faces_clip.hdf5")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.0002)
    parser.add_argument("--l1_coef",    type=float, default=50.0,
                        help="L1 reconstruction loss weight")
    parser.add_argument("--save_every", type=int,   default=10)
    parser.add_argument("--output",     default="D:/EchoFace/checkpoints/face_gan")
    parser.add_argument("--gen_ckpt",   default=None, help="Resume from generator checkpoint")
    parser.add_argument("--disc_ckpt",  default=None, help="Resume from discriminator checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
