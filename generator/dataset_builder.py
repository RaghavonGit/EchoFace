"""
Build HDF5 training dataset from face images using CLIP image embeddings.

Each image is embedded with CLIP's image encoder (ViT-B/32).  At inference
time the generator receives CLIP text embeddings, which live in the same
embedding space — enabling cross-modal face generation without text labels.

Usage:
    conda run -n stylehuman python generator/dataset_builder.py ^
        --images_dir "D:/EchoFace/Speech 2 Face/Speech 2 Face/Real Images" ^
        --output     datasets/faces_clip.hdf5 ^
        --val_split  0.2 ^
        --img_size   64 ^
        --batch_size 64

Output HDF5 layout:
    train/
        faces       (N_train, 3, img_size, img_size)  float32 in [-1, 1]
        embeds      (N_train, 512)                     float32, L2-normalised
        wrong_faces (N_train, 3, img_size, img_size)  float32 (shuffled)
    val/
        (same structure)
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import clip  # openai/CLIP installed locally


# ── Helpers ───────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_paths(*images_dirs: str):
    """Return sorted list of image file paths across one or more directories."""
    paths = []
    for images_dir in images_dirs:
        if not os.path.isdir(images_dir):
            print(f"  Warning: directory not found, skipping: {images_dir}")
            continue
        for fname in os.listdir(images_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                paths.append(os.path.join(images_dir, fname))
    return sorted(paths)


def _load_face_tensor(path: str, img_size: int) -> np.ndarray:
    """Load image, resize to img_size x img_size, normalise to [-1, 1].

    Returns: (3, img_size, img_size) float32 ndarray.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [0,255] → [-1,1]
    return arr.transpose(2, 0, 1)                          # HWC → CHW


def _compute_embeddings(paths, preprocess, model, device, batch_size):
    """Compute CLIP image embeddings for all paths.

    Returns:
        face_arrays: list of (3, img_size, img_size) float32 ndarrays
        embeds:      (N, 512) float32 ndarray, L2-normalised
    """
    img_size = None  # derived from first successful load
    face_arrays = []
    all_embeds = []
    skipped = 0

    for batch_start in tqdm(range(0, len(paths), batch_size), desc="  Embedding"):
        batch_paths = paths[batch_start: batch_start + batch_size]
        clip_inputs = []
        batch_faces = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                clip_tensor = preprocess(img)         # CLIP's own preprocess
                clip_inputs.append(clip_tensor)

                # Derive img_size from first CLIP preprocess result
                if img_size is None:
                    # CLIP preprocesses to 224x224; we store at separate size
                    # img_size is set by caller — peek via face tensor
                    pass

                batch_faces.append(path)              # store path, load later
            except Exception as exc:
                print(f"\n  Skip {os.path.basename(path)}: {exc}")
                skipped += 1
                continue

        if not clip_inputs:
            continue

        clip_batch = torch.stack(clip_inputs).to(device)
        with torch.no_grad():
            embeds = model.encode_image(clip_batch)
            embeds = F.normalize(embeds.float(), dim=-1)

        all_embeds.extend(embeds.cpu().numpy())
        face_arrays.extend(batch_faces)   # still paths at this point

    if skipped:
        print(f"\n  Skipped {skipped} corrupt/unreadable images.")

    return face_arrays, np.array(all_embeds, dtype=np.float32)


def _load_face_arrays(paths, img_size):
    """Load and resize all face images to ndarrays."""
    arrays = []
    for path in tqdm(paths, desc="  Loading faces"):
        try:
            arrays.append(_load_face_tensor(path, img_size))
        except Exception as exc:
            print(f"\n  Skip {os.path.basename(path)}: {exc}")
    return arrays


def _write_split(grp, face_arrays, embeds):
    """Write faces, embeds, and wrong_faces into an open HDF5 group."""
    n = len(face_arrays)
    face_np = np.array(face_arrays, dtype=np.float32)   # (N, 3, H, W)

    wrong_indices = list(range(n))
    random.shuffle(wrong_indices)
    wrong_np = face_np[wrong_indices]

    grp.create_dataset("faces",       data=face_np,  compression="gzip", compression_opts=4)
    grp.create_dataset("embeds",      data=embeds,   compression="gzip", compression_opts=4)
    grp.create_dataset("wrong_faces", data=wrong_np, compression="gzip", compression_opts=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading images from: {args.images_dirs}")

    paths = _collect_paths(*args.images_dirs)
    if not paths:
        raise ValueError(f"No images found in: {args.images_dirs}")
    print(f"Found {len(paths)} images")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(paths)

    n_val = max(1, int(len(paths) * args.val_split))
    val_paths   = paths[:n_val]
    train_paths = paths[n_val:]
    print(f"Train: {len(train_paths)} | Val: {n_val}")

    # Load CLIP
    print("\nLoading CLIP ViT-B/32 ...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with h5py.File(args.output, "w") as hf:
        for split_name, split_paths in [("train", train_paths), ("val", val_paths)]:
            print(f"\n── {split_name} split ({len(split_paths)} images) ──")

            # Step 1: compute CLIP embeddings (paths → embeds)
            valid_paths, embeds = _compute_embeddings(
                split_paths, preprocess, model, device, args.batch_size
            )

            # Step 2: load & resize face images to storage size
            face_arrays = _load_face_arrays(valid_paths, args.img_size)

            # Align: drop any that failed at step 2
            min_len = min(len(face_arrays), len(embeds))
            face_arrays = face_arrays[:min_len]
            embeds = embeds[:min_len]

            print(f"  Writing {min_len} samples ...")
            grp = hf.create_group(split_name)
            _write_split(grp, face_arrays, embeds)
            print(f"  Done.")

    print(f"\nDataset saved → {args.output}")
    print("Next step:")
    print(f"  conda run -n stylehuman python generator/train_face_gan.py "
          f"--dataset {args.output}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Build face+CLIP HDF5 dataset")
    parser.add_argument(
        "--images_dirs",
        nargs="+",
        default=[
            "D:/EchoFace/Speech 2 Face/Speech 2 Face/Real Images",
            "D:/EchoFace/Speech 2 Face/Speech 2 Face/AI-Generated Images",
        ],
        help="One or more directories containing face JPEG/PNG images",
    )
    parser.add_argument(
        "--output",
        default="D:/EchoFace/datasets/faces_clip.hdf5",
        help="Output HDF5 file path",
    )
    parser.add_argument("--val_split",  type=float, default=0.2)
    parser.add_argument("--img_size",   type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    return parser.parse_args()


if __name__ == "__main__":
    build(_parse_args())
