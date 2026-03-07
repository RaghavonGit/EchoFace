import sys
import os

# Ensure project root is on sys.path so gpu_utils is importable when this
# script is run directly (python utils/metrics.py ...) as well as via import.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths
import gpu_utils

_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _count_images(directory):
    """Count supported image files in directory via glob — no PIL loading."""
    return sum(
        1 for f in Path(directory).glob('*')
        if f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def compute_fid(real_dir, gen_dir='outputs/'):
    """Compute the Frechet Inception Distance between real and generated images.

    Args:
        real_dir: Path to directory of real reference images.
        gen_dir: Path to directory of generated images (default: 'outputs/').

    Returns:
        float: FID score.

    Raises:
        ValueError: If fewer than 2048 real images found, or gen_dir is empty.
    """
    cfg = gpu_utils.get_device_config()
    device = str(cfg['device'])

    real_img_count = _count_images(real_dir)
    if real_img_count < 2048:
        raise ValueError(
            f"FID requires at least 2048 real images; found {real_img_count} in {real_dir}. "
            "Download the full Kaggle Human Faces Dataset."
        )

    gen_img_count = _count_images(gen_dir)
    if gen_img_count == 0:
        raise ValueError(
            f"No generated images found in {gen_dir}. Run optimization first."
        )

    fid_value = calculate_fid_given_paths(
        [str(real_dir), str(gen_dir)],
        batch_size=50,
        device=device,
        dims=2048,
        num_workers=0,
    )
    return fid_value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('real_dir', help='Path to real images directory')
    parser.add_argument('gen_dir', nargs='?', default='outputs/',
                        help='Path to generated images directory (default: outputs/)')
    args = parser.parse_args()
    score = compute_fid(args.real_dir, args.gen_dir)
    print(score)
