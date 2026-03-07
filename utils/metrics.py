from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths
import gpu_utils


def compute_fid(real_dir, gen_dir='outputs/'):
    raise NotImplementedError


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('real_dir', help='Path to real images directory')
    parser.add_argument('gen_dir', nargs='?', default='outputs/',
                        help='Path to generated images directory (default: outputs/)')
    args = parser.parse_args()
    score = compute_fid(args.real_dir, args.gen_dir)
    print(score)
