from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_dataset(path, max_images=None):
    """Load images from a directory and return as a list of PIL Images.

    Args:
        path: Path to directory containing image files.
        max_images: Optional cap on the number of images returned.

    Returns:
        List[PIL.Image]: Images in sorted filesystem order, all in RGB mode.

    Raises:
        FileNotFoundError: If directory is missing or contains no supported images.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        msg = (
            f"Dataset not found at {path}. Download the Kaggle "
            "'Human Faces Dataset' by kaustubhdhote and place images there."
        )
        print(msg)
        raise FileNotFoundError(msg)

    files = sorted(
        (f for f in p.iterdir()
         if f.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=str,
    )
    if not files:
        msg = (
            f"Dataset not found at {path}. Download the Kaggle "
            "'Human Faces Dataset' by kaustubhdhote and place images there."
        )
        print(msg)
        raise FileNotFoundError(msg)

    images = []
    skip_count = 0
    for f in files:
        if max_images is not None and len(images) >= max_images:
            break
        try:
            img = Image.open(f).convert('RGB').copy()
            images.append(img)
        except Exception:
            skip_count += 1

    if skip_count:
        print(f"Skipped {skip_count} unreadable files")

    return images
