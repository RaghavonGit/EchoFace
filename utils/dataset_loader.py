from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_dataset(path, max_images=None):
    raise NotImplementedError
