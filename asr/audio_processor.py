import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)


def process_audio(path: str) -> np.ndarray:
    """Load, resample, trim, and normalize audio to 16kHz mono float32.

    Args:
        path: Path to audio file (any librosa-supported format: WAV, MP3, OGG, FLAC).

    Returns:
        1-D float32 numpy array at 16kHz, amplitude in [-1, 1].

    Raises:
        ValueError: If file cannot be loaded (unsupported format, corrupt file, missing file).
    """
    try:
        y, _ = librosa.load(path, sr=16000, mono=True)
    except Exception as e:
        raise ValueError(f"Cannot load audio: {e}") from e

    y, _ = librosa.effects.trim(y, top_db=20)

    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val

    return y
