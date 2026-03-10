import gc
import logging
import torch

# ── Whisper / PyTorch 1.9.x compatibility ────────────────────────────────────
# openai-whisper >= 20231117 passes weights_only=True to torch.load, but
# PyTorch 1.9.1 does not support that keyword argument.  Strip it silently.
_real_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    kwargs.pop("weights_only", None)
    extra = {"pickle_module": pickle_module} if pickle_module is not None else {}
    return _real_torch_load(f, map_location=map_location, **extra)

torch.load = _patched_torch_load
# ─────────────────────────────────────────────────────────────────────────────

import whisper

logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps Whisper medium for local ASR. Loads and unloads per call for VRAM safety.

    Args:
        cfg: Config dict from gpu_utils.get_device_config(). Uses 'device' key.
    """

    def __init__(self, cfg: dict) -> None:
        # Whisper load_model expects a string, not torch.device
        self.device = str(cfg['device'])

    def transcribe(self, audio_array) -> str:
        """Transcribe a float32 16kHz mono numpy array to text.

        Loads Whisper medium, transcribes, then unloads from VRAM before returning.

        Args:
            audio_array: 1-D float32 numpy array at 16kHz (output of process_audio()).

        Returns:
            Plain text transcription string. Stripped of leading/trailing whitespace.
            Returns as-is (with WARNING) if fewer than 3 words detected.
        """
        if torch.cuda.is_available():
            logger.debug(
                "VRAM before Whisper load: %d bytes",
                torch.cuda.memory_allocated(),
            )

        model = whisper.load_model("medium", device=self.device)
        result = model.transcribe(audio_array)
        text = result["text"].strip()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            logger.debug(
                "VRAM after Whisper unload: %d bytes",
                torch.cuda.memory_allocated(),
            )

        words = text.split()
        if len(words) < 3:
            logger.warning(
                "Short transcription (%d word(s)): %r — returning as-is",
                len(words),
                text,
            )

        return text
