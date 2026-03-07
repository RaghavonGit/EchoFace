import gc
import logging
import torch
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
