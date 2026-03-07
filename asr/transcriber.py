import gc
import logging
import numpy as np
import torch
import whisper


class Transcriber:
    def __init__(self, cfg: dict) -> None:
        raise NotImplementedError

    def transcribe(self, audio_array) -> str:
        raise NotImplementedError
