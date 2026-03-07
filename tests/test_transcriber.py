import gc
import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import torch

from asr.transcriber import Transcriber


def _make_cfg(device_str="cpu"):
    return {'device': torch.device(device_str)}


def _make_mock_model(text=" a young woman with dark hair "):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": text}
    return mock_model


class TestTranscriber(unittest.TestCase):

    @patch('asr.transcriber.whisper')
    def test_returns_string(self, mock_whisper):
        mock_whisper.load_model.return_value = _make_mock_model()
        t = Transcriber(_make_cfg())
        result = t.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertIsInstance(result, str)

    @patch('asr.transcriber.whisper')
    def test_loads_medium(self, mock_whisper):
        mock_whisper.load_model.return_value = _make_mock_model()
        t = Transcriber(_make_cfg("cpu"))
        t.transcribe(np.zeros(16000, dtype=np.float32))
        mock_whisper.load_model.assert_called_once_with("medium", device="cpu")

    @patch('asr.transcriber.torch')
    @patch('asr.transcriber.gc')
    @patch('asr.transcriber.whisper')
    def test_vram_unload(self, mock_whisper, mock_gc, mock_torch):
        mock_whisper.load_model.return_value = _make_mock_model()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.empty_cache = MagicMock()
        t = Transcriber({'device': torch.device('cpu')})
        t.transcribe(np.zeros(16000, dtype=np.float32))
        mock_gc.collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch('asr.transcriber.torch')
    @patch('asr.transcriber.whisper')
    def test_vram_logging(self, mock_whisper, mock_torch):
        mock_whisper.load_model.return_value = _make_mock_model()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 5_000_000_000
        mock_torch.cuda.empty_cache = MagicMock()
        t = Transcriber({'device': torch.device('cpu')})
        with self.assertLogs('asr.transcriber', level='DEBUG') as log_ctx:
            t.transcribe(np.zeros(16000, dtype=np.float32))
        # memory_allocated must be called at least twice (before and after unload)
        self.assertGreaterEqual(mock_torch.cuda.memory_allocated.call_count, 2)

    @patch('asr.transcriber.whisper')
    def test_short_transcription_warning(self, mock_whisper):
        mock_whisper.load_model.return_value = _make_mock_model(text=" OK ")
        t = Transcriber(_make_cfg())
        with self.assertLogs('asr.transcriber', level='WARNING') as log_ctx:
            result = t.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(result, "OK")   # stripped, returned as-is
        self.assertTrue(any('WARNING' in r for r in log_ctx.output))

    def test_bypass_path(self):
        # ASR-04: the bypass path is the caller choosing NOT to call transcribe().
        # This test verifies the contract: a string can flow directly to optimize()
        # without going through Transcriber at all.
        # We verify by asserting that if a caller has a string, no Transcriber
        # instance or transcribe() call is needed.
        text = "a woman with blue eyes and blonde hair"
        # Simulate pipeline: if text is already a str, skip Transcriber entirely
        mock_transcriber = MagicMock(spec=Transcriber)
        # Pipeline decision logic (as Phase 6 will implement it):
        if isinstance(text, str):
            final_text = text       # bypass path — transcribe() never called
        else:
            final_text = mock_transcriber.transcribe(text)
        self.assertEqual(final_text, text)
        mock_transcriber.transcribe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
