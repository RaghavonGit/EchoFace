import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from asr.audio_processor import process_audio


class TestProcessAudio(unittest.TestCase):

    @patch('asr.audio_processor.librosa')
    def test_resamples_16k(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_librosa.effects.trim.return_value = (np.zeros(16000, dtype=np.float32), np.array([0, 16000]))
        result = process_audio("dummy.wav")
        mock_librosa.load.assert_called_once_with("dummy.wav", sr=16000, mono=True)
        self.assertEqual(result.dtype, np.float32)

    @patch('asr.audio_processor.librosa')
    def test_trim_silence(self, mock_librosa):
        audio = np.ones(16000, dtype=np.float32) * 0.5
        mock_librosa.load.return_value = (audio, 16000)
        mock_librosa.effects.trim.return_value = (audio[:8000], np.array([0, 8000]))
        process_audio("dummy.wav")
        mock_librosa.effects.trim.assert_called_once()
        call_kwargs = mock_librosa.effects.trim.call_args
        # top_db=20 must be passed
        self.assertIn(20, call_kwargs.args or list(call_kwargs.kwargs.values()))

    @patch('asr.audio_processor.librosa')
    def test_peak_normalize(self, mock_librosa):
        audio = np.array([0.0, 0.3, 0.6, -0.9, 0.3], dtype=np.float32)
        mock_librosa.load.return_value = (audio, 16000)
        mock_librosa.effects.trim.return_value = (audio, np.array([0, len(audio)]))
        result = process_audio("dummy.wav")
        self.assertLessEqual(np.max(np.abs(result)), 1.0 + 1e-6)
        self.assertEqual(result.dtype, np.float32)

    @patch('asr.audio_processor.librosa')
    def test_near_silent_no_nan(self, mock_librosa):
        # near-silent audio: max abs is ~1e-8, guard must prevent NaN
        audio = np.full(16000, 1e-9, dtype=np.float32)
        mock_librosa.load.return_value = (audio, 16000)
        mock_librosa.effects.trim.return_value = (audio, np.array([0, len(audio)]))
        result = process_audio("dummy.wav")
        self.assertFalse(np.any(np.isnan(result)), "Output contains NaN")
        self.assertFalse(np.any(np.isinf(result)), "Output contains Inf")

    @patch('asr.audio_processor.librosa')
    def test_invalid_file(self, mock_librosa):
        mock_librosa.load.side_effect = Exception("audioread.NoBackendError")
        with self.assertRaises(ValueError) as ctx:
            process_audio("bad_file.xyz")
        self.assertIn("Cannot load audio", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
