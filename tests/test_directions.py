"""
Tests for StyleCLIPWrapper.apply_direction() and _load_directions().
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock, patch


def _make_wrapper_with_directions(directions_npz_path: str):
    from generator.styleclip_wrapper import StyleCLIPWrapper
    wrapper = object.__new__(StyleCLIPWrapper)
    wrapper.device = torch.device("cpu")

    mock_G = MagicMock()
    mock_G.synthesis.return_value = torch.zeros(1, 3, 1024, 1024)
    wrapper.G = mock_G

    num_ws = 18
    wrapper._w_avg           = torch.zeros(512)
    original_w               = torch.randn(1, num_ws, 512)
    wrapper._original_w      = original_w.clone()
    wrapper._current_w       = original_w.clone()
    wrapper._original_prompt = "a test face"
    wrapper._original_sim    = 0.30
    wrapper._directions      = None
    wrapper._directions_path = directions_npz_path
    return wrapper


class TestApplyDirection(unittest.TestCase):

    def setUp(self):
        self.tmpdir  = tempfile.mkdtemp()
        self.npz_path = os.path.join(self.tmpdir, "directions.npz")
        fake_dir = np.random.randn(18, 512).astype(np.float32)
        np.savez(self.npz_path, glasses=fake_dir, smile=fake_dir)

    def _make(self):
        return _make_wrapper_with_directions(self.npz_path)

    def test_raises_if_no_state(self):
        """apply_direction() raises RuntimeError when called before generate()."""
        from generator.styleclip_wrapper import StyleCLIPWrapper
        wrapper = object.__new__(StyleCLIPWrapper)
        wrapper._current_w       = None
        wrapper._directions      = None
        wrapper._directions_path = self.npz_path
        wrapper.device           = torch.device("cpu")
        with self.assertRaises(RuntimeError) as ctx:
            wrapper.apply_direction("glasses", 2.0)
        self.assertIn("Generate a face first", str(ctx.exception))

    def test_raises_if_npz_missing(self):
        """apply_direction() raises FileNotFoundError when npz is absent."""
        wrapper = _make_wrapper_with_directions("/nonexistent/path/directions.npz")
        with self.assertRaises(FileNotFoundError):
            wrapper.apply_direction("glasses", 2.0)

    def test_raises_if_attr_not_found(self):
        """apply_direction() raises KeyError for unknown attribute."""
        wrapper = self._make()
        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))
        wrapper._encode_text       = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        wrapper._clip_encode_image = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        with self.assertRaises(KeyError):
            with patch("generator.styleclip_wrapper.Image") as mock_img:
                mock_img.fromarray.return_value = pil_img
                wrapper.apply_direction("nonexistent_attr", 2.0)

    def test_updates_current_w(self):
        """apply_direction() updates _current_w but leaves _original_w frozen."""
        wrapper           = self._make()
        original_w_before = wrapper._current_w.clone()
        original_w_frozen = wrapper._original_w.clone()
        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))
        wrapper._encode_text       = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        wrapper._clip_encode_image = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            wrapper.apply_direction("glasses", 2.0)
        self.assertFalse(
            torch.allclose(wrapper._current_w, original_w_before),
            "_current_w must differ after apply_direction()"
        )
        self.assertTrue(
            torch.allclose(wrapper._original_w, original_w_frozen),
            "_original_w must stay frozen"
        )

    def test_directions_cached(self):
        """_load_directions() returns the same dict on second call."""
        wrapper = self._make()
        d1 = wrapper._load_directions()
        d2 = wrapper._load_directions()
        self.assertIs(d1, d2)

    def test_returns_pil_image_and_float(self):
        """apply_direction() returns (PIL.Image, float)."""
        wrapper = self._make()
        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))
        wrapper._encode_text       = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        wrapper._clip_encode_image = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            result_pil, sim = wrapper.apply_direction("glasses", 2.0)
        self.assertIsInstance(result_pil, Image.Image)
        self.assertIsInstance(sim, float)

    def test_negative_strength_opposite_direction(self):
        """Positive and negative strength produce equal and opposite _current_w deltas."""
        start_w = torch.randn(1, 18, 512)
        wp = self._make(); wp._current_w = start_w.clone()
        wn = _make_wrapper_with_directions(self.npz_path); wn._current_w = start_w.clone()
        wp._load_directions(); wn._directions = wp._directions
        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))
        for w in [wp, wn]:
            w._encode_text       = MagicMock(
                return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
            )
            w._clip_encode_image = MagicMock(
                return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
            )
        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            wp.apply_direction("glasses", +2.0)
            wn.apply_direction("glasses", -2.0)
        delta_pos = wp._current_w - start_w
        delta_neg = wn._current_w - start_w
        self.assertTrue(
            torch.allclose(delta_pos, -delta_neg, atol=1e-5),
            "Positive and negative strength must be equal and opposite"
        )


if __name__ == "__main__":
    unittest.main()
