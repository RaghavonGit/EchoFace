"""
Tests for StyleCLIPWrapper.edit() and reset() methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch, PropertyMock


def _make_wrapper_with_state():
    """
    Build a StyleCLIPWrapper-like object with pre-set state, bypassing
    the heavy __init__ (no StyleGAN / CLIP loading required).
    """
    from generator.styleclip_wrapper import StyleCLIPWrapper

    wrapper = object.__new__(StyleCLIPWrapper)

    # Minimal device / model stubs
    wrapper.device = torch.device("cpu")

    # Stub G.synthesis to return a plausible [-1,1] image tensor
    mock_G = MagicMock()
    mock_G.synthesis.return_value = torch.zeros(1, 3, 1024, 1024)
    wrapper.G = mock_G

    # Pre-set edit state (simulates a successful generate() call)
    num_ws = 18
    wrapper._w_avg = torch.zeros(512)
    original_w = torch.randn(1, num_ws, 512)
    wrapper._original_w     = original_w.clone()
    wrapper._current_w      = original_w.clone()
    wrapper._original_prompt = "a young man with dark hair"
    wrapper._original_sim   = 0.3500

    return wrapper


class TestEdit(unittest.TestCase):

    def test_raises_if_no_state(self):
        """edit() raises RuntimeError when called before generate()."""
        from generator.styleclip_wrapper import StyleCLIPWrapper
        wrapper = object.__new__(StyleCLIPWrapper)
        wrapper._current_w = None
        wrapper._original_w = None
        wrapper._original_prompt = None
        wrapper._original_sim = None
        wrapper.device = torch.device("cpu")
        wrapper._w_avg = torch.zeros(512)

        with self.assertRaises(RuntimeError) as ctx:
            list_result = wrapper.edit("change eyes to blue")
        self.assertIn("Generate a face first", str(ctx.exception))

    def test_constructs_combined_prompt(self):
        """edit() builds: 'a photo of a face, {original_prompt}, {edit_instruction}'."""
        wrapper = _make_wrapper_with_state()

        captured_prompts = []

        def _fake_encode_text(prompt):
            captured_prompts.append(prompt)
            return torch.randn(1, 512)

        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))
        fake_arr = np.array(pil_img)

        wrapper._encode_text = _fake_encode_text
        wrapper._synthesise_small = MagicMock(return_value=torch.rand(1, 3, 256, 256))
        wrapper._clip_encode_image = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )

        # Patch Image.fromarray to avoid numpy overhead
        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            wrapper.edit("change eyes to blue", steps=1, lr=0.05)

        self.assertTrue(len(captured_prompts) > 0)
        prompt_used = captured_prompts[0]
        self.assertIn("a photo of a face", prompt_used)
        self.assertIn("a young man with dark hair", prompt_used)
        self.assertIn("change eyes to blue", prompt_used)

    def test_updates_current_w(self):
        """edit() updates _current_w to the optimized latent."""
        wrapper = _make_wrapper_with_state()
        original_current = wrapper._current_w.clone()

        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))

        wrapper._encode_text = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        wrapper._synthesise_small = MagicMock(return_value=torch.rand(1, 3, 256, 256))
        wrapper._clip_encode_image = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )

        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            wrapper.edit("change eyes to blue", steps=2, lr=0.05)

        # _current_w must have been updated (at least shape is preserved)
        self.assertEqual(wrapper._current_w.shape, original_current.shape)

    def test_does_not_overwrite_original_w(self):
        """edit() must never overwrite _original_w."""
        wrapper = _make_wrapper_with_state()
        original_w_before = wrapper._original_w.clone()

        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))

        wrapper._encode_text = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )
        wrapper._synthesise_small = MagicMock(return_value=torch.rand(1, 3, 256, 256))
        wrapper._clip_encode_image = MagicMock(
            return_value=torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        )

        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            wrapper.edit("change hair to red", steps=2, lr=0.05)

        self.assertTrue(
            torch.allclose(wrapper._original_w, original_w_before),
            "_original_w must stay frozen after edit()"
        )


class TestReset(unittest.TestCase):

    def test_raises_if_no_state(self):
        """reset() raises RuntimeError when called before generate()."""
        from generator.styleclip_wrapper import StyleCLIPWrapper
        wrapper = object.__new__(StyleCLIPWrapper)
        wrapper._current_w = None
        wrapper._original_w = None
        wrapper._original_prompt = None
        wrapper._original_sim = None
        wrapper.device = torch.device("cpu")
        wrapper._w_avg = torch.zeros(512)

        with self.assertRaises(RuntimeError) as ctx:
            wrapper.reset()
        self.assertIn("Generate a face first", str(ctx.exception))

    def test_restores_current_w_to_original(self):
        """reset() copies _original_w back into _current_w."""
        wrapper = _make_wrapper_with_state()

        # Simulate drift: mutate _current_w away from original
        wrapper._current_w = torch.randn_like(wrapper._original_w)
        original_w_snapshot = wrapper._original_w.clone()

        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))

        # Stub synthesis
        wrapper._synthesise_small = MagicMock(return_value=torch.rand(1, 3, 256, 256))

        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            wrapper.reset()

        self.assertTrue(
            torch.allclose(wrapper._current_w, original_w_snapshot),
            "_current_w must equal _original_w after reset()"
        )

    def test_returns_original_sim(self):
        """reset() returns the cached _original_sim without recomputing."""
        wrapper = _make_wrapper_with_state()
        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))

        wrapper._synthesise_small = MagicMock(return_value=torch.rand(1, 3, 256, 256))

        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            _, sim = wrapper.reset()

        self.assertAlmostEqual(sim, 0.3500, places=4)

    def test_returns_pil_image(self):
        """reset() returns a PIL Image as first element."""
        wrapper = _make_wrapper_with_state()
        pil_img = Image.new("RGB", (256, 256), (100, 80, 60))

        wrapper._synthesise_small = MagicMock(return_value=torch.rand(1, 3, 256, 256))

        with patch("generator.styleclip_wrapper.Image") as mock_img:
            mock_img.fromarray.return_value = pil_img
            result_pil, _ = wrapper.reset()

        self.assertIsInstance(result_pil, Image.Image)
