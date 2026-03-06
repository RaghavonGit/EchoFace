import torch
import torch.nn.functional as F
from PIL import Image
from unittest.mock import patch, MagicMock, call
import unittest

from optimizer.clip_optimizer import optimize


def _make_mock_wrapper():
    """Return a MagicMock mimicking StyleGANWrapper."""
    wrapper = MagicMock()

    # sample_w returns [1, 18, 512] (detachable, cloneable)
    wrapper.sample_w.return_value = torch.zeros(1, 18, 512)

    # synthesize(w) returns [H, W, 3] uint8 tensor
    wrapper.synthesize.return_value = torch.zeros(256, 256, 3, dtype=torch.uint8)

    # G.synthesis returns [1, 3, H, W] float32
    wrapper.G.synthesis.return_value = torch.zeros(1, 3, 256, 256, dtype=torch.float32)

    # G.parameters returns frozen params
    p1 = torch.zeros(4, 4, requires_grad=False)
    p2 = torch.zeros(8, requires_grad=False)
    wrapper.G.parameters.return_value = iter([p1, p2])

    return wrapper


def _make_mock_cfg():
    return {
        'device': torch.device('cpu'),
        'dtype': torch.float32,
        'use_fp16': False,
        'resolution': 256,
        'truncation_psi': 0.5,
    }


def _make_mock_encoder_cls(img_feat=None):
    """Return a MagicMock CLIPEncoder class whose instances behave correctly."""
    normalized_text = F.normalize(torch.ones(1, 512, dtype=torch.float32), dim=-1)
    if img_feat is None:
        img_feat = torch.ones(1, 512, dtype=torch.float32)

    encoder_instance = MagicMock()
    encoder_instance.encode_text.return_value = normalized_text
    encoder_instance.model = MagicMock()
    encoder_instance.model.encode_image.return_value = img_feat

    mock_cls = MagicMock(return_value=encoder_instance)
    return mock_cls, encoder_instance


class TestOptimize(unittest.TestCase):

    def test_calls_sample_w(self):
        """optimize() calls wrapper.sample_w() exactly once."""
        mock_cls, _ = _make_mock_encoder_cls()
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        with patch('optimizer.clip_optimizer.CLIPEncoder', mock_cls):
            optimize("a test prompt", wrapper, cfg)
        wrapper.sample_w.assert_called_once()

    def test_runs_150_steps(self):
        """optimize() runs exactly 150 gradient steps when no NaN occurs.

        Strategy: count encode_image calls — it is called once per step inside
        the gradient path (synthesize → PIL → encode_image).
        """
        mock_cls, encoder_instance = _make_mock_encoder_cls()
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        with patch('optimizer.clip_optimizer.CLIPEncoder', mock_cls):
            optimize("a test prompt", wrapper, cfg)
        # encode_image is called once per optimization step
        assert encoder_instance.model.encode_image.call_count == 150, (
            f"Expected 150 encode_image calls, got "
            f"{encoder_instance.model.encode_image.call_count}"
        )

    def test_return_tuple(self):
        """optimize() returns a 3-tuple: (PIL.Image, torch.Tensor, float)."""
        mock_cls, _ = _make_mock_encoder_cls()
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        with patch('optimizer.clip_optimizer.CLIPEncoder', mock_cls):
            result = optimize("a test prompt", wrapper, cfg)
        assert isinstance(result, tuple) and len(result) == 3, (
            f"Expected 3-tuple, got {type(result)} of len {len(result) if hasattr(result, '__len__') else '?'}"
        )
        pil_img, w_tensor, sim_score = result
        assert isinstance(pil_img, Image.Image), f"Expected PIL.Image, got {type(pil_img)}"
        assert isinstance(w_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(w_tensor)}"
        assert isinstance(sim_score, float), f"Expected float, got {type(sim_score)}"

    def test_generator_frozen(self):
        """After optimize() returns, all G parameters have requires_grad=False."""
        mock_cls, _ = _make_mock_encoder_cls()
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()

        # Use real tensors so requires_grad can be checked properly
        p1 = torch.zeros(4, 4, requires_grad=False)
        p2 = torch.zeros(8, requires_grad=False)
        wrapper.G.parameters.return_value = [p1, p2]

        with patch('optimizer.clip_optimizer.CLIPEncoder', mock_cls):
            optimize("a test prompt", wrapper, cfg)

        for param in wrapper.G.parameters():
            assert not param.requires_grad, (
                f"Generator parameter has requires_grad=True after optimize()"
            )

    def test_nan_recovery(self):
        """When loss is NaN at step 1, optimize() stops early, logs WARNING, returns valid 3-tuple."""
        # Make encode_image return NaN to force NaN loss
        nan_feat = torch.tensor([[float('nan')] * 512])
        mock_cls, encoder_instance = _make_mock_encoder_cls(img_feat=nan_feat)
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()

        with self.assertLogs(level='WARNING'):
            with patch('optimizer.clip_optimizer.CLIPEncoder', mock_cls):
                result = optimize("a test prompt", wrapper, cfg)

        assert isinstance(result, tuple) and len(result) == 3, (
            f"Expected 3-tuple on NaN recovery, got {type(result)}"
        )
        pil_img, w_tensor, sim_score = result
        assert isinstance(pil_img, Image.Image), f"Expected PIL.Image, got {type(pil_img)}"
        assert isinstance(w_tensor, torch.Tensor), f"Expected torch.Tensor, got {type(w_tensor)}"
        assert isinstance(sim_score, float), f"Expected float, got {type(sim_score)}"

    def test_callback_frequency(self):
        """callback is called exactly 15 times (steps 10, 20, ..., 150) with keyword args."""
        mock_cls, _ = _make_mock_encoder_cls()
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        callback = MagicMock()

        with patch('optimizer.clip_optimizer.CLIPEncoder', mock_cls):
            optimize("a test prompt", wrapper, cfg, callback=callback)

        assert callback.call_count == 15, (
            f"Expected 15 callback calls (every 10 steps over 150), "
            f"got {callback.call_count}"
        )
        # Verify keyword args step, loss, sim_score are present in each call
        for i, c in enumerate(callback.call_args_list):
            kwargs = c.kwargs
            assert 'step' in kwargs, f"Call {i}: missing 'step' kwarg"
            assert 'loss' in kwargs, f"Call {i}: missing 'loss' kwarg"
            assert 'sim_score' in kwargs, f"Call {i}: missing 'sim_score' kwarg"


if __name__ == '__main__':
    unittest.main()
