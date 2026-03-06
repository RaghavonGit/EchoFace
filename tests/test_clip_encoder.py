import torch
import torch.nn.functional as F
from PIL import Image
from unittest.mock import patch, MagicMock
import unittest

from encoder.clip_encoder import CLIPEncoder


def _make_mock_clip():
    """Return (mock_model, mock_preprocess) mimicking clip.load() output."""
    mock_model = MagicMock()

    # encode_text returns a float32 [1, 512] tensor; normalize will give norm 1.0
    raw_text_feat = F.normalize(torch.ones(1, 512, dtype=torch.float32), dim=-1)
    mock_model.encode_text.return_value = raw_text_feat

    # encode_image returns a float32 [1, 512] tensor (normalized)
    raw_img_feat = F.normalize(torch.ones(1, 512, dtype=torch.float32), dim=-1)
    mock_model.encode_image.return_value = raw_img_feat

    # parameters() returns an iterator of 1 frozen parameter
    p1 = torch.zeros(4, 4, requires_grad=False)
    mock_model.parameters.return_value = iter([p1])

    # eval() returns the model itself
    mock_model.eval.return_value = mock_model

    # mock_preprocess is callable, returns [3, 224, 224] tensor
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))

    return mock_model, mock_preprocess


def _make_cfg():
    return {
        'device': torch.device('cpu'),
        'dtype': torch.float32,
        'use_fp16': False,
        'resolution': 256,
        'truncation_psi': 0.5,
    }


class TestCLIPEncoder(unittest.TestCase):

    def test_encode_text_shape(self):
        mock_model, mock_preprocess = _make_mock_clip()
        with patch('encoder.clip_encoder.clip') as mock_clip_module:
            mock_clip_module.load.return_value = (mock_model, mock_preprocess)
            mock_clip_module.tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)
            encoder = CLIPEncoder(_make_cfg())
            out = encoder.encode_text("a young woman with red hair")
        assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"

    def test_encode_text_normalized(self):
        mock_model, mock_preprocess = _make_mock_clip()
        with patch('encoder.clip_encoder.clip') as mock_clip_module:
            mock_clip_module.load.return_value = (mock_model, mock_preprocess)
            mock_clip_module.tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)
            encoder = CLIPEncoder(_make_cfg())
            out = encoder.encode_text("a young woman with red hair")
        norm = torch.linalg.norm(out.float(), dim=-1)
        assert abs(norm.item() - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm.item()}"

    def test_encode_image_shape(self):
        mock_model, mock_preprocess = _make_mock_clip()
        with patch('encoder.clip_encoder.clip') as mock_clip_module:
            mock_clip_module.load.return_value = (mock_model, mock_preprocess)
            mock_clip_module.tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)
            encoder = CLIPEncoder(_make_cfg())
            pil_img = Image.new("RGB", (256, 512))
            out = encoder.encode_image(pil_img)
        assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"

    def test_encode_image_normalized(self):
        mock_model, mock_preprocess = _make_mock_clip()
        with patch('encoder.clip_encoder.clip') as mock_clip_module:
            mock_clip_module.load.return_value = (mock_model, mock_preprocess)
            mock_clip_module.tokenize.return_value = torch.zeros(1, 77, dtype=torch.long)
            encoder = CLIPEncoder(_make_cfg())
            pil_img = Image.new("RGB", (256, 512))
            out = encoder.encode_image(pil_img)
        norm = torch.linalg.norm(out.float(), dim=-1)
        assert abs(norm.item() - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm.item()}"


if __name__ == '__main__':
    unittest.main()
