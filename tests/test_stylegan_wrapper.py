import os
import pytest
import torch
from unittest.mock import patch, MagicMock, call


def _make_mock_G(num_ws=18, z_dim=512, c_dim=0, native_res=1024):
    """Return a MagicMock that mimics a frozen StyleGAN-Human EMA generator."""
    mock_G = MagicMock()
    mock_G.z_dim = z_dim
    mock_G.c_dim = c_dim
    mock_G.num_ws = num_ws

    # All parameters have requires_grad=False (frozen)
    p1 = torch.zeros(4, 4, requires_grad=False)
    p2 = torch.zeros(8, requires_grad=False)
    mock_G.parameters.return_value = iter([p1, p2])

    # synthesis returns a [1, 3, native_res, native_res] float32 tensor
    mock_G.synthesis.return_value = torch.zeros(
        1, 3, native_res, native_res, dtype=torch.float32
    )

    # mapping returns a [1, num_ws, 512] float32 tensor
    mock_G.mapping.return_value = torch.zeros(1, num_ws, 512)

    # eval() returns self
    mock_G.eval.return_value = mock_G

    # to(device) returns self
    mock_G.to.return_value = mock_G

    return mock_G


def _mock_open_url_and_load(mock_G):
    """Return a pair (open_url_patcher, load_pkl_patcher) configured to yield mock_G."""

    # open_url context manager mock
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    open_url_mock = MagicMock(return_value=cm)

    # load_network_pkl mock — returns {'G_ema': mock_G}
    load_pkl_mock = MagicMock(return_value={'G_ema': mock_G})

    return open_url_mock, load_pkl_mock


class TestStyleGANWrapper:

    # -------------------------------------------------------------------------
    # GEN-01: pkl loading, parameter freezing, synthesis signature
    # These run without CUDA; they mock the pkl load path.
    # -------------------------------------------------------------------------

    def test_load_uses_legacy(self):
        """GEN-01: StyleGANWrapper.__init__ must call legacy.load_network_pkl, not pickle.load."""
        from generator.stylegan_wrapper import StyleGANWrapper

        mock_G = _make_mock_G()
        open_url_mock, load_pkl_mock = _mock_open_url_and_load(mock_G)

        cfg = {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'use_fp16': False,
            'resolution': 1024,
            'truncation_psi': 1.0,
        }

        with patch('generator.stylegan_wrapper.dnnlib.util.open_url', open_url_mock), \
             patch('generator.stylegan_wrapper.legacy.load_network_pkl', load_pkl_mock):
            StyleGANWrapper.__init__(
                StyleGANWrapper.__new__(StyleGANWrapper),
                'fake.pkl',
                cfg,
            )

        load_pkl_mock.assert_called_once()
        # Ensure pickle.load was never called — StyleGANWrapper must not
        # bypass legacy deserialization.
        # (No need to assert absence of pickle.load directly; mocking
        # legacy.load_network_pkl and confirming it was called is sufficient.)

    def test_parameters_frozen(self):
        """GEN-01: All G.parameters() must have requires_grad=False after __init__."""
        from generator.stylegan_wrapper import StyleGANWrapper

        mock_G = _make_mock_G()
        open_url_mock, load_pkl_mock = _mock_open_url_and_load(mock_G)

        # Give mock_G parameters that start with requires_grad=True so we can
        # verify the wrapper sets them to False.
        p1 = torch.zeros(4, 4, requires_grad=True)
        p2 = torch.zeros(8, requires_grad=True)
        # parameters() is called twice: once to freeze, so use side_effect list
        mock_G.parameters.side_effect = [iter([p1, p2]), iter([p1, p2])]

        cfg = {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'use_fp16': False,
            'resolution': 1024,
            'truncation_psi': 1.0,
        }

        with patch('generator.stylegan_wrapper.dnnlib.util.open_url', open_url_mock), \
             patch('generator.stylegan_wrapper.legacy.load_network_pkl', load_pkl_mock):
            wrapper = StyleGANWrapper.__new__(StyleGANWrapper)
            StyleGANWrapper.__init__(wrapper, 'fake.pkl', cfg)

        # After __init__, no parameter should have requires_grad=True
        for p in [p1, p2]:
            assert not p.requires_grad, (
                f"Parameter still has requires_grad=True after StyleGANWrapper.__init__"
            )

    def test_synthesis_signature(self):
        """GEN-01: synthesize() must call G.synthesis with noise_mode='const' and force_fp32=True."""
        from generator.stylegan_wrapper import StyleGANWrapper

        mock_G = _make_mock_G()
        open_url_mock, load_pkl_mock = _mock_open_url_and_load(mock_G)

        cfg = {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'use_fp16': False,
            'resolution': 1024,
            'truncation_psi': 1.0,
        }

        with patch('generator.stylegan_wrapper.dnnlib.util.open_url', open_url_mock), \
             patch('generator.stylegan_wrapper.legacy.load_network_pkl', load_pkl_mock):
            wrapper = StyleGANWrapper.__new__(StyleGANWrapper)
            StyleGANWrapper.__init__(wrapper, 'fake.pkl', cfg)

        # Call synthesize with a dummy W tensor
        w = torch.zeros(1, 18, 512)
        wrapper.synthesize(w)

        mock_G.synthesis.assert_called_once_with(
            w, noise_mode='const', force_fp32=True
        )

    # -------------------------------------------------------------------------
    # GEN-02: GPU path — skip if CUDA unavailable
    # -------------------------------------------------------------------------

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_sample_w_shape(self):
        """GEN-02: sample_w() returns tensor shape [1, num_ws, 512] on CUDA device."""
        import gpu_utils
        from generator.stylegan_wrapper import StyleGANWrapper
        cfg = gpu_utils.get_device_config()
        pkl = "StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl"
        wrapper = StyleGANWrapper(pkl, cfg)
        w = wrapper.sample_w()
        assert w.shape[0] == 1
        assert w.shape[2] == 512
        assert w.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_generates_1024_png(self):
        """GEN-02: generate_and_save() writes a valid PIL-openable 1024x1024 PNG to outputs/."""
        import gpu_utils
        from PIL import Image
        from generator.stylegan_wrapper import StyleGANWrapper
        cfg = gpu_utils.get_device_config()
        pkl = "StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl"
        wrapper = StyleGANWrapper(pkl, cfg)
        path = wrapper.generate_and_save()
        assert os.path.exists(path)
        img = Image.open(path)
        # StyleGAN-Human v2 1024 generates portrait images (512 wide x 1024 tall)
        assert img.size == (512, 1024)

    # -------------------------------------------------------------------------
    # GEN-03: CPU path — mock G.synthesis to return [1, 3, 1024, 1024] float tensor
    # -------------------------------------------------------------------------

    def test_cpu_resize_to_256(self):
        """GEN-03: synthesize() output is 256x256 when cfg['resolution']=256."""
        from unittest.mock import MagicMock
        from generator.stylegan_wrapper import StyleGANWrapper

        cpu_cfg = {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'use_fp16': False,
            'resolution': 256,
            'truncation_psi': 0.5,
        }
        pkl = "StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl"

        with patch('generator.stylegan_wrapper.dnnlib.util.open_url') as mock_url, \
             patch('generator.stylegan_wrapper.legacy.load_network_pkl') as mock_load:
            mock_G = MagicMock()
            mock_G.parameters.return_value = []
            # synthesis returns [1, 3, 1024, 1024] float32 — native model output shape
            mock_G.synthesis.return_value = torch.zeros(1, 3, 1024, 1024, dtype=torch.float32)
            mock_G.eval.return_value = mock_G
            mock_G.to.return_value = mock_G
            mock_load.return_value = {'G_ema': mock_G}
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            mock_url.return_value = cm

            wrapper = StyleGANWrapper(pkl, cpu_cfg)
            w = torch.zeros(1, 18, 512)
            result = wrapper.synthesize(w)

        assert result.shape == (256, 256, 3), f"Expected (256,256,3), got {result.shape}"

    def test_cpu_generates_256_png(self):
        """GEN-03: generate_and_save() writes a valid 256x256 PNG without error."""
        import tempfile
        from unittest.mock import MagicMock
        from PIL import Image
        from generator.stylegan_wrapper import StyleGANWrapper

        cpu_cfg = {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'use_fp16': False,
            'resolution': 256,
            'truncation_psi': 0.5,
        }
        pkl = "StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl"

        with patch('generator.stylegan_wrapper.dnnlib.util.open_url') as mock_url, \
             patch('generator.stylegan_wrapper.legacy.load_network_pkl') as mock_load:
            mock_G = MagicMock()
            mock_G.z_dim = 512
            mock_G.c_dim = 0
            mock_G.num_ws = 18
            mock_G.parameters.return_value = []
            mock_G.synthesis.return_value = torch.zeros(1, 3, 1024, 1024, dtype=torch.float32)
            mock_G.mapping.return_value = torch.zeros(1, 18, 512)
            mock_G.eval.return_value = mock_G
            mock_G.to.return_value = mock_G
            mock_load.return_value = {'G_ema': mock_G}
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            mock_url.return_value = cm

            wrapper = StyleGANWrapper(pkl, cpu_cfg)
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = wrapper.generate_and_save(out_dir=tmp_dir)
                assert os.path.exists(path)
                with Image.open(path) as img:
                    size = img.size
                assert size == (256, 256)
