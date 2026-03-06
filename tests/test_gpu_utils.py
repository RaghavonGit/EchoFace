# tests/test_gpu_utils.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_utils import get_device_config


class TestGetDeviceConfig:
    def test_returns_dict(self):
        cfg = get_device_config()
        assert isinstance(cfg, dict), "get_device_config() must return a plain dict"

    def test_required_keys_present(self):
        cfg = get_device_config()
        required = {"device", "dtype", "use_fp16", "resolution", "truncation_psi"}
        assert required == set(cfg.keys()), f"Keys mismatch. Got: {set(cfg.keys())}"

    def test_device_is_torch_device(self):
        cfg = get_device_config()
        assert isinstance(cfg["device"], torch.device), "device must be torch.device instance"

    def test_dtype_is_torch_dtype(self):
        cfg = get_device_config()
        assert isinstance(cfg["dtype"], torch.dtype), "dtype must be torch.dtype instance"

    def test_use_fp16_is_bool(self):
        cfg = get_device_config()
        assert isinstance(cfg["use_fp16"], bool), "use_fp16 must be bool"

    def test_resolution_is_int(self):
        cfg = get_device_config()
        assert isinstance(cfg["resolution"], int), "resolution must be int"

    def test_truncation_psi_is_float(self):
        cfg = get_device_config()
        assert isinstance(cfg["truncation_psi"], float), "truncation_psi must be float"

    def test_cuda_path_values(self):
        """If CUDA is available, verify GPU-path values."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA — skipping GPU path test")
        cfg = get_device_config()
        assert cfg["device"] == torch.device("cuda")
        assert cfg["dtype"] == torch.float16
        assert cfg["use_fp16"] is True
        assert cfg["resolution"] == 1024
        assert cfg["truncation_psi"] == 1.0

    def test_cpu_path_values(self):
        """If CUDA is NOT available, verify CPU-path values."""
        if torch.cuda.is_available():
            pytest.skip("CUDA present — skipping CPU path test")
        cfg = get_device_config()
        assert cfg["device"] == torch.device("cpu")
        assert cfg["dtype"] == torch.float32
        assert cfg["use_fp16"] is False
        assert cfg["resolution"] == 256
        assert cfg["truncation_psi"] == 0.5

    def test_idempotent(self):
        """Calling twice returns equivalent values."""
        cfg1 = get_device_config()
        cfg2 = get_device_config()
        assert cfg1["device"] == cfg2["device"]
        assert cfg1["dtype"] == cfg2["dtype"]
        assert cfg1["use_fp16"] == cfg2["use_fp16"]
        assert cfg1["resolution"] == cfg2["resolution"]
        assert cfg1["truncation_psi"] == cfg2["truncation_psi"]
