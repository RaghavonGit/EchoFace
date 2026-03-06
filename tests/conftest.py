# tests/conftest.py
import os
import sys
import pytest

# Ensure project root is on sys.path so gpu_utils and validate_env are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCAFFOLD_DIRS = [
    "asr",
    "encoder",
    "generator",
    "optimizer",
    "ui",
    "utils",
    "outputs",
    os.path.join("data", "human_faces"),
    os.path.join("StyleGAN-Human", "pretrained_models"),
]

@pytest.fixture
def project_root():
    return PROJECT_ROOT

@pytest.fixture
def scaffold_dirs():
    return SCAFFOLD_DIRS
