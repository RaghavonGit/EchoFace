# tests/test_env.py
# Covers ENV-01 (validation script), ENV-02 (scaffold), ENV-03 (requirements / pip check)
import importlib
import os
import subprocess
import sys
import pytest

# Ensure project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCAFFOLD_DIRS = [
    "asr",
    "encoder",
    "generator",
    "ui",
    "outputs",
    os.path.join("StyleGAN-Human", "pretrained_models"),
]


class TestImports:
    """ENV-03: All extra dependencies importable after requirements_extra.txt install."""

    def test_gradio_importable(self):
        gradio = importlib.import_module("gradio")
        assert gradio is not None

    def test_whisper_importable(self):
        whisper = importlib.import_module("whisper")
        assert whisper is not None

    def test_clip_importable(self):
        clip = importlib.import_module("clip")
        assert clip is not None

    def test_librosa_importable(self):
        librosa = importlib.import_module("librosa")
        assert librosa is not None

    def test_soundfile_importable(self):
        sf = importlib.import_module("soundfile")
        assert sf is not None

    def test_numpy_importable(self):
        np = importlib.import_module("numpy")
        assert np is not None


class TestVersionPins:
    """ENV-03: Critical version pins are enforced."""

    def test_gradio_version_is_3x(self):
        import gradio
        version_parts = tuple(int(x) for x in gradio.__version__.split(".")[:2])
        assert version_parts >= (3, 40), (
            f"gradio {gradio.__version__} is below 3.40 floor"
        )
        assert version_parts < (4, 0), (
            f"gradio {gradio.__version__} is 4.x or higher — incompatible with Python 3.8"
        )

    def test_numpy_version_below_1_24(self):
        import numpy as np
        version_parts = tuple(int(x) for x in np.__version__.split(".")[:2])
        assert version_parts < (1, 24), (
            f"numpy {np.__version__} >= 1.24 — will break StyleGAN-Human internals "
            "(np.bool/np.int aliases removed in 1.24)"
        )

    def test_librosa_version_is_0_9_2(self):
        import librosa
        assert librosa.__version__ == "0.9.2", (
            f"librosa {librosa.__version__} != 0.9.2 — other versions may conflict with scipy 1.7.1"
        )


class TestPipCheck:
    """ENV-03: pip check reports no broken requirements."""

    def test_no_pip_conflicts(self):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"pip check failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )


class TestScaffold:
    """ENV-02: All scaffold directories and .gitkeep files exist."""

    @pytest.mark.parametrize("rel_dir", SCAFFOLD_DIRS)
    def test_directory_exists(self, rel_dir):
        full_path = os.path.join(PROJECT_ROOT, rel_dir)
        assert os.path.isdir(full_path), f"Scaffold directory missing: {full_path}"

    @pytest.mark.parametrize("rel_dir", SCAFFOLD_DIRS)
    def test_gitkeep_exists(self, rel_dir):
        gitkeep = os.path.join(PROJECT_ROOT, rel_dir, ".gitkeep")
        assert os.path.exists(gitkeep), f".gitkeep missing in: {os.path.join(PROJECT_ROOT, rel_dir)}"


