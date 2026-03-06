# validate_env.py
# EchoFace Phase 1 environment validation script.
# Run from D:/EchoFace/ with: python validate_env.py
# Must exit 0 (ALL CHECKS PASSED) before Phase 2 begins.

import os
import sys

# Ensure project root is importable (handles running from subdirectories)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


def check(label: str, fn):
    """Run fn(), print PASS or FAIL, return True/False."""
    try:
        result = fn()
        suffix = f": {result}" if result is not None and result is not True else ""
        print(f"  [PASS] {label}{suffix}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main():
    print("=== EchoFace Environment Validation ===\n")
    failures = 0

    # --- Core PyTorch ---
    print("-- PyTorch --")
    failures += 0 if check("torch importable", lambda: __import__("torch").__version__) else 1
    failures += 0 if check("torch.cuda.is_available", lambda: __import__("torch").cuda.is_available()) else 1

    # --- Extra Dependencies ---
    print("\n-- Extra Dependencies (requirements_extra.txt) --")
    failures += 0 if check("gradio importable", lambda: __import__("gradio").__version__) else 1
    failures += 0 if check("whisper importable", lambda: __import__("whisper") and "ok") else 1
    failures += 0 if check("clip importable", lambda: __import__("clip") and "ok") else 1
    failures += 0 if check("librosa importable", lambda: __import__("librosa").__version__) else 1
    failures += 0 if check("soundfile importable", lambda: __import__("soundfile").__version__) else 1
    failures += 0 if check("pytorch_fid importable", lambda: __import__("pytorch_fid") and "ok") else 1

    # --- Version Pins ---
    print("\n-- Version Pins --")

    def check_gradio_version():
        import gradio
        v = gradio.__version__
        parts = tuple(int(x) for x in v.split(".")[:2])
        if parts < (3, 40):
            raise RuntimeError(f"gradio {v} < 3.40 floor")
        if parts >= (4, 0):
            raise RuntimeError(f"gradio {v} >= 4.0 — incompatible with Python 3.8")
        return v

    def check_numpy_version():
        import numpy as np
        v = np.__version__
        parts = tuple(int(x) for x in v.split(".")[:2])
        if parts >= (1, 24):
            raise RuntimeError(f"numpy {v} >= 1.24 — StyleGAN-Human internals use np.bool/np.int")
        return v

    def check_librosa_version():
        import librosa
        v = librosa.__version__
        if v != "0.9.2":
            raise RuntimeError(f"librosa {v} != 0.9.2")
        return v

    failures += 0 if check("gradio version >=3.40,<4.0", check_gradio_version) else 1
    failures += 0 if check("numpy version <1.24", check_numpy_version) else 1
    failures += 0 if check("librosa version ==0.9.2", check_librosa_version) else 1

    # --- CUDA Detection ---
    print("\n-- CUDA / Device --")
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram_gb:.1f} GB")
        if vram_gb < 6.0:
            print(f"  [WARN] VRAM < 6 GB — minimum recommended is 6 GB; 8 GB for comfortable operation")

    # --- gpu_utils Smoke Test ---
    print("\n-- gpu_utils --")
    try:
        import gpu_utils
        cfg = gpu_utils.get_device_config()
        required_keys = {"device", "dtype", "use_fp16", "resolution", "truncation_psi"}
        missing_keys = required_keys - set(cfg.keys())
        if missing_keys:
            raise RuntimeError(f"get_device_config() missing keys: {missing_keys}")
        print(f"  [PASS] gpu_utils.get_device_config() returned:")
        for k, v in cfg.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  [FAIL] gpu_utils: {e}")
        failures += 1

    # --- Repo Scaffold ---
    print("\n-- Repo Scaffold --")
    for rel_dir in SCAFFOLD_DIRS:
        full_dir = os.path.join(_PROJECT_ROOT, rel_dir)
        gitkeep = os.path.join(full_dir, ".gitkeep")
        if os.path.isdir(full_dir) and os.path.exists(gitkeep):
            print(f"  [PASS] {rel_dir}/.gitkeep")
        elif os.path.isdir(full_dir):
            print(f"  [FAIL] {rel_dir}/ exists but .gitkeep missing")
            failures += 1
        else:
            print(f"  [FAIL] {rel_dir}/ directory missing")
            failures += 1

    # --- Final Result ---
    print(f"\n{'=' * 42}")
    if failures == 0:
        print("RESULT: ALL CHECKS PASSED — environment is ready")
    else:
        print(f"RESULT: {failures} CHECK(S) FAILED — resolve before proceeding to Phase 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
