# Phase 4: ASR and Audio Preprocessing - Research

**Researched:** 2026-03-07
**Domain:** Audio preprocessing (librosa 0.9.2) + local Whisper ASR (openai-whisper 20250625) + VRAM lifecycle management
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Noise filtering library:** librosa only — no noisereduce or additional deps
- **Silence trimming:** `librosa.effects.trim()` with default `top_db=20`
- **Amplitude normalization:** peak normalize to [-1, 1] (divide by max absolute value)
- **Resampling:** to 16kHz mono float32
- **Input formats:** any format librosa supports (WAV, MP3, OGG, FLAC, etc.)
- **Short transcription handling:** < 3 words → return text as-is, log WARNING — do not block pipeline
- **Corrupt audio handling:** raise ValueError with descriptive message
- **Transcriber return type:** plain text string only — no Whisper result dict, no segments, no language code
- **VRAM load strategy:** load_model() at start of each transcribe call, unload after — no singleton, no lazy caching
- **Unload sequence:** `del model` → `gc.collect()` → `torch.cuda.empty_cache()`
- **VRAM logging:** `torch.cuda.memory_allocated()` before and after unload at DEBUG level
- **Model size:** "medium" hardcoded — not configurable
- **Module pattern:** `Transcriber(cfg)` class with `transcribe(audio_array)` method
- **AudioProcessor pattern:** standalone function `process_audio(path) -> np.ndarray` (no persistent state)

### Claude's Discretion
- Exact logging format and logger name
- `gc` import and call placement within unload sequence
- Internal helper structure within `audio_processor.py`

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ASR-01 | `audio_processor.py` applies noise filtering, amplitude normalization, and silence trimming to input audio using librosa + soundfile, outputting 16kHz mono float32 array | `librosa.load(path, sr=16000, mono=True, dtype=np.float32)` handles resampling and mono conversion in one call; `librosa.effects.trim()` handles silence trimming; peak normalization is a simple divide-by-max |
| ASR-02 | `transcriber.py` loads Whisper medium model locally (no API key), transcribes preprocessed audio to text, and implements lazy load + unload to free VRAM before optimization begins | `whisper.load_model("medium")` loads locally from `~/.cache/whisper/`; `model.transcribe(audio_array)` accepts `np.ndarray` directly; unload via `del model` + `gc.collect()` + `torch.cuda.empty_cache()` |
| ASR-03 | User can provide audio via real-time microphone recording or .wav/.mp3 file upload; both paths are resampled to 16kHz mono before transcription | `librosa.load()` handles both file types transparently; mic recording path will be wired in Phase 6 UI — Phase 4 only needs `process_audio(path)` that accepts any file path |
| ASR-04 | User can type a text description directly to bypass ASR entirely when audio quality is insufficient | Bypass path is a pure Python concern — `transcribe()` is simply not called; the text string is passed directly to `optimize()`; test verifies calling contract, not UI |
</phase_requirements>

---

## Summary

Phase 4 implements a two-file ASR pipeline: `asr/audio_processor.py` (stateless function) and `asr/transcriber.py` (class with cfg dict). Both libraries are already installed and verified in the `stylehuman` conda env: librosa 0.9.2 and openai-whisper 20250625.

The audio preprocessing path is straightforward: `librosa.load(path, sr=16000, mono=True)` handles format detection, channel downmix, and resampling in a single call. float32 is the default dtype in librosa 0.9.2, so no dtype argument is needed (though explicit is fine). `librosa.effects.trim()` removes leading/trailing silence. Peak normalization is `y / np.max(np.abs(y))` with a near-zero guard. The output is a 1-D float32 numpy array at 16kHz.

The Whisper transcription path is: load model → transcribe numpy array → extract `result["text"]` → unload. Whisper's `model.transcribe()` accepts `Union[str, np.ndarray, torch.Tensor]` directly — no intermediate file or FFmpeg needed when the array is already float32 16kHz mono. VRAM management is the critical concern: Whisper medium requires ~5 GB VRAM; the StyleGAN optimizer also needs VRAM, so they cannot coexist. The decided unload sequence (`del model` + `gc.collect()` + `torch.cuda.empty_cache()`) is the standard PyTorch pattern for this.

**Primary recommendation:** Implement `process_audio(path) -> np.ndarray` as a module-level function in `audio_processor.py` and `Transcriber(cfg)` class with `transcribe(audio_array: np.ndarray) -> str` in `transcriber.py`. Load and unload Whisper inside every `transcribe()` call. Test both modules with `unittest.mock.patch` targeting `asr.audio_processor.librosa` and `asr.transcriber.whisper` namespace bindings.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| librosa | 0.9.2 (installed) | Audio loading, resampling, silence trimming | Project decision; handles all common audio formats via soundfile/audioread backends |
| openai-whisper | 20250625 (installed) | Local ASR — float array → text | Project decision; fully local, no API key, supports numpy array input directly |
| soundfile | 0.13.1 (installed) | Backend for librosa on WAV/FLAC/OGG | Auto-used by librosa; no direct calls needed in Phase 4 |
| torch | 1.9.1 (installed) | VRAM management after Whisper unload | Already in env; `torch.cuda.empty_cache()` and `torch.cuda.memory_allocated()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| gc (stdlib) | built-in | Force CPython reference collection | Between `del model` and `torch.cuda.empty_cache()` in unload sequence |
| logging (stdlib) | built-in | DEBUG VRAM logging, WARNING for short transcriptions | Module-level logger pattern (matches optimizer) |
| numpy | 1.23.5 (pinned) | Array dtype, peak normalization arithmetic | Already in env; `np.max`, `np.abs` for normalization |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| librosa only | noisereduce | noisereduce is more aggressive but adds a dep; locked out by user decision |
| load-per-call | singleton Whisper instance | Singleton saves 2-4s per call but risks OOM when optimizer runs; locked out by user decision |

**Installation:** No new packages needed — all dependencies are already installed in `stylehuman` env.

---

## Architecture Patterns

### Recommended Project Structure
```
asr/
├── __init__.py          # empty package marker (follows encoder/ pattern)
├── audio_processor.py   # process_audio(path) -> np.ndarray
└── transcriber.py       # Transcriber(cfg) class
tests/
├── test_audio_processor.py  # new — ASR-01
└── test_transcriber.py      # new — ASR-02, ASR-03, ASR-04
```

### Pattern 1: Audio Processor as Standalone Function
**What:** `process_audio(path)` — no class, no state, pure transformation
**When to use:** Module has no persistent state (no model to hold)
**Example:**
```python
# Mirrors optimizer/clip_optimizer.py standalone function pattern
import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)

def process_audio(path: str) -> np.ndarray:
    """Load, resample, trim, and normalize audio to 16kHz mono float32.

    Args:
        path: Path to audio file (any librosa-supported format).

    Returns:
        1-D float32 numpy array at 16kHz, amplitude range [-1, 1].

    Raises:
        ValueError: If file cannot be loaded (unsupported format, corrupt file).
    """
    try:
        y, _ = librosa.load(path, sr=16000, mono=True)  # dtype=np.float32 by default
    except Exception as e:
        raise ValueError(f"Cannot load audio: {e}") from e

    y, _ = librosa.effects.trim(y, top_db=20)

    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val

    return y
```

### Pattern 2: Transcriber as Class with cfg Dict
**What:** `Transcriber(cfg)` with `transcribe(audio_array)` method — follows CLIPEncoder pattern
**When to use:** Module needs device selection from cfg dict
**Example:**
```python
# Mirrors encoder/clip_encoder.py class pattern
import gc
import logging
import whisper
import torch

logger = logging.getLogger(__name__)

class Transcriber:
    """Wraps Whisper medium for local ASR. Loads and unloads per call.

    Args:
        cfg: Config dict from gpu_utils.get_device_config(). Uses 'device' key.
    """

    def __init__(self, cfg: dict) -> None:
        self.device = str(cfg['device'])  # whisper accepts "cuda" / "cpu" strings

    def transcribe(self, audio_array) -> str:
        """Transcribe float32 16kHz mono numpy array to text.

        Loads Whisper medium, transcribes, then unloads from VRAM.

        Args:
            audio_array: 1-D float32 numpy array at 16kHz.

        Returns:
            Plain text transcription string.
        """
        if torch.cuda.is_available():
            logger.debug("VRAM before Whisper load: %d bytes", torch.cuda.memory_allocated())

        model = whisper.load_model("medium", device=self.device)
        result = model.transcribe(audio_array)
        text = result["text"].strip()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            logger.debug("VRAM after Whisper unload: %d bytes", torch.cuda.memory_allocated())

        words = text.split()
        if len(words) < 3:
            logger.warning("Short transcription (%d words): %r — returning as-is", len(words), text)

        return text
```

### Pattern 3: Test Mocking — Namespace Binding Patch
**What:** Patch `asr.audio_processor.librosa` and `asr.transcriber.whisper`, not top-level modules
**When to use:** All unit tests for this phase — avoids touching the real model
**Example:**
```python
# Source: established project pattern from tests/test_stylegan_wrapper.py and test_clip_encoder.py
from unittest.mock import patch, MagicMock
import numpy as np

class TestProcessAudio(unittest.TestCase):

    @patch('asr.audio_processor.librosa')
    def test_resamples_to_16kHz(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_librosa.effects.trim.return_value = (np.zeros(16000, dtype=np.float32), np.array([0, 16000]))
        result = process_audio("dummy.wav")
        mock_librosa.load.assert_called_once_with("dummy.wav", sr=16000, mono=True)
        assert result.dtype == np.float32

class TestTranscriber(unittest.TestCase):

    @patch('asr.transcriber.whisper')
    def test_transcribe_returns_string(self, mock_whisper):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " a young woman with dark hair "}
        mock_whisper.load_model.return_value = mock_model
        t = Transcriber({'device': torch.device('cpu')})
        result = t.transcribe(np.zeros(16000, dtype=np.float32))
        assert isinstance(result, str)
        mock_whisper.load_model.assert_called_once_with("medium", device="cpu")
```

### Anti-Patterns to Avoid
- **Caching the Whisper model as a class attribute:** The model must not be stored on `self` — the user decision locks load-per-call. Caching would defeat the VRAM unload requirement.
- **Calling `wrapper.synthesize()` in audio path:** Audio processor is upstream of the generator — no generator involvement here.
- **Patching top-level `whisper` module:** `@patch('whisper.load_model')` will not intercept the import already bound in `asr.transcriber`. Must patch `asr.transcriber.whisper`.
- **Returning Whisper result dict:** `transcribe()` must return `str`, not the full dict. The downstream caller (Phase 6) expects a plain string.
- **Dividing by zero in peak normalization:** If audio is silent, `max_val` is 0.0. Guard with `if max_val > 1e-6` to avoid NaN/inf array.
- **Forgetting `__init__.py`:** `asr/` must have `__init__.py` or tests cannot import `from asr.audio_processor import process_audio`. Follows `encoder/` and `optimizer/` pattern.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio file format detection | Custom file header parser | `librosa.load()` | librosa delegates to soundfile (WAV/FLAC/OGG) and audioread (MP3 via FFmpeg) — handles codec detection automatically |
| Resampling algorithm | Custom interpolation | `librosa.load(sr=16000)` | Kaiser-best quality resampling built in; resampy handles multi-rate correctly |
| Stereo-to-mono downmix | Average channels manually | `librosa.load(mono=True)` | Proper channel mixing with numerical stability |
| ASR model | Train or fine-tune | `whisper.load_model("medium")` | 769M parameter model, pre-trained on 680K hours of audio |
| GPU memory release | Custom CUDA free() | `del model` + `gc.collect()` + `torch.cuda.empty_cache()` | PyTorch's allocator holds cached blocks; empty_cache() is the only supported release path |

**Key insight:** Both librosa and Whisper have well-defined interfaces for the exact transformations required. The only custom code in this phase is the thin wiring between them.

---

## Common Pitfalls

### Pitfall 1: Whisper numpy array requires float32 and 16kHz — wrong dtype silently gives bad output
**What goes wrong:** Passing int16 or float64 array to `model.transcribe()` either raises `RuntimeError: expected floating point` or silently produces garbled transcriptions.
**Why it happens:** Whisper internally converts to log-Mel spectrogram using float arithmetic; wrong dtype or scale produces garbage input to the encoder.
**How to avoid:** `librosa.load()` already returns float32 by default. Verify `audio_array.dtype == np.float32` in the transcriber before passing to Whisper.
**Warning signs:** `RuntimeError: expected a tensor of floating point or complex values` in test output.

### Pitfall 2: `del model` alone does not release CUDA memory
**What goes wrong:** After `del model`, `torch.cuda.memory_allocated()` still shows ~5 GB. The optimizer then OOMs.
**Why it happens:** PyTorch keeps a cache of freed CUDA blocks for reuse. `del` removes the Python reference but PyTorch's allocator does not immediately return memory to CUDA.
**How to avoid:** Always call `torch.cuda.empty_cache()` after `del model` + `gc.collect()`. Log `memory_allocated()` before and after to verify the drop.
**Warning signs:** `CUDA out of memory` in optimizer even after transcription completes.

### Pitfall 3: Patching the wrong namespace in tests
**What goes wrong:** `@patch('whisper.load_model')` does nothing — the mock is not seen by `asr.transcriber` because it already bound the name at import time.
**Why it happens:** Python module-level imports bind names in the importing module's namespace. Patching the source module after the fact doesn't affect already-bound names.
**How to avoid:** Always patch `asr.transcriber.whisper` (the whole module object as bound in `asr.transcriber`). Follows the established project pattern from `test_stylegan_wrapper.py` and `test_clip_encoder.py`.
**Warning signs:** Mock's `call_count` is 0 even though the function was called.

### Pitfall 4: Peak normalization on near-silent audio produces NaN
**What goes wrong:** `y / np.max(np.abs(y))` raises a divide-by-zero warning and produces `NaN` or `inf` when audio is silence-only (max is 0.0 or ~1e-10).
**Why it happens:** Librosa trims silence, so after `effects.trim()` the signal may be nearly empty.
**How to avoid:** Guard normalization: `if max_val > 1e-6: y = y / max_val` (else leave y as-is — it's already near-zero).
**Warning signs:** `RuntimeWarning: invalid value encountered in true_divide`.

### Pitfall 5: librosa.effects.trim returns a tuple, not just the array
**What goes wrong:** `y = librosa.effects.trim(y, top_db=20)` assigns the tuple `(trimmed_array, trim_indices)` to `y`, causing downstream shape errors.
**Why it happens:** `librosa.effects.trim()` always returns `(y_trimmed, index)`.
**How to avoid:** Unpack: `y, _ = librosa.effects.trim(y, top_db=20)`.
**Warning signs:** `TypeError: expected np.ndarray, got tuple` on the next operation.

### Pitfall 6: Windows file-locking with temp audio files
**What goes wrong:** On Windows, creating a named temp file and passing its path to librosa fails with `PermissionError` if the file is still open.
**Why it happens:** Windows does not allow a second process/handle to open a file that's already open by another handle (unlike POSIX).
**How to avoid:** Close temp file before passing path to `librosa.load()`. Use `delete=False` + explicit cleanup after. Follows the PIL Image close-before-tempdir pattern already established in Phase 2.
**Warning signs:** `PermissionError: [WinError 32]` in tests that use temp audio files.

---

## Code Examples

Verified patterns from installed libraries and official sources:

### librosa.load — exact signature in v0.9.2
```python
# Verified by inspecting installed librosa 0.9.2 in stylehuman env
y, sr = librosa.load(path, sr=16000, mono=True)
# Returns: y shape (n,), dtype float32, sr=16000
# Default dtype IS float32 — no dtype arg needed, but explicit is fine
# Default res_type='kaiser_best' in 0.9.2 (not 'soxr_hq' which is later versions)
```

### librosa.effects.trim — exact signature in v0.9.2
```python
# Verified by inspecting installed librosa 0.9.2 in stylehuman env
# Params: y, top_db=60, ref=np.max, frame_length=2048, hop_length=512, aggregate=np.max
y_trimmed, index = librosa.effects.trim(y, top_db=20)
# Returns tuple — always unpack
```

### Peak normalization with zero guard
```python
# Source: standard audio normalization pattern; verified safe for near-silent audio
max_val = np.max(np.abs(y))
if max_val > 1e-6:
    y = y / max_val
# Result: y in [-1, 1] range, float32 preserved
```

### Whisper transcribe with numpy array
```python
# Source: whisper/transcribe.py — audio param is Union[str, np.ndarray, torch.Tensor]
# Verified: transcribe params include 'audio' accepting ndarray
model = whisper.load_model("medium", device="cuda")  # or "cpu"
result = model.transcribe(audio_array)  # audio_array: float32, 1-D, 16kHz
text = result["text"].strip()
```

### VRAM unload sequence
```python
# Source: standard PyTorch memory management; confirmed by whisper community (GitHub #5, #180)
# Whisper medium requires ~5 GB VRAM — must release before optimizer runs
import gc
del model
gc.collect()
torch.cuda.empty_cache()
# After this, torch.cuda.memory_allocated() should drop by ~5 GB
```

### Namespace patching for Transcriber tests
```python
# Source: established project pattern (test_stylegan_wrapper.py, test_clip_encoder.py)
@patch('asr.transcriber.whisper')         # patch the bound name in asr.transcriber
def test_unloads_model(self, mock_whisper):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "a man with blue eyes"}
    mock_whisper.load_model.return_value = mock_model
    t = Transcriber({'device': torch.device('cpu')})
    t.transcribe(np.zeros(16000, dtype=np.float32))
    del mock_whisper  # verify del was called on model
    # assert mock_model was deleted — use weakref or call_count on gc.collect mock
```

### whisper.load_model device string format
```python
# Whisper load_model accepts device as string, not torch.device object
# Source: whisper API — device param is str ("cuda", "cpu", "cuda:0")
model = whisper.load_model("medium", device=str(cfg['device']))
# cfg['device'] is torch.device — convert with str() to get "cuda" or "cpu"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual FFmpeg subprocess for resampling | `librosa.load(sr=16000)` | librosa 0.6+ | No external process needed for format conversion |
| Pass file path to Whisper only | Pass `np.ndarray` directly | Supported since initial Whisper release | No need to write temp files for in-memory audio |
| Manual PyTorch weight deletion | `del` + `gc.collect()` + `empty_cache()` | PyTorch 1.x standard | Three-step sequence required; single `del` is insufficient |

**Deprecated/outdated:**
- `librosa.resample()` called separately after `librosa.load()`: No longer needed — `librosa.load(sr=16000)` handles resampling in one pass.

---

## Open Questions

1. **Does Whisper model.transcribe require fp16 audio or fp32?**
   - What we know: Whisper converts internally; input float32 array is accepted; the model itself runs in fp16 on GPU
   - What's unclear: Whether passing fp16 array is also accepted
   - Recommendation: Use fp32 input (output of librosa.load) — it is the confirmed safe path

2. **VRAM verification testability in unit tests**
   - What we know: `torch.cuda.memory_allocated()` returns 0 on CPU; the decided approach is DEBUG logging
   - What's unclear: How to assert VRAM was logged in a CPU-only test environment
   - Recommendation: Test that `logger.debug` was called with `memory_allocated` — mock `torch.cuda.is_available` to return True and `torch.cuda.memory_allocated` to return a sentinel value, then assert the log message

3. **ASR-03 scope boundary**
   - What we know: ASR-03 mentions mic recording — but Gradio mic input is Phase 6 scope
   - What's unclear: Does Phase 4 need to test the mic path?
   - Recommendation: Phase 4 tests only the file path. Both paths converge at `process_audio(path)` — the mic path simply writes to a temp file first (Phase 6 concern). Phase 4 verifies the file→array pipeline only.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, used in Phases 1-3) |
| Config file | none — conftest.py handles sys.path |
| Quick run command | `conda run -n stylehuman --no-capture-output python -m pytest tests/test_audio_processor.py tests/test_transcriber.py -x -q` |
| Full suite command | `conda run -n stylehuman --no-capture-output python -m pytest tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ASR-01 | `process_audio()` resamples to 16kHz mono float32 with trim and normalization | unit | `pytest tests/test_audio_processor.py -x -q` | Wave 0 |
| ASR-01 | `process_audio()` raises ValueError on corrupt/unsupported file | unit | `pytest tests/test_audio_processor.py::TestProcessAudio::test_invalid_file -x` | Wave 0 |
| ASR-02 | `Transcriber.transcribe()` returns plain string from Whisper result dict | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_returns_string -x` | Wave 0 |
| ASR-02 | Whisper model is deleted and VRAM cleared after transcribe call | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_vram_unload -x` | Wave 0 |
| ASR-02 | `whisper.load_model("medium")` called with correct device string | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_loads_medium -x` | Wave 0 |
| ASR-03 | `librosa.load` called with `sr=16000, mono=True` | unit | `pytest tests/test_audio_processor.py::TestProcessAudio::test_resamples_16k -x` | Wave 0 |
| ASR-04 | Text bypass path: `optimize()` receives string without calling `Transcriber.transcribe()` | unit (contract) | `pytest tests/test_transcriber.py::TestTranscriber::test_bypass_path -x` | Wave 0 |
| ASR-02 | Short transcription (< 3 words) logs WARNING and returns text | unit | `pytest tests/test_transcriber.py::TestTranscriber::test_short_transcription_warning -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `conda run -n stylehuman --no-capture-output python -m pytest tests/test_audio_processor.py tests/test_transcriber.py -x -q`
- **Per wave merge:** `conda run -n stylehuman --no-capture-output python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_audio_processor.py` — covers ASR-01, ASR-03
- [ ] `tests/test_transcriber.py` — covers ASR-02, ASR-04
- [ ] `asr/__init__.py` — package marker (import path `asr.audio_processor`, `asr.transcriber`)

*(Framework and conftest.py already exist from Phase 1)*

---

## Sources

### Primary (HIGH confidence)
- Installed `librosa==0.9.2` in `stylehuman` env — `inspect.signature(librosa.load)` and `inspect.signature(librosa.effects.trim)` confirmed parameters directly
- Installed `openai-whisper 20250625` in `stylehuman` env — `inspect.signature(whisper.transcribe)` confirmed audio accepts ndarray
- [deepwiki.com/openai/whisper/2.4-api-reference](https://deepwiki.com/openai/whisper/2.4-api-reference) — transcribe function signature and return dict
- [librosa.org/doc/0.9.2/generated/librosa.load.html](https://librosa.org/doc/0.9.2/generated/librosa.load.html) — official 0.9.2 docs

### Secondary (MEDIUM confidence)
- [github.com/openai/whisper/discussions/450](https://github.com/openai/whisper/discussions/450) — float32 16kHz array requirement for transcribe
- [github.com/openai/whisper/discussions/5](https://github.com/openai/whisper/discussions/5) — Whisper medium ~5 GB VRAM requirement
- [github.com/openai/whisper/discussions/380](https://github.com/openai/whisper/discussions/380) — numpy array input to transcribe

### Tertiary (LOW confidence)
- Community reports of `del` + `gc.collect()` + `torch.cuda.empty_cache()` sequence for VRAM release — consistent across multiple GitHub discussions but no single authoritative reference

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries installed and verified in target env; API signatures confirmed via `inspect`
- Architecture: HIGH — patterns are direct extensions of Phase 2/3 established conventions; locked by user decisions
- Pitfalls: HIGH for namespace patching and tuple unpacking (verified against project tests); MEDIUM for VRAM release (community-confirmed, not official PyTorch docs for Whisper specifically)

**Research date:** 2026-03-07
**Valid until:** 2026-06-07 (stable libraries; openai-whisper API has been stable; librosa 0.9.2 pinned)
