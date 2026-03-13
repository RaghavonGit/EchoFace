import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock, patch

from ui.app import generate_fn, export_image, edit_fn, reset_fn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cfg():
    return {
        'device': torch.device('cpu'),
        'dtype': torch.float32,
        'use_fp16': False,
        'resolution': 256,
        'truncation_psi': 0.5,
    }


def _pil():
    return Image.new('RGB', (256, 256), (120, 80, 60))


def _mock_wrapper(pil_image=None, sim=0.7777, steps_to_fire=(0, 10)):
    """Return a wrapper mock whose .generate() fires callbacks then returns."""
    pil_image = pil_image or _pil()
    wrapper = MagicMock()

    def _generate(prompt, callback=None, **kwargs):
        if callback:
            for s in steps_to_fire:
                callback(s, 0.8 - s * 0.01, pil_image)
        return (pil_image, sim)

    wrapper.generate.side_effect = _generate
    return wrapper


def _all_yields(mic=None, file=None, text="", wrapper=None, cfg=None):
    wrapper = wrapper or _mock_wrapper()
    cfg = cfg or _cfg()
    return list(generate_fn(mic, file, text, wrapper, cfg))


# ── generate_fn tests ─────────────────────────────────────────────────────────

class TestGenerateFn(unittest.TestCase):

    def test_empty_input_guard(self):
        """No audio and no text → single yield with guidance message."""
        yields = _all_yields()
        self.assertEqual(len(yields), 1)
        self.assertIn("Please provide", yields[0][0])

    def test_asr_path_calls_transcriber(self):
        """Audio only (no text) → transcribes, fills text box, stops before generate."""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        transcription = "a young woman with dark hair"

        with patch('ui.app.process_audio', return_value=dummy_audio) as mock_audio, \
             patch('ui.app.Transcriber') as mock_t_cls:
            mock_t_cls.return_value.transcribe.return_value = transcription
            yields = _all_yields(file="/tmp/x.wav", text="")

        # ASR called with audio path
        mock_audio.assert_called_once_with("/tmp/x.wav")
        # Transcriber.transcribe called
        mock_t_cls.return_value.transcribe.assert_called_once()
        # wrapper.generate must NOT be called — two-stage flow stops here
        # (we don't pass a wrapper so default mock, but generate is never reached)
        # Exactly two yields: "Transcribing..." then the transcription result
        self.assertEqual(len(yields), 2)
        # Second yield puts transcription in text_in slot (index 1)
        self.assertEqual(yields[1][1], transcription)
        # Status signals user to click Generate again
        status = yields[1][0].lower()
        self.assertTrue(
            any(w in status for w in ("review", "click", "transcription", "complete")),
            f"Unexpected status: {yields[1][0]!r}"
        )

    def test_asr_skipped_when_text_provided(self):
        """Text provided → ASR never called, goes straight to generate."""
        with patch('ui.app.process_audio') as mock_audio, \
             patch('ui.app.Transcriber') as mock_t:
            yields = _all_yields(text="a young man with short hair")

        mock_audio.assert_not_called()
        mock_t.assert_not_called()

    def test_generation_complete_status(self):
        """Final yield status contains 'Generation complete'."""
        yields = _all_yields(text="a young woman with blue eyes")
        self.assertIn("Generation complete", yields[-1][0])

    def test_final_yield_clip_score(self):
        """Final yield score (index 2) contains the sim value to 4 decimal places."""
        w = _mock_wrapper(sim=0.3142)
        yields = _all_yields(text="an elderly man with grey beard", wrapper=w)
        self.assertIn("0.3142", yields[-1][2])

    def test_final_yield_image_is_ndarray(self):
        """Final yield image (index 3) is a numpy ndarray."""
        yields = _all_yields(text="a young woman with dark hair")
        self.assertIsInstance(yields[-1][3], np.ndarray)

    def test_export_btn_visible_on_done(self):
        """Final yield export_btn (index 5) has visible=True."""
        yields = _all_yields(text="a young woman with dark hair")
        export_update = yields[-1][5]
        # gr.update returns a dict-like with visible key
        visible = (
            export_update.get("visible")
            if isinstance(export_update, dict)
            else getattr(export_update, "visible", None)
        )
        self.assertTrue(visible, "export_btn update must set visible=True on done")

    def test_progress_yields_before_done(self):
        """Intermediate yields appear before the final done yield."""
        w = _mock_wrapper(steps_to_fire=(0, 10, 20))
        yields = _all_yields(text="an elderly man with grey hair", wrapper=w)
        # At least 3 progress yields + 1 done yield
        self.assertGreaterEqual(len(yields), 4)
        # All but last have "Optimizing" in status
        for y in yields[:-1]:
            self.assertIn("Optimizing", y[0], f"Expected 'Optimizing' in: {y[0]!r}")

    def test_error_handling(self):
        """wrapper.generate raising RuntimeError yields 'Generation failed:' status."""
        w = MagicMock()
        w.generate.side_effect = RuntimeError("out of memory")
        yields = _all_yields(text="a young man with short hair", wrapper=w)
        error_yields = [y for y in yields if "Generation failed:" in y[0]]
        self.assertTrue(len(error_yields) > 0, "Expected at least one error yield")
        self.assertIn("out of memory", error_yields[0][0])

    def test_loader_off_on_done(self):
        """Final yield loader_html (index 6) does not contain the loader animation."""
        yields = _all_yields(text="a woman")
        loader_val = yields[-1][6]
        html = (
            loader_val.get("value", "")
            if isinstance(loader_val, dict)
            else getattr(loader_val, "value", "")
        )
        self.assertNotIn("ef-loader-center", html,
                         "Loader should be hidden in the final done yield")

    def test_loader_on_during_progress(self):
        """Progress yields have loader_html (index 6) containing the loader animation."""
        w = _mock_wrapper(steps_to_fire=(0,))
        yields = _all_yields(text="an elderly man with grey hair", wrapper=w)
        progress_yields = yields[:-1]
        self.assertTrue(len(progress_yields) > 0)
        loader_val = progress_yields[0][6]
        html = (
            loader_val.get("value", "")
            if isinstance(loader_val, dict)
            else getattr(loader_val, "value", "")
        )
        self.assertIn("loader", html,
                      "Loader HTML should be present in progress yields")


# ── export_image tests ────────────────────────────────────────────────────────

class TestExportImage(unittest.TestCase):

    def test_saves_timestamped_png(self):
        """Saves a YYYYMMDD_HHMMSS.png file to the given directory."""
        tmp = tempfile.mkdtemp()
        export_image(_pil(), tmp)
        files = os.listdir(tmp)
        png_files = [f for f in files if f.endswith('.png')]
        self.assertTrue(len(png_files) > 0, f"No PNG in {tmp}")
        self.assertTrue(
            any(re.match(r'^\d{8}_\d{6}\.png$', f) for f in png_files),
            f"No timestamped PNG found: {png_files}"
        )

    def test_returns_two_tuple(self):
        """Returns a 2-tuple (export_btn_update, path_update)."""
        tmp = tempfile.mkdtemp()
        result = export_image(_pil(), tmp)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_path_update_points_to_existing_file(self):
        """Second return element's value is a path to an existing .png file."""
        tmp = tempfile.mkdtemp()
        _, path_update = export_image(_pil(), tmp)
        value = (
            path_update.get("value")
            if isinstance(path_update, dict)
            else getattr(path_update, "value", None)
        )
        self.assertIsNotNone(value)
        self.assertTrue(str(value).endswith('.png'))
        self.assertTrue(os.path.exists(str(value)))

    def test_no_image_returns_gracefully(self):
        """export_image(None) returns without raising."""
        result = export_image(None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ── edit_fn tests ─────────────────────────────────────────────────────────────

def _mock_edit_wrapper(pil_image=None, sim=0.5000, steps_to_fire=(0,)):
    """Return a wrapper mock whose .edit() fires callbacks then returns."""
    pil_image = pil_image or _pil()
    wrapper = MagicMock()

    def _edit(instruction, callback=None, **kwargs):
        if callback:
            for s in steps_to_fire:
                callback(s, 0.6 - s * 0.01, pil_image)
        return (pil_image, sim)

    wrapper.edit.side_effect = _edit
    return wrapper


def _all_edit_yields(mic=None, text="", wrapper=None, cfg=None):
    wrapper = wrapper or _mock_edit_wrapper()
    cfg = cfg or _cfg()
    return list(edit_fn(mic, text, wrapper, cfg))


class TestEditFn(unittest.TestCase):

    def test_empty_input_guard(self):
        """No audio and no text → single yield with guidance message."""
        yields = _all_edit_yields()
        self.assertEqual(len(yields), 1)
        self.assertIn("Please provide", yields[0][0])

    def test_short_instruction_guard(self):
        """Instruction < 4 words → single yield with short-prompt warning."""
        yields = _all_edit_yields(text="blue eyes")
        self.assertEqual(len(yields), 1)
        self.assertIn("short", yields[0][0].lower())

    def test_edit_complete_status(self):
        """Final yield status contains 'Edit applied'."""
        yields = _all_edit_yields(text="change eye color to blue")
        self.assertIn("Edit applied", yields[-1][0])

    def test_edit_updates_stored_image(self):
        """Final yield stored_image (index 4) is a PIL Image."""
        from PIL import Image as PILImage
        yields = _all_edit_yields(text="change eye color to blue")
        self.assertIsInstance(yields[-1][4], PILImage.Image)


class TestResetFn(unittest.TestCase):

    def test_reset_returns_5_tuple(self):
        """reset_fn returns a 5-element tuple."""
        wrapper = MagicMock()
        pil_img = _pil()
        wrapper.reset.return_value = (pil_img, 0.3500)
        result = reset_fn(wrapper)
        self.assertEqual(len(result), 5)

    def test_reset_image_is_ndarray(self):
        """First element of reset_fn result is a numpy ndarray."""
        wrapper = MagicMock()
        wrapper.reset.return_value = (_pil(), 0.35)
        result = reset_fn(wrapper)
        self.assertIsInstance(result[0], np.ndarray)

    def test_reset_error_handled(self):
        """reset_fn handles RuntimeError gracefully (no generate called)."""
        wrapper = MagicMock()
        wrapper.reset.side_effect = RuntimeError("Generate a face first.")
        result = reset_fn(wrapper)
        self.assertIn("Generate a face first", result[1])


if __name__ == '__main__':
    unittest.main()
