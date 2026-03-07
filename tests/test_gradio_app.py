import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch, MagicMock, call

from ui.app import generate_fn, export_image


def _make_mock_wrapper():
    """Return a MagicMock mimicking StyleGANWrapper."""
    wrapper = MagicMock()
    wrapper.sample_w.return_value = torch.zeros(1, 18, 512)
    wrapper.synthesize.return_value = torch.zeros(256, 512, 3, dtype=torch.uint8)
    wrapper.G.synthesis.return_value = torch.zeros(1, 3, 256, 512, dtype=torch.float32)
    return wrapper


def _make_mock_cfg():
    return {
        'device': torch.device('cpu'),
        'dtype': torch.float32,
        'use_fp16': False,
        'resolution': 256,
        'truncation_psi': 0.5,
    }


def _make_pil_image():
    """Return a 256x512 portrait PIL image matching StyleGAN-Human v2 output."""
    return Image.new('RGB', (256, 512), color=(128, 64, 32))


class TestGenerate(unittest.TestCase):

    def test_text_bypass(self):
        """When text_input is non-empty, generate_fn skips ASR entirely.

        Final yielded status must be 'Done.'
        process_audio and Transcriber must never be called.
        """
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        pil_image = _make_pil_image()

        def _fake_optimize(prompt, wrapper, cfg, callback=None):
            return (pil_image, MagicMock(), 0.7777)

        with patch('ui.app.optimize') as mock_opt, \
             patch('ui.app.process_audio') as mock_process_audio, \
             patch('ui.app.Transcriber') as mock_transcriber:
            mock_opt.side_effect = _fake_optimize
            yields = list(generate_fn(None, None, "a young woman", wrapper, cfg))

        # ASR path must not be triggered
        mock_process_audio.assert_not_called()
        mock_transcriber.assert_not_called()

        # Final yield status must signal completion
        final_status = yields[-1][0]
        self.assertEqual(final_status, "Done.")

    def test_asr_path(self):
        """When text_input is empty and file_audio is provided, ASR runs and fills text box.

        The two-stage UX flow:
        1. generate_fn calls process_audio and Transcriber.transcribe
        2. Yields exactly once with the transcription in the text_in slot (index 1)
        3. Status (index 0) signals the user to review/edit
        4. optimize() is NOT called — generator stops after ASR
        """
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        dummy_audio = np.zeros(16000, dtype=np.float32)
        transcription = "a young woman with dark hair"

        with patch('ui.app.optimize') as mock_opt, \
             patch('ui.app.process_audio') as mock_process_audio, \
             patch('ui.app.Transcriber') as mock_transcriber:

            mock_process_audio.return_value = dummy_audio
            mock_transcriber_instance = MagicMock()
            mock_transcriber_instance.transcribe.return_value = transcription
            mock_transcriber.return_value = mock_transcriber_instance

            yields = list(generate_fn(None, "/tmp/x.wav", "", wrapper, cfg))

        # 1. process_audio called with the audio file path
        self.assertTrue(mock_process_audio.called)
        self.assertEqual(mock_process_audio.call_args[0][0], "/tmp/x.wav")

        # 2. Transcriber(cfg).transcribe() called with the returned audio array
        self.assertTrue(mock_transcriber_instance.transcribe.called)

        # 3. Generator yields exactly once then stops
        self.assertEqual(len(yields), 1,
            f"Expected 1 yield for ASR path, got {len(yields)}")

        # 4. text_in_update (index 1) equals the transcription string
        self.assertEqual(yields[0][1], transcription,
            f"Expected text_in_update='{transcription}', got '{yields[0][1]}'")

        # 5. Status (index 0) signals user to review/edit
        status_lower = yields[0][0].lower()
        self.assertTrue(
            "review" in status_lower or "edit" in status_lower or "transcription" in status_lower,
            f"Expected status to contain 'review', 'edit', or 'transcription', got: '{yields[0][0]}'"
        )

        # 6. optimize() was NOT called
        self.assertEqual(mock_opt.call_count, 0,
            f"optimize() should not be called in ASR path, got call_count={mock_opt.call_count}")

    def test_empty_input(self):
        """When mic_audio=None, file_audio=None, text_input='', yield an error status."""
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()

        with patch('ui.app.optimize'), \
             patch('ui.app.process_audio'), \
             patch('ui.app.Transcriber'):
            yields = list(generate_fn(None, None, "", wrapper, cfg))

        first_status = yields[0][0]
        self.assertIn("Please provide a description or record audio.", first_status)

    def test_exception_handling(self):
        """When optimize() raises RuntimeError, generate_fn yields a readable error status."""
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()

        with patch('ui.app.optimize') as mock_opt, \
             patch('ui.app.process_audio'), \
             patch('ui.app.Transcriber'):
            mock_opt.side_effect = RuntimeError("out of memory")
            yields = list(generate_fn(None, None, "a young man", wrapper, cfg))

        error_statuses = [y[0] for y in yields if "Generation failed:" in y[0]]
        self.assertTrue(len(error_statuses) > 0,
            "Expected at least one yield with 'Generation failed:' in status")
        self.assertIn("out of memory", error_statuses[0])

    def test_score_yielded(self):
        """During optimization, generate_fn yields intermediate score updates.

        When callback fires at step=10 with sim_score=0.42, a yield containing
        '0.42' in the score string (index 2) must appear before the final yield.
        """
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        pil_image = _make_pil_image()

        def _fake_optimize(prompt, wrapper, cfg, callback=None):
            if callback:
                callback(10, 0.5, 0.42)
            return (pil_image, MagicMock(), 0.7777)

        with patch('ui.app.optimize') as mock_opt, \
             patch('ui.app.process_audio'), \
             patch('ui.app.Transcriber'):
            mock_opt.side_effect = _fake_optimize
            yields = list(generate_fn(None, None, "a young woman", wrapper, cfg))

        score_strings = [y[2] for y in yields if y[2] and "0.42" in str(y[2])]
        self.assertTrue(len(score_strings) > 0,
            f"Expected a yield with '0.42' in score string (index 2). All score values: "
            f"{[y[2] for y in yields]}")

    def test_final_score(self):
        """Final yielded tuple has:
        - score (index 2) containing the final_sim formatted to 4 decimal places
        - image (index 3) as np.ndarray
        - export_btn_update (index 5) as gr.update(visible=True)
        """
        wrapper = _make_mock_wrapper()
        cfg = _make_mock_cfg()
        pil_image = _make_pil_image()

        def _fake_optimize(prompt, wrapper, cfg, callback=None):
            return (pil_image, MagicMock(), 0.7777)

        with patch('ui.app.optimize') as mock_opt, \
             patch('ui.app.process_audio'), \
             patch('ui.app.Transcriber'):
            mock_opt.side_effect = _fake_optimize
            yields = list(generate_fn(None, None, "a young woman", wrapper, cfg))

        final = yields[-1]

        # Score at index 2 must contain the final sim formatted to 4 decimal places
        self.assertIn("0.7777", str(final[2]),
            f"Expected '0.7777' in final score string, got: {final[2]!r}")

        # Image at index 3 must be np.ndarray
        self.assertIsInstance(final[3], np.ndarray,
            f"Expected np.ndarray for image, got {type(final[3])}")

        # export_btn_update at index 5 must be visible
        import gradio as gr
        export_btn = final[5]
        # gr.update returns a dict-like object; check visible=True is set
        # Gradio 3.x gr.update returns a dict with '__type__' key
        self.assertIsNotNone(export_btn,
            "export_btn_update (index 5) must not be None in final yield")


class TestExport(unittest.TestCase):

    def test_export_saves_file(self):
        """export_image saves a timestamped PNG to outputs_dir and the file exists on disk."""
        import re
        pil_image = _make_pil_image()
        tmp_dir = tempfile.mkdtemp()

        import gradio as gr
        with patch('ui.app.process_audio'), patch('ui.app.Transcriber'):
            result = export_image(pil_image, tmp_dir)

        # A file matching YYYYMMDD_HHMMSS.png must exist in tmp_dir
        files = os.listdir(tmp_dir)
        png_files = [f for f in files if f.endswith('.png')]
        self.assertTrue(len(png_files) > 0,
            f"No PNG files found in {tmp_dir}. Files: {files}")

        pattern = re.compile(r'^\d{8}_\d{6}\.png$')
        matching = [f for f in png_files if pattern.match(f)]
        self.assertTrue(len(matching) > 0,
            f"No file matching YYYYMMDD_HHMMSS.png found. PNG files: {png_files}")

        saved_path = os.path.join(tmp_dir, matching[0])
        self.assertTrue(os.path.exists(saved_path),
            f"Saved file does not exist: {saved_path}")

    def test_export_shows_path(self):
        """export_image returns (gr.update(visible=True), gr.update(visible=True, value=path)).

        Second element must have visible=True and value equal to the saved file path.
        """
        import re
        pil_image = _make_pil_image()
        tmp_dir = tempfile.mkdtemp()

        result = export_image(pil_image, tmp_dir)

        self.assertIsInstance(result, tuple,
            f"export_image must return a tuple, got {type(result)}")
        self.assertEqual(len(result), 2,
            f"export_image must return a 2-tuple, got length {len(result)}")

        _, path_update = result

        # The second element must have a value equal to the saved file path
        # Gradio 3.x gr.update returns a dict with 'value' key when value= is set
        if hasattr(path_update, '__getitem__'):
            value = path_update.get('value') or path_update.get('__value__')
        else:
            value = getattr(path_update, 'value', None)

        self.assertIsNotNone(value,
            f"Second return element must contain saved file path, got: {path_update!r}")
        self.assertTrue(str(value).endswith('.png'),
            f"Saved path must end with .png, got: {value!r}")
        self.assertTrue(os.path.exists(str(value)),
            f"Saved file path does not exist on disk: {value!r}")


if __name__ == '__main__':
    unittest.main()
