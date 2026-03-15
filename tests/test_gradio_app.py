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

from ui.app import generate_fn, export_image, accessories_fn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cfg():
    return {
        'device': torch.device('cpu'),
        'dtype': torch.float32,
        'use_fp16': False,
        'resolution': 256,
        'truncation_psi': 0.5,
    }


def _pil(size=256):
    return Image.new('RGB', (size, size), (120, 80, 60))


def _mock_wrapper(pil_image=None, sim=0.7777, steps_to_fire=(0, 10)):
    pil_image = pil_image or _pil()
    wrapper   = MagicMock()

    def _generate(prompt, callback=None, **kwargs):
        if callback:
            for s in steps_to_fire:
                callback(s, 0.8 - s * 0.01, pil_image)
        return (pil_image, sim)

    wrapper.generate.side_effect = _generate
    return wrapper


def _all_yields(mic=None, file=None, text="", wrapper=None, cfg=None):
    wrapper = wrapper or _mock_wrapper()
    cfg     = cfg     or _cfg()
    return list(generate_fn(mic, file, text, wrapper, cfg))


# ── generate_fn tests ─────────────────────────────────────────────────────────

class TestGenerateFn(unittest.TestCase):

    def test_empty_input_guard(self):
        yields = _all_yields()
        self.assertEqual(len(yields), 1)
        self.assertIn("Please provide", yields[0][0])

    def test_asr_path_calls_transcriber(self):
        dummy   = np.zeros(16000, dtype=np.float32)
        script  = "a young woman with dark hair"
        with patch('ui.app.process_audio', return_value=dummy), \
             patch('ui.app.Transcriber') as MockT:
            MockT.return_value.transcribe.return_value = script
            yields = _all_yields(file="/tmp/x.wav", text="")
        self.assertEqual(len(yields), 2)
        self.assertEqual(yields[1][1], script)
        status = yields[1][0].lower()
        self.assertTrue(any(w in status for w in
                            ("review", "click", "transcription", "complete")))

    def test_asr_skipped_when_text_provided(self):
        with patch('ui.app.process_audio') as mock_audio, \
             patch('ui.app.Transcriber') as mock_t:
            _all_yields(text="a young man with short hair")
        mock_audio.assert_not_called()
        mock_t.assert_not_called()

    def test_generation_complete_status(self):
        yields = _all_yields(text="a young woman with blue eyes")
        self.assertIn("Generation complete", yields[-1][0])

    def test_final_yield_clip_score(self):
        w = _mock_wrapper(sim=0.3142)
        yields = _all_yields(text="an elderly man with grey beard", wrapper=w)
        self.assertIn("0.3142", yields[-1][2])

    def test_final_yield_image_is_ndarray(self):
        yields = _all_yields(text="a young woman with dark hair")
        self.assertIsInstance(yields[-1][3], np.ndarray)

    def test_export_btn_visible_on_done(self):
        """export_btn is output index 5 — must be visible=True on done."""
        yields = _all_yields(text="a young woman with dark hair")
        upd    = yields[-1][5]
        visible = (upd.get("visible") if isinstance(upd, dict)
                   else getattr(upd, "visible", None))
        self.assertTrue(visible)

    def test_progress_yields_before_done(self):
        w = _mock_wrapper(steps_to_fire=(0, 10, 20))
        yields = _all_yields(text="an elderly man with grey hair", wrapper=w)
        self.assertGreaterEqual(len(yields), 4)
        for y in yields[:-1]:
            self.assertIn("Optimizing", y[0])

    def test_error_handling(self):
        w = MagicMock()
        w.generate.side_effect = RuntimeError("out of memory")
        yields = _all_yields(text="a young man with short hair", wrapper=w)
        errors = [y for y in yields if "Generation failed:" in y[0]]
        self.assertTrue(len(errors) > 0)
        self.assertIn("out of memory", errors[0][0])

    def test_loader_off_on_done(self):
        """loader_html is output index 6 — loader must be hidden on done."""
        yields = _all_yields(text="a woman with red hair")
        upd    = yields[-1][6]
        html   = (upd.get("value", "") if isinstance(upd, dict)
                  else getattr(upd, "value", ""))
        self.assertNotIn("ef-loader-center", html)

    def test_loader_on_during_progress(self):
        """loader_html is output index 6 — loader must be present in progress."""
        w = _mock_wrapper(steps_to_fire=(0,))
        yields = _all_yields(text="an elderly man with grey hair", wrapper=w)
        upd  = yields[0][6]
        html = (upd.get("value", "") if isinstance(upd, dict)
                else getattr(upd, "value", ""))
        self.assertIn("loader", html)

    def test_base_face_set_on_done(self):
        """base_face is output index 7 — must be the PIL image on done."""
        pil    = _pil()
        w      = _mock_wrapper(pil_image=pil)
        yields = _all_yields(text="a young woman with dark hair", wrapper=w)
        self.assertIs(yields[-1][7], pil)

    def test_dropdowns_reset_to_none_on_done(self):
        """dd_glasses (index 8) and dd_eye (index 9) must be 'none' on done."""
        yields = _all_yields(text="a woman with blue eyes")
        self.assertEqual(yields[-1][8], "none")
        self.assertEqual(yields[-1][9], "none")

    def test_ten_outputs_per_yield(self):
        """Every yielded tuple must have exactly 10 elements."""
        w = _mock_wrapper(steps_to_fire=(0, 10))
        yields = _all_yields(text="a young man with short hair", wrapper=w)
        for i, y in enumerate(yields):
            self.assertEqual(len(y), 10, f"Yield {i} has {len(y)} elements, expected 10")


# ── export_image tests ────────────────────────────────────────────────────────

class TestExportImage(unittest.TestCase):

    def test_saves_timestamped_png(self):
        tmp = tempfile.mkdtemp()
        export_image(_pil(), tmp)
        pngs = [f for f in os.listdir(tmp) if f.endswith('.png')]
        self.assertTrue(any(re.match(r'^\d{8}_\d{6}\.png$', f) for f in pngs))

    def test_returns_two_tuple(self):
        tmp    = tempfile.mkdtemp()
        result = export_image(_pil(), tmp)
        self.assertEqual(len(result), 2)

    def test_path_update_points_to_existing_file(self):
        tmp       = tempfile.mkdtemp()
        _, upd    = export_image(_pil(), tmp)
        value     = (upd.get("value") if isinstance(upd, dict)
                     else getattr(upd, "value", None))
        self.assertTrue(str(value).endswith('.png'))
        self.assertTrue(os.path.exists(str(value)))

    def test_no_image_returns_gracefully(self):
        result = export_image(None)
        self.assertEqual(len(result), 2)


# ── accessories_fn tests ──────────────────────────────────────────────────────

class TestAccessoriesFn(unittest.TestCase):

    def _face(self):
        return Image.new("RGB", (1024, 1024), (180, 140, 110))

    def test_returns_five_tuple(self):
        result = accessories_fn(self._face(), "none", "none")
        self.assertEqual(len(result), 5)

    def test_no_face_returns_guidance(self):
        result = accessories_fn(None, "round", "blue")
        self.assertIsNone(result[0])
        self.assertIn("Generate a face first", result[1])

    def test_none_accessories_returns_image(self):
        result = accessories_fn(self._face(), "none", "none")
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIn("cleared", result[1].lower())

    def test_glasses_only_status(self):
        result = accessories_fn(self._face(), "round", "none")
        self.assertIn("round", result[1].lower())
        self.assertNotIn("eye", result[1].lower())

    def test_eye_only_status(self):
        result = accessories_fn(self._face(), "none", "blue")
        self.assertIn("blue", result[1].lower())
        self.assertNotIn("glasses", result[1].lower())

    def test_both_accessories_status(self):
        result = accessories_fn(self._face(), "square", "green")
        self.assertIn("square", result[1].lower())
        self.assertIn("green", result[1].lower())

    def test_stored_image_is_pil(self):
        result = accessories_fn(self._face(), "round", "blue")
        self.assertIsInstance(result[3], Image.Image)

    def test_score_out_is_update(self):
        """score_out (index 2) must be gr.update() so CLIP score is preserved."""
        import gradio as gr
        result = accessories_fn(self._face(), "none", "none")
        # gr.update() returns a dict-like — should not be a plain string
        self.assertNotIsInstance(result[2], str)


# ── apply_accessories unit tests ──────────────────────────────────────────────

class TestApplyAccessories(unittest.TestCase):

    def _face(self):
        return Image.new("RGB", (1024, 1024), (180, 140, 110))

    def test_apply_glasses_round_returns_rgb(self):
        from generator.accessories import apply_glasses
        result = apply_glasses(self._face(), "round")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (1024, 1024))

    def test_apply_glasses_square_returns_rgb(self):
        from generator.accessories import apply_glasses
        result = apply_glasses(self._face(), "square")
        self.assertEqual(result.mode, "RGB")

    def test_apply_glasses_aviator_returns_rgb(self):
        from generator.accessories import apply_glasses
        result = apply_glasses(self._face(), "aviator")
        self.assertEqual(result.mode, "RGB")

    def test_apply_glasses_none_equals_copy(self):
        from generator.accessories import apply_glasses
        face   = self._face()
        result = apply_glasses(face, "none")
        self.assertEqual(np.array(face).tolist(), np.array(result).tolist())

    def test_apply_glasses_does_not_mutate_base(self):
        from generator.accessories import apply_glasses
        face   = self._face()
        before = np.array(face).copy()
        apply_glasses(face, "round")
        self.assertEqual(before.tolist(), np.array(face).tolist())

    def test_apply_glasses_modifies_frame_region(self):
        """Glasses frame pixels must differ from the plain face."""
        from generator.accessories import apply_glasses, _L_EYE
        face   = self._face()
        result = apply_glasses(face, "round")
        arr_f  = np.array(face)
        arr_r  = np.array(result)
        # Sample near the left eye position — frame ring should be dark
        lx, ly = _L_EYE
        patch_result = arr_r[ly - 5:ly + 5, lx - 5:lx + 5]
        patch_face   = arr_f[ly - 5:ly + 5, lx - 5:lx + 5]
        self.assertFalse(np.array_equal(patch_result, patch_face),
                         "Frame area unchanged — glasses not rendered")

    def test_apply_eye_color_none_equals_copy(self):
        from generator.accessories import apply_eye_color
        face   = self._face()
        result = apply_eye_color(face, "none")
        self.assertEqual(np.array(face).tolist(), np.array(result).tolist())

    def test_apply_eye_color_does_not_mutate_base(self):
        from generator.accessories import apply_eye_color
        face   = self._face()
        before = np.array(face).copy()
        apply_eye_color(face, "blue")
        self.assertEqual(before.tolist(), np.array(face).tolist())

    def test_apply_eye_color_modifies_iris_region(self):
        """Eye-colour shift must change pixels at the iris centre."""
        from generator.accessories import apply_eye_color, _L_EYE, _IRIS_R
        face   = self._face()
        result = apply_eye_color(face, "blue")
        lx, ly = _L_EYE
        r      = _IRIS_R // 2          # inner half of iris — strongest shift
        patch_f = np.array(face)  [ly - r:ly + r, lx - r:lx + r]
        patch_r = np.array(result)[ly - r:ly + r, lx - r:lx + r]
        self.assertFalse(np.array_equal(patch_f, patch_r),
                         "Iris region unchanged — eye colour not applied")

    def test_apply_accessories_order(self):
        """apply_accessories == apply_glasses(apply_eye_color(base))."""
        from generator.accessories import (
            apply_accessories, apply_glasses, apply_eye_color,
        )
        face     = self._face()
        expected = apply_glasses(apply_eye_color(face, "green"), "square")
        result   = apply_accessories(face, glasses="square", eye_color="green")
        self.assertEqual(np.array(expected).tolist(), np.array(result).tolist())


# ── _find_iris_centre unit tests ───────────────────────────────────────────────

class TestFindIrisCentre(unittest.TestCase):
    """Tests for dynamic pupil-based iris centre detection."""

    def _uniform_arr(self, brightness=120):
        """1024×1024 RGB array, all pixels the same grey."""
        return np.full((1024, 1024, 3), brightness, dtype=np.uint8)

    def test_uniform_image_returns_expected_centre(self):
        """On a flat image there is no dark cluster → return expected coords."""
        from generator.accessories import _find_iris_centre
        arr = self._uniform_arr(brightness=120)
        cx, cy = _find_iris_centre(arr, expected_cx=340, expected_cy=480, search_r=60)
        # Must stay close to the expected position
        self.assertAlmostEqual(cx, 340, delta=5)
        self.assertAlmostEqual(cy, 480, delta=5)

    def test_dark_blob_found(self):
        """A dark patch at a known offset from the expected centre is detected."""
        from generator.accessories import _find_iris_centre
        arr = self._uniform_arr(brightness=200)
        # Place a 24×24 dark patch 15px right of expected centre.
        # 24×24 = 576 pixels vs 120×120 = 14400-pixel window → clearly detected
        # by the min+20 threshold (min=10, thresh=30, 576 pixels qualify).
        true_cx, true_cy = 355, 480
        arr[true_cy - 12:true_cy + 12, true_cx - 12:true_cx + 12] = 10
        cx, cy = _find_iris_centre(arr, expected_cx=340, expected_cy=480, search_r=60)
        self.assertAlmostEqual(cx, true_cx, delta=8)
        self.assertAlmostEqual(cy, true_cy, delta=8)

    def test_dark_blob_outside_search_window_ignored(self):
        """A dark patch far outside the search radius must not move the result."""
        from generator.accessories import _find_iris_centre
        arr = self._uniform_arr(brightness=200)
        # Patch 200px away — outside search_r=60
        arr[200:215, 100:115] = 5
        cx, cy = _find_iris_centre(arr, expected_cx=340, expected_cy=480, search_r=60)
        self.assertAlmostEqual(cx, 340, delta=5)
        self.assertAlmostEqual(cy, 480, delta=5)

    def test_result_is_int_tuple(self):
        from generator.accessories import _find_iris_centre
        arr = self._uniform_arr()
        cx, cy = _find_iris_centre(arr, 340, 480, 60)
        self.assertIsInstance(cx, int)
        self.assertIsInstance(cy, int)


# ── Lab colour-space conversion tests ─────────────────────────────────────────

class TestLabConversion(unittest.TestCase):
    """Round-trip and known-value tests for RGB↔Lab numpy helpers."""

    def _arr(self, *rgb):
        """Return a (1,1,3) float32 array from integer RGB values."""
        return np.array([[[r / 255.0 for r in rgb]]], dtype=np.float32)

    def test_white_to_lab(self):
        """sRGB white (1,1,1) → Lab L≈100, a≈0, b≈0."""
        from generator.accessories import _rgb_to_lab_np
        lab = _rgb_to_lab_np(np.ones((1, 1, 3), dtype=np.float32))
        self.assertAlmostEqual(float(lab[0, 0, 0]), 100.0, delta=1.0)   # L
        self.assertAlmostEqual(float(lab[0, 0, 1]),   0.0, delta=2.0)   # a
        self.assertAlmostEqual(float(lab[0, 0, 2]),   0.0, delta=2.0)   # b

    def test_black_to_lab(self):
        """sRGB black (0,0,0) → Lab L≈0, a≈0, b≈0."""
        from generator.accessories import _rgb_to_lab_np
        lab = _rgb_to_lab_np(np.zeros((1, 1, 3), dtype=np.float32))
        self.assertAlmostEqual(float(lab[0, 0, 0]), 0.0, delta=1.0)
        self.assertAlmostEqual(float(lab[0, 0, 1]), 0.0, delta=2.0)
        self.assertAlmostEqual(float(lab[0, 0, 2]), 0.0, delta=2.0)

    def test_round_trip(self):
        """RGB → Lab → RGB recovers original within 1/255 error."""
        from generator.accessories import _rgb_to_lab_np, _lab_to_rgb_np
        original = np.array([[[0.4, 0.2, 0.7],
                               [0.9, 0.1, 0.3]]], dtype=np.float32)
        recovered = _lab_to_rgb_np(_rgb_to_lab_np(original))
        np.testing.assert_allclose(recovered, original, atol=1/255.0)

    def test_blue_has_negative_b(self):
        """Pure blue in Lab should have strongly negative b axis."""
        from generator.accessories import _rgb_to_lab_np
        blue = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        lab = _rgb_to_lab_np(blue)
        self.assertLess(float(lab[0, 0, 2]), -50.0)   # b should be very negative

    def test_output_shapes_preserved(self):
        from generator.accessories import _rgb_to_lab_np, _lab_to_rgb_np
        rgb = np.random.rand(16, 16, 3).astype(np.float32)
        lab = _rgb_to_lab_np(rgb)
        self.assertEqual(lab.shape, rgb.shape)
        back = _lab_to_rgb_np(lab)
        self.assertEqual(back.shape, rgb.shape)


# ── _shift_iris_lab unit tests ─────────────────────────────────────────────────

class TestShiftIrisLab(unittest.TestCase):
    """Tests for the Lab-space iris colour replacement."""

    def _grey_face(self, brightness=120):
        """Uniform 1024×1024 RGB array, all pixels the same grey."""
        return np.full((1024, 1024, 3), brightness, dtype=np.uint8)

    def test_centre_pixel_changes(self):
        """The pixel at the iris centre must change colour."""
        from generator.accessories import _shift_iris_lab
        arr = self._grey_face(120)
        # a_tgt=-15, b_tgt=-55 → blueish
        out = _shift_iris_lab(arr, 340, 480, r=26, a_tgt=-15.0, b_tgt=-55.0)
        cx, cy = 340, 480
        self.assertFalse(np.array_equal(arr[cy, cx], out[cy, cx]),
                         "Centre pixel was not modified")

    def test_far_pixel_unchanged(self):
        """Pixels far from the iris must not be touched."""
        from generator.accessories import _shift_iris_lab
        arr = self._grey_face(120)
        out = _shift_iris_lab(arr, 340, 480, r=26, a_tgt=-15.0, b_tgt=-55.0)
        # (0, 0) is hundreds of pixels away
        np.testing.assert_array_equal(arr[0, 0], out[0, 0])

    def test_input_not_mutated(self):
        from generator.accessories import _shift_iris_lab
        arr = self._grey_face(120)
        orig = arr.copy()
        _shift_iris_lab(arr, 340, 480, r=26, a_tgt=-15.0, b_tgt=-55.0)
        np.testing.assert_array_equal(arr, orig)

    def test_output_uint8(self):
        from generator.accessories import _shift_iris_lab
        arr = self._grey_face(120)
        out = _shift_iris_lab(arr, 340, 480, r=26, a_tgt=-15.0, b_tgt=-55.0)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, arr.shape)

    def test_dark_iris_gets_visible_colour(self):
        """
        A very dark iris (brightness ≈ 15/255) should receive visible colour —
        the old HSV approach returned near-black; Lab preserves some L.
        """
        from generator.accessories import _shift_iris_lab
        arr = self._grey_face(200)                        # bright background
        cx, cy, r = 340, 480, 26
        arr[cy - r:cy + r, cx - r:cx + r] = 15           # very dark iris patch
        out = _shift_iris_lab(arr, cx, cy, r=r, a_tgt=-15.0, b_tgt=-55.0)
        # Centre pixel must NOT be near-black after blue shift
        centre = out[cy, cx].astype(float)
        self.assertGreater(centre[2], 30,
            "Blue channel too low — Lab replacement failed on dark iris")


if __name__ == '__main__':
    unittest.main()
