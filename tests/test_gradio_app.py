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
        """Every yielded tuple must have exactly 13 elements (extended for landmarks)."""
        w = _mock_wrapper(steps_to_fire=(0, 10))
        yields = _all_yields(text="a young man with short hair", wrapper=w)
        for i, y in enumerate(yields):
            self.assertEqual(len(y), 13, f"Yield {i} has {len(y)} elements, expected 13")


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
        result = accessories_fn(self._face(), None, "none", "original", "none", "none")
        self.assertEqual(len(result), 5)

    def test_no_face_returns_guidance(self):
        result = accessories_fn(None, None, "round", "original", "none", "blue")
        self.assertIsNone(result[0])
        self.assertIn("Generate a face first", result[1])

    def test_none_accessories_returns_image(self):
        result = accessories_fn(self._face(), None, "none", "original", "none", "none")
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIn("cleared", result[1].lower())

    def test_glasses_only_status(self):
        result = accessories_fn(self._face(), None, "round", "original", "none", "none")
        self.assertIn("round", result[1].lower())
        self.assertNotIn("eye", result[1].lower())

    def test_eye_only_status(self):
        result = accessories_fn(self._face(), None, "none", "original", "none", "blue")
        self.assertIn("blue", result[1].lower())
        self.assertNotIn("glasses", result[1].lower())

    def test_both_accessories_status(self):
        result = accessories_fn(self._face(), None, "round", "original", "none", "green")
        self.assertIn("round", result[1].lower())
        self.assertIn("green", result[1].lower())

    def test_stored_image_is_pil(self):
        result = accessories_fn(self._face(), None, "round", "original", "none", "blue")
        self.assertIsInstance(result[3], Image.Image)

    def test_score_out_is_update(self):
        """score_out (index 2) must be gr.update() so CLIP score is preserved."""
        import gradio as gr
        result = accessories_fn(self._face(), None, "none", "original", "none", "none")
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
        """Glasses frame pixels must differ from the plain face somewhere in the eye area."""
        from generator.accessories import apply_glasses
        face   = self._face()
        result = apply_glasses(face, "round")
        arr_f  = np.array(face)
        arr_r  = np.array(result)
        # PNG-based renderer places the frame in the upper half of the face
        # (rows 288-535, cols 182-840 for FFHQ fallback).  Assert at least
        # one pixel in the broad eye band is modified.
        diff = ~(arr_r == arr_f).all(axis=2)
        eye_band = diff[280:540, 180:850]
        self.assertTrue(eye_band.any(),
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


class TestGlassesOverlayLandmarks(unittest.TestCase):
    """Tests for landmark-based glasses overlay and colorization functions."""

    def _face(self):
        return Image.new("RGB", (1024, 1024), (180, 140, 110))

    def _lms(self):
        return {
            "left_iris_center":        [335.0, 480.0],
            "right_iris_center":       [685.0, 480.0],
            "interpupillary_distance": 350.0,
            "nose_bridge_top":         [512.0, 390.0],
            "nose_bridge_bottom":      [512.0, 530.0],
            "left_brow_peak":          [330.0, 420.0],
            "right_brow_peak":         [690.0, 420.0],
            "left_face_edge":          [100.0, 800.0],
            "right_face_edge":         [900.0, 800.0],
            "chin":                    [512.0, 900.0],
            "jaw_outline":             [[100 + i*50, 800] for i in range(17)],
        }

    def _glasses_path(self, style="wayfarer"):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "assets", "glasses", f"{style}.png")

    def test_overlay_with_landmarks_returns_rgb_1024(self):
        from generator.glasses_overlay import overlay_glasses_with_landmarks
        path = self._glasses_path()
        if not os.path.exists(path):
            self.skipTest("assets/glasses/wayfarer.png not found — run Task 4")
        result = overlay_glasses_with_landmarks(self._face(), path, self._lms())
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (1024, 1024))

    def test_overlay_fallback_when_landmarks_none_does_not_raise(self):
        from generator.glasses_overlay import overlay_glasses_with_landmarks
        path = self._glasses_path()
        if not os.path.exists(path):
            self.skipTest("assets/glasses/wayfarer.png not found — run Task 4")
        result = overlay_glasses_with_landmarks(self._face(), path, None)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (1024, 1024))

    def test_colorize_frame_black_sets_rgb_to_near_black(self):
        from generator.glasses_overlay import colorize_frame
        rgba = np.zeros((100, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 80
        rgba[:, :, 3]  = 255
        result = colorize_frame(rgba.copy(), "black")
        self.assertEqual(result.shape, rgba.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue((result[:, :, 0] <= 15).all())

    def test_colorize_frame_gold_differs_from_black(self):
        from generator.glasses_overlay import colorize_frame
        rgba = np.zeros((100, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 80
        rgba[:, :, 3]  = 255
        black = colorize_frame(rgba.copy(), "black")
        gold  = colorize_frame(rgba.copy(), "gold")
        self.assertFalse(np.array_equal(black, gold))

    def test_colorize_frame_original_returns_unchanged(self):
        from generator.glasses_overlay import colorize_frame
        rgba = np.zeros((50, 100, 4), dtype=np.uint8)
        rgba[:, :, :3] = 80
        rgba[:, :, 3]  = 200
        result = colorize_frame(rgba.copy(), "original")
        np.testing.assert_array_equal(result, rgba)

    def test_apply_lens_tint_modifies_pixels(self):
        from generator.glasses_overlay import apply_lens_tint
        face   = self._face()
        result = apply_lens_tint(face, self._lms(), "grey")
        self.assertEqual(result.size, face.size)
        diff = np.abs(np.array(result).astype(int) - np.array(face).astype(int))
        self.assertGreater(diff.sum(), 0)

    def test_apply_lens_tint_none_returns_identical_copy(self):
        from generator.glasses_overlay import apply_lens_tint
        face   = self._face()
        result = apply_lens_tint(face, self._lms(), "none")
        np.testing.assert_array_equal(np.array(result), np.array(face))


class TestApplyEyeColorWithLandmarks(unittest.TestCase):

    def _face(self, size=1024):
        return Image.new("RGB", (size, size), (120, 80, 60))

    def _lms(self):
        return {
            "left_iris_center":        [335.0, 480.0],
            "right_iris_center":       [685.0, 480.0],
            "interpupillary_distance": 350.0,
            "nose_bridge_top":         [512.0, 390.0],
            "nose_bridge_bottom":      [512.0, 530.0],
            "left_brow_peak":          [330.0, 420.0],
            "right_brow_peak":         [690.0, 420.0],
            "left_face_edge":          [100.0, 800.0],
            "right_face_edge":         [900.0, 800.0],
            "chin":                    [512.0, 900.0],
            "jaw_outline":             [[100 + i*50, 800] for i in range(17)],
        }

    def test_none_color_returns_copy_unchanged(self):
        from generator.accessories import apply_eye_color
        face   = self._face()
        result = apply_eye_color(face, "none", self._lms())
        np.testing.assert_array_equal(np.array(result), np.array(face))

    def test_unknown_color_returns_copy_unchanged(self):
        from generator.accessories import apply_eye_color
        face   = self._face()
        result = apply_eye_color(face, "purple", self._lms())
        np.testing.assert_array_equal(np.array(result), np.array(face))

    def test_apply_changes_pixels_at_iris_coords(self):
        from generator.accessories import apply_eye_color
        # Dark patch at landmark iris centre simulates the iris
        face_arr = np.full((1024, 1024, 3), (120, 80, 60), dtype=np.uint8)
        cx, cy   = 335, 480
        face_arr[cy-15:cy+15, cx-15:cx+15] = (30, 20, 15)   # dark patch (iris-sized)
        face  = Image.fromarray(face_arr)
        result = apply_eye_color(face, "blue", self._lms())
        result_arr = np.array(result)
        self.assertFalse(
            np.array_equal(face_arr[cy-5:cy+5, cx-5:cx+5],
                           result_arr[cy-5:cy+5, cx-5:cx+5]),
            "Eye colour must change pixels at the landmark iris centre",
        )

    def test_output_is_rgb_same_size(self):
        from generator.accessories import apply_eye_color
        face   = self._face()
        result = apply_eye_color(face, "green", self._lms())
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, face.size)

    def test_fallback_when_landmarks_none_does_not_raise(self):
        from generator.accessories import apply_eye_color
        face   = self._face()
        result = apply_eye_color(face, "hazel", None)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, face.size)

    def test_fallback_when_landmarks_none_changes_pixels(self):
        """Fallback must actually apply colour at FFHQ hardcoded coords."""
        from generator.accessories import apply_eye_color
        face_arr = np.full((1024, 1024, 3), (120, 80, 60), dtype=np.uint8)
        # Dark patch at FFHQ fallback left-eye (340, 480)
        face_arr[460:500, 320:360] = (30, 20, 15)
        face   = Image.fromarray(face_arr)
        result = apply_eye_color(face, "blue", None)
        diff   = np.abs(np.array(result).astype(int) - face_arr.astype(int))
        self.assertGreater(diff.sum(), 0, "Fallback should modify pixels near FFHQ iris coords")

    def test_input_not_mutated(self):
        from generator.accessories import apply_eye_color
        face     = self._face()
        original = np.array(face).copy()
        apply_eye_color(face, "grey", self._lms())
        np.testing.assert_array_equal(np.array(face), original)


class TestApplyGlassesWrapper(unittest.TestCase):

    def _face(self):
        return Image.new("RGB", (1024, 1024), (180, 140, 110))

    def _lms(self):
        return {
            "left_iris_center":        [335.0, 480.0],
            "right_iris_center":       [685.0, 480.0],
            "interpupillary_distance": 350.0,
            "nose_bridge_top":         [512.0, 390.0],
            "nose_bridge_bottom":      [512.0, 530.0],
            "left_brow_peak":          [330.0, 420.0],
            "right_brow_peak":         [690.0, 420.0],
            "left_face_edge":          [100.0, 800.0],
            "right_face_edge":         [900.0, 800.0],
            "chin":                    [512.0, 900.0],
            "jaw_outline":             [[100 + i*50, 800] for i in range(17)],
        }

    def test_none_style_returns_identical_copy(self):
        from generator.accessories import apply_glasses
        face   = self._face()
        result = apply_glasses(face, "none", "original", "none", self._lms())
        np.testing.assert_array_equal(np.array(result), np.array(face))

    def test_returns_rgb_same_size_when_asset_present(self):
        from generator.accessories import apply_glasses
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "assets", "glasses", "wayfarer.png")
        if not os.path.exists(path):
            self.skipTest("wayfarer.png not found — run Task 4")
        result = apply_glasses(self._face(), "wayfarer", "black", "grey", self._lms())
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (1024, 1024))

    def test_unknown_style_returns_identical_copy(self):
        from generator.accessories import apply_glasses
        face   = self._face()
        result = apply_glasses(face, "monocle", "original", "none", self._lms())
        np.testing.assert_array_equal(np.array(result), np.array(face))


class TestApplyAccessoriesPipeline(unittest.TestCase):

    def _face(self):
        return Image.new("RGB", (1024, 1024), (150, 120, 90))

    def _lms(self):
        return {
            "left_iris_center":        [335.0, 480.0],
            "right_iris_center":       [685.0, 480.0],
            "interpupillary_distance": 350.0,
            "nose_bridge_top":         [512.0, 390.0],
            "nose_bridge_bottom":      [512.0, 530.0],
            "left_brow_peak":          [330.0, 420.0],
            "right_brow_peak":         [690.0, 420.0],
            "left_face_edge":          [100.0, 800.0],
            "right_face_edge":         [900.0, 800.0],
            "chin":                    [512.0, 900.0],
            "jaw_outline":             [[100 + i*50, 800] for i in range(17)],
        }

    def test_all_none_returns_identical_copy(self):
        from generator.accessories import apply_accessories
        face   = self._face()
        result = apply_accessories(face, None,
                                   glasses="none", frame_color="original",
                                   lens_tint="none", eye_color="none")
        np.testing.assert_array_equal(np.array(result), np.array(face))

    def test_returns_rgb_same_size(self):
        from generator.accessories import apply_accessories
        face   = self._face()
        result = apply_accessories(face, self._lms(),
                                   glasses="none", frame_color="original",
                                   lens_tint="none", eye_color="blue")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, face.size)

    def test_eye_color_applied_changes_result_vs_no_color(self):
        """Eye colour changes must produce a different image than no accessories."""
        from generator.accessories import apply_accessories
        face_arr = np.full((1024, 1024, 3), (120, 80, 60), dtype=np.uint8)
        face_arr[460:500, 320:360] = (30, 20, 15)   # dark patch at FFHQ L-eye
        face = Image.fromarray(face_arr)
        no_color = apply_accessories(face, None,
                                     glasses="none", frame_color="original",
                                     lens_tint="none", eye_color="none")
        with_color = apply_accessories(face, None,
                                       glasses="none", frame_color="original",
                                       lens_tint="none", eye_color="blue")
        diff = np.abs(np.array(with_color).astype(int) -
                      np.array(no_color).astype(int))
        self.assertGreater(diff.sum(), 0,
                           "Eye colour must produce pixel changes vs. no accessories")


class TestAccessoriesFnNewSignature(unittest.TestCase):
    """Tests for the updated accessories_fn that accepts landmarks + 4 dropdowns."""

    def _face(self):
        return Image.new("RGB", (1024, 1024), (140, 110, 80))

    def _lms(self):
        return {
            "left_iris_center":        [335.0, 480.0],
            "right_iris_center":       [685.0, 480.0],
            "interpupillary_distance": 350.0,
            "nose_bridge_top":         [512.0, 390.0],
            "nose_bridge_bottom":      [512.0, 530.0],
            "left_brow_peak":          [330.0, 420.0],
            "right_brow_peak":         [690.0, 420.0],
            "left_face_edge":          [100.0, 800.0],
            "right_face_edge":         [900.0, 800.0],
            "chin":                    [512.0, 900.0],
            "jaw_outline":             [[100 + i*50, 800] for i in range(17)],
        }

    def test_returns_5_element_tuple(self):
        from ui.app import accessories_fn
        result = accessories_fn(self._face(), self._lms(),
                                "none", "original", "none", "none")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    def test_none_base_returns_none_image_and_generate_prompt(self):
        from ui.app import accessories_fn
        img, status, *_ = accessories_fn(None, None,
                                         "none", "original", "none", "none")
        self.assertIsNone(img)
        self.assertIn("Generate", status)

    def test_eye_color_does_not_raise(self):
        from ui.app import accessories_fn
        result = accessories_fn(self._face(), self._lms(),
                                "none", "original", "none", "blue")
        self.assertIsNotNone(result[0])

    def test_status_message_includes_selection(self):
        from ui.app import accessories_fn
        _, status, *_ = accessories_fn(self._face(), self._lms(),
                                       "none", "original", "none", "blue")
        self.assertIn("blue", status.lower())

    def test_all_none_status_contains_cleared(self):
        from ui.app import accessories_fn
        _, status, *_ = accessories_fn(self._face(), self._lms(),
                                       "none", "original", "none", "none")
        self.assertIn("clear", status.lower())


if __name__ == '__main__':
    unittest.main()
