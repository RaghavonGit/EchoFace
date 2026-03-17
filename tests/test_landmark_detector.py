import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from generator.landmark_detector import (
    detect_landmarks,
    save_landmark_cache,
    load_landmark_cache,
)


def _fake_lms():
    lms = np.zeros((68, 2), dtype=np.float32)
    for i, (x, y) in enumerate([(310,480),(325,470),(345,470),(360,480),(345,490),(325,490)]):
        lms[36 + i] = [x, y]
    for i, (x, y) in enumerate([(660,480),(675,470),(695,470),(710,480),(695,490),(675,490)]):
        lms[42 + i] = [x, y]
    lms[27] = [512, 390]
    lms[30] = [512, 480]
    lms[19] = [330, 420]
    lms[24] = [690, 420]
    for i in range(17):
        lms[i] = [100 + i * 52, 800 + abs(i - 8) * 10]
    return lms


def _pil(size=1024):
    return Image.new("RGB", (size, size), (120, 80, 60))


class TestDetectLandmarks(unittest.TestCase):

    def test_returns_none_when_detection_returns_empty_list(self):
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.return_value = []
            result = detect_landmarks(_pil())
        self.assertIsNone(result)

    def test_returns_none_when_detection_returns_none(self):
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.return_value = None
            result = detect_landmarks(_pil())
        self.assertIsNone(result)

    def test_returns_none_when_detection_raises(self):
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.side_effect = RuntimeError("boom")
            result = detect_landmarks(_pil())
        self.assertIsNone(result)

    def test_returns_dict_with_all_required_keys(self):
        lms = _fake_lms()
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.return_value = [lms]
            result = detect_landmarks(_pil())
        for key in ["left_iris_center","right_iris_center","nose_bridge_top",
                    "nose_bridge_bottom","left_brow_peak","right_brow_peak",
                    "left_face_edge","right_face_edge","chin","jaw_outline",
                    "interpupillary_distance"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_left_iris_center_is_centroid_of_indices_36_to_41(self):
        lms = _fake_lms()
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.return_value = [lms]
            result = detect_landmarks(_pil())
        self.assertAlmostEqual(result["left_iris_center"][0], float(lms[36:42,0].mean()), places=3)
        self.assertAlmostEqual(result["left_iris_center"][1], float(lms[36:42,1].mean()), places=3)

    def test_ipd_matches_euclidean_distance(self):
        lms = _fake_lms()
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.return_value = [lms]
            result = detect_landmarks(_pil())
        li, ri = result["left_iris_center"], result["right_iris_center"]
        expected = math.sqrt((ri[0]-li[0])**2 + (ri[1]-li[1])**2)
        self.assertAlmostEqual(result["interpupillary_distance"], expected, places=3)

    def test_jaw_outline_has_17_points(self):
        lms = _fake_lms()
        with patch("generator.landmark_detector._get_fa") as mock_fa:
            mock_fa.return_value.get_landmarks.return_value = [lms]
            result = detect_landmarks(_pil())
        self.assertEqual(len(result["jaw_outline"]), 17)


class TestSaveLoadCache(unittest.TestCase):

    def setUp(self):
        self.lms = {
            "left_iris_center": [335.0, 480.0],
            "right_iris_center": [685.0, 480.0],
            "interpupillary_distance": 350.0,
            "nose_bridge_top": [512.0, 390.0],
            "nose_bridge_bottom": [512.0, 480.0],
            "left_brow_peak": [330.0, 420.0],
            "right_brow_peak": [690.0, 420.0],
            "left_face_edge": [100.0, 800.0],
            "right_face_edge": [900.0, 800.0],
            "chin": [512.0, 900.0],
            "jaw_outline": [[100+i*50, 800] for i in range(17)],
        }

    def test_save_creates_json_file_with_landmarks_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = os.path.join(d, "face.png")
            open(img_path, "w").close()
            json_path = save_landmark_cache(self.lms, img_path)
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(json_path.endswith("_landmarks.json"))

    def test_load_returns_same_data(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = os.path.join(d, "face.png")
            open(img_path, "w").close()
            save_landmark_cache(self.lms, img_path)
            loaded = load_landmark_cache(img_path)
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded["interpupillary_distance"], 350.0, places=3)
        self.assertEqual(loaded["left_iris_center"], [335.0, 480.0])

    def test_load_returns_none_when_file_missing(self):
        self.assertIsNone(load_landmark_cache("/nonexistent/path/face.png"))


if __name__ == "__main__":
    unittest.main()
