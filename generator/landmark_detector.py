"""
generator/landmark_detector.py

Detects 68 facial landmarks using the face-alignment library.
Runs once per generated face; caches results to a JSON file alongside the image.

Public API
----------
detect_landmarks(pil_image) -> dict | None
save_landmark_cache(landmarks, image_path) -> str
load_landmark_cache(image_path) -> dict | None
"""
from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np
from PIL import Image

import gpu_utils

_fa_instance = None


def _get_fa():
    """Return the shared FaceAlignment instance, creating it on first call."""
    global _fa_instance
    if _fa_instance is None:
        import face_alignment
        cfg    = gpu_utils.get_device_config()
        device = cfg["device"].type   # "cuda" or "cpu" — single source of truth
        try:
            lm_type = face_alignment.LandmarksType._2D
        except AttributeError:
            lm_type = face_alignment.LandmarksType.TWO_D
        _fa_instance = face_alignment.FaceAlignment(
            lm_type, flip_input=False, device=device,
        )
    return _fa_instance


def detect_landmarks(pil_image: Image.Image) -> dict | None:
    """
    Run 2-D landmark detection on *pil_image*.
    Returns a dict of named landmark positions, or None if detection fails.
    """
    try:
        fa      = _get_fa()
        arr     = np.array(pil_image.convert("RGB"))
        lms_all = fa.get_landmarks(arr)
        if not lms_all:
            return None
        lms = lms_all[0]
    except Exception as exc:
        warnings.warn(f"[EchoFace] landmark detection failed: {exc}")
        return None

    def _pt(i: int) -> list[float]:
        return [float(lms[i, 0]), float(lms[i, 1])]

    def _centroid(indices) -> list[float]:
        pts = lms[indices]
        return [float(pts[:, 0].mean()), float(pts[:, 1].mean())]

    left_iris  = _centroid(list(range(36, 42)))
    right_iris = _centroid(list(range(42, 48)))
    ipd = math.sqrt(
        (right_iris[0] - left_iris[0]) ** 2 +
        (right_iris[1] - left_iris[1]) ** 2
    )

    return {
        "left_iris_center":        left_iris,
        "right_iris_center":       right_iris,
        "nose_bridge_top":         _pt(27),
        "nose_bridge_bottom":      _pt(30),
        "left_brow_peak":          _pt(19),
        "right_brow_peak":         _pt(24),
        "left_face_edge":          _pt(0),
        "right_face_edge":         _pt(16),
        "chin":                    _pt(8),
        "jaw_outline":             lms[0:17].tolist(),
        "interpupillary_distance": ipd,
    }


def save_landmark_cache(landmarks: dict, image_path: str) -> str:
    """Write landmarks to JSON adjacent to image_path. Returns JSON path."""
    base      = os.path.splitext(image_path)[0]
    json_path = base + "_landmarks.json"
    with open(json_path, "w") as f:
        json.dump(landmarks, f)
    return json_path


def load_landmark_cache(image_path: str) -> dict | None:
    """Load landmark cache for image_path. Returns None if missing."""
    base      = os.path.splitext(image_path)[0]
    json_path = base + "_landmarks.json"
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)
