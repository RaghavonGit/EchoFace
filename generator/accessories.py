"""
generator/accessories.py

Accessory overlays for EchoFace — glasses frames and eye colour shifts.
All public functions are pure: the input PIL image is never mutated.

Eye colour
----------
Iris chrominance (a, b in CIE Lab) is replaced inside a soft circular mask
centred on the coordinates from the landmark cache.  L (lightness) is
preserved pixel-by-pixel so natural iris shading is retained.
Falls back to hardcoded FFHQ proportions when landmarks=None.

Glasses
-------
Delegates to generator.glasses_overlay.overlay_glasses_with_landmarks.
Lens tint is applied inside apply_glasses after the PNG overlay.
Falls back to FFHQ proportions when landmarks=None.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from PIL import Image

from generator.glasses_overlay import (
    apply_lens_tint,
    overlay_glasses_with_landmarks,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_SZ     = 1024
_IRIS_R = 26          # iris colour-shift radius (px)

# FFHQ fallback iris positions — used when landmark cache is unavailable
_FFHQ_L_EYE = (340, 480)
_FFHQ_R_EYE = (684, 480)

# Backward-compatible aliases (used by legacy tests)
_L_EYE = _FFHQ_L_EYE
_R_EYE = _FFHQ_R_EYE
_EYE_SPAN = _R_EYE[0] - _L_EYE[0]   # 344 px  centre-to-centre

# Map: UI style name → PNG filename in assets/glasses/
_GLASSES_FILES: dict[str, str] = {
    "aviator":    "aviator.png",
    "round":      "round.png",
    "wayfarer":   "wayfarer.png",
    "clubmaster": "clubmaster.png",
    "cat-eye":    "cat_eye.png",
}

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "glasses",
)

# ── Search radius for dynamic pupil detection ─────────────────────────────────
_SEARCH_R = 60   # px — window around expected eye position to scan


# ── CIE Lab colour-space conversion (pure numpy, D65 illuminant) ──────────────

def _rgb_to_lab_np(rgb_f: np.ndarray) -> np.ndarray:
    """sRGB float32 [..., 3] in [0, 1]  →  CIE Lab float32 [..., 3]."""
    mask = rgb_f > 0.04045
    lin  = np.where(mask,
                    ((rgb_f + 0.055) / 1.055) ** 2.4,
                    rgb_f / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz   = lin @ M.T
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz   = xyz / white
    delta  = 6.0 / 29.0
    delta3 = delta ** 3
    coeff  = 1.0 / (3.0 * delta ** 2)
    f = np.where(xyz > delta3,
                 xyz ** (1.0 / 3.0),
                 coeff * xyz + (4.0 / 29.0))
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb_np(lab: np.ndarray) -> np.ndarray:
    """CIE Lab float32 [..., 3]  →  sRGB float32 [..., 3] clipped to [0, 1]."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    delta  = 6.0 / 29.0
    delta2 = delta ** 2
    def _f_inv(t):
        return np.where(t > delta, t ** 3, 3.0 * delta2 * (t - 4.0 / 29.0))
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz   = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * white
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    lin = np.clip(xyz @ M_inv.T, 0.0, 1.0)
    mask = lin > 0.0031308
    rgb  = np.where(mask,
                    1.055 * lin ** (1.0 / 2.4) - 0.055,
                    12.92 * lin)
    return np.clip(rgb, 0.0, 1.0)


# ── Eye-colour Lab target values ──────────────────────────────────────────────

_EYE_COLOURS: dict[str, tuple[float, float]] = {
    "blue":  (-15.0, -55.0),
    "green": (-38.0,  28.0),
    "hazel": ( 12.0,  32.0),
    "grey":  ( -3.0,  -8.0),
    "brown": ( 18.0,  34.0),
}


def _find_iris_centre(arr: np.ndarray,
                      expected_cx: int, expected_cy: int,
                      search_r: int = _SEARCH_R) -> tuple:
    """
    Find the iris centre by locating the darkest pixel cluster in a search
    window around the expected position.

    Works on any RGB uint8 numpy array.  Falls back to the expected coordinates
    on uniform / featureless images (e.g. test mocks).

    Returns
    -------
    (cx, cy) : tuple of int
    """
    h, w = arr.shape[:2]
    y0 = max(0, expected_cy - search_r)
    y1 = min(h,  expected_cy + search_r)
    x0 = max(0, expected_cx - search_r)
    x1 = min(w,  expected_cx + search_r)

    patch = arr[y0:y1, x0:x1]                        # (sh, sw, 3)
    grey  = patch.mean(axis=2)                        # (sh, sw)  brightness

    # Find the absolute minimum brightness in the window.
    # If the minimum is bright (> 100), no dark pupil exists — return expected.
    min_val = float(grey.min())
    if min_val > 100:
        return expected_cx, expected_cy

    # Threshold: anything within +20 of the minimum (captures the pupil cluster).
    thresh = min_val + 20.0
    mask = grey <= thresh

    ys_local, xs_local = np.where(mask)

    if len(xs_local) == 0:
        return expected_cx, expected_cy

    # Centroid of dark pixels → iris centre
    cx = int(round(xs_local.mean())) + x0
    cy = int(round(ys_local.mean())) + y0
    return cx, cy


def _shift_iris_lab(arr: np.ndarray,
                    cx: int, cy: int, r: int,
                    a_tgt: float, b_tgt: float) -> np.ndarray:
    """
    Replace Lab chrominance (a, b) of one iris while preserving lightness (L).
    Uses a soft circular alpha mask with quadratic falloff.
    Returns a new uint8 array (input not mutated).
    """
    y0 = max(0, cy - r);  y1 = min(arr.shape[0], cy + r)
    x0 = max(0, cx - r);  x1 = min(arr.shape[1], cx + r)
    patch_u8 = arr[y0:y1, x0:x1]
    patch_f  = patch_u8.astype(np.float32) / 255.0
    lab      = _rgb_to_lab_np(patch_f)
    lab_tgt       = lab.copy()
    lab_tgt[..., 1] = a_tgt
    lab_tgt[..., 2] = b_tgt
    rgb_tgt  = _lab_to_rgb_np(lab_tgt)
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    xx, yy  = np.meshgrid(xs, ys)
    dist    = np.sqrt(xx ** 2 + yy ** 2)
    alpha   = np.clip(1.0 - dist / r, 0.0, 1.0) ** 1.8
    alpha3  = alpha[:, :, np.newaxis]
    blended = patch_f * (1.0 - alpha3) + rgb_tgt * alpha3
    out = arr.copy()
    out[y0:y1, x0:x1] = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public eye colour API
# ─────────────────────────────────────────────────────────────────────────────

def apply_eye_color(base: Image.Image,
                    color: str,
                    landmarks: dict | None = None) -> Image.Image:
    """
    Shift iris colour on both eyes using landmark-detected centres.

    Parameters
    ----------
    base      : RGB PIL Image (1024×1024 FFHQ-aligned face).
    color     : "blue" | "green" | "hazel" | "grey" | "brown" | "none"
    landmarks : dict from detect_landmarks(), or None to use FFHQ fallback.

    Returns a new RGB PIL Image; *base* is not modified.
    """
    if color == "none" or color not in _EYE_COLOURS:
        return base.copy()

    if landmarks is None:
        warnings.warn(
            "[EchoFace] landmark detection unavailable — "
            "using FFHQ fallback coordinates for eye colour"
        )
        eye_centres = [_FFHQ_L_EYE, _FFHQ_R_EYE]
    else:
        li = landmarks["left_iris_center"]
        ri = landmarks["right_iris_center"]
        eye_centres = [(int(li[0]), int(li[1])), (int(ri[0]), int(ri[1]))]

    a_tgt, b_tgt = _EYE_COLOURS[color]
    arr = np.array(base.convert("RGB"), dtype=np.uint8)

    for (ex, ey) in eye_centres:
        cx, cy = _find_iris_centre(arr, ex, ey, _SEARCH_R)
        arr = _shift_iris_lab(arr, cx, cy, _IRIS_R, a_tgt, b_tgt)

    return Image.fromarray(arr, "RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Public glasses API
# ─────────────────────────────────────────────────────────────────────────────

def apply_glasses(base: Image.Image,
                  style: str,
                  frame_color: str = "original",
                  lens_tint: str = "none",
                  landmarks: dict | None = None) -> Image.Image:
    """
    Composite a glasses PNG overlay onto the face using landmark alignment.

    Parameters
    ----------
    base        : RGB PIL Image (1024×1024 FFHQ-aligned face).
    style       : "aviator" | "round" | "wayfarer" | "clubmaster" | "cat-eye" | "none"
    frame_color : "original" | "black" | "gold" | "silver" | "tortoise"
    lens_tint   : "none" | "grey" | "brown" | "blue" | "green"
    landmarks   : dict from detect_landmarks(), or None for FFHQ fallback.

    Returns a new RGB PIL Image; *base* is not modified.
    """
    if style == "none":
        return base.copy()

    filename = _GLASSES_FILES.get(style)
    if filename is None:
        warnings.warn(f"[EchoFace] Unknown glasses style '{style}' — skipping")
        return base.copy()

    glasses_path = os.path.join(_ASSETS_DIR, filename)
    if not os.path.exists(glasses_path):
        warnings.warn(
            f"[EchoFace] Glasses asset not found: {glasses_path} — skipping"
        )
        return base.copy()

    result = overlay_glasses_with_landmarks(base, glasses_path, landmarks, frame_color)

    if lens_tint != "none":
        # Build a minimal landmark dict for apply_lens_tint when landmarks=None.
        lms_for_tint = landmarks if landmarks is not None else {
            "left_iris_center":        list(_FFHQ_L_EYE),
            "right_iris_center":       list(_FFHQ_R_EYE),
            "interpupillary_distance": _FFHQ_R_EYE[0] - _FFHQ_L_EYE[0],
        }
        result = apply_lens_tint(result, lms_for_tint, lens_tint)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Combined entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_accessories(base: Image.Image,
                      landmarks: dict | None = None,
                      glasses:     str = "none",
                      frame_color: str = "original",
                      lens_tint:   str = "none",
                      eye_color:   str = "none") -> Image.Image:
    """
    Apply accessories in compositing order starting from the clean base image:
      1. Eye colour (Lab iris chrominance shift via apply_eye_color)
      2. Glasses overlay (PNG alignment + colourisation + shadow + nose occlusion
         + lens tint — all handled inside apply_glasses)

    Parameters
    ----------
    base        : RGB PIL Image (1024×1024 FFHQ-aligned face) — never mutated.
    landmarks   : dict from detect_landmarks(), or None for FFHQ fallback.
    glasses     : "aviator" | "round" | "wayfarer" | "clubmaster" | "cat-eye" | "none"
    frame_color : "original" | "black" | "gold" | "silver" | "tortoise"
    lens_tint   : "none" | "grey" | "brown" | "blue" | "green"
    eye_color   : "blue" | "green" | "hazel" | "grey" | "brown" | "none"

    Returns a new RGB PIL Image; *base* is not modified.
    """
    img = apply_eye_color(base, eye_color, landmarks)
    img = apply_glasses(img, glasses, frame_color, lens_tint, landmarks)
    return img
