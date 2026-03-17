"""
glasses_overlay.py — Realistic glasses compositing using face landmarks.

Pipeline:
  1. Auto-crop glasses image to bounding box of dark frame pixels.
  2. Darkness-based alpha (skipped for pre-generated transparent PNGs).
  3. Scale using interpupillary distance from landmark cache.
  4. Frame colourisation (black / gold / silver / tortoise / original).
  5. Head-tilt rotation.
  6. Optional perspective warp for face yaw (requires cv2).
  7. Drop shadow.
  8. Nose-bridge occlusion (original nose pixels pasted back on top).

Public API
----------
overlay_glasses_with_landmarks(face_img, glasses_path, landmarks, frame_color) -> PIL.Image
colorize_frame(rgba_array, color_name) -> np.ndarray
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLASSES_DIR  = os.path.join(PROJECT_ROOT, "assets", "glasses")

_FRAME_COLOURS: dict[str, tuple[int, int, int] | None] = {
    "black":    (15,  15,  15),
    "gold":     (212, 175,  55),
    "silver":   (192, 192, 192),
    "original": None,
}


def _crop_to_frame(img: Image.Image, dark_thresh: int = 120, pad: int = 14) -> Image.Image:
    arr  = np.array(img.convert("RGB"), dtype=np.float32)
    dark = arr.mean(axis=2) < dark_thresh
    rows = np.any(dark, axis=1)
    cols = np.any(dark, axis=0)
    if not rows.any():
        return img
    h, w = arr.shape[:2]
    r0 = max(0,   int(np.argmax(rows)) - pad)
    r1 = min(h-1, int(len(rows) - 1 - np.argmax(rows[::-1])) + pad)
    c0 = max(0,   int(np.argmax(cols)) - pad)
    c1 = min(w-1, int(len(cols) - 1 - np.argmax(cols[::-1])) + pad)
    return img.crop((c0, r0, c1 + 1, r1 + 1))


def _make_frame_alpha(img: Image.Image) -> Image.Image:
    """Return RGBA. Skips darkness extraction if image is already RGBA."""
    if img.mode == "RGBA":
        return img.copy()
    rgb  = img.convert("RGB")
    arr  = np.array(rgb, dtype=np.float32)
    dark = 255.0 - arr.mean(axis=2)
    lo, hi = 25.0, 95.0
    alpha  = np.clip((dark - lo) / (hi - lo), 0.0, 1.0) * 255.0
    alpha_pil  = Image.fromarray(alpha.astype(np.uint8), mode="L")
    alpha_soft = alpha_pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    r, g, b = rgb.split()
    return Image.merge("RGBA", (r, g, b, alpha_soft))


def _make_drop_shadow(frames_rgba: Image.Image,
                      blur: int = 6, opacity: float = 0.28) -> Image.Image:
    arr    = np.array(frames_rgba, dtype=np.float32)
    shadow = np.zeros_like(arr)
    shadow[:, :, 3] = arr[:, :, 3] * opacity
    return Image.fromarray(shadow.astype(np.uint8), "RGBA").filter(
        ImageFilter.GaussianBlur(radius=blur)
    )


def _yaw_perspective_warp(glass_arr: np.ndarray, yaw: float) -> np.ndarray:
    if abs(yaw) < 0.01:
        return glass_arr
    try:
        import cv2
    except ImportError:
        warnings.warn(
            "[EchoFace] cv2 not available — yaw perspective warp skipped. "
            "Run: pip install opencv-python"
        )
        return glass_arr

    H, W   = glass_arr.shape[:2]
    shrink = max(1, int(abs(yaw) * 0.15 * W))
    src    = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    if yaw > 0:
        dst = np.float32([[shrink, 0], [W, 0], [W, H], [shrink, H]])
    else:
        dst = np.float32([[0, 0], [W - shrink, 0], [W - shrink, H], [0, H]])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(glass_arr, M, (W, H),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


def colorize_frame(rgba_array: np.ndarray, color_name: str) -> np.ndarray:
    """Replace RGB values of opaque frame pixels with color_name. Returns new array."""
    out  = rgba_array.copy()
    mask = out[:, :, 3] > 64

    if color_name == "original":
        return out

    if color_name == "tortoise":
        rng = np.random.default_rng()
        H, W = out.shape[:2]
        noise = rng.uniform(0.6, 1.0, size=(H, W))
        for ch, base in enumerate((120, 60, 20)):
            channel = out[:, :, ch].astype(np.float32)
            channel[mask] = np.clip(base * noise[mask], 0, 255)
            out[:, :, ch] = channel.astype(np.uint8)
        return out

    colour = _FRAME_COLOURS.get(color_name)
    if colour is None:
        warnings.warn(f"[EchoFace] Unknown frame color '{color_name}' — returning unchanged.")
        return out

    for ch, val in enumerate(colour):
        channel = out[:, :, ch].copy()
        channel[mask] = val
        out[:, :, ch] = channel
    return out


def overlay_glasses_with_landmarks(face_img:     Image.Image,
                                   glasses_path: str,
                                   landmarks:    dict | None,
                                   frame_color:  str = "original") -> Image.Image:
    """
    Overlay glasses PNG onto face_img using landmark-aligned placement.
    Falls back to FFHQ proportion estimates when landmarks is None.
    """
    face_img = face_img.convert("RGB")
    face_arr = np.array(face_img)
    fh, fw   = face_arr.shape[:2]

    if landmarks is not None:
        li   = landmarks["left_iris_center"]
        ri   = landmarks["right_iris_center"]
        left_outer  = (int(li[0]), int(li[1]))
        right_outer = (int(ri[0]), int(ri[1]))
        iris_mid_y  = (left_outer[1] + right_outer[1]) // 2
        face_cx     = (
            landmarks.get("left_face_edge",  [fw * 0.1])[0] +
            landmarks.get("right_face_edge", [fw * 0.9])[0]
        ) / 2.0
        yaw = (face_cx - fw / 2.0) / fw
    else:
        warnings.warn(
            "[EchoFace] landmark detection unavailable — "
            "using FFHQ fallback coordinates for glasses placement"
        )
        iris_mid_y  = int(fh * 0.40)
        left_outer  = (int(fw * 0.27), iris_mid_y)
        right_outer = (int(fw * 0.73), iris_mid_y)
        yaw         = 0.0

    eye_span = max(right_outer[0] - left_outer[0], 1)

    glasses_src = Image.open(glasses_path)
    if glasses_src.mode != "RGBA":
        glasses_src = _crop_to_frame(glasses_src.convert("RGB"))
    glasses_rgba = _make_frame_alpha(glasses_src)

    # The PNG canvas has lens centres at 28 % and 72 % of its width,
    # so the lens-to-lens fraction is 0.44.  Scale so that eye_span maps
    # exactly onto the inter-lens distance in the PNG.
    _LENS_CENTRE_FRAC = 0.44
    g_w    = min(int(eye_span / _LENS_CENTRE_FRAC), int(fw * 0.90))
    aspect = glasses_rgba.height / max(glasses_rgba.width, 1)
    g_h    = int(g_w * aspect)
    glasses_rgba = glasses_rgba.resize((g_w, g_h), Image.LANCZOS)

    if frame_color != "original":
        glasses_rgba = Image.fromarray(
            colorize_frame(np.array(glasses_rgba, dtype=np.uint8), frame_color), "RGBA"
        )

    dy    = right_outer[1] - left_outer[1]
    dx    = right_outer[0] - left_outer[0] + 1e-9
    angle = math.degrees(math.atan2(dy, dx))
    if abs(angle) > 0.5:
        glasses_rgba = glasses_rgba.rotate(
            -angle, expand=True, resample=Image.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

    if abs(yaw) > 0.01:
        glasses_rgba = Image.fromarray(
            _yaw_perspective_warp(np.array(glasses_rgba, dtype=np.uint8), yaw), "RGBA"
        )

    mid_x   = (left_outer[0] + right_outer[0]) // 2
    paste_x = max(0, min(mid_x - glasses_rgba.width  // 2, fw - glasses_rgba.width))
    paste_y = max(0, min(iris_mid_y - glasses_rgba.height // 2,
                         fh - glasses_rgba.height))

    face_rgba = face_img.convert("RGBA")
    shadow    = _make_drop_shadow(glasses_rgba)
    sx = max(0, min(paste_x,     fw - shadow.width))
    sy = max(0, min(paste_y + 3, fh - shadow.height))
    face_rgba.paste(shadow, (sx, sy), shadow)
    face_rgba.paste(glasses_rgba, (paste_x, paste_y), glasses_rgba)

    nose_cx = fw // 2
    nose_cy = iris_mid_y + int(fh * 0.015)
    rx_px   = max(int(fw * 0.035), 12)
    ry_px   = max(int(fh * 0.025), 8)
    yy, xx  = np.mgrid[0:fh, 0:fw]
    dist2   = ((xx - nose_cx) / rx_px) ** 2 + ((yy - nose_cy) / ry_px) ** 2
    nose_a  = np.clip((1.4 - dist2) / 1.4, 0.0, 1.0) * 255.0
    nose_a  = Image.fromarray(nose_a.astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=4)
    )
    orig_rgba          = np.array(face_img.convert("RGBA"), dtype=np.uint8)
    orig_rgba[:, :, 3] = np.array(nose_a, dtype=np.uint8)
    nose_pil           = Image.fromarray(orig_rgba, "RGBA")
    face_rgba.paste(nose_pil, (0, 0), nose_pil)

    return face_rgba.convert("RGB")
