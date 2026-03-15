"""
generator/accessories.py

Accessory overlays for EchoFace — glasses frames and eye colour shifts.
All public functions are pure: the input PIL image is never mutated.

Landmark strategy
-----------------
FFHQ alignment places every 1024×1024 face identically:
  Left  iris centre : (340, 480)
  Right iris centre : (684, 480)
These coordinates are hard-coded constants derived from the FFHQ alignment
script and hold for all images produced by the FFHQ StyleGAN2-ADA model.

Glasses rendering
-----------------
Frames are drawn programmatically at 2× resolution (2048×2048) then
LANCZOS-downsampled to 1024×1024 to get smooth, anti-aliased edges.
Each lens is rendered as two overlapping fills:
  1. A fully-opaque dark ring  → the physical frame
  2. A semi-transparent tint   → the glass lens interior  (alpha ≈ 11 %)

Eye colour
----------
Iris chrominance (a, b in CIE Lab) is replaced inside a soft circular mask
centred on each iris.  The L (lightness) channel is preserved pixel-by-pixel
so natural lighting is retained.  Dynamic pupil detection (darkest-cluster
centroid) is used to find the actual iris centre before colouring.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

# ── FFHQ 1024×1024 landmark constants ────────────────────────────────────────

_SZ       = 1024
_L_EYE    = (340, 480)   # left  iris centre (x, y)
_R_EYE    = (684, 480)   # right iris centre (x, y)
_EYE_SPAN = _R_EYE[0] - _L_EYE[0]   # 344 px  centre-to-centre
_IRIS_R   = 26           # iris colour-shift radius (px)

# ── Glasses drawing constants ─────────────────────────────────────────────────

_SCALE     = 2                      # supersampling factor
_FRAME_CLR = (20, 20, 20, 255)      # near-black, fully opaque
_LENS_TINT = (90, 100, 120, 28)     # cool-grey tint, ~11 % opacity


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lens_hw() -> int:
    """Half-width of each lens at 1× (px).  ≈ 127 px."""
    return int(_EYE_SPAN * 0.37)


def _aviator_pts(cx: float, cy: float,
                 hw: float, hh_top: float, hh_bot: float,
                 steps: int = 120) -> list[tuple[float, float]]:
    """
    Polygon vertices for one aviator / teardrop lens.
    Wider at the bottom (hh_bot > hh_top), circular at the top.
    """
    pts = []
    for i in range(steps):
        angle = 2.0 * math.pi * i / steps
        ca, sa = math.cos(angle), math.sin(angle)
        ex = cx + hw * ca
        ey = cy + (hh_top if sa < 0.0 else hh_bot) * sa
        pts.append((ex, ey))
    return pts


# ── Search radius for dynamic pupil detection ─────────────────────────────────
_SEARCH_R = 60   # px — window around expected eye position to scan


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
    # Using min+20 instead of a fixed percentile so even a small dark patch
    # (12×12 pixels in a 120×120 window = 1 %) is reliably detected.
    thresh = min_val + 20.0
    mask = grey <= thresh

    ys_local, xs_local = np.where(mask)

    if len(xs_local) == 0:
        return expected_cx, expected_cy

    # Centroid of dark pixels → iris centre
    cx = int(round(xs_local.mean())) + x0
    cy = int(round(ys_local.mean())) + y0
    return cx, cy


# ── CIE Lab colour-space conversion (pure numpy, D65 illuminant) ──────────────

def _rgb_to_lab_np(rgb_f: np.ndarray) -> np.ndarray:
    """
    sRGB float32 [..., 3] in [0, 1]  →  CIE Lab float32 [..., 3].
    L in [0, 100],  a/b in roughly [-128, 128].
    """
    # Step 1: gamma expand (linearise)
    mask = rgb_f > 0.04045
    lin  = np.where(mask,
                    ((rgb_f + 0.055) / 1.055) ** 2.4,
                    rgb_f / 12.92)

    # Step 2: linear sRGB → XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = lin @ M.T                                   # [..., 3]

    # Step 3: normalise by D65 white point
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz   = xyz / white

    # Step 4: f function
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
    """
    CIE Lab float32 [..., 3]  →  sRGB float32 [..., 3] clipped to [0, 1].
    """
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

    # XYZ → linear sRGB
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    lin = np.clip(xyz @ M_inv.T, 0.0, 1.0)

    # Gamma compress
    mask = lin > 0.0031308
    rgb  = np.where(mask,
                    1.055 * lin ** (1.0 / 2.4) - 0.055,
                    12.92 * lin)
    return np.clip(rgb, 0.0, 1.0)


# ── Eye-colour Lab target values ──────────────────────────────────────────────
# Stored as (a, b) in CIE Lab.  L (lightness) is always taken from the
# original pixel so natural iris shading is preserved.
_EYE_COLOURS: dict[str, tuple[float, float]] = {
    "blue":  (-15.0, -55.0),
    "green": (-38.0,  28.0),
    "hazel": ( 12.0,  32.0),
    "grey":  ( -3.0,  -8.0),
    "brown": ( 18.0,  34.0),
}


def _shift_iris_lab(arr: np.ndarray,
                    cx: int, cy: int, r: int,
                    a_tgt: float, b_tgt: float) -> np.ndarray:
    """
    Replace the chrominance (a, b in Lab) of one iris while preserving
    per-pixel lightness (L).  Uses a soft circular alpha mask with quadratic
    falloff.

    Parameters
    ----------
    arr   : (H, W, 3) uint8 RGB array — not mutated.
    cx,cy : iris centre (pixels).
    r     : iris radius (pixels).
    a_tgt, b_tgt : target Lab chrominance values.

    Returns a new uint8 array.
    """
    y0 = max(0, cy - r);  y1 = min(arr.shape[0], cy + r)
    x0 = max(0, cx - r);  x1 = min(arr.shape[1], cx + r)

    patch_u8  = arr[y0:y1, x0:x1]                         # (h, w, 3) uint8
    patch_f   = patch_u8.astype(np.float32) / 255.0       # [0, 1]

    # Convert to Lab
    lab = _rgb_to_lab_np(patch_f)                          # (h, w, 3)

    # Build target Lab: keep L, replace a and b
    lab_tgt       = lab.copy()
    lab_tgt[..., 1] = a_tgt
    lab_tgt[..., 2] = b_tgt

    # Convert back to RGB
    rgb_tgt = _lab_to_rgb_np(lab_tgt)                     # (h, w, 3) float

    # Soft circular alpha mask (quadratic falloff)
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    xx, yy  = np.meshgrid(xs, ys)
    dist    = np.sqrt(xx ** 2 + yy ** 2)
    alpha   = np.clip(1.0 - dist / r, 0.0, 1.0) ** 1.8   # (h, w)
    alpha3  = alpha[:, :, np.newaxis]                      # (h, w, 1)

    blended = patch_f * (1.0 - alpha3) + rgb_tgt * alpha3  # (h, w, 3)

    out = arr.copy()
    out[y0:y1, x0:x1] = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _draw_glasses_2x(style: str) -> Image.Image:
    """
    Render glasses on a transparent 2048×2048 RGBA canvas using the chosen
    style.  Returns the raw RGBA image before downsampling.
    """
    S  = _SZ * _SCALE                              # 2048
    hw = _lens_hw() * _SCALE                       # ≈ 254 px at 2×
    tk = 16                                        # frame ring thickness at 2×
    fc = _FRAME_CLR
    lc = _LENS_TINT

    lx = _L_EYE[0] * _SCALE   # 680
    ly = _L_EYE[1] * _SCALE   # 820
    rx = _R_EYE[0] * _SCALE   # 1368
    ry = _R_EYE[1] * _SCALE   # 820

    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(canvas)

    # ── Lens bodies ───────────────────────────────────────────────────────────
    if style == "round":
        hh = hw
        for cx, cy in ((lx, ly), (rx, ry)):
            draw.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=fc)
            draw.ellipse([cx - hw + tk, cy - hh + tk,
                          cx + hw - tk, cy + hh - tk], fill=lc)

    elif style == "square":
        hh  = int(hw * 0.68)
        rnd = max(8, hw // 10)
        for cx, cy in ((lx, ly), (rx, ry)):
            try:
                draw.rounded_rectangle(
                    [cx - hw, cy - hh, cx + hw, cy + hh],
                    radius=rnd, fill=fc)
                draw.rounded_rectangle(
                    [cx - hw + tk, cy - hh + tk,
                     cx + hw - tk, cy + hh - tk],
                    radius=max(2, rnd - tk // 2), fill=lc)
            except AttributeError:
                # Pillow < 8.2 safety fallback (not expected on this project)
                draw.rectangle([cx - hw, cy - hh, cx + hw, cy + hh], fill=fc)
                draw.rectangle([cx - hw + tk, cy - hh + tk,
                                cx + hw - tk, cy + hh - tk], fill=lc)

    elif style == "aviator":
        hh_top = int(hw * 0.43)
        hh_bot = int(hw * 0.80)
        for cx, cy in ((lx, ly), (rx, ry)):
            outer = _aviator_pts(cx, cy, hw,       hh_top,       hh_bot)
            inner = _aviator_pts(cx, cy, hw - tk,
                                 max(4, hh_top - tk), max(4, hh_bot - tk))
            draw.polygon(outer, fill=fc)
            draw.polygon(inner, fill=lc)

    else:
        # Unknown style — return blank overlay
        return Image.new("RGBA", (_SZ, _SZ), (0, 0, 0, 0))

    # ── Nose bridge ───────────────────────────────────────────────────────────
    # Three-segment gentle curve: ramps up from each lens, flat at centre.
    bx1   = lx + hw                            # inner right edge of left lens
    bx2   = rx - hw                            # inner left  edge of right lens
    by    = ly + tk // 2                       # just below lens centre
    mid_x = (bx1 + bx2) // 2
    drop  = tk * 3                             # how far the bridge dips down

    draw.line([(bx1, by),               (mid_x - tk * 2, by + drop)],
              fill=fc, width=tk)
    draw.line([(mid_x - tk * 2, by + drop), (mid_x + tk * 2, by + drop)],
              fill=fc, width=tk)
    draw.line([(mid_x + tk * 2, by + drop), (bx2, by)],
              fill=fc, width=tk)

    # ── Temple arms ───────────────────────────────────────────────────────────
    # Start at the outer lens edge (same height as lens centre) and run nearly
    # horizontal to the canvas edge with a gentle downward slope toward the ear.
    arm_oy    = ly                             # same height as lens centre
    arm_slope = tk                             # tiny downward drop to canvas edge

    draw.line([(lx - hw, arm_oy), (0,  arm_oy + arm_slope)], fill=fc, width=tk)
    draw.line([(rx + hw, arm_oy), (S, arm_oy + arm_slope)], fill=fc, width=tk)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Public glasses API
# ─────────────────────────────────────────────────────────────────────────────

def apply_glasses(base: Image.Image, style: str) -> Image.Image:
    """
    Composite a glasses frame overlay onto a 1024×1024 FFHQ-aligned face.

    Parameters
    ----------
    base  : RGB PIL Image (1024×1024).
    style : "round" | "square" | "aviator" | "none"

    Returns a new RGB PIL Image; *base* is not modified.
    """
    if style == "none":
        return base.copy()

    overlay_2x = _draw_glasses_2x(style)
    overlay    = overlay_2x.resize((_SZ, _SZ), Image.LANCZOS)
    face_rgba  = base.convert("RGBA")
    return Image.alpha_composite(face_rgba, overlay).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Eye colour
# ─────────────────────────────────────────────────────────────────────────────

def apply_eye_color(base: Image.Image, color: str) -> Image.Image:
    """
    Shift the iris colour on both eyes of a 1024×1024 FFHQ-aligned face.

    Uses dynamic pupil detection (darkest-cluster centroid) to find the actual
    iris centre, then replaces chrominance in CIE Lab space so that dark irises
    receive a visible colour shift (which the old HSV approach could not do).

    Parameters
    ----------
    base  : RGB PIL Image (1024×1024).
    color : "blue" | "green" | "hazel" | "grey" | "brown" | "none"

    Returns a new RGB PIL Image; *base* is not modified.
    """
    if color == "none" or color not in _EYE_COLOURS:
        return base.copy()

    a_tgt, b_tgt = _EYE_COLOURS[color]
    arr = np.array(base.convert("RGB"), dtype=np.uint8)

    for (ex, ey) in (_L_EYE, _R_EYE):
        cx, cy = _find_iris_centre(arr, ex, ey, _SEARCH_R)
        arr = _shift_iris_lab(arr, cx, cy, _IRIS_R, a_tgt, b_tgt)

    return Image.fromarray(arr, "RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Combined entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_accessories(base: Image.Image,
                      glasses:   str = "none",
                      eye_color: str = "none") -> Image.Image:
    """
    Apply accessories in the correct layering order:
      1. Eye colour first  (visible through the lens)
      2. Glasses frame on top

    Parameters
    ----------
    base      : RGB PIL Image (1024×1024 FFHQ-aligned face).
    glasses   : "round" | "square" | "aviator" | "none"
    eye_color : "blue"  | "green"  | "hazel"   | "grey" | "brown" | "none"

    Returns a new RGB PIL Image; *base* is not modified.
    """
    img = apply_eye_color(base, eye_color)
    img = apply_glasses(img, glasses)
    return img
