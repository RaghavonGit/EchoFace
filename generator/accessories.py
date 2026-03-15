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
A vectorised NumPy hue-shift is applied inside a soft circular mask centred
on each iris.  The V (brightness) channel is preserved so natural lighting
is retained; only H and S are replaced with the target colour values.
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

# Target (hue, saturation) in [0, 1].  Value (brightness) is always preserved
# from the original pixel so lighting looks natural.
_EYE_COLOURS: dict[str, tuple[float, float]] = {
    "blue":  (210 / 360, 0.62),
    "green": (130 / 360, 0.55),
    "hazel": ( 35 / 360, 0.52),
    "grey":  (200 / 360, 0.16),
    "brown": ( 25 / 360, 0.68),
}


def _hsv_to_rgb_np(h: np.ndarray, s: np.ndarray,
                   v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully vectorised HSV → RGB conversion.
    All three inputs must be numpy arrays of identical shape, values in [0, 1].
    """
    h6 = h * 6.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t  = v * (1.0 - s * (1.0 - f))

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [v,      q,      p,      p,      t,      v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [t,      v,      v,      q,      p,      p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [p,      p,      t,      v,      v,      q])
    return r, g, b


def _shift_iris(arr: np.ndarray,
                cx: int, cy: int, r: int,
                h_tgt: float, s_tgt: float) -> np.ndarray:
    """
    Shift the hue of a single iris region on a (H, W, 3) uint8 array.

    Uses a soft circular mask (quadratic falloff from the centre) so the
    colour change fades smoothly toward the edge of the iris, avoiding hard
    boundaries.  The V (brightness) channel is preserved pixel-by-pixel.

    Returns a *new* array; the input is not mutated.
    """
    y0 = max(0, cy - r);  y1 = min(arr.shape[0], cy + r)
    x0 = max(0, cx - r);  x1 = min(arr.shape[1], cx + r)

    patch  = arr[y0:y1, x0:x1].astype(np.float32) / 255.0   # (h, w, 3)

    # Soft circular alpha mask ─────────────────────────────────────────────
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    xx, yy = np.meshgrid(xs, ys)
    dist   = np.sqrt(xx ** 2 + yy ** 2)
    alpha  = np.clip(1.0 - dist / r, 0.0, 1.0) ** 1.8   # quadratic falloff
    alpha3 = alpha[:, :, np.newaxis]                       # (h, w, 1)

    # Preserve per-pixel brightness ────────────────────────────────────────
    R, G, B = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    V = np.maximum(np.maximum(R, G), B)

    # Build target colour at (h_tgt, s_tgt, original V) ───────────────────
    H_t = np.full_like(V, h_tgt)
    S_t = np.full_like(V, s_tgt)
    r_t, g_t, b_t = _hsv_to_rgb_np(H_t, S_t, V)
    target = np.stack([r_t, g_t, b_t], axis=2)

    blended = patch * (1.0 - alpha3) + target * alpha3

    out              = arr.copy()
    out[y0:y1, x0:x1] = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return out


def apply_eye_color(base: Image.Image, color: str) -> Image.Image:
    """
    Shift the iris hue on both eyes of a 1024×1024 FFHQ-aligned face.

    Parameters
    ----------
    base  : RGB PIL Image (1024×1024).
    color : "blue" | "green" | "hazel" | "grey" | "brown" | "none"

    Returns a new RGB PIL Image; *base* is not modified.
    """
    if color == "none" or color not in _EYE_COLOURS:
        return base.copy()

    h_tgt, s_tgt = _EYE_COLOURS[color]
    arr = np.array(base.convert("RGB"), dtype=np.uint8)
    arr = _shift_iris(arr, _L_EYE[0], _L_EYE[1], _IRIS_R, h_tgt, s_tgt)
    arr = _shift_iris(arr, _R_EYE[0], _R_EYE[1], _IRIS_R, h_tgt, s_tgt)
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
