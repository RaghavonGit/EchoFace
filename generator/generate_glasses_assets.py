"""
One-off script: generate 5 glasses frame PNGs with transparent backgrounds.
Run: python generator/generate_glasses_assets.py
Output: assets/glasses/{aviator,round,wayfarer,clubmaster,cat_eye}.png
"""
import math
import os

from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets", "glasses")
W, H    = 800, 300
FRAME   = (20, 20, 20)
TK      = 14


def _supersampled(fn, w=W, h=H):
    big = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    fn(big)
    return big.resize((w, h), Image.LANCZOS)


def _aviator():
    def _draw(img):
        d  = ImageDraw.Draw(img)
        W2, H2 = img.size
        hw = int(W2 * 0.20)
        hh_t = int(hw * 0.43)
        hh_b = int(hw * 0.80)
        cx_l, cx_r = int(W2 * 0.28), int(W2 * 0.72)
        cy = H2 // 2
        tk = TK * 2
        for cx in (cx_l, cx_r):
            steps = 120
            outer, inner = [], []
            for i in range(steps):
                a = 2 * math.pi * i / steps
                ca, sa = math.cos(a), math.sin(a)
                outer.append((cx + hw * ca, cy + (hh_t if sa < 0 else hh_b) * sa))
                inner.append((cx + (hw - tk) * ca,
                               cy + (max(4, hh_t - tk) if sa < 0 else max(4, hh_b - tk)) * sa))
            d.polygon(outer, fill=FRAME + (255,))
            d.polygon(inner, fill=(0, 0, 0, 0))
        bx1, bx2 = cx_l + hw - tk, cx_r - hw + tk
        by = cy
        d.line([(bx1, by), ((bx1+bx2)//2, by + tk*3), (bx2, by)],
               fill=FRAME + (255,), width=tk)
        d.line([(cx_l - hw, cy), (0, cy + tk)], fill=FRAME + (255,), width=tk)
        d.line([(cx_r + hw, cy), (W2, cy + tk)], fill=FRAME + (255,), width=tk)
    return _supersampled(_draw)


def _round():
    def _draw(img):
        d  = ImageDraw.Draw(img)
        W2, H2 = img.size
        r  = int(W2 * 0.18)
        cx_l, cx_r = int(W2 * 0.28), int(W2 * 0.72)
        cy = H2 // 2
        tk = TK * 2
        for cx in (cx_l, cx_r):
            d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=FRAME + (255,))
            d.ellipse([cx-r+tk, cy-r+tk, cx+r-tk, cy+r-tk], fill=(0, 0, 0, 0))
        bx1, bx2 = cx_l + r, cx_r - r
        d.line([(bx1, cy), ((bx1+bx2)//2, cy + tk*2), (bx2, cy)],
               fill=FRAME + (255,), width=tk)
        d.line([(cx_l - r, cy), (0, cy + tk)], fill=FRAME + (255,), width=tk)
        d.line([(cx_r + r, cy), (W2, cy + tk)], fill=FRAME + (255,), width=tk)
    return _supersampled(_draw)


def _wayfarer():
    def _draw(img):
        d  = ImageDraw.Draw(img)
        W2, H2 = img.size
        hw = int(W2 * 0.20)
        hh = int(hw * 0.70)
        cx_l, cx_r = int(W2 * 0.28), int(W2 * 0.72)
        cy = H2 // 2
        tk = TK * 2
        rnd = hw // 8
        for cx in (cx_l, cx_r):
            try:
                d.rounded_rectangle([cx-hw, cy-hh, cx+hw, cy+hh], radius=rnd,
                                    fill=FRAME + (255,))
                d.rounded_rectangle([cx-hw+tk, cy-hh+tk, cx+hw-tk, cy+hh-tk],
                                    radius=max(2, rnd-tk//2), fill=(0, 0, 0, 0))
            except AttributeError:
                d.rectangle([cx-hw, cy-hh, cx+hw, cy+hh], fill=FRAME + (255,))
                d.rectangle([cx-hw+tk, cy-hh+tk, cx+hw-tk, cy+hh-tk], fill=(0, 0, 0, 0))
        bx1, bx2 = cx_l + hw, cx_r - hw
        by = cy - hh // 4
        d.line([(bx1, by), ((bx1+bx2)//2, by + tk*2), (bx2, by)],
               fill=FRAME + (255,), width=tk)
        d.line([(cx_l - hw, cy), (0, cy + tk)], fill=FRAME + (255,), width=tk)
        d.line([(cx_r + hw, cy), (W2, cy + tk)], fill=FRAME + (255,), width=tk)
    return _supersampled(_draw)


def _clubmaster():
    def _draw(img):
        d  = ImageDraw.Draw(img)
        W2, H2 = img.size
        hw = int(W2 * 0.20)
        hh = int(hw * 0.70)
        cx_l, cx_r = int(W2 * 0.28), int(W2 * 0.72)
        cy = H2 // 2
        tk = TK * 2
        for cx in (cx_l, cx_r):
            d.rectangle([cx-hw, cy-hh, cx+hw, cy-hh//3], fill=FRAME + (255,))
            d.ellipse([cx-hw, cy-hh, cx+hw, cy+hh//2], outline=FRAME + (255,), width=tk)
        bx1, bx2 = cx_l + hw, cx_r - hw
        by = cy - hh // 3
        d.line([(bx1, by), ((bx1+bx2)//2, by + tk), (bx2, by)],
               fill=FRAME + (255,), width=tk)
        d.line([(cx_l - hw, cy - hh//2), (0, cy)], fill=FRAME + (255,), width=tk)
        d.line([(cx_r + hw, cy - hh//2), (W2, cy)], fill=FRAME + (255,), width=tk)
    return _supersampled(_draw)


def _cat_eye():
    def _draw(img):
        d  = ImageDraw.Draw(img)
        W2, H2 = img.size
        hw = int(W2 * 0.20)
        hh = int(hw * 0.65)
        cx_l, cx_r = int(W2 * 0.28), int(W2 * 0.72)
        cy = H2 // 2
        tk = TK * 2
        for cx, flip in ((cx_l, -1), (cx_r, 1)):
            d.ellipse([cx-hw, cy-hh, cx+hw, cy+hh//2], fill=FRAME + (255,))
            d.ellipse([cx-hw+tk, cy-hh+tk, cx+hw-tk, cy+hh//2-tk], fill=(0, 0, 0, 0))
            ox = cx + flip * hw
            d.polygon([
                (ox, cy - hh//2),
                (ox + flip * hw//3, cy - hh - hh//4),
                (ox + flip * hw//3 - flip*tk*2, cy - hh),
                (ox - flip*tk, cy - hh//2 + tk),
            ], fill=FRAME + (255,))
        bx1, bx2 = cx_l + hw, cx_r - hw
        by = cy - hh // 4
        d.line([(bx1, by), ((bx1+bx2)//2, by + tk), (bx2, by)],
               fill=FRAME + (255,), width=tk)
        d.line([(cx_l - hw, cy), (0, cy + tk)], fill=FRAME + (255,), width=tk)
        d.line([(cx_r + hw, cy), (W2, cy + tk)], fill=FRAME + (255,), width=tk)
    return _supersampled(_draw)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assets = {
        "aviator":    _aviator(),
        "round":      _round(),
        "wayfarer":   _wayfarer(),
        "clubmaster": _clubmaster(),
        "cat_eye":    _cat_eye(),
    }
    for name, img in assets.items():
        path = os.path.join(OUT_DIR, f"{name}.png")
        img.save(path)
        print(f"Saved: {path}  {img.size}")


if __name__ == "__main__":
    main()
