"""16:9 LinkedIn article cover — indices and Treasury yields."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "indices-yields-cover.png"

W, H = 1920, 1080
SCALE = 2
SW, SH = W * SCALE, H * SCALE

BG = (8, 14, 28)
TEXT = (244, 246, 248)
AMBER = (224, 122, 61)

HN = "/System/Library/Fonts/HelveticaNeue.ttc"


def _hn(size: int, *, index: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(HN, size * SCALE, index=index)
    except OSError:
        return ImageFont.load_default()


def _glow(size: tuple[int, int], cx: float, cy: float, radius: float, color: tuple[int, int, int], alpha: int) -> Image.Image:
    w, h = size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    a = np.clip(1.0 - d, 0, 1) ** 2.2
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = (a * alpha).astype(np.uint8)
    return Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=int(80 * SCALE / 2)))


def _grain(size: tuple[int, int], amount: int = 12) -> Image.Image:
    w, h = size
    noise = np.random.default_rng(11).integers(0, amount, (h, w), dtype=np.uint8)
    arr = np.dstack([noise, noise, noise, np.full((h, w), 22, dtype=np.uint8)])
    return Image.fromarray(arr, "RGBA")


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _curve(draw: ImageDraw.ImageDraw) -> None:
    """A quiet par curve on the right. Tenor runs left to right, yield runs up."""
    pts = []
    x0, x1 = int(SW * 0.62), int(SW * 0.90)
    y_top, y_bot = int(SH * 0.70), int(SH * 0.84)
    for i in range(64):
        t = i / 63
        # Front end a bit high, belly dips, long end rises.
        yld = 0.55 - 0.35 * np.sin(t * np.pi) + 0.40 * t
        x = x0 + t * (x1 - x0)
        y = y_bot - yld * (y_bot - y_top)
        pts.append((x, y))
    draw.line(pts, fill=AMBER + (180,), width=3 * SCALE)


def build_cover() -> Image.Image:
    base = Image.new("RGB", (SW, SH), BG).convert("RGBA")
    base = Image.alpha_composite(base, _glow((SW, SH), SW * 0.22, SH * 0.30, SW * 0.34, AMBER, 42))
    base = Image.alpha_composite(base, _glow((SW, SH), SW * 0.78, SH * 0.78, SW * 0.36, (28, 64, 112), 70))
    base = Image.alpha_composite(base, _grain((SW, SH)))

    draw = ImageDraw.Draw(base)
    _curve(draw)
    kicker_font = _hn(22, index=1)
    thin = _hn(84, index=12)
    bold = _hn(84, index=1)

    kicker = "US MARKETS"
    line1 = "Major indices"
    line2 = "Treasury yields"
    kicker_h = kicker_font.getbbox(kicker)[3] - kicker_font.getbbox(kicker)[1]
    h1 = thin.getbbox(line1)[3] - thin.getbbox(line1)[1]
    h2 = bold.getbbox(line2)[3] - bold.getbbox(line2)[1]
    gap_kicker = 36 * SCALE
    gap_title = 16 * SCALE
    block = kicker_h + gap_kicker + h1 + gap_title + h2
    y0 = (SH - block) // 2 - 20 * SCALE

    draw.text(((SW - _text_width(draw, kicker, kicker_font)) // 2, y0), kicker, font=kicker_font, fill=AMBER)
    y1 = y0 + kicker_h + gap_kicker
    draw.text(((SW - _text_width(draw, line1, thin)) // 2, y1), line1, font=thin, fill=TEXT)
    draw.text(((SW - _text_width(draw, line2, bold)) // 2, y1 + h1 + gap_title), line2, font=bold, fill=TEXT)

    return base.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)


def main() -> None:
    cover = build_cover()
    cover.save(OUT, quality=96, subsampling=0)
    print(f"Wrote {OUT} ({cover.size[0]}x{cover.size[1]})")


if __name__ == "__main__":
    main()
