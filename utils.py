"""Shared helpers: thumbnails, tile overlays, base64 encoding for split viewer."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw


def wsi_thumbnail(slide, max_size: int = 1024) -> Image.Image:
    """Return a RGB thumbnail from an already-open openslide.OpenSlide."""
    thumb = slide.get_thumbnail((max_size, max_size))
    return thumb.convert("RGB")


def overlay_tiles_on_thumb(
    thumb: Image.Image,
    coords_px: list[tuple[int, int]],
    tile_size_slide_px: int,
    slide_dims: tuple[int, int],
    color: tuple[int, int, int, int] = (0, 210, 0, 100),
) -> Image.Image:
    """Draw semi-transparent green rectangles for each accepted tile on thumb.

    coords_px   : list of (x, y) in full-slide pixel coordinates
    slide_dims  : (width, height) of full slide in pixels
    """
    sw, sh = slide_dims
    tw, th = thumb.size
    scale_x = tw / sw
    scale_y = th / sh

    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    tsw = max(1, int(tile_size_slide_px * scale_x))
    tsh = max(1, int(tile_size_slide_px * scale_y))
    for x, y in coords_px:
        tx = int(x * scale_x)
        ty = int(y * scale_y)
        draw.rectangle([tx, ty, tx + tsw, ty + tsh], fill=color, outline=(0, 210, 0, 220))

    return Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


_SLIDER_SCRIPT = '<script type="module" src="https://unpkg.com/img-comparison-slider@8/dist/index.js"></script>'
_SLIDER_STYLE = """
<style>
  img-comparison-slider { --divider-width: 3px; --divider-color: #fff; outline: none; }
  .slider-block { display:inline-block; margin:6px; vertical-align:top; }
  .slider-label { font-family:sans-serif; font-size:11px; color:#333; margin-bottom:3px; text-align:center; }
</style>
"""


def comparison_slider_html(
    img_orig: Image.Image,
    img_manip: Image.Image,
    label: str = "",
    width: int = 400,
) -> str:
    """Return HTML for a before/after slider comparing original and counterfactual."""
    b64_orig = pil_to_base64(img_orig)
    b64_manip = pil_to_base64(img_manip)
    return (
        f'<div class="slider-block">'
        f'<div class="slider-label">{label}</div>'
        f'<img-comparison-slider style="width:{width}px;display:block">'
        f'  <img slot="first"  src="data:image/png;base64,{b64_orig}"  width="{width}">'
        f'  <img slot="second" src="data:image/png;base64,{b64_manip}" width="{width}">'
        f'</img-comparison-slider>'
        f"</div>"
    )


def sliders_row_html(
    orig_img: Image.Image,
    results: list[dict],
    categories: list[str],
    width: int = 380,
    probs_orig: list[float] | None = None,
    stoch_orig_img: Image.Image | None = None,
) -> str:
    """Build a row of per-amplitude comparison sliders."""
    blocks = [_SLIDER_SCRIPT, _SLIDER_STYLE]
    if probs_orig is not None:
        orig_prob_str = " | ".join(f"{categories[i]}: {probs_orig[i]:.2f}" for i in range(len(categories)))
        blocks.append(
            f'<div style="font-family:sans-serif;font-size:11px;color:#333;margin-bottom:4px">'
            f'Original: {orig_prob_str}</div>'
        )
    blocks.append('<div style="display:flex;flex-wrap:wrap;gap:4px">')
    for r in results:
        amp = r["amp"]
        prob_enc   = " | ".join(f"{categories[i]}: {r['probs_img'][i]:.2f}" for i in range(len(categories)))
        label_enc  = f"amp={amp:.1f} &nbsp; encoded noise &nbsp; img: {prob_enc}"
        blocks.append(comparison_slider_html(orig_img, r["manip_img"], label=label_enc, width=width))
        if stoch_orig_img is not None and "stoch_manip_img" in r:
            prob_stoch  = " | ".join(f"{categories[i]}: {r['probs_stoch_img'][i]:.2f}" for i in range(len(categories)))
            label_stoch = f"amp={amp:.1f} &nbsp; random noise &nbsp; img: {prob_stoch}"
            blocks.append(comparison_slider_html(stoch_orig_img, r["stoch_manip_img"], label=label_stoch, width=width))
    blocks.append("</div>")
    blocks.append('<hr style="border:none;border-top:1px solid #e0e0e8;margin:12px 0">')
    return "\n".join(blocks)
