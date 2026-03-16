"""STAMP feature extraction + heatmap wrappers for the GUI."""

import sys
import tempfile
from pathlib import Path

import openslide

STAMP_SRC = Path(__file__).resolve().parents[1] / "STAMP" / "src"
if str(STAMP_SRC) not in sys.path:
    sys.path.insert(0, str(STAMP_SRC))

from stamp.preprocessing import extract_
from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.tiling import _foreground_coords, _has_enough_texture, get_slide_mpp_
from stamp.heatmaps import heatmaps_
from stamp.types import Microns, SlideMPP, SlidePixels, TilePixels

from utils import overlay_tiles_on_thumb, wsi_thumbnail

EXTRACTOR_CHOICES = [
    "uni2", "h-optimus-0", "h-optimus-1",
    "conch", "conch1_5", "virchow2", "virchow",
    "gigapath", "ctranspath", "plip",
]


def stream_tile_preview(
    wsi_path: str,
    tile_size_um: float = 256.0,
    brightness_cutoff: int = 240,
    canny_cutoff: float = 0.02,
    thumb_max: int = 1024,
):
    """Generator: yields (thumb, n_tiles_so_far, slide_mpp, tile_grid_or_None).

    First yield  → bare thumbnail immediately after opening slide.
    Middle yields → tiles appear one batch at a time (brightness + Canny filtered).
    Final yield   → fully overlaid thumbnail + complete tile_grid dict.
    """
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        slide_mpp = get_slide_mpp_(slide, default_mpp=None)
        if slide_mpp is None:
            raise ValueError("Could not read MPP from slide. Set a default MPP in the STAMP config.")
        tile_size_slide_px = SlidePixels(int(tile_size_um / slide_mpp))
        thumb = wsi_thumbnail(slide, max_size=thumb_max)
        dims = slide.dimensions

        # Yield bare thumbnail immediately so the UI isn't blocked
        yield thumb.copy(), 0, float(slide_mpp), None

        # Brightness pass (fast)
        bright_coords = sorted(
            [(c.x, c.y) for c in _foreground_coords(slide, tile_size_slide_px, brightness_cutoff)],
            key=lambda c: (c[1], c[0]),
        )

        # Canny pass — read small tiles (128px) for speed, stream partial overlays
        check_px = 128
        best_level = slide.get_best_level_for_downsample(tile_size_slide_px / check_px)
        level_ds = slide.level_downsamples[best_level]
        region_px = max(1, int(tile_size_slide_px / level_ds))

        all_coords: list[tuple[int, int]] = []
        batch = max(1, len(bright_coords) // 50)
        for i, (x, y) in enumerate(bright_coords):
            tile_img = slide.read_region(
                (x, y), best_level, (region_px, region_px)
            ).convert("RGB").resize((check_px, check_px))
            if _has_enough_texture(tile_img, cutoff=canny_cutoff):
                all_coords.append((x, y))
            if (i + 1) % batch == 0:
                partial = overlay_tiles_on_thumb(thumb, all_coords, tile_size_slide_px, dims)
                yield partial, len(all_coords), float(slide_mpp), None

    finally:
        slide.close()

    # Final yield with complete overlay + tile_grid
    final_thumb = overlay_tiles_on_thumb(thumb, all_coords, tile_size_slide_px, dims)
    tile_grid = {
        "wsi_path": str(wsi_path),
        "slide_dims": dims,
        "tile_size_slide_px": int(tile_size_slide_px),
        "thumb_size": thumb.size,
        "coords": all_coords,
    }
    yield final_thumb, len(all_coords), float(slide_mpp), tile_grid


def extract_tile_at_click(
    click_x: int,
    click_y: int,
    tile_grid: dict,
    out_tile_px: int = 224,
):
    """Given a click on the thumbnail, extract the nearest foreground tile from the WSI.

    Returns (PIL Image, tile_key str) or raises ValueError if no tile found nearby.
    """
    import math

    sw, sh = tile_grid["slide_dims"]
    tw, th = tile_grid["thumb_size"]
    tile_size_slide_px = tile_grid["tile_size_slide_px"]
    coords = tile_grid["coords"]

    # Map thumbnail click → slide pixel space
    sx = click_x / tw * sw
    sy = click_y / th * sh

    # Snap to nearest tile grid position
    snapped_x = int(round(sx / tile_size_slide_px) * tile_size_slide_px)
    snapped_y = int(round(sy / tile_size_slide_px) * tile_size_slide_px)

    # Find nearest accepted foreground tile
    best = min(coords, key=lambda c: math.hypot(c[0] - snapped_x, c[1] - snapped_y))
    dist = math.hypot(best[0] - snapped_x, best[1] - snapped_y)
    if dist > tile_size_slide_px * 1.5:
        raise ValueError("No foreground tile near click location.")

    bx, by = best
    slide = openslide.OpenSlide(tile_grid["wsi_path"])
    try:
        region = slide.read_region((bx, by), 0, (tile_size_slide_px, tile_size_slide_px))
    finally:
        slide.close()

    img = region.convert("RGB").resize((out_tile_px, out_tile_px))
    tile_key = f"tile_x{bx}_y{by}"
    return img, tile_key


def extract_tile_at_slide_coords(
    slide_x: int,
    slide_y: int,
    tile_grid: dict,
    out_tile_px: int = 224,
):
    """Extract nearest foreground tile given a click in slide pixel coordinates.

    OSD provides coordinates in slide (image) space directly, so no thumbnail
    remapping is needed — simpler than extract_tile_at_click.
    """
    import math

    tile_size_slide_px = tile_grid["tile_size_slide_px"]
    coords = tile_grid["coords"]

    # floor division: snap to the tile that *contains* the click point
    snapped_x = int(slide_x // tile_size_slide_px) * tile_size_slide_px
    snapped_y = int(slide_y // tile_size_slide_px) * tile_size_slide_px

    best = min(coords, key=lambda c: math.hypot(c[0] - snapped_x, c[1] - snapped_y))
    dist = math.hypot(best[0] - snapped_x, best[1] - snapped_y)
    if dist > tile_size_slide_px * 1.5:
        raise ValueError("No foreground tile near click location.")

    bx, by = best
    slide = openslide.OpenSlide(tile_grid["wsi_path"])
    try:
        region = slide.read_region((bx, by), 0, (tile_size_slide_px, tile_size_slide_px))
    finally:
        slide.close()

    img = region.convert("RGB").resize((out_tile_px, out_tile_px))
    return img, f"tile_x{bx}_y{by}"


def run_extraction(
    wsi_path: str,
    extractor_name: str,
    output_dir: str,
    device: str = "cuda",
    tile_size_um: float = 256.0,
    tile_size_px: int = 224,
    brightness_cutoff: int = 240,
    canny_cutoff: float = 0.02,
    max_workers: int = 4,
):
    """Run STAMP feature extraction for a single WSI.

    Tiles are extracted at tile_size_px (default 224) from the WSI and cached
    to disk at 224px, matching the autoencoder's expected input size directly.

    Returns (h5_path, tile_cache_dir).
    """
    wsi_path = Path(wsi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_cache_dir = output_dir / "tile_cache"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        link = Path(tmp_dir) / wsi_path.name
        link.symlink_to(wsi_path.resolve())

        extract_(
            wsi_dir=Path(tmp_dir),
            output_dir=output_dir,
            wsi_list=None,
            cache_dir=tile_cache_dir,
            cache_tiles_ext="jpg",
            extractor=ExtractorName(extractor_name),
            tile_size_px=TilePixels(tile_size_px),
            tile_size_um=Microns(tile_size_um),
            max_workers=max_workers,
            device=device,
            default_slide_mpp=None,
            brightness_cutoff=brightness_cutoff,
            canny_cutoff=canny_cutoff,
            generate_hash=False,
        )

    # STAMP saves to output_dir/{extractor_id}/{slide_name}.h5
    h5_files = list(output_dir.glob("**/*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 file produced in {output_dir}")
    return h5_files[0], tile_cache_dir


def run_heatmap(
    wsi_path: str,
    h5_dir: str,
    ckpt_path: str,
    output_dir: str,
    device: str = "cuda",
    topk: int = 5,
    bottomk: int = 0,
    opacity: float = 0.5,
):
    """Generate STAMP heatmaps for a single WSI.

    Returns:
        overlay_paths  : list[Path] — per-class overlay PNGs
        top_tile_paths : list[Path] — top-k tile JPGs
    """
    wsi_path = Path(wsi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmaps_(
        feature_dir=Path(h5_dir),
        wsi_dir=wsi_path.parent,
        checkpoint_path=Path(ckpt_path),
        output_dir=output_dir,
        slide_paths=[wsi_path],
        device=device,
        default_slide_mpp=None,
        opacity=opacity,
        topk=topk,
        bottomk=bottomk,
    )

    overlay_paths = sorted(output_dir.rglob("plots/overlay-*.png"))
    top_tile_paths = sorted(output_dir.rglob("tiles/top_*.jpg"))
    return overlay_paths, top_tile_paths
