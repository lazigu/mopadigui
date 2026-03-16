"""
FastAPI DeepZoom tile server — mounted on the same port as Gradio.
Endpoints:
  GET /dzi?wsi=<path>               → DZI XML descriptor
  GET /tile/{level}/{col}_{row}.jpeg?wsi=<path>  → tile JPEG
  GET /image?path=<path>            → serve any local image (heatmap overlay)
"""
import io
import threading
from pathlib import Path

import openslide
from openslide.deepzoom import DeepZoomGenerator
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

tile_app = FastAPI()
tile_app.add_middleware(CORSMiddleware, allow_origins=["*"])

_cache: dict[str, DeepZoomGenerator] = {}
_lock = threading.Lock()


def _get_dz(wsi_path: str) -> DeepZoomGenerator:
    with _lock:
        if wsi_path not in _cache:
            slide = openslide.OpenSlide(wsi_path)
            _cache[wsi_path] = DeepZoomGenerator(
                slide, tile_size=256, overlap=1, limit_bounds=True
            )
        return _cache[wsi_path]


@tile_app.get("/dzi")
def dzi_info(wsi: str = Query(...)):
    try:
        dz = _get_dz(wsi)
        return Response(content=dz.get_dzi("jpeg"), media_type="application/xml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@tile_app.get("/tile/{level}/{col_row}.jpeg")
def get_tile(level: int, col_row: str, wsi: str = Query(...)):
    try:
        col, row = map(int, col_row.split("_"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid tile coordinates")
    try:
        dz = _get_dz(wsi)
        img = dz.get_tile(level, (col, row))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@tile_app.get("/image")
def serve_image(path: str = Query(...)):
    """Serve a local image file — used to display heatmap overlays in OSD."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return Response(content=p.read_bytes(), media_type=media_type)
