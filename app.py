"""STAMP + MoPaDi GUI — FastAPI + Jinja2."""

import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openslide
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from starlette.requests import Request

# ── path setup ────────────────────────────────────────────────────────────────
GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUI_DIR))

from dotenv import load_dotenv
load_dotenv(GUI_DIR / ".env")

import mopadi_runner
import stamp_runner
from utils import pil_to_base64, sliders_row_html

# ── app ───────────────────────────────────────────────────────────────────────
app  = FastAPI()
tmpl = Jinja2Templates(directory=str(GUI_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(GUI_DIR)), name="static")

# ── DZI tile server ───────────────────────────────────────────────────────────
_dz_cache: dict[str, DeepZoomGenerator] = {}
_dz_lock  = threading.Lock()


def _get_dz(wsi_path: str) -> DeepZoomGenerator:
    with _dz_lock:
        if wsi_path not in _dz_cache:
            slide = openslide.OpenSlide(wsi_path)
            _dz_cache[wsi_path] = DeepZoomGenerator(
                slide, tile_size=256, overlap=1, limit_bounds=True
            )
        return _dz_cache[wsi_path]


@app.get("/current.dzi")
def current_dzi():
    """DZI descriptor for the currently loaded slide.
    OSD constructs tile URLs as /current_files/{level}/{col}_{row}.jpeg — handled below."""
    wsi_path = _state.get("wsi_path")
    if not wsi_path:
        raise HTTPException(status_code=404, detail="No slide loaded yet")
    try:
        dz = _get_dz(wsi_path)
        return Response(content=dz.get_dzi("jpeg"), media_type="application/xml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/current_files/{level}/{col_row}.jpeg")
def current_tile(level: int, col_row: str):
    """Tile endpoint matching the standard DZI URL pattern OSD expects."""
    wsi_path = _state.get("wsi_path")
    if not wsi_path:
        raise HTTPException(status_code=404, detail="No slide loaded yet")
    try:
        col, row = map(int, col_row.split("_"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid tile coordinates")
    try:
        dz  = _get_dz(wsi_path)
        img = dz.get_tile(level, (col, row))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/image")
def serve_image(path: str = Query(...)):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    mt = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return Response(content=p.read_bytes(), media_type=mt)


# ── CF log (thread-safe, polled by frontend) ─────────────────────────────────
_cf_log: list[str] = []
_cf_log_lock = threading.Lock()

def _cf_log_fn(msg: str) -> None:
    print(msg)
    with _cf_log_lock:
        _cf_log.append(msg)

@app.get("/api/cf-log")
def api_cf_log():
    with _cf_log_lock:
        msgs = _cf_log.copy()
        _cf_log.clear()
    return {"messages": msgs}


# ── server-side session state (single-user research tool) ────────────────────
_state: dict = {
    "tile_grid":      None,
    "h5_path":        None,
    "tile_cache":     None,
    "heatmap_path":   None,
    "top_tile_paths": [],
    "selected_tiles": [],
    "n_tiles":        0,
    "wsi_path":       None,
}

# ── main page ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return tmpl.TemplateResponse("index.html", {"request": request})


MOPADI_CONFIG_PATH = os.environ["MOPADI_CONFIG_PATH"]

CLASSIFIER_CHOICES = {
    "TCGA-CRC BRAF (Virchow2)": {
        "ckpt":        os.environ["CLASSIFIER_BRAF_CKPT"],
        "target_dict": {"MUT": 0, "WT": 1},
    },
    "TCGA-CRC MSI (Virchow2)": {
        "ckpt":        os.environ["CLASSIFIER_MSI_CKPT"],
        "target_dict": {"MSIH": 0, "nonMSIH": 1},
    },
}


@app.get("/api/config")
def get_config():
    return {
        "extractor_choices":   stamp_runner.EXTRACTOR_CHOICES,
        "classifier_choices":  CLASSIFIER_CHOICES,
        "example_wsi_path":    os.environ.get("EXAMPLE_WSI_PATH", ""),
        "defaults": {
            "device":            "cuda:0",
            "tile_size_um":      256.0,
            "brightness_cutoff": 230,
            "output_dir":        str(GUI_DIR.parent / "gui_output"),
            "T_inv":             200,
            "T_step":            100,
        },
    }


# ── streaming tile preview (SSE) ──────────────────────────────────────────────
@app.get("/api/tile-preview")
def api_tile_preview(
    wsi_path:          str   = Query(...),
    tile_size_um:      float = Query(256.0),
    brightness_cutoff: int   = Query(240),
    canny_cutoff:      float = Query(0.02),
):
    def generate():
        _state["tile_grid"] = None
        _state["n_tiles"]   = 0
        try:
            for thumb, n, mpp, tg in stamp_runner.stream_tile_preview(
                wsi_path,
                tile_size_um=tile_size_um,
                brightness_cutoff=brightness_cutoff,
                canny_cutoff=canny_cutoff,
            ):
                _state["n_tiles"] = n
                payload: dict = {
                    "thumb":   pil_to_base64(thumb, fmt="JPEG"),
                    "n_tiles": n,
                    "mpp":     mpp,
                }
                if tg is not None:
                    _state["tile_grid"] = tg
                    _state["wsi_path"]  = wsi_path
                    payload["done"]      = True
                    payload["tile_grid"] = {
                        "coords":            tg["coords"],
                        "tile_size_slide_px": tg["tile_size_slide_px"],
                        "slide_dims":         list(tg["slide_dims"]),
                    }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── feature extraction (SSE) ──────────────────────────────────────────────────
@app.get("/api/extract")
def api_extract(
    wsi_path:          str   = Query(...),
    extractor:         str   = Query("uni2"),
    device:            str   = Query("cuda:0"),
    tile_size_um:      float = Query(256.0),
    brightness_cutoff: int   = Query(240),
    canny_cutoff:      float = Query(0.02),
    output_dir:        str   = Query(...),
):
    def generate():
        result_q: queue.Queue = queue.Queue()
        h5_dir    = Path(output_dir) / "features"
        cache_dir = h5_dir / "tile_cache"

        def do_extract():
            try:
                h5, cache = stamp_runner.run_extraction(
                    wsi_path=wsi_path,
                    extractor_name=extractor,
                    output_dir=str(h5_dir),
                    device=device,
                    tile_size_um=tile_size_um,
                    brightness_cutoff=brightness_cutoff,
                    canny_cutoff=canny_cutoff,
                )
                result_q.put(("done", str(h5), str(cache)))
            except Exception as e:
                result_q.put(("error", str(e)))

        t = threading.Thread(target=do_extract, daemon=True)
        t.start()

        start = time.time()
        while t.is_alive():
            yield f"data: {json.dumps({'running': True, 'elapsed': int(time.time() - start)})}\n\n"
            time.sleep(1.0)

        result = result_q.get()
        if result[0] == "done":
            _state["h5_path"]    = result[1]
            _state["tile_cache"] = result[2]
            yield f"data: {json.dumps({'pct': 1.0, 'done': True, 'h5_path': result[1], 'tile_cache': result[2]})}\n\n"
        else:
            yield f"data: {json.dumps({'error': result[1]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── heatmap ───────────────────────────────────────────────────────────────────
@app.post("/api/heatmap")
def api_heatmap(
    wsi_path:        str = Form(...),
    classifier_name: str = Form(...),
    output_dir:      str = Form(...),
    device:          str = Form("cuda:0"),
):
    clf = CLASSIFIER_CHOICES.get(classifier_name)
    if not clf:
        raise HTTPException(status_code=400, detail=f"Unknown classifier: {classifier_name!r}")
    stamp_ckpt = clf["ckpt"]

    # Start preloading MoPaDi while heatmap runs (5 copies = max top tiles)
    mopadi_runner.start_preload(MOPADI_CONFIG_PATH, stamp_ckpt, n=3, device=device)

    h5_dir      = str(Path(_state["h5_path"]).parent) if _state.get("h5_path") else str(Path(output_dir) / "features")
    heatmap_out = Path(output_dir) / "heatmaps"
    try:
        overlay_paths, top_tile_paths = stamp_runner.run_heatmap(
            wsi_path=wsi_path,
            h5_dir=h5_dir,
            ckpt_path=stamp_ckpt,
            output_dir=str(heatmap_out),
            device=device,
            topk=5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # use the pure RGBA score image (RdBu_r, NEAREST upscale, transparent bg)
    import re as _re
    import numpy as _np
    from PIL import Image as _Image
    score_imgs = sorted(heatmap_out.rglob("raw/*-MUT=*.png"))
    if not score_imgs:
        score_imgs = sorted(heatmap_out.rglob("raw/*=*.png"))
    heatmap_path  = str(score_imgs[0]) if score_imgs else None
    top_tile_strs = [str(p) for p in top_tile_paths]
    _state["heatmap_path"]   = heatmap_path
    _state["top_tile_paths"] = top_tile_strs

    # match STAMP top-tile JPEGs against the tile cache via nearest-neighbour MSE
    top_tile_coords: list[list[int]] = []
    if top_tile_paths:
        try:
            import zipfile as _zf, io as _io, openslide as _osl2
            thumb_sz = (32, 32)
            cache_dir = Path(output_dir) / "features" / "tile_cache"
            cache_zips = list(cache_dir.glob(f"{Path(wsi_path).stem}*.zip"))
            print(f"[top_tile_coords] top_tile_paths={len(top_tile_paths)} cache_dir={cache_dir} zips={cache_zips}")
            if cache_zips:
                _sl2 = _osl2.OpenSlide(wsi_path)
                _mpp = float(_sl2.properties.get("openslide.mpp-x", 0.25))
                _sl2.close()
                # build cache: list of (thumb_array float32, [sx, sy])
                cache_thumbs: list[tuple[_np.ndarray, list[int]]] = []
                with _zf.ZipFile(cache_zips[0]) as zf:
                    for name in zf.namelist():
                        m = _re.match(r"tile_\(([0-9.]+),\s*([0-9.]+)\)", name)
                        if not m:
                            continue
                        sx = round(float(m.group(1)) / _mpp)
                        sy = round(float(m.group(2)) / _mpp)
                        th = _np.array(
                            _Image.open(_io.BytesIO(zf.read(name)))
                            .resize(thumb_sz, _Image.BILINEAR), dtype=_np.float32
                        )
                        cache_thumbs.append((th, [sx, sy]))
                # vectorise: stack into (N, H*W*C)
                cache_mat = _np.stack([t.ravel() for t, _ in cache_thumbs])  # (N, D)
                for tp in top_tile_paths:
                    q = _np.array(
                        _Image.open(tp).resize(thumb_sz, _Image.BILINEAR),
                        dtype=_np.float32
                    ).ravel()
                    mse = ((cache_mat - q) ** 2).mean(axis=1)
                    best_idx = int(mse.argmin())
                    top_tile_coords.append(cache_thumbs[best_idx][1])
                print(f"[top_tile_coords] matched: {top_tile_coords}")
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"[top_tile_coords] matching failed: {_e}")

    # parse class scores from filenames like  stem-MUT=0.68.png  stem-WT=0.32.png
    all_score_imgs = sorted(heatmap_out.rglob("raw/*=*.png"))
    scores = {}
    for p in all_score_imgs:
        m = _re.search(r"-([^-=]+)=([0-9.]+)\.png$", p.name)
        if m:
            scores[m.group(1)] = float(m.group(2))
    predicted_class = max(scores, key=scores.get) if scores else None

    # compute tiled-area bounds in OSD viewport coords (width=1 = full WSI width)
    tg = _state.get("tile_grid")
    overlay_bounds = None
    if tg and heatmap_path:
        import openslide as _osl
        _sl   = _osl.OpenSlide(wsi_path)
        wsi_w = _sl.dimensions[0]
        _sl.close()
        tile_px = tg["tile_size_slide_px"]
        coords  = tg["coords"]
        max_x   = max(c[0] for c in coords) + tile_px
        max_y   = max(c[1] for c in coords) + tile_px
        overlay_bounds = {"x": 0, "y": 0,
                          "w": max_x / wsi_w,
                          "h": max_y / wsi_w}

    return {
        "heatmap_path":    heatmap_path,
        "top_tile_paths":  top_tile_strs,
        "overlay_bounds":  overlay_bounds,
        "scores":          scores,
        "predicted_class": predicted_class,
        "top_tile_coords": top_tile_coords,  # [[x,y], ...] in slide pixels
        "target_dict":     clf["target_dict"],
    }


# ── tile click ────────────────────────────────────────────────────────────────
@app.post("/api/tile-click")
def api_tile_click(x: int = Form(...), y: int = Form(...)):
    tg = _state.get("tile_grid")
    if tg is None:
        raise HTTPException(status_code=400, detail="No tile grid — run tiling first")
    try:
        img, tile_key = stamp_runner.extract_tile_at_slide_coords(
            slide_x=x, slide_y=y, tile_grid=tg
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    tmp  = Path(tempfile.mkdtemp()) / f"{tile_key}.png"
    img.save(tmp)
    path = str(tmp)
    if path not in _state["selected_tiles"]:
        _state["selected_tiles"].append(path)
    return {"tile_key": tile_key, "path": path, "b64": pil_to_base64(img)}


@app.post("/api/clear-tiles")
def api_clear_tiles():
    _state["selected_tiles"] = []
    return {"ok": True}


@app.get("/api/selected-tiles")
def api_selected_tiles():
    return {"paths": _state["selected_tiles"]}


# ── counterfactuals ───────────────────────────────────────────────────────────
@app.post("/api/counterfactuals")
def api_counterfactuals(
    tile_paths_str:  str = Form(...),  # JSON array of tile paths
    classifier_name: str = Form(...),
    source_class:    str = Form(...),
    target_dict_str: str = Form(...),
    amplitudes_str:  str = Form(...),
    T_inv:           int = Form(200),
    T_step:          int = Form(100),
    device:          str = Form("cuda:0"),
    output_dir:      str = Form(...),
):
    try:
        tile_paths = json.loads(tile_paths_str)
        target_dict = json.loads(target_dict_str)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON param: {e}")

    clf = CLASSIFIER_CHOICES.get(classifier_name)
    if not clf:
        raise HTTPException(status_code=400, detail=f"Unknown classifier: {classifier_name}")
    mil_path = clf["ckpt"]

    amplitudes = sorted(float(a) for a in json.loads(amplitudes_str))
    categories = list(target_dict.keys())
    n = len(tile_paths)

    try:
        models = mopadi_runner.acquire_models(MOPADI_CONFIG_PATH, mil_path, n, device)
        print(f"[CF] {len(models)} model(s) ready, starting parallel inference")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load MoPaDi: {e}")

    # Clear log for this run
    with _cf_log_lock:
        _cf_log.clear()

    def run_one(idx: int, tile_path: str, manipulator) -> dict:
        cf_out = str(Path(output_dir) / "counterfactuals" / Path(tile_path).stem)
        orig_img, probs_orig, results, target_class, stoch_orig_img = mopadi_runner.generate_counterfactuals(
            manipulator=manipulator,
            tile_path=tile_path,
            source_class=source_class,
            target_dict=target_dict,
            amplitudes=amplitudes,
            T_inv=T_inv,
            T_step=T_step,
            out_dir=cf_out,
            log_fn=_cf_log_fn,
        )
        orig_prob_str = " | ".join(f"{categories[i]}: {probs_orig[i]:.3f}" for i in range(len(categories)))
        return {
            "idx":       idx,
            "tile_path": tile_path,
            "status":    f"Done · {Path(tile_path).name}\nOriginal ({source_class}): {orig_prob_str}\nTarget: {target_class}",
            "html":      sliders_row_html(orig_img, results, categories, width=224, probs_orig=probs_orig, stoch_orig_img=stoch_orig_img),
        }

    results_all = [None] * n
    errors = []
    with ThreadPoolExecutor(max_workers=n) as exe:
        futs = {exe.submit(run_one, i, tp, models[i]): i for i, tp in enumerate(tile_paths)}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results_all[r["idx"]] = r
            except Exception as e:
                i = futs[fut]
                errors.append(f"Tile {i+1}: {e}")
                results_all[i] = {"idx": i, "status": f"Error: {e}", "html": f"<p style='color:red'>{e}</p>"}

    return {"results": results_all, "errors": errors}


# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3010, log_level="warning")
