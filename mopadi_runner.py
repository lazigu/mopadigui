"""MoPaDi counterfactual generation wrapper for the GUI."""

import os
import sys
import threading
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

_GUI_DIR = Path(__file__).resolve().parent
load_dotenv(_GUI_DIR / ".env")

MOPADI_SRC = Path(__file__).resolve().parents[1] / "mopadi" / "src"
if str(MOPADI_SRC) not in sys.path:
    sys.path.insert(0, str(MOPADI_SRC))

from mopadi.mil.manipulate.manipulator_stamp import ImageManipulatorSTAMP
from mopadi.configs.templates import default_autoenc

AUTOENC_CKPT = os.environ["AUTOENC_CKPT"]

# ── model pool ────────────────────────────────────────────────────────────────
_pool: dict[str, list[ImageManipulatorSTAMP]] = {}   # mil_path → [models]
_pool_lock = threading.Lock()


def _resolve_config(config_yaml_path: str) -> dict:
    with open(config_yaml_path) as f:
        config = yaml.safe_load(f)
    base_dir = Path(config.get("base_dir", ""))
    if not base_dir.is_absolute():
        config["base_dir"] = str(Path(config_yaml_path).parent / base_dir)
    return config


def _load_one(config_yaml_path: str, mil_path: str, device: str) -> ImageManipulatorSTAMP:
    # No global lock — PyTorch handles CUDA thread safety internally.
    # The pool_lock is sufficient to protect the pool list itself.
    config       = _resolve_config(config_yaml_path)
    autoenc_conf = default_autoenc(config)
    return ImageManipulatorSTAMP(
        autoenc_config = autoenc_conf,
        autoenc_path   = AUTOENC_CKPT,
        mil_path       = mil_path,
        dataset        = None,
        device         = device,
    )


def start_preload(config_yaml_path: str, mil_path: str, n: int = 5,
                  device: str = "cuda:0") -> None:
    """Kick off background loading of N model copies into the pool.
    Call this at the start of the heatmap run so models are ready by Generate time.
    Skips copies that are already in the pool.
    """
    def _fill():
        for i in range(n):
            with _pool_lock:
                already = len(_pool.get(mil_path, []))
            if already >= n:
                break
            print(f"[preload] loading model copy {already+1}/{n}…")
            m = _load_one(config_yaml_path, mil_path, device)
            with _pool_lock:
                _pool.setdefault(mil_path, []).append(m)
                print(f"[preload] pool now has {len(_pool[mil_path])} copy(ies) for {Path(mil_path).name}")

    threading.Thread(target=_fill, daemon=True).start()


def acquire_models(config_yaml_path: str, mil_path: str, n: int,
                   device: str = "cuda:0") -> list[ImageManipulatorSTAMP]:
    """Take N models from pool (loading any that are still missing)."""
    with _pool_lock:
        available      = _pool.pop(mil_path, [])
        taken          = available[:n]
        leftover       = available[n:]
        if leftover:
            _pool[mil_path] = leftover

    n_missing = n - len(taken)
    n_from_pool = len(taken)
    for i in range(n_missing):
        print(f"[CF] pool miss — loading model {n_from_pool+i+1}/{n} on demand")
        taken.append(_load_one(config_yaml_path, mil_path, device))

    return taken


def _to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    t = transforms.ToTensor()(img)
    t = transforms.Normalize((0.5,) * 3, (0.5,) * 3)(t)
    return t.unsqueeze(0).to(device)


def _save_rgb(tensor: torch.Tensor, path: str):
    im = tensor.clone()
    if im.min() < 0:
        im = (im + 1) / 2
    im = (im.clamp(0, 1) * 255).byte()[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(im).save(path)


def generate_counterfactuals(
    manipulator: ImageManipulatorSTAMP,
    tile_path: str,
    source_class: str,
    target_dict: dict,
    amplitudes: list[float],
    T_inv: int = 200,
    T_step: int = 100,
    out_dir: str = "/tmp/mopadi_gui_cf",
    log_fn=None,
) -> tuple[Image.Image, list[float], list[dict], str]:
    if log_fn is None:
        log_fn = print
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device        = manipulator.device
    target_class  = next(k for k in target_dict if k != source_class)
    target_cls_id = target_dict[target_class]

    img      = Image.open(tile_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    img_01   = transforms.ToTensor()(img).unsqueeze(0).to(device)   # [0,1] → Virchow2
    img_neg1 = _to_tensor(img, device)                               # [-1,1] → autoenc

    with torch.no_grad():
        feats = manipulator.model.feat_extractor.extract_feats(img_01, need_grad=False).float()

    coords     = torch.zeros(1, 2, device=device, dtype=torch.float32)
    feats_bag  = feats.unsqueeze(0)
    coords_bag = coords.unsqueeze(0)

    with torch.no_grad():
        probs_ori = torch.softmax(
            manipulator.classifier(feats_bag, coords=coords_bag, mask=None), dim=-1)

    with torch.no_grad():
        xT      = manipulator.model.encode_stochastic(img_neg1, feats, T=T_inv)
        x_id    = manipulator.model.render(xT, feats, T=T_step)
        xT_rand = torch.randn_like(xT)
        x_stoch = manipulator.model.render(xT_rand, feats, T=T_step)

    ori_path   = os.path.join(out_dir, f"{Path(tile_path).stem}_0_original_{source_class}.png")
    stoch_path = os.path.join(out_dir, f"{Path(tile_path).stem}_0_stoch_{source_class}.png")
    _save_rgb(x_id, ori_path)
    _save_rgb(x_stoch, stoch_path)

    log_fn(f"[CF] {Path(tile_path).name}  {source_class}→{target_class}  amps={amplitudes}")
    log_fn(f"[CF]   orig  {dict(zip(target_dict, [f'{p:.3f}' for p in probs_ori[0].tolist()]))}")

    results = []
    for amp in amplitudes:
        manip_feats = manipulator.manipulate_latent_feats_stamp_tile_level(
            vit_model    = manipulator.classifier,
            feats        = feats,
            coords       = coords,
            tile_indices = [0],
            man_amp      = amp,
            cls_id       = target_cls_id,
        )
        with torch.no_grad():
            probs_feat = torch.softmax(
                manipulator.classifier(manip_feats.unsqueeze(0), coords=coords_bag, mask=None), dim=-1)

        log_fn(f"[CF]   amp={amp}  feat {dict(zip(target_dict, [f'{p:.3f}' for p in probs_feat[0].tolist()]))}")

        log_fn(f"[CF]   amp={amp}  rendering ({T_step} steps)…")
        amp_tag = str(amp).replace('.', ',')
        with torch.no_grad():
            x_manip       = manipulator.model.render(xT, manip_feats, T=T_step)
            x_manip_stoch = manipulator.model.render(xT_rand, manip_feats, T=T_step)
            log_fn(f"[CF]   amp={amp}  render done, re-encoding…")
            x_manip_01       = (x_manip.clamp(-1, 1) + 1) / 2
            x_manip_stoch_01 = (x_manip_stoch.clamp(-1, 1) + 1) / 2
            img_feats2       = manipulator.model.feat_extractor.extract_feats(x_manip_01.to(device), need_grad=False).float()
            img_feats_stoch  = manipulator.model.feat_extractor.extract_feats(x_manip_stoch_01.to(device), need_grad=False).float()
            probs_img        = torch.softmax(
                manipulator.classifier(img_feats2.unsqueeze(0), coords=coords_bag, mask=None), dim=-1)
            probs_stoch_img  = torch.softmax(
                manipulator.classifier(img_feats_stoch.unsqueeze(0), coords=coords_bag, mask=None), dim=-1)

        out_img_path       = os.path.join(out_dir, f"{Path(tile_path).stem}_manip_to_{target_class}_amp_{amp_tag}.png")
        out_stoch_img_path = os.path.join(out_dir, f"{Path(tile_path).stem}_stoch_to_{target_class}_amp_{amp_tag}.png")
        _save_rgb(x_manip, out_img_path)
        _save_rgb(x_manip_stoch, out_stoch_img_path)
        results.append({
            "amp":            amp,
            "manip_img":      Image.open(out_img_path).convert("RGB"),
            "stoch_manip_img":  Image.open(out_stoch_img_path).convert("RGB"),
            "probs_feat":       probs_feat[0].tolist(),
            "probs_img":        probs_img[0].tolist(),
            "probs_stoch_img":  probs_stoch_img[0].tolist(),
        })

    return Image.open(ori_path).convert("RGB"), probs_ori[0].tolist(), results, target_class, Image.open(stoch_path).convert("RGB")
