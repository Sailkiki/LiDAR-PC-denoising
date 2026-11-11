"""Evaluation helpers shared by training-time validation and final testing."""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .data import CloudStore
from .ops import cd_rmse_nc


def stable_seed(key: str, base: int = 2020) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") ^ int(base)


def make_noisy(clean: torch.Tensor, noise: float, key: str, base_seed: int = 2020) -> torch.Tensor:
    rng = np.random.default_rng(stable_seed(key, base_seed))
    arr = clean.detach().cpu().numpy()
    noisy = (arr + rng.normal(0.0, float(noise), arr.shape)).astype(np.float32)
    return torch.from_numpy(noisy).to(clean.device)


@torch.no_grad()
def evaluate_store(
    model,
    store: CloudStore,
    *,
    noise: float = 0.01,
    base_seed: int = 2020,
    max_clouds: Optional[int] = None,
    infer_kwargs: Optional[Dict] = None,
    tta_rotations: int = 0,
    progress: bool = False,
) -> Dict:
    infer_kwargs = dict(infer_kwargs or {})
    rows: List[Dict] = []
    clouds = store.clouds[: int(max_clouds)] if max_clouds else store.clouds
    for cloud in clouds:
        clean = cloud.data
        noisy = make_noisy(clean, noise, f"{cloud.split}/{cloud.points}/{cloud.name}/{noise}", base_seed)
        if tta_rotations and int(tta_rotations) > 1:
            denoised = model.infer_cloud_tta(
                noisy, n_rotations=int(tta_rotations), seed=stable_seed(cloud.name, base_seed), **infer_kwargs
            )
        else:
            denoised = model.infer_cloud(noisy, **infer_kwargs)
        gen_np = denoised.detach().cpu().numpy()
        ref_np = clean.detach().cpu().numpy()
        inp_np = noisy.detach().cpu().numpy()
        out_m = cd_rmse_nc(gen_np, ref_np)
        in_m = cd_rmse_nc(inp_np, ref_np)
        if progress:
            print(f"[eval] {cloud.name:14s} CD={out_m['cd']:.4g} RMSE={out_m['rmse']:.4g} NC={out_m['nc']:.4f}", flush=True)
        rows.append({
            "name": cloud.name,
            "points": cloud.points,
            "split": cloud.split,
            "noise": float(noise),
            "cd": out_m["cd"],
            "rmse": out_m["rmse"],
            "nc": out_m["nc"],
            "input_cd": in_m["cd"],
            "input_rmse": in_m["rmse"],
            "input_nc": in_m["nc"],
        })
    if not rows:
        raise RuntimeError("No clouds evaluated")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("cd", "rmse", "nc", "input_cd", "input_rmse", "input_nc")}
    std = {k: float(np.std([r[k] for r in rows], ddof=0)) for k in ("cd", "rmse", "nc")}
    return {"rows": rows, "mean": mean, "std": std, "n": len(rows)}
