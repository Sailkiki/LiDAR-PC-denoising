#!/usr/bin/env python3
"""Train R2-Net v2 on PlantN-R.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m r2net.train --config configs/v2_base.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .data import CloudStore
from .evaluation import evaluate_store
from .models import DenoiseNet


DEFAULTS: Dict[str, Any] = {
    "run_name": "v2_base",
    "seed": 2020,
    "device": "cuda",
    "data_root": "data/plantn_r",
    "out_dir": "runs",
    "points": [70000, 90000, 110000],
    "patch_size": 1000,
    "clean_patch_size": 1200,
    "batch_size": 32,
    "noise_min": 0.005,
    "noise_max": 0.025,
    "scale_min": 0.8,
    "scale_max": 1.2,
    "rotate": True,
    "max_iters": 150000,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "warmup_iters": 2000,
    "grad_clip": 1.0,
    "amp": True,
    "loss_type": "weighted_huber",
    "rbf_sigma": 0.02,
    "alpha": 0.5,
    "huber_delta": 1.0,
    "loss_scale": 100.0,
    "frame_knn": 32,
    "num_train_points": 128,
    "num_clean_nbs": 4,
    "score_net_hidden_dim": 128,
    "score_net_num_blocks": 4,
    "conv_channels": 24,
    "num_convs": 4,
    "num_fc_layers": 3,
    "growth_rate": 12,
    "conv_knn": 16,
    "use_oc": True,
    "use_ace": True,
    "candidate_ratio": 1.5,
    "ace_reduction": 8,
    "val_every": 5000,
    "val_clouds": 5,
    "val_noise": 0.01,
    "val_steps": 30,
    "val_patch_batch_size": 4,
    "log_every": 50,
    "save_every": 5000,
    "early_stop_patience": 30,
    "resume": None,
    "dry_run": False,
    "no_amp": False,
    "config": None,
    "ema_decay": 0.999,
    "init_from": None,
    "balance_species": False,
}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def parse_args() -> argparse.Namespace:
    # YAML values provide defaults; explicit command-line flags take precedence.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = dict(DEFAULTS)
    if pre_args.config:
        cfg.update(load_yaml(pre_args.config))

    p = argparse.ArgumentParser(description="Train R2-Net v2")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--run-name", type=str, default=cfg["run_name"])
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--device", type=str, default=cfg["device"])
    p.add_argument("--data-root", type=str, default=cfg["data_root"])
    p.add_argument("--out-dir", type=str, default=cfg["out_dir"])
    p.add_argument("--points", type=int, nargs="+", default=cfg["points"])
    p.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    p.add_argument("--max-iters", type=int, default=cfg["max_iters"])
    p.add_argument("--lr", type=float, default=cfg["lr"])
    p.add_argument("--warmup-iters", type=int, default=cfg["warmup_iters"])
    p.add_argument("--loss-type", type=str, default=cfg["loss_type"], choices=["weighted_huber", "plain_huber"])
    p.add_argument("--rbf-sigma", type=float, default=cfg["rbf_sigma"])
    p.add_argument("--alpha", type=float, default=cfg["alpha"])
    p.add_argument("--loss-scale", type=float, default=cfg["loss_scale"])
    p.add_argument("--log-every", type=int, default=cfg["log_every"])
    p.add_argument("--frame-knn", type=int, default=cfg["frame_knn"])
    p.add_argument("--num-train-points", type=int, default=cfg["num_train_points"])
    p.add_argument("--num-clean-nbs", type=int, default=cfg["num_clean_nbs"])
    p.add_argument("--score-hidden", type=int, dest="score_net_hidden_dim", default=cfg["score_net_hidden_dim"])
    p.add_argument("--score-blocks", type=int, dest="score_net_num_blocks", default=cfg["score_net_num_blocks"])
    p.add_argument("--conv-channels", type=int, default=cfg["conv_channels"])
    p.add_argument("--growth-rate", type=int, default=cfg["growth_rate"])
    p.add_argument("--conv-knn", type=int, default=cfg["conv_knn"])
    p.add_argument("--val-every", type=int, default=cfg["val_every"])
    p.add_argument("--val-clouds", type=int, default=cfg["val_clouds"])
    p.add_argument("--save-every", type=int, default=cfg["save_every"])
    p.add_argument("--early-stop-patience", type=int, default=cfg["early_stop_patience"])
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init-from", type=str, default=cfg.get("init_from"))
    p.add_argument("--ema-decay", type=float, default=cfg.get("ema_decay", 0.999))
    p.add_argument("--balance-species", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Merge YAML keys that do not have an explicit CLI flag.
    merged = dict(cfg)
    for k, v in vars(args).items():
        if v is not None:
            merged[k] = v
    merged["points"] = [int(x) for x in merged["points"]]
    merged["amp"] = bool(merged.get("amp", True)) and not args.no_amp
    merged["balance_species"] = bool(cfg.get("balance_species", False) or bool(args.balance_species))
    if merged["dry_run"]:
        merged["max_iters"] = min(5, int(merged["max_iters"]))
        merged["val_every"] = 5
        merged["val_clouds"] = 1
        merged["save_every"] = 5
    return argparse.Namespace(**merged)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


class ModelEMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if k not in msd:
                continue
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def build_model(cfg: argparse.Namespace) -> DenoiseNet:
    return DenoiseNet(
        frame_knn=cfg.frame_knn,
        num_train_points=cfg.num_train_points,
        num_clean_nbs=cfg.num_clean_nbs,
        score_net_hidden_dim=cfg.score_net_hidden_dim,
        score_net_num_blocks=cfg.score_net_num_blocks,
        conv_channels=cfg.conv_channels,
        num_convs=cfg.num_convs,
        num_fc_layers=cfg.num_fc_layers,
        growth_rate=cfg.growth_rate,
        conv_knn=cfg.conv_knn,
        use_oc=cfg.use_oc,
        use_ace=cfg.use_ace,
        candidate_ratio=cfg.candidate_ratio,
        ace_reduction=cfg.ace_reduction,
    )


def make_scheduler(optimizer, max_iters: int, warmup_iters: int) -> LambdaLR:
    warmup = max(1, int(warmup_iters))
    total = max(2, int(max_iters))

    def lr_lambda(it: int) -> float:
        if it < warmup:
            return float(it + 1) / float(warmup)
        progress = min(1.0, max(0.0, (it - warmup) / float(max(1, total - warmup))))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, model, optimizer, scheduler, cfg, iteration: int, best_cd: float, history,
                    ema: ModelEMA | None = None, use_ema: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = ema.module.state_dict() if (use_ema and ema is not None) else model.state_dict()
    torch.save({
        "model": model_state,
        "ema": ema.module.state_dict() if ema is not None else None,
        "use_ema": bool(use_ema and ema is not None),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(cfg),
        "iteration": int(iteration),
        "best_cd": float(best_cd),
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)


def main() -> None:
    cfg = parse_args()
    seed_everything(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(cfg), indent=2), encoding="utf-8")

    print(f"[R2Net-v2] run={cfg.run_name} device={device}", flush=True)
    # Keep the clouds on the GPU and sample noisy/clean patches on the fly.  This
    # avoids a CPU data-loader bottleneck for the small PlantN-R benchmark.
    train_store = CloudStore.from_split(
        cfg.data_root, "train", cfg.points, device, normalize=True,
        balance_species=bool(getattr(cfg, "balance_species", False)),
    )
    val_store = CloudStore.from_split(cfg.data_root, "validation", [70000], device, normalize=True)
    print(f"[data] train clouds={len(train_store.clouds)} val clouds={len(val_store.clouds)}", flush=True)

    model = build_model(cfg).to(device)
    if cfg.init_from:
        init_ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init_ckpt["model"])
        print(f"[init] loaded weights from {cfg.init_from} (iter={init_ckpt.get('iteration')})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} ({n_params / 1e6:.3f}M)", flush=True)
    (run_dir / "model_summary.json").write_text(json.dumps({
        "params": int(n_params),
        "architecture": repr(model),
    }, indent=2), encoding="utf-8")

    optimizer = Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = make_scheduler(optimizer, int(cfg.max_iters), int(cfg.warmup_iters))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp) and device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    ema = ModelEMA(model, decay=float(cfg.ema_decay)) if float(cfg.ema_decay) > 0 else None

    start_iter = 0
    best_cd = float("inf")
    history = []
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = int(ckpt.get("iteration", 0))
        best_cd = float(ckpt.get("best_cd", float("inf")))
        history = list(ckpt.get("history", []))
        if ema is not None and ckpt.get("ema") is not None:
            ema.module.load_state_dict(ckpt["ema"])
        print(f"[resume] {cfg.resume} @ iter {start_iter}, best_cd={best_cd:.6g}", flush=True)

    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("iter,lr,loss,sec,sigma_mean\n", encoding="utf-8")

    stale = 0
    t_start = time.time()
    for it in range(start_iter + 1, int(cfg.max_iters) + 1):
        model.train()
        pn, pc, metas = train_store.sample_batch(
            int(cfg.batch_size),
            patch_size=int(cfg.patch_size),
            clean_patch_size=int(cfg.clean_patch_size),
            noise_min=float(cfg.noise_min),
            noise_max=float(cfg.noise_max),
            scale_min=float(cfg.scale_min),
            scale_max=float(cfg.scale_max),
            rotate=bool(cfg.rotate),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        # AMP is used for the network forward/backward.  kNN searches in
        # r2net.ops are explicitly performed in FP32, so neighbour indices stay
        # stable while the MLPs run in half precision.
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=bool(cfg.amp) and device.type == "cuda"):
            loss = model.get_supervised_loss(
                pn, pc,
                loss_type=cfg.loss_type,
                rbf_sigma=float(cfg.rbf_sigma),
                alpha=float(cfg.alpha),
                huber_delta=float(cfg.huber_delta),
                loss_scale=float(cfg.loss_scale),
            )
        scaler.scale(loss).backward()
        if float(cfg.grad_clip) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        dt = time.time() - t0

        if it % int(cfg.log_every) == 0 or it == 1:
            lr = optimizer.param_groups[0]["lr"]
            sigma_mean = float(np.mean([m["sigma"] for m in metas]))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{it},{lr:.8g},{float(loss.item()):.8g},{dt:.4f},{sigma_mean:.6g}\n")
            print(f"[train] iter={it:7d} loss={float(loss.item()):.5f} lr={lr:.2e} {dt*1000:.1f}ms sigma={sigma_mean:.4f}", flush=True)

        if it % int(cfg.val_every) == 0 or it == int(cfg.max_iters):
            # Checkpoint selection uses only the validation split.  The test
            # split is never touched during training.
            eval_model = ema.module if ema is not None else model
            metrics = evaluate_store(
                eval_model, val_store,
                noise=float(cfg.val_noise),
                max_clouds=int(cfg.val_clouds),
                infer_kwargs=dict(
                    patch_size=int(cfg.patch_size),
                    patch_batch_size=int(cfg.val_patch_batch_size),
                    num_steps=int(cfg.val_steps),
                    step_size=0.2,
                    step_decay=0.95,
                    denoise_knn=4,
                    noise_scale=0.0,
                ),
            )
            mean = metrics["mean"]
            entry = {"iteration": it, "n": metrics["n"], **mean, **{f"std_{k}": v for k, v in metrics["std"].items()}}
            history.append(entry)
            (run_dir / "validation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"[val] iter={it} n={metrics['n']} CD={mean['cd']:.6g} RMSE={mean['rmse']:.6g} "
                f"NC={mean['nc']:.6f} input_CD={mean['input_cd']:.6g}",
                flush=True,
            )
            if mean["cd"] < best_cd:
                best_cd = mean["cd"]
                stale = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, cfg, it, best_cd, history, ema=ema, use_ema=ema is not None)
                save_checkpoint(run_dir / "best_raw.pt", model, optimizer, scheduler, cfg, it, best_cd, history, ema=None, use_ema=False)
                print(f"[ckpt] new best CD={best_cd:.6g} at iter {it} ({'EMA' if ema is not None else 'raw'})", flush=True)
            else:
                stale += 1
            if stale >= int(cfg.early_stop_patience):
                print(f"[early-stop] no improvement for {stale} validations at iter {it}", flush=True)
                break

        if it % int(cfg.save_every) == 0:
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, cfg, it, best_cd, history, ema=ema, use_ema=False)

    save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, cfg, int(cfg.max_iters), best_cd, history, ema=ema, use_ema=False)
    total = time.time() - t_start
    print(f"[done] run={cfg.run_name} best_cd={best_cd:.6g} elapsed={total/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
