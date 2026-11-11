#!/usr/bin/env python3
"""Evaluate an R2-Net v2 checkpoint on PlantN-R.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m r2net.evaluate \
      --checkpoint runs/v2_base/best.pt --split test --points 70000 \
      --noise 0.01 0.03 0.05 --out results/v2_base_test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import CloudStore
from .evaluation import evaluate_store
from .models import DenoiseNet


def build_model(cfg: dict) -> DenoiseNet:
    return DenoiseNet(
        frame_knn=cfg.get("frame_knn", 32),
        num_train_points=cfg.get("num_train_points", 128),
        num_clean_nbs=cfg.get("num_clean_nbs", 4),
        score_net_hidden_dim=cfg.get("score_net_hidden_dim", 128),
        score_net_num_blocks=cfg.get("score_net_num_blocks", 4),
        conv_channels=cfg.get("conv_channels", 24),
        num_convs=cfg.get("num_convs", 4),
        num_fc_layers=cfg.get("num_fc_layers", 3),
        growth_rate=cfg.get("growth_rate", 12),
        conv_knn=cfg.get("conv_knn", 16),
        use_oc=cfg.get("use_oc", True),
        use_ace=cfg.get("use_ace", True),
        candidate_ratio=cfg.get("candidate_ratio", 1.5),
        ace_reduction=cfg.get("ace_reduction", 8),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data-root", type=str, default="data/plantn_r")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--points", type=int, default=70000)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    ap.add_argument("--out", type=str, default="results/eval")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-clouds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--step-size", type=float, default=0.2)
    ap.add_argument("--step-decay", type=float, default=0.95)
    ap.add_argument("--patch-batch-size", type=int, default=8)
    ap.add_argument("--denoise-knn", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=1000)
    ap.add_argument("--seed-k", type=float, default=3.0)
    ap.add_argument("--tta-rotations", type=int, default=0)
    ap.add_argument("--noise-scale", type=float, default=0.0)
    ap.add_argument("--save-clouds", action="store_true")
    args = ap.parse_args()

    # Checkpoints contain both the model state and the architecture config, so
    # evaluation does not depend on a separately maintained YAML file.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()
    print(f"[eval] checkpoint={args.checkpoint} iter={ckpt.get('iteration')} params={sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)

    store = CloudStore.from_split(args.data_root, args.split, [args.points], device, normalize=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for noise in args.noise:
        # Metrics are computed in the same unit-sphere coordinate system used
        # for training and inference.
        metrics = evaluate_store(
            model, store,
            noise=float(noise),
            max_clouds=args.max_clouds,
            tta_rotations=args.tta_rotations,
            progress=True,
            infer_kwargs=dict(
                patch_size=args.patch_size,
                seed_k=args.seed_k,
                patch_batch_size=args.patch_batch_size,
                num_steps=args.steps,
                step_size=args.step_size,
                step_decay=args.step_decay,
                noise_scale=args.noise_scale,
            ),
        )
        mean, std = metrics["mean"], metrics["std"]
        print(
            f"[eval] noise={noise:.2f} n={metrics['n']} "
            f"CD={mean['cd']:.6g}±{std['cd']:.2g} "
            f"RMSE={mean['rmse']:.6g}±{std['rmse']:.2g} "
            f"NC={mean['nc']:.6f}±{std['nc']:.4g} | "
            f"input CD={mean['input_cd']:.6g} RMSE={mean['input_rmse']:.6g} NC={mean['input_nc']:.6f}",
            flush=True,
        )
        all_results.append({"noise": noise, "mean": mean, "std": std, "rows": metrics["rows"]})

        if args.save_clouds:
            for cloud in store.clouds:
                from .evaluation import make_noisy
                noisy = make_noisy(cloud.data, float(noise), f"{cloud.split}/{cloud.points}/{cloud.name}/{noise}", 2020)
                if args.tta_rotations and args.tta_rotations > 1:
                    den = model.infer_cloud_tta(
                        noisy, n_rotations=args.tta_rotations,
                        patch_size=args.patch_size, seed_k=args.seed_k,
                        patch_batch_size=args.patch_batch_size,
                        num_steps=args.steps, step_size=args.step_size,
                        step_decay=args.step_decay, denoise_knn=args.denoise_knn,
                        noise_scale=args.noise_scale,
                    )
                else:
                    den = model.infer_cloud(
                        noisy, patch_size=args.patch_size, seed_k=args.seed_k,
                        patch_batch_size=args.patch_batch_size,
                        num_steps=args.steps, step_size=args.step_size,
                        step_decay=args.step_decay, denoise_knn=args.denoise_knn,
                        noise_scale=args.noise_scale,
                    )
                d = out_dir / f"noise_{noise:.2f}" / cloud.name
                d.mkdir(parents=True, exist_ok=True)
                np.savetxt(d / "noisy.xyz", noisy.detach().cpu().numpy(), fmt="%.8f")
                np.savetxt(d / "denoised.xyz", den.detach().cpu().numpy(), fmt="%.8f")
                np.savetxt(d / "clean.xyz", cloud.data.detach().cpu().numpy(), fmt="%.8f")

    summary = {"checkpoint": str(args.checkpoint), "split": args.split, "points": args.points,
               "results": all_results}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise", "name", "cd", "rmse", "nc", "input_cd", "input_rmse", "input_nc"])
        for res in all_results:
            for row in res["rows"]:
                w.writerow([res["noise"], row["name"], row["cd"], row["rmse"], row["nc"],
                            row["input_cd"], row["input_rmse"], row["input_nc"]])
    print(f"[eval] wrote {out_dir/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
