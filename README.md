# R²-Net v2

Official code release for **R²-Net v2**, a clean and reproducible
implementation of the R²-Net point-cloud denoising method for individual crops
and vegetation.

> More than a year has passed since the original paper submission. We have
> continued to maintain and refine the method, and we are now releasing
> R²-Net v2.

This repository contains **core code and configuration files only**. It does
not include datasets, trained checkpoints, training logs, or evaluation
records.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_smoke.py

# Prepare data from the original submission archive
python scripts/prepare_data.py \
    --zip /path/to/Botany3D_LiDAR_backup.zip \
    --splits-dir /path/to/splits \
    --out data/plantn_r

# Train from a configuration
bash scripts/train.sh 0 configs/v2_wide.yaml

# Evaluate the checkpoint produced by training
python -m r2net.evaluate \
    --checkpoint runs/v2_wide/best.pt \
    --split test --points 70000 --noise 0.01 \
    --tta-rotations 4 --step-size 0.15 --seed-k 3 \
    --patch-batch-size 16
```

## Documentation

- Official English release document: [`R2NET_V2.md`](R2NET_V2.md)
- Training configuration guide: [`configs/README.md`](configs/README.md)

## What is in v2

R²-Net v2 includes a corrected OC-Conv implementation, a completed ACE module,
the structure-aware weighted Huber objective, identity-preserving patch
merging, explicit training/validation/test separation, self-contained
numerical operations, and reproducible training/evaluation scripts.

See [`R2NET_V2.md`](R2NET_V2.md) for the full release note.

## License

MIT. See [`LICENSE`](LICENSE).
