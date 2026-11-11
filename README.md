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


## License

MIT. See [`LICENSE`](LICENSE).
