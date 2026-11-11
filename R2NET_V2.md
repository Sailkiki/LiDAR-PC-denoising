# R²-Net v2 — Official Release

## A note from the authors

More than a year has passed since the original R²-Net manuscript was submitted.
During this period, we have continued to maintain, test, and refine the method
and its implementation. We are now releasing **R²-Net v2**, a complete,
cleaned, and reproducible version of the method described in the paper.

This release is intended to make R²-Net easier to run, inspect, and reproduce.
It follows the same two-stage design as the published work: a Reconstruct stage
that learns robust geometric features from noisy plant point clouds, and a
Refine stage that iteratively projects points toward the underlying clean
surface under a conditional score-based formulation.

R²-Net v2 is a code and reproducibility update. It preserves the core
methodology of the paper and does not introduce a new scientific claim.

## Scope of this repository

This repository contains the core implementation, training configurations,
data-preparation utilities, evaluation utilities, and smoke tests. It does
**not** contain datasets, data-split records, trained checkpoints, training
logs, or evaluation records. Those artefacts are generated or supplied
locally by the user and are deliberately excluded from this code-only release.

## What is included

The release provides a self-contained PyTorch implementation of the complete
pipeline:

- OC-Conv feature extraction with offset-corrected neighbourhood construction.
- Adaptive Channel Enhancement (ACE) with channel, spatial, and global
  calibration branches.
- Conditional residual refinement for the displacement/score field.
- The structure-aware weighted Huber objective.
- Patch-based training and inference for individual plant point clouds.
- Invertible normalisation and block-wise processing for large scenes.
- Dataset preparation utilities for the PlantN-R benchmark.
- Training, validation, and evaluation scripts.
- Training configurations for the base, wide, plain-Huber, and high-noise
  variants.

## What is new in v2

R²-Net v2 focuses on implementation correctness, completeness, and
reproducibility.

### Corrected OC-Conv implementation

- The offset predictor now produces a three-dimensional geometric offset and
  applies it to point coordinates, as described in the paper.
- The oversampled candidate neighbourhood is retained, and the
  attention-selected top-K neighbours are used by the dense aggregation block.
- The attention network produces one importance score per candidate edge.
- The graph is rebuilt in the corrected coordinate space at every layer.

### Completed ACE module

- The channel, spatial, and global calibration branches follow the formulation
  in the paper.
- The calibration is applied inside the dense feature stream and followed by a
  residual connection.
- The module is configurable for ablation studies.

### Completed training objective

- The noisy-to-clean displacement vectors are computed from clean
  nearest-neighbour correspondences.
- The radial-basis weighting and composite weighting factors are implemented.
- The weighted Huber loss is used for training, with an optional plain-Huber
  ablation mode.

### Cleaner inference pipeline

- Overlapping patch predictions are merged by the original point identity with
  distance-based weights.
- Uncovered points are retained instead of being discarded.
- The implementation supports deterministic refinement and an optional
  stochastic Langevin mode.
- Multi-rotation test-time augmentation is available as an optional inference
  enhancement.

### Reproducible training protocol

- Training, validation, and test subsets are explicitly separated.
- Checkpoint selection uses the validation subset only.
- Random seeds, AMP settings, optimiser configuration, and data splits are
  recorded in the training configuration.
- The training code supports EMA checkpointing and resuming from a checkpoint.

### Self-contained numerical operations

- kNN search, farthest-point sampling, normalisation, and evaluation utilities
  are implemented in pure PyTorch and NumPy.
- The release does not require PyTorch3D, torch-cluster, or a compiled CUDA
  extension.

## Repository layout

```text
R2Net_v2/
├── r2net/
│   ├── models/          # OC-Conv, ACE, feature extractor, score net, DenoiseNet
│   ├── data.py          # PlantN-R loading and patch sampling
│   ├── evaluation.py    # evaluation utilities
│   ├── ops.py           # pure-PyTorch kNN, FPS, normalisation, evaluation utilities
│   ├── train.py         # training entry point
│   └── evaluate.py      # evaluation entry point
├── configs/
│   ├── README.md
│   ├── v2_base.yaml
│   ├── v2_wide.yaml
│   ├── v2_plain.yaml
│   └── v2_wide_hn.yaml
├── scripts/             # data preparation, training, evaluation
├── tests/               # smoke tests
├── README.md            # short repository landing page
├── R2NET_V2.md          # this release document
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Requirements and installation

```bash
python -m pip install -r requirements.txt
# or install the package in editable mode
python -m pip install -e .
```

A CUDA-capable GPU is recommended for training and evaluation. The code also
runs on CPU for smoke tests and small examples.

## Data preparation

The PlantN-R data can be prepared from the original submission archive. The
data-split records are not distributed with this code-only repository; users
should provide their own split directory containing the benchmark table and
the train/validation/test lists.

```bash
python scripts/prepare_data.py \
    --zip /path/to/Botany3D_LiDAR_backup.zip \
    --splits-dir /path/to/splits \
    --out data/plantn_r
```

The script extracts xyz coordinates, stores them as float32 arrays, and writes
a manifest with checksums and split assignments.

## Training

```bash
# Single-GPU training; choose an available GPU index
bash scripts/train.sh 0 configs/v2_wide.yaml
```

Training configurations are provided in `configs/`. The training script
supports:

- configurable network width and depth;
- weighted Huber and plain Huber objectives;
- AMP, gradient clipping, warm-up, cosine learning-rate decay, and EMA;
- validation-based checkpoint selection;
- resume from a saved checkpoint.

Training outputs, including checkpoints and logs, are written to `runs/` and
are not part of this repository.

## Evaluation

After training a model, evaluate its checkpoint with:

```bash
python -m r2net.evaluate \
    --checkpoint runs/v2_wide/best.pt \
    --split test \
    --points 70000 \
    --noise 0.01 \
    --tta-rotations 4 \
    --step-size 0.15 \
    --seed-k 3 \
    --patch-batch-size 16
```

Evaluation outputs are written to the directory passed with `--out`.

## Reproducibility notes

- Training, validation, and test subsets are kept identity-disjoint by the
  user-provided split records.
- The training configuration and validation history are written to `runs/`
  for each local training run.
- Smoke tests can be executed with:

```bash
python -m pytest -q tests/test_smoke.py
```

## Relationship to the published paper

R²-Net v2 is an updated implementation of the method described in the
published paper. The core two-stage design, OC-Conv, ACE, and conditional
refinement formulation are retained. Some implementation details have been
corrected or reorganised to improve readability, numerical stability, and
reproducibility. Users who refer to the method in scientific work should cite
the original paper.

## Citation

If you use this code, please cite the original R²-Net paper:

```text
Bolin Fu, Mingzhe Sui, Fei Yang, Xing Zhang, Tengfang Deng.
A novel probabilistic generative framework of denoising point clouds for
individual crops and large-scale vegetation.
Computers and Electronics in Agriculture.
```

## License

This project is released under the MIT License. See `LICENSE` for details.
