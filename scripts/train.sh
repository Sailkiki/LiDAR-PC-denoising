#!/usr/bin/env bash
# Usage: bash scripts/train.sh <gpu_id> <config.yaml> [extra train args...]
set -euo pipefail
GPU="${1:-0}"
CONFIG="${2:-configs/v2_base.yaml}"
shift 2 || true
cd "$(dirname "$0")/.."
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -m r2net.train --config "${CONFIG}" "$@"
