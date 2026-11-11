#!/usr/bin/env bash
# Usage: bash scripts/evaluate.sh <gpu_id> <checkpoint> [extra evaluate args...]
set -euo pipefail
GPU="${1:-0}"
CKPT="${2:?checkpoint path required}"
shift 2 || true
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES="${GPU}" python3 -m r2net.evaluate --checkpoint "${CKPT}" "$@"
