#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/instrumental_v6/clean_bach_large_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-out/instrumental_v6_voice_aware_v2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-8000}"
RESUME="${RESUME:-0}"

if [[ ! -f "$DATA_DIR/tokenized/train.pt" || ! -f "$DATA_DIR/tokenized/val.pt" ]]; then
  echo "Missing tokenized dataset under $DATA_DIR" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader

CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -c \
  'import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing CPU training")
torch.cuda.init()
print(f"PyTorch CUDA ready: {torch.cuda.get_device_name(0)}")'

mkdir -p "$OUTPUT_DIR"

ARGS=(
  scripts/train_instrumental_v6.py
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --architecture voice_aware_v2
  --d-model 256
  --n-heads 8
  --n-layers 6
  --n-cross-layers 2
  --dropout 0.1
  --max-seq-len 256
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --lr 2e-4
  --device cuda
  --amp
  --balanced-voice-counts
  --log-every 25
  --val-every 250
  --save-every 500
)

if [[ "$RESUME" == "1" ]]; then
  CHECKPOINT="$OUTPUT_DIR/checkpoint_latest.pt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "RESUME=1 but no checkpoint exists at $CHECKPOINT" >&2
    exit 1
  fi
  ARGS+=(--resume "$CHECKPOINT")
fi

exec env CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach \
  python -u "${ARGS[@]}"
