#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-bach}"
DEVICE="${DEVICE:-cuda}"

CLEANED_DIR="${CLEANED_DIR:-data/musicxml_cleaned/bach_counterpoint_v1}"
CLEANED_ROOT="${CLEANED_ROOT:-$CLEANED_DIR/files}"
APPROVED_LIST="${APPROVED_LIST:-$CLEANED_DIR/approved_files.txt}"

DATA_DIR="${DATA_DIR:-data/instrumental_v6/clean_bach_long_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-out/instrumental_v6_voice_aware_384_long_v1}"

MAX_BARS="${MAX_BARS:-96}"
SEQ_LEN="${SEQ_LEN:-512}"
STRIDE="${STRIDE:-256}"
MAX_VOICES="${MAX_VOICES:-6}"
MIN_SLICES="${MIN_SLICES:-64}"
LIMIT_MOVEMENTS_PER_WORK="${LIMIT_MOVEMENTS_PER_WORK:-0}"

D_MODEL="${D_MODEL:-384}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-8}"
N_CROSS_LAYERS="${N_CROSS_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-16000}"
LR="${LR:-2e-4}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.05}"

REBUILD_CLEANED="${REBUILD_CLEANED:-0}"
REBUILD_DATASET="${REBUILD_DATASET:-auto}"
RESUME="${RESUME:-0}"

CONDA_RUN=(env CONDA_NO_PLUGINS=true conda run --no-capture-output -n "$CONDA_ENV")

echo "v6 long experiment"
echo "  data:   $DATA_DIR"
echo "  output: $OUTPUT_DIR"
echo "  model:  voice_aware_v2 d=$D_MODEL layers=$N_LAYERS cross=$N_CROSS_LAYERS seq=$SEQ_LEN"

if [[ "$REBUILD_CLEANED" == "1" ]]; then
  echo "Rebuilding cleaned MusicXML mirror: $CLEANED_DIR"
  "${CONDA_RUN[@]}" python -u scripts/clean_musicxml_corpus.py \
    --output-dir "$CLEANED_DIR"
fi

if [[ ! -f "$APPROVED_LIST" ]]; then
  echo "Approved MusicXML list not found: $APPROVED_LIST" >&2
  echo "Run with REBUILD_CLEANED=1 or set APPROVED_LIST=/path/to/approved_files.txt" >&2
  exit 1
fi

should_build_dataset=0
if [[ "$REBUILD_DATASET" == "1" ]]; then
  should_build_dataset=1
elif [[ "$REBUILD_DATASET" == "auto" ]]; then
  if [[ ! -f "$DATA_DIR/tokenized/train.pt" || ! -f "$DATA_DIR/tokenized/val.pt" ]]; then
    should_build_dataset=1
  fi
fi

if [[ "$should_build_dataset" == "1" ]]; then
  echo "Building v6 dataset: max_bars=$MAX_BARS seq_len=$SEQ_LEN stride=$STRIDE"
  "${CONDA_RUN[@]}" python -u scripts/make_instrumental_v6_dataset.py \
    --cleaned-root "$CLEANED_ROOT" \
    --approved-list "$APPROVED_LIST" \
    --output-dir "$DATA_DIR" \
    --max-voices "$MAX_VOICES" \
    --max-bars "$MAX_BARS" \
    --min-slices "$MIN_SLICES" \
    --seq-len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --limit-movements-per-work "$LIMIT_MOVEMENTS_PER_WORK"
else
  echo "Using existing tokenized dataset under $DATA_DIR"
fi

if [[ ! -f "$DATA_DIR/tokenized/train.pt" || ! -f "$DATA_DIR/tokenized/val.pt" ]]; then
  echo "Missing tokenized dataset under $DATA_DIR" >&2
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  echo "Checking NVIDIA GPU..."
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader
  "${CONDA_RUN[@]}" python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing CPU training")
torch.cuda.init()
print(f"PyTorch CUDA ready: {torch.cuda.get_device_name(0)}")
PY
fi

mkdir -p "$OUTPUT_DIR"

TRAIN_ARGS=(
  scripts/train_instrumental_v6.py
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --architecture voice_aware_v2
  --d-model "$D_MODEL"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --n-cross-layers "$N_CROSS_LAYERS"
  --dropout "$DROPOUT"
  --max-seq-len "$SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --lr-min-ratio "$LR_MIN_RATIO"
  --device "$DEVICE"
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
  TRAIN_ARGS+=(--resume "$CHECKPOINT")
fi

echo "Starting training..."
exec "${CONDA_RUN[@]}" python -u "${TRAIN_ARGS[@]}"
