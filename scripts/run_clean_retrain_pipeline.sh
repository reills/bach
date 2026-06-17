#!/usr/bin/env bash
# Run the clean Bach dataset rebuild -> audit -> train -> generate -> eval flow.
#
# Default run:
#   bash scripts/run_clean_retrain_pipeline.sh
#
# Quick sanity run:
#   bash scripts/run_clean_retrain_pipeline.sh --smoke
#
# The script logs everything to out/pipeline_logs/<timestamp>.log while also
# printing to the terminal.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_INPUT="data/tobis_xml"
PROCESSED_DIR="data/processed_rebuilt"
TRAIN_OUT_DIR="out/notelm_clean_v1"
EXAMPLE_OUT_DIR="${EXAMPLE_OUT_DIR:-}"
LOG_DIR="out/pipeline_logs"

CONDA_ENV="${CONDA_ENV:-bach}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
BARS_PER_SEQ="${BARS_PER_SEQ:-8}"
VAL_EVERY="${VAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100000}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.75}"
GEN_TOP_P="${GEN_TOP_P:-0.85}"
GEN_MAX_LENGTH="${GEN_MAX_LENGTH:-512}"
GEN_KEY="${GEN_KEY:-C}"
GEN_STYLE="${GEN_STYLE:-}"
GEN_MEASURES="${GEN_MEASURES:-8}"

SMOKE=0
SKIP_DATASET=0
SKIP_TRAIN=0
SKIP_GENERATE=0
RESUME_CHECKPOINT=""

usage() {
  cat <<'USAGE'
Run the clean Bach dataset rebuild -> audit -> train -> generate -> eval flow.

Default run:
  bash scripts/run_clean_retrain_pipeline.sh

Quick sanity run:
  bash scripts/run_clean_retrain_pipeline.sh --smoke

Options:
  --smoke                 Use a small dataset/training run to verify plumbing.
  --skip-dataset          Reuse existing PROCESSED_DIR events/vocab and start at audit.
  --skip-train            Reuse an existing checkpoint and only generate/evaluate.
  --skip-generate         Stop after training.
  --resume CHECKPOINT     Resume train_v1.py from a checkpoint.
  --help                  Show this help.

Environment overrides:
  DATA_INPUT, PROCESSED_DIR, TRAIN_OUT_DIR, CONDA_ENV
  MAX_STEPS, BATCH_SIZE, MAX_SEQ_LEN, BARS_PER_SEQ, VAL_EVERY, SAVE_EVERY, TRAIN_EPOCHS
  GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_LENGTH, GEN_KEY, GEN_STYLE, GEN_MEASURES
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --skip-dataset) SKIP_DATASET=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --skip-generate) SKIP_GENERATE=1 ;;
    --resume)
      RESUME_CHECKPOINT="${2:-}"
      [[ -n "$RESUME_CHECKPOINT" ]] || { echo "missing checkpoint after --resume" >&2; exit 2; }
      shift
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ "$SMOKE" == "1" ]]; then
  PROCESSED_DIR="data/processed_smoke"
  TRAIN_OUT_DIR="out/notelm_clean_smoke"
  MAX_STEPS="500"
  BATCH_SIZE="2"
  MAX_SEQ_LEN="512"
  BARS_PER_SEQ="4"
  VAL_EVERY="100"
  SAVE_EVERY="100"
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/clean_retrain_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo
  echo "[$(date '+%F %T')] FAILED with exit code ${exit_code}"
  echo "Log: $LOG_FILE"
  exit "$exit_code"
}
trap on_error ERR

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
  log "RUN: $*"
  "$@"
}

py() {
  CONDA_NO_PLUGINS=true conda run --no-capture-output -n "$CONDA_ENV" python -u "$@"
}

EVENTS_PATH="$PROCESSED_DIR/events.parquet"
DATASET_VOCAB_PATH="$PROCESSED_DIR/vocab.json"
TRAIN_VOCAB_PATH="$TRAIN_OUT_DIR/vocab.json"
CHECKPOINT_PATH=""
if [[ -z "$EXAMPLE_OUT_DIR" ]]; then
  EXAMPLE_OUT_DIR="out/examples/$(basename "$TRAIN_OUT_DIR")_step${MAX_STEPS}"
fi

log "Starting clean retrain pipeline"
echo "Repo:              $ROOT_DIR"
echo "Log:               $LOG_FILE"
echo "Conda env:         $CONDA_ENV"
echo "Input XML:         $DATA_INPUT"
echo "Processed dir:     $PROCESSED_DIR"
echo "Train out dir:     $TRAIN_OUT_DIR"
echo "Max steps:         $MAX_STEPS"
echo "Example out dir:   $EXAMPLE_OUT_DIR"

run bash docs/skills/python-test-env/scripts/run_tests.sh --check

if [[ "$SKIP_DATASET" == "0" ]]; then
  mkdir -p "$PROCESSED_DIR"
  MAKE_DATASET_ARGS=(
    scripts/make_dataset.py
    --input "$DATA_INPUT"
    --output "$PROCESSED_DIR"
    --voice-mode auto
    --max-voices 8
  )
  if [[ "$SMOKE" == "1" ]]; then
    MAKE_DATASET_ARGS+=(--limit 20 --validate-roundtrip 2)
  else
    MAKE_DATASET_ARGS+=(--validate-roundtrip 20)
  fi
  py "${MAKE_DATASET_ARGS[@]}"

  py scripts/build_vocab.py \
    --events "$EVENTS_PATH" \
    --output "$DATASET_VOCAB_PATH"
else
  log "Skipping dataset rebuild; using $EVENTS_PATH"
fi

py scripts/audit_dataset.py \
  --events "$EVENTS_PATH" \
  --vocab "$DATASET_VOCAB_PATH" \
  --output-json "$PROCESSED_DIR/stats.json"

if [[ "$SKIP_TRAIN" == "0" ]]; then
  mkdir -p "$TRAIN_OUT_DIR"
  TRAIN_ARGS=(
    scripts/train_v1.py
    --events "$EVENTS_PATH"
    --vocab "$DATASET_VOCAB_PATH"
    --output-dir "$TRAIN_OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --max-seq-len "$MAX_SEQ_LEN"
    --bars-per-seq "$BARS_PER_SEQ"
    --val-split "$VAL_SPLIT"
    --val-every "$VAL_EVERY"
    --save-every "$SAVE_EVERY"
    --epochs "$TRAIN_EPOCHS"
    --max-steps "$MAX_STEPS"
    --prepend-bos
    --append-eos
    --bos-token "<bos>"
    --eos-token "<eos>"
    --mask-prefix-loss
  )
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
  fi
  py "${TRAIN_ARGS[@]}"
else
  log "Skipping training; will use latest checkpoint in $TRAIN_OUT_DIR"
fi

if [[ "$SKIP_GENERATE" == "0" ]]; then
  CHECKPOINT_PATH="$(find "$TRAIN_OUT_DIR" -maxdepth 1 -type f -name 'notelm_step*.pt' \
    | sort -V \
    | tail -1)"
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "No checkpoints found in $TRAIN_OUT_DIR" >&2
    exit 1
  fi
  CHECKPOINT_STEP="$(basename "$CHECKPOINT_PATH" .pt | sed 's/^notelm_step//')"
  if [[ "$EXAMPLE_OUT_DIR" == "out/examples/$(basename "$TRAIN_OUT_DIR")_step${MAX_STEPS}" ]]; then
    EXAMPLE_OUT_DIR="out/examples/$(basename "$TRAIN_OUT_DIR")_step${CHECKPOINT_STEP}"
  fi
  log "Generating from checkpoint: $CHECKPOINT_PATH"
  if [[ ! -f "$TRAIN_VOCAB_PATH" ]]; then
    echo "Expected trained vocab not found: $TRAIN_VOCAB_PATH" >&2
    exit 1
  fi

  GENERATE_ARGS=(
    scripts/generate_example.py
    --checkpoint "$CHECKPOINT_PATH" \
    --vocab "$TRAIN_VOCAB_PATH" \
    --out-dir "$EXAMPLE_OUT_DIR" \
    --key "$GEN_KEY" \
    --measures "$GEN_MEASURES" \
    --max-length "$GEN_MAX_LENGTH" \
    --temperature "$GEN_TEMPERATURE" \
    --top-p "$GEN_TOP_P" \
    --use-grammar-mask
  )
  if [[ -n "$GEN_STYLE" ]]; then
    STYLE_TOKEN="STYLE_$(printf '%s' "$GEN_STYLE" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//' | tr '[:lower:]' '[:upper:]')"
    if CONDA_NO_PLUGINS=true conda run --no-capture-output -n "$CONDA_ENV" python - "$TRAIN_VOCAB_PATH" "$STYLE_TOKEN" <<'PY'
import json
import sys
from pathlib import Path
vocab = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if sys.argv[2] in vocab else 1)
PY
    then
      GENERATE_ARGS+=(--style "$GEN_STYLE")
    else
      echo "Warning: skipping GEN_STYLE=$GEN_STYLE because $STYLE_TOKEN is not in $TRAIN_VOCAB_PATH"
    fi
  fi
  py "${GENERATE_ARGS[@]}"

  py scripts/eval_basic.py \
    --token-file "$EXAMPLE_OUT_DIR/tokens.txt" \
    --vocab "$TRAIN_VOCAB_PATH" \
    --output-json "$EXAMPLE_OUT_DIR/metrics_with_vocab.json" \
    --quiet

  log "Key generated metrics"
  CONDA_NO_PLUGINS=true conda run --no-capture-output -n "$CONDA_ENV" python - "$EXAMPLE_OUT_DIR/metrics_with_vocab.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
for key in (
    "interval_range_ok",
    "harm_mismatch_count",
    "counterpoint_parallel_fifths",
    "counterpoint_parallel_octaves",
    "counterpoint_voice_crossings",
    "counterpoint_static_voice_rate",
    "counterpoint_avg_active_voices",
):
    print(f"{key}: {metrics.get(key)}")
PY
fi

log "DONE"
echo "Log: $LOG_FILE"
echo "Dataset stats: $PROCESSED_DIR/stats.json"
echo "Train dir: $TRAIN_OUT_DIR"
if [[ "$SKIP_GENERATE" == "0" ]]; then
  echo "Example dir: $EXAMPLE_OUT_DIR"
fi
