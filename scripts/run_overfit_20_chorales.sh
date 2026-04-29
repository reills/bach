#!/usr/bin/env bash
set -euo pipefail

# Reproducible "can the model overfit 20 clean chorales?" experiment.
#
# Override knobs from the shell, for example:
#   MAX_STEPS=6000 D_MODEL=384 N_LAYERS=6 bash scripts/run_overfit_20_chorales.sh

SOURCE_EVENTS="${SOURCE_EVENTS:-data/processed_rebuilt/events.parquet}"
SOURCE_BARPLANS="${SOURCE_BARPLANS:-data/processed_rebuilt/barplans.parquet}"
DATA_DIR="${DATA_DIR:-data/overfit_20_chorales}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-out/overfit_20_chorales}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-out/eval/overfit_20_chorales}"

MAX_STEPS="${MAX_STEPS:-3000}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BARS_PER_SEQ="${BARS_PER_SEQ:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-4}"
LR="${LR:-5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-1337}"

PROMPT_BARS="${PROMPT_BARS:-2}"
CONTINUATION_BARS="${CONTINUATION_BARS:-6}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-1.0}"

py() {
  CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u "$@"
}

mkdir -p "$DATA_DIR" "$TRAIN_OUT_DIR" "$EVAL_OUT_DIR"

echo "[1/5] Preparing 20 clean four-part chorales"
py scripts/prepare_overfit_chorales.py \
  --events "$SOURCE_EVENTS" \
  --barplans "$SOURCE_BARPLANS" \
  --output "$DATA_DIR" \
  --limit 20 \
  --min-bars 8 \
  --min-pct-4plus 1.0

echo "[2/5] Building overfit vocab"
py scripts/build_vocab.py \
  --events "$DATA_DIR/events.parquet" \
  --output "$DATA_DIR/vocab.json" \
  --special-tokens "<pad>,<unk>"

echo "[3/5] Auditing selected dataset"
py scripts/audit_dataset.py \
  --events "$DATA_DIR/events.parquet" \
  --vocab "$DATA_DIR/vocab.json" \
  --output-json "$DATA_DIR/audit.json"

echo "[4/5] Training overfit model"
py scripts/train_v1.py \
  --events "$DATA_DIR/events.parquet" \
  --vocab "$DATA_DIR/vocab.json" \
  --output-dir "$TRAIN_OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --bars-per-seq "$BARS_PER_SEQ" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --d-model "$D_MODEL" \
  --n-heads "$N_HEADS" \
  --n-layers "$N_LAYERS" \
  --dropout 0.0 \
  --weight-decay 0.0 \
  --lr "$LR" \
  --warmup-steps "$WARMUP_STEPS" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --log-every 25 \
  --save-every 500 \
  --seed "$SEED" \
  --device "$DEVICE"

CHECKPOINT_PATH="$(find "$TRAIN_OUT_DIR" -maxdepth 1 -name 'notelm_step*.pt' -print | sort -V | tail -n 1)"
if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "No checkpoint found in $TRAIN_OUT_DIR" >&2
  exit 1
fi

echo "[5/5] Evaluating continuation from first ${PROMPT_BARS} bars"
py scripts/eval_overfit_continuation.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --vocab "$TRAIN_OUT_DIR/vocab.json" \
  --events "$DATA_DIR/events.parquet" \
  --out-dir "$EVAL_OUT_DIR" \
  --samples 20 \
  --prompt-bars "$PROMPT_BARS" \
  --continuation-bars "$CONTINUATION_BARS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --use-grammar-mask \
  --texture 4 \
  --device "$DEVICE" \
  --seed "$SEED"

echo "checkpoint: $CHECKPOINT_PATH"
echo "summary: $EVAL_OUT_DIR/summary.json"
