# Instrumental v6

Instrumental v6 is the factorized variable-voice counterpoint engine. One checkpoint
has a six-voice capacity, while each piece carries an active `voice_count` mask.
Voice and pair heads are shared, so the model does not contain separate `v0`, `v1`,
or two-part-only output fields.

The current model architecture is `voice_aware_v2`: causal temporal attention runs
for each voice before cross-voice attention. The legacy `pooled_v1` checkpoint path
is loadable for comparison, but it loses separate long-term voice histories and is
not the recommended training target.

The representation includes:

- Compound global, voice, and all-pair events.
- Relative melodic interval, absolute register, duration, scale degree, and state.
- Vertical interval, motion, consonance, crossing, spacing, and perfect-parallel labels.
- Form, section role, development operation, entry voice, and local key context.
- Work-level train/validation splitting and SHA-256 source deduplication.

Build or audit a clean MusicXML mirror first; see
[`docs/MUSICXML_CLEANING.md`](MUSICXML_CLEANING.md).

## Tiny Overfit Gate

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u \
  scripts/make_instrumental_v6_dataset.py \
  --output-dir data/instrumental_v6/tiny_mixed_gate_v2 \
  --max-voices 6 \
  --max-bars 32 \
  --limit-per-source 1 \
  --limit-movements-per-work 1 \
  --seq-len 128 \
  --stride 64 \
  --overfit-all

CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u \
  scripts/train_instrumental_v6.py \
  --data-dir data/instrumental_v6/tiny_mixed_gate_v2 \
  --output-dir out/instrumental_v6_tiny_gate_v2 \
  --architecture voice_aware_v2 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 6 \
  --n-cross-layers 2 \
  --dropout 0 \
  --max-seq-len 256 \
  --batch-size 1 \
  --max-steps 1500 \
  --lr 3e-4 \
  --device cuda \
  --amp
```

## Full GPU Training

```bash
OUTPUT_DIR=out/instrumental_v6_voice_aware_v2 \
MAX_STEPS=8000 \
BATCH_SIZE=2 \
bash scripts/run_instrumental_v6_voice_aware_training.sh
```

## Generate

Change `--voices` and `--form` without changing checkpoints:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u \
  scripts/generate_instrumental_v6.py \
  --checkpoint out/instrumental_v6_voice_aware_v2/checkpoint_best.pt \
  --data-dir data/instrumental_v6/clean_bach_large_v1 \
  --out-dir out/instrumental_v6_voice_aware_v2/generated_best/bwv850 \
  --voices 4 \
  --form fugue \
  --piece-id BWV_0850_m01 \
  --prompt-rows 64 \
  --max-new-rows 192 \
  --candidates 8 \
  --temperature 0.4 \
  --duration-temperature 0.8 \
  --duration-prior-strength 0.1 \
  --device cuda
```

## EMI Signature Memory

Build a v6 EMI-style fragment database from the same dataset used by the
checkpoint:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u \
  scripts/build_emi_v6_fragments.py \
  --data-dir data/instrumental_v6/clean_bach_long_v1 \
  --output data/instrumental_v6/clean_bach_long_v1/emi_v6_fragments.jsonl \
  --length-slices 8 \
  --hop-slices 4
```

Generate with transformer logits softly biased by compatible EMI signatures:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u \
  scripts/generate_instrumental_v6.py \
  --checkpoint out/instrumental_v6_voice_aware_384_long_v1/checkpoint_best.pt \
  --data-dir data/instrumental_v6/clean_bach_long_v1 \
  --out-dir out/instrumental_v6_voice_aware_384_long_v1/generated_emi \
  --voices 2 \
  --form invention \
  --prompt-rows 64 \
  --max-new-rows 192 \
  --candidates 8 \
  --temperature 0.4 \
  --duration-temperature 0.8 \
  --duration-prior-strength 0.1 \
  --emi-fragments data/instrumental_v6/clean_bach_long_v1/emi_v6_fragments.jsonl \
  --emi-bias-strength 0.8 \
  --device cuda
```

Use the same memory from the frontend/backend:

```bash
BACH_GEN_ENGINE=instrumental_v6 \
BACH_GEN_DEVICE=cuda \
BACH_GEN_V6_CHECKPOINT=out/instrumental_v6_voice_aware_384_long_v1/checkpoint_best.pt \
BACH_GEN_V6_DATA_DIR=data/instrumental_v6/clean_bach_long_v1 \
BACH_GEN_V6_EMI_FRAGMENTS=data/instrumental_v6/clean_bach_long_v1/emi_v6_fragments.jsonl \
BACH_GEN_V6_EMI_BIAS_STRENGTH=0.8 \
BACH_GEN_V6_EMI_FRAGMENT_LIMIT=4 \
uvicorn src.api.compose_app:app --port 8001
```

Generation uses a dynamic all-pair beam. Crossings and parallel perfect intervals
remain hard constraints for every active voice pair; candidate reranking also reports
activity, repeated sonorities, strong-beat dissonance, source overlap, and the matching
source-window baseline. It also reports tonal outliers, transposition-invariant
subject-head recurrence, and a global coherence report covering subject recurrence
across opening/middle/closing sections, cadence placement, development-role
coverage, local-key arc, and phrase-role balance. If an EMI fragment database is
supplied, compatible short interval/rhythm signatures are retrieved by voice,
role, key, and local continuity, then applied as a soft pitch bias inside the
transformer decoder. The duration prior is learned from pieces with the requested
form and voice count; duration temperature is separate from pitch temperature so
rhythmic variety does not require randomizing pitch.
