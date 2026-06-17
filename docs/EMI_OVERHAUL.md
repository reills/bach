# EMI-Inspired Counterpoint Overhaul

The current `instrumental_v4` path is the right base: it already uses fixed-grid
compound events and per-field heads. The missing layer is EMI-style symbolic
analysis: short fragments, style signatures, compatibility scoring, and retrieval
conditioned generation.

## Target Shape

```text
MusicXML corpus
  -> v3 fixed-grid voice slices
  -> v4 measure plans
  -> EMI fragment/signature database
  -> typed v5 Parquet event table
  -> retrieval-conditioned compound Transformer
  -> counterpoint + novelty verifier
  -> MusicXML/MIDI export
```

This keeps the neural model, but stops asking it to discover all form, motif,
cadence, and compatibility structure from raw slices alone.

## Dataset Formats

Use Parquet as the canonical dataset source for model paths after v4. JSON and
JSONL stay useful for debugging/interchange, but the main event table should be
typed columns:

- `events.parquet`: canonical v5 source table with one fixed-grid row per slice.
- `vocab.json`: bounded field specs and EMI bucket maps, not raw fragment IDs.
- `emi_fragments.jsonl`: retrieval/debug metadata with source fragment IDs.
- `train_emi_fragments.jsonl`: retrieval source for training/generation.
- `val_emi_fragments.jsonl`: held-out diagnostic fragments only.
- `metadata.json`: split and build provenance.
  - Includes `conditioning_coverage` so builds report non-default rates for
    `phrase_role`, `speac_label`, `cadence_target`, `harmonic_function`,
    `local_key_pc`, and retrieval buckets.

Later training can add pre-tokenized tensor/binary shards derived from Parquet.
Those shards are training caches, not the canonical dataset.

## Current First Step

`src/emi/fragments.py` mines short cells from v3/v4 pieces:

- voice id
- start bar/position
- phrase role heuristic
- melodic interval contour
- rhythm steps
- vertical interval context
- start/end scale degree
- compact contour hash
- source fingerprint for novelty checks

Build a sample fragment database:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/build_emi_fragments.py \
  --dataset data/instrumental_v4/keyboard_overture_cnorm_outer2_v4.json \
  --format v4 \
  --output data/emi_fragments/keyboard_overture_cnorm_outer2.fragments.jsonl \
  --length-slices 8 \
  --hop-slices 4
```

Default `length-slices=8` is two quarter-note beats at the project grid
(`grid_ticks=6`, `tpq=24`). That is deliberate: use signatures and cells, not
long copied passages.

Build a Parquet-first v5 dataset from an existing v4 dataset:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/make_instrumental_v5_dataset.py \
  --v4-dataset data/instrumental_v4/keyboard_overture_cnorm_outer2_v4.json \
  --output-dir data/instrumental_v5/keyboard_overture_cnorm_outer2_v5 \
  --length-slices 8 \
  --hop-slices 4
```

Pre-tokenize the v5 Parquet table into fixed-length tensor windows:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/tokenize_instrumental_v5.py \
  --events data/instrumental_v5/keyboard_overture_cnorm_outer2_v5/events.parquet \
  --vocab data/instrumental_v5/keyboard_overture_cnorm_outer2_v5/vocab.json \
  --output-dir data/instrumental_v5/keyboard_overture_cnorm_outer2_v5/tokenized \
  --seq-len 512
```

Run a small v5 overfit check:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/train_instrumental_v5.py \
  --data-dir data/instrumental_v5/keyboard_overture_cnorm_outer2_v5 \
  --tokenized-dir data/instrumental_v5/keyboard_overture_cnorm_outer2_v5/tokenized \
  --output-dir out/instrumental_v5_overfit \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 2 \
  --batch-size 1 \
  --max-seq-len 512 \
  --max-steps 500 \
  --lr 0.0003 \
  --device cuda
```

Generate/export from a v5 checkpoint:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python scripts/generate_instrumental_v5.py \
  --checkpoint out/instrumental_v5_overfit/checkpoint_latest.pt \
  --data-dir data/instrumental_v5/keyboard_overture_cnorm_outer2_v5 \
  --out-dir out/instrumental_v5_overfit/generated \
  --samples 10 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --device cuda
```

## Phase 1: Better Fragment Analysis

Improve `infer_phrase_role` from simple heuristics to Baroque-specific labels:

- `SUBJECT_ENTRY`
- `ANSWER_ENTRY`
- `COUNTERSUBJECT`
- `EPISODE`
- `SEQUENCE`
- `CADENTIAL_PREPARATION`
- `CADENCE`
- `CLOSING`

Implementation notes:

- Detect subject entries by repeated contour/rhythm hashes across voices and
  transpositions.
- Detect sequences by contour repetition under transposition, not exact pitch.
- Detect cadences from local scale-degree endpoints and vertical interval motion.
- Keep labels probabilistic/heuristic at first; do not block model work on a
  perfect musicological analyzer.

## Phase 2: Retrieval-Conditioned Representation

Add a v5 representation rather than mutating v4 in place. The v5 EMI fields are
bounded symbolic plan/retrieval fields:

- `phrase_role`
- `speac_label`
- `cadence_target`
- `harmonic_function`
- `local_key_pc`
- `retrieved_contour_bucket`
- `retrieved_rhythm_bucket`

Do not put raw fragment IDs, source piece names, or exact measure origins in the
main LM fields. Raw IDs belong in `emi_fragments.jsonl` for retrieval,
diagnostics, and novelty checks.

## Phase 3: Generation Loop

Generation should become plan-first:

1. Build or sample a high-level role plan.
2. For each region, query compatible fragments using role, key/mode, start/end
   degree, register, and previous endpoint.
3. Condition the generator on the selected fragment signatures.
4. Generate compound rows.
5. Reject/rerank with existing counterpoint metrics plus source-overlap checks.

The current `rank_fragments()` API is intentionally small so it can be used by a
future generator without importing training code.

Validation checks now cover:

- `hybrid` transformer failures do not call EMI unless debug fallback is
  explicitly enabled.
- `hybridAllowEmiFallback=true` is labeled
  `transformer_exception_debug_only` in diagnostics.
- Hybrid context can be written into actual v5 rows via bounded field IDs.
- v5 dataset metadata reports conditioning coverage so sparse/default labels are
  visible before training.

## API Engine Switch

The compose API can now run the legacy transformer path, the EMI-style symbolic
baseline path, or the retrieval-conditioned hybrid path:

```json
{
  "render_mode": "piano",
  "constraints": {
    "engine": "hybrid",
    "key": "D minor",
    "measures": 8,
    "texture": 4
  }
}
```

Runtime defaults:

- `BACH_GEN_ENGINE=transformer|emi|hybrid`
- `BACH_GEN_EMI_FRAGMENTS=data/emi_fragments/example.fragments.jsonl`
- `BACH_GEN_HYBRID_EMI_DEBUG_FALLBACK=false`

Current semantics:

- `transformer` keeps the existing NoteLM/token path.
- `emi` bypasses checkpoints and composes a notation-first canonical score using
  protected subject/countersubject cells, SPEAC-like role sequencing, optional
  fragment retrieval, and contrapuntal cleanup. This is a diagnostic/baseline
  engine, not the preferred composer.
- `hybrid` builds a phrase/SPEAC/cadence plan, retrieves compatible fragment
  signatures, attaches bounded conditioning fields to the transformer generation
  config, generates/reranks candidates, reports validity and novelty metrics, and
  rejects excessive source overlap. It does not fall back to EMI unless
  `BACH_GEN_HYBRID_EMI_DEBUG_FALLBACK=true` or `hybridAllowEmiFallback=true` is
  set explicitly for debugging.

This is not exact historical EMI. It is the practical integration layer that lets
EMI-style symbolic structure coexist with transformer work while the v5
retrieval-conditioned model matures.

## Required Gates

Do not run long training before these pass:

- Tiny overfit on 1-5 pieces reaches near-perfect per-head accuracy.
- Tiny continuation uses the requested phrase-role/fragment conditioning.
- Held-out eval reports stuck-voice rate, repeated-sonority rate, interval
  distribution match, crossing/spacing, parallel fifth/octave rate, and source
  overlap.
- Source novelty gate: reject generations with excessive exact fragment chains or
  high contiguous source match.

## Non-Goals

- Do not revive flat-token NoteLM as the main path.
- Do not add more decoder rules to hide collapse.
- Do not train guitar ergonomics and counterpoint in one model yet; compose first,
  arrange/tab second.
