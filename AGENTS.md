# AGENTS.md - bach-gen

Project goal: generate MusicXML/sheet-music scores in a Bach-like contrapuntal style.

## Environment

- Preferred Python runtime: `conda run -n bach python`
- For commands that should stream output, use:
  `CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u ...`
- Optional local venv fallback:
  `python -m venv .venv && source .venv/bin/activate`
- Environment check:
  `bash docs/skills/python-test-env/scripts/run_tests.sh --check`
- Full Python test suite:
  `bash docs/skills/python-test-env/scripts/run_tests.sh`

## Backend

Run the backend from the repo root using `compose_app.py`, not `app.py`:

```bash
conda activate bach
uvicorn src.api.compose_app:app --reload --port 8001
```

To use a specific checkpoint:

```bash
BACH_GEN_CHECKPOINT=out/notelm_clean_v1/notelm_step7895.pt \
BACH_GEN_VOCAB=out/notelm_clean_v1/vocab.json \
BACH_GEN_USE_GRAMMAR_MASK=true \
uvicorn src.api.compose_app:app --reload --port 8001
```

Useful backend environment variables:

- `BACH_GEN_CHECKPOINT`
- `BACH_GEN_VOCAB`
- `BACH_GEN_DEVICE`
- `BACH_GEN_USE_GRAMMAR_MASK`
- `BACH_GEN_QUALITY_PASSES`

## Frontend

Run separately:

```bash
cd frontend
npm run dev
```

The Vite dev server proxies API calls to `http://localhost:8001`.

## Data And Training

Clean rebuild/audit/train/generate pipeline:

```bash
bash scripts/run_clean_retrain_pipeline.sh
```

Reuse an existing rebuilt dataset:

```bash
bash scripts/run_clean_retrain_pipeline.sh --skip-dataset
```

Generate/evaluate from the latest checkpoint without retraining:

```bash
bash scripts/run_clean_retrain_pipeline.sh --skip-dataset --skip-train
```

Dataset audit only:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u scripts/audit_dataset.py \
  --events data/processed_rebuilt/events.parquet \
  --vocab data/processed_rebuilt/vocab.json \
  --output-json data/processed_rebuilt/stats.json
```

Batch generation evaluation:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u scripts/eval_generation_batch.py \
  --checkpoint out/notelm_clean_v1/notelm_step7895.pt \
  --vocab out/notelm_clean_v1/vocab.json \
  --samples 20 \
  --texture 4 \
  --quality-passes 4 \
  --use-grammar-mask \
  --out-dir out/eval/notelm_clean_v1_step7895
```

## Development Rules

- Prefer small diffs.
- Do not refactor unrelated code.
- Do not weaken or remove tests to make them pass.
- If adding a feature, add focused tests in the same run.
- Do not hard-code tests; test real behavior.
- Keep raw source data under `data/tobis_xml` intact. Deduplicate during dataset build by SHA-256 rather than deleting source files.
- Prefer existing tokenization/eventization/round-trip helpers over creating parallel parsing logic.

## Current Architecture To Reuse

- Token schema/tokenizer: `src/tokens/schema.py`, `src/tokens/tokenizer.py`
- MusicXML eventization: `src/tokens/eventizer.py`
- Round-trip helpers: `src/tokens/roundtrip.py`
- Dataset/vocab builders: `scripts/make_dataset.py`, `scripts/build_vocab.py`
- Dataset audit: `scripts/audit_dataset.py`
- Bar descriptors: `src/dataio/descriptors.py`
- Dataset loading/packing/collation: `src/dataio/dataset.py`, `src/dataio/collate_miditok.py`
- NoteLM model: `src/models/notelm/model.py`
- Sampling/decoding: `src/utils/decoding/sampler.py`, `src/utils/decoding/rules.py`, `src/utils/decoding/scg.py`
- Counterpoint metrics: `src/music/counterpoint.py`
- Candidate reranking: `src/inference/rerank.py`
- Compose API: `src/api/compose_launcher.py`, `src/api/compose_service.py`, `src/api/compose_app.py`
- Frontend shell/viewer: `frontend/src/App.tsx`, `frontend/src/components/ScoreViewer.tsx`
- Local frontend mock data: `frontend/src/mock/localData.ts`

## Current Quality Focus

The dataset now audits cleanly. The main generation-quality work is:

- Preserve 3-4 active voices during generation.
- Reduce voice crossings and spacing violations.
- Avoid parallel fifths/octaves.
- Keep harmonic metadata repaired and valid.
- Use repeatable batch evaluation instead of judging one sample.
