# AGENTS.md - bach-gen

Project goal: generate clean MusicXML/sheet-music scores in a Bach-like instrumental contrapuntal style.

Primary target is **not** Bach chorale harmonization. The desired output is 2-4 voice keyboard/instrumental counterpoint: inventions, sinfonias, fugue-like textures, suites, partitas, and related Baroque writing.

Current strategic decision:

- Keep the frontend/API/canonical-score/export infrastructure.
- Treat flat-token NoteLM and chorale-v2 SATB-token experiments as legacy/prototype paths.
- Do not keep adding decoder rules to hide model collapse.
- The next model engine should be interval-aware symbolic instrumental counterpoint with compound musical events and type-specific prediction heads.
- Audio-generation models may be useful later as critics/rerankers, but they are not the primary generator because the output must be clean notation.

## Environment

- Preferred Python runtime: `conda run -n bach python`
- For commands that should stream output, use:
  `CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u ...`
- When using CUDA in Python scripts, initialize/resolve CUDA before reading parquet datasets with pandas/pyarrow. In this project, reading parquet first can make later `torch.cuda` initialization fail with `CUDA driver initialization failed` even when `nvidia-smi` sees the GPU. Prefer resolving the device and calling the seed/CUDA initialization before dataset construction/loading.
- If CUDA fails unexpectedly, check `nvidia-smi` and a minimal PyTorch CUDA probe before falling back to CPU. Do not silently run full training on CPU.
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

Legacy NoteLM/v1 pipeline exists and may still be useful for reference, tests, and infrastructure. It is not the preferred research direction for the next model engine.

Preferred next model direction:

- Build an instrumental dataset from MusicXML/MIDI keyboard works, starting with inventions/sinfonias and expanding to WTC, suites, partitas, fugues, and related Baroque/classical contrapuntal works.
- Represent one timestep as a compound musical event/slice, not a flat text-token sequence.
- Include relative melodic intervals per voice, vertical intervals between voices, local harmonic/scale-degree context, absolute register, duration, and NOTE/HOLD/REST state.
- Use separate model heads/losses for voice state, melodic interval, duration, register/pitch, harmonic context, and next-position/advance.
- Gate training with tiny-overfit and held-out objective metrics before listening.

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

The immediate quality focus is model-engine redesign, not decoder patching.

- Preserve independent 2-4 voice motion.
- Avoid stuck soprano/bass and repeated sonority collapse.
- Model relative melodic intervals and vertical harmonic intervals explicitly.
- Maintain local harmonic continuity and cadence behavior.
- Reduce voice crossings, spacing violations, and parallel fifths/octaves.
- Use repeatable batch evaluation instead of judging one sample.

Required objective gates for any new model path:

- Tiny overfit on 1-5 instrumental pieces must reach near-perfect per-head accuracy and generate coherent continuations.
- Held-out eval must report per-voice accuracy, stuck-voice rate, repeated-sonority rate, interval-distribution match, crossing/spacing rates, and parallel fifth/octave rates.
- Do not proceed to long training runs if tiny overfit fails.
