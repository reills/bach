# Bach Gen

MVP for Bach-style classical guitar generation.

What is shipped in this repo today:

- MusicXML to tokenized bar dataset conversion
- Vocabulary building for token streams
- NoteLM v1 training with resume, validation split, and dry-run support
- Example generation to MusicXML, MIDI, ASCII tab, raw tokens, and eval metrics
- FastAPI score, draft, inpaint, and fingering route contract
- React + Vite + AlphaTab frontend with API mode, local test-data mode, and demo mode

This README is now both the setup guide and the short-form design record for the project. The frontend contract and browser workflow live in [frontend/README.md](/mnt/c/Users/Admin/dev/bach_gen/frontend/README.md).

## 1. Project Shape

The original project plan had more long-range material than the repo needed day to day. The important parts worth preserving are:

- `NoteLM` composes symbolic music only. In v1 it does not generate string/fret tokens.
- The guitar `Tabber` runs after composition and assigns playable `(string, fret)` positions under guitar constraints.
- MusicXML is a render/export format. The backend/frontend editing flow is built around a canonical score model, not MusicXML as the source of truth.
- The generation loop is meant to be fast preview -> inspect -> inpaint/repair -> export, not one giant offline batch step.

## 2. Core Representation Notes

These are the main historical design decisions from the original planning docs that still matter when reading the code:

- `VOICE_v` means an internal continuity track, not a score part or SATB label.
- Absolute pitch is reconstructed from `ABS_VOICE_*` anchors plus `MEL_INT12` deltas.
- `HARM_*` tokens are derived interval features, not the source of pitch truth.
- Token timing uses `TPQ = 24` so binary and ternary subdivisions both quantize cleanly.
- Generated output maps `VOICE_v` directly to canonical `voiceId = v` for the single guitar part.
- Exact doublings may be collapsed during preprocessing, so round-trip reconstruction targets a collapsed symbolic score, not original orchestration detail.

Practical v1 token bundle shape for a pitched onset:

```text
VOICE_v, DUR_{ticks}, MEL_INT12_{delta}, HARM_OCT_{o|NA}, HARM_CLASS_{c|NA}
```

Practical implications:

- illegal generations are usually token-grammar or playability problems, not just "bad MusicXML"
- `/compose` is symbolic generation followed by token parsing and tab assignment
- if symbolic notes parse but cannot be fingered, failure happens in the tab stage, not the language model stage

## 3. Prerequisites

Preferred Python runtime:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python -V
```

Fallback:

```bash
python -m venv .venv
source .venv/bin/activate
```

Environment smoke check:

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh --check
```

That helper verifies the Python env can import the packages the current MVP expects: `pytest`, `torch`, `music21`, `pandas`, and `pyarrow`.

Frontend runtime:

- `frontend/package.json` pins Volta Node `25.3.0`
- `npm install` in `frontend/` is the supported setup path
- running `npm install` at the repo root fails by design because there is no root `package.json`

## 4. Repository Map

- [scripts/make_dataset.py](/mnt/c/Users/Admin/dev/bach_gen/scripts/make_dataset.py): MusicXML to bar-level parquet dataset
- [scripts/build_vocab.py](/mnt/c/Users/Admin/dev/bach_gen/scripts/build_vocab.py): token vocab builder
- [scripts/train_v1.py](/mnt/c/Users/Admin/dev/bach_gen/scripts/train_v1.py): NoteLM v1 trainer
- [scripts/generate_example.py](/mnt/c/Users/Admin/dev/bach_gen/scripts/generate_example.py): end-to-end compose pipeline smoke path
- [scripts/eval_basic.py](/mnt/c/Users/Admin/dev/bach_gen/scripts/eval_basic.py): lightweight token/eval metrics
- [src/api/app.py](/mnt/c/Users/Admin/dev/bach_gen/src/api/app.py): FastAPI app factory and default app
- [src/api/routes/scores.py](/mnt/c/Users/Admin/dev/bach_gen/src/api/routes/scores.py): compose, inpaint, draft, and fingering routes
- [frontend/src/App.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/App.tsx): browser workflow shell

## 5. Build Data

Convert MusicXML into the parquet dataset used by training:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/make_dataset.py \
  --input data/tobis_xml \
  --output data/processed
```

Useful optional flags:

- `--limit N`: process only the first `N` files
- `--shuffle --seed 1337`: shuffle before limiting
- `--validate-roundtrip N`: roundtrip `N` files back to MIDI as a smoke check
- `--voice-mode auto|parts|pitch|events`: choose voice assignment strategy

Build a vocab from the resulting events parquet:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/build_vocab.py \
  --events data/processed/events.parquet \
  --special-tokens "<pad>,<unk>"
```

Default outputs:

- `data/processed/events.parquet`
- `data/processed/plans.parquet`
- `data/processed/vocab.json`

## 6. Train NoteLM v1

Minimal practical first run:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/train_v1.py \
  --events data/processed/events.parquet \
  --vocab data/processed/vocab.json \
  --output-dir out/notelm_v1 \
  --batch-size 4 \
  --max-seq-len 512 \
  --bars-per-seq 4 \
  --shuffle \
  --max-steps 1000 \
  --save-every 200
```

Dry-run the full input pipeline without committing to a real training run:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/train_v1.py \
  --events data/processed/events.parquet \
  --vocab data/processed/vocab.json \
  --output-dir out/notelm_v1 \
  --batch-size 2 \
  --max-seq-len 256 \
  --bars-per-seq 2 \
  --dry-run-batches 2
```

Resume from a checkpoint:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/train_v1.py \
  --events data/processed/events.parquet \
  --vocab data/processed/vocab.json \
  --output-dir out/notelm_v1 \
  --resume out/notelm_v1/notelm_step200.pt
```

Enable a small validation split:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/train_v1.py \
  --events data/processed/events.parquet \
  --vocab data/processed/vocab.json \
  --output-dir out/notelm_v1 \
  --val-split 0.1 \
  --val-every 100
```

Checkpoint files land in `out/notelm_v1/notelm_step<N>.pt` and include:

- model weights
- optimizer state
- serialized config
- vocab path
- UTC timestamp
- original training args

## 7. Compose and Evaluate

The concrete shipped compose path is [scripts/generate_example.py](/mnt/c/Users/Admin/dev/bach_gen/scripts/generate_example.py).

Run it without a model to exercise the full symbolic pipeline:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/generate_example.py \
  --out-dir out/examples
```

Run it with a trained checkpoint:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/generate_example.py \
  --checkpoint out/notelm_v1/notelm_step1000.pt \
  --vocab data/processed/vocab.json \
  --key C \
  --style baroque \
  --difficulty easy \
  --measures 8 \
  --out-dir out/examples/model
```

Outputs written per run:

- `example.musicxml`
- `example.mid`
- `example_tab.txt`
- `tokens.txt`
- `metrics.json`

Quick eval on generated tokens:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/eval_basic.py \
  --token-file out/examples/tokens.txt
```

Or score the dataset/events parquet:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/eval_basic.py \
  --parquet data/processed/events.parquet \
  --vocab data/processed/vocab.json
```

## 8. Run the Backend

Serve the default FastAPI app:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python -m uvicorn src.api.app:app --reload
```

If port 8000 is already in use, specify a different port:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python -m uvicorn src.api.app:app --reload --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Current backend reality:

- `GET /healthz` is live on the default app
- score, draft, inpaint, and fingering route shapes are implemented in [src/api/routes/scores.py](/mnt/c/Users/Admin/dev/bach_gen/src/api/routes/scores.py)
- the default app uses an in-memory repository
- `/compose` exists in the contract, but `src.api.app:app` does not bind a default `compose_service`, so the default app returns `503 compose service is not configured` on `/compose`

To run `/compose` locally against a trained checkpoint, use the supported compose launcher instead:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python -m uvicorn src.api.compose_app:app --reload
```

By default, the launcher uses:

- `out/notelm_v1/notelm_step5000.pt`
- `out/notelm_v1/vocab.json`

You only need env vars if you want to override those defaults:

```bash
BACH_GEN_CHECKPOINT=out/notelm_v1/notelm_step5000.pt \
BACH_GEN_VOCAB=out/notelm_v1/vocab.json \
CONDA_NO_PLUGINS=true conda run -n bach python -m uvicorn src.api.compose_app:app --reload
```

Optional launcher env vars:

- `BACH_GEN_DEVICE` defaults to `cuda` when available, else `cpu`
- `BACH_GEN_MAX_LENGTH` defaults to `512`
- `BACH_GEN_TEMPERATURE` defaults to `1.0`
- `BACH_GEN_TOP_P` defaults to `0.9`

Important routes and payload families:

- `POST /compose`
- `POST /inpaint_preview`
- `POST /commit_draft`
- `POST /discard_draft`
- `POST /alt_positions`
- `POST /apply_fingering`

## 9. Run the Frontend

Install and start the Vite app:

```bash
cd frontend
npm install
npm run dev
```

Run against a backend:

```bash
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Run without a backend:

```bash
cd frontend
VITE_USE_LOCAL_DATA=true npm run dev
```

See [frontend/README.md](/mnt/c/Users/Admin/dev/bach_gen/frontend/README.md) for the browser workflow and frontend/backend contract.

## 10. Exercise the MVP

### A. Compose pipeline

Use the CLI compose path:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/generate_example.py \
  --out-dir out/examples
```

Inspect the generated MusicXML, MIDI, ASCII tab, and metrics in `out/examples/`.

### B. Inpaint flow in the browser

1. Start the frontend in local mode with `VITE_USE_LOCAL_DATA=true npm run dev`.
2. Choose `Local test-data`.
3. Click `Load Test Data`.
4. Click a measure in the score.
5. Click `Generate Preview`.
6. Use `Keep` or `Discard` on the draft banner.

This path uses [frontend/public/test-data/manifest.json](/mnt/c/Users/Admin/dev/bach_gen/frontend/public/test-data/manifest.json) and does not need a backend.

### C. Fingering flow

The shipped fingering flow is backend-driven and requires an `eventHitMap` plus live `/alt_positions` and `/apply_fingering` responses. The current concrete verification path is:

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_fingering.py
```

That test covers:

- resolving a clicked note hit key to an event id
- returning alternate string/fret positions
- applying one or more fingering selections
- confirming MusicXML pitch content stays unchanged while technical tags update

## 11. Focused Test Commands

Backend and training smoke coverage used most often:

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh \
  tests/test_train_v1_smoke.py \
  tests/test_compose_service.py \
  tests/test_api_scores.py \
  tests/test_api_fingering.py
```

Frontend workflow tests:

```bash
cd frontend
npm test -- --run src/App.test.ts src/state/types.test.ts
```
