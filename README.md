# Bach Gen

Symbolic sheet-music generation for Bach-like contrapuntal keyboard music.

The project goal is **not** just Bach chorale harmonization. The target is clean
MusicXML/MIDI generation for 2-4 voice instrumental counterpoint: inventions,
sinfonias, fugue-like textures, suites, partitas, and related Baroque keyboard
writing.

## Current Status

The frontend and API shell are still useful. The model engine needs a redesign.

What is worth keeping:

- React/Vite frontend and score-viewing workflow.
- FastAPI compose service structure.
- Canonical score/MusicXML/MIDI export utilities.
- MusicXML parsing/eventization helpers.
- Dataset audit and evaluation infrastructure.
- Counterpoint/voice-leading metrics.

What is now considered legacy:

- Flat autoregressive token-stream NoteLM as the primary music model.
- Chorale-v2 vertical SATB token experiments.
- Decoder-rule fixes for collapsed generation.
- Guitar-first assumptions in older docs.

The old model path can memorize tiny datasets, so the training loop is not
fundamentally broken. The failure is representation/objective mismatch: flat
tokens let easy metadata, duration, position, and held-note predictions dominate
while soprano/bass/inner-voice motion remains weak.

## Target Architecture

The next engine should be an **interval-aware symbolic instrumental
counterpoint model**.

One model timestep should represent a musical event/slice, not a sequence of
text-like tokens:

```text
bar_position
voice_count
key / local_harmony / scale_degree_context

for each active voice:
  state = NOTE | HOLD | REST
  absolute_register
  melodic_interval_from_previous_note
  duration
  tie_or_hold

vertical:
  bass_to_upper_intervals
  adjacent_voice_intervals
  consonance_or_dissonance_class
  spacing_features
```

The model should use separate losses/heads for musically distinct outputs:

- voice state: note/hold/rest
- melodic interval per voice
- absolute pitch/register correction
- duration/rhythm
- local harmonic class or scale-degree context
- next position/advance

This is different from the legacy flat-token model, where one shared softmax
predicts unrelated tokens such as `BAR`, `DUR_24`, `SOP_69`, and `KEY_Bb`.

## Practical Roadmap

1. Build a curated instrumental dataset.
   - Start with Bach inventions and sinfonias.
   - Add WTC fugues/preludes, suites, partitas, and other keyboard works.
   - Keep chorales only as auxiliary/simple validation data, not the main goal.

2. Create `instrumental_v3` representation.
   - Preserve voice identity where possible.
   - Encode relative melodic intervals and vertical intervals.
   - Include absolute register so interval sequences do not drift.
   - Include harmonic/measure context for continuity.

3. Train in strict stages.
   - Tiny overfit: 1-5 pieces must reproduce.
   - Small corpus: inventions only.
   - Broader corpus: keyboard/instrumental works.
   - Fine-tune by texture: 2-part invention, 3-part sinfonia, fugue-like.

4. Gate with objective metrics before listening.
   - per-voice prediction accuracy
   - stuck-voice rate
   - repeated sonority rate
   - voice crossing rate
   - spacing violations
   - parallel fifth/octave rate
   - interval distribution vs reference
   - cadence/harmonic continuity

5. Use symbolic generation as the main path.
   - Audio/Suno-style models are not the primary path because the output must be
     clean sheet music.
   - Audio models may later help as critics/rerankers or preference signals.
   - ACE-Step 1.5 integration is downstream only: v5 writes MusicXML/MIDI/tab
     and ACE-Step sidecars for audio styling or LoRA rendering.

## Environment

Preferred Python runtime:

```bash
conda run -n bach python
```

For streaming output:

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n bach python -u ...
```

CUDA/GPU rules:

- Ask for explicit escalated permission before GPU/CUDA commands.
- Initialize CUDA before reading parquet datasets with pandas/pyarrow.
- If CUDA fails, check `nvidia-smi` and a minimal PyTorch CUDA probe before
  falling back.
- Do not silently run full training on CPU.

Environment check:

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh --check
```

Full Python test suite:

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh
```

## Backend

Run the compose backend from the repo root:

```bash
conda activate bach
uvicorn src.api.compose_app:app --reload --port 8001
```

The Vite frontend proxies API calls to `http://localhost:8001`.

Useful backend environment variables:

- `BACH_GEN_CHECKPOINT`
- `BACH_GEN_VOCAB`
- `BACH_GEN_DEVICE`
- `BACH_GEN_ENGINE`
- `BACH_GEN_V6_CHECKPOINT`
- `BACH_GEN_V6_DATA_DIR`
- `BACH_GEN_V6_CANDIDATES`
- `BACH_GEN_V6_EMI_FRAGMENTS`
- `BACH_GEN_V6_EMI_BIAS_STRENGTH`
- `BACH_GEN_V6_EMI_FRAGMENT_LIMIT`
- `BACH_GEN_USE_GRAMMAR_MASK`
- `BACH_GEN_QUALITY_PASSES`

For the current voice-aware instrumental v6 model:

```bash
BACH_GEN_ENGINE=instrumental_v6 \
BACH_GEN_DEVICE=cuda \
BACH_GEN_V6_CHECKPOINT=out/instrumental_v6_voice_aware_v2/checkpoint_best.pt \
BACH_GEN_V6_DATA_DIR=data/instrumental_v6/clean_bach_large_v1 \
uvicorn src.api.compose_app:app --port 8001
```

To add EMI-style signature retrieval to v6, first build
`data/instrumental_v6/clean_bach_long_v1/emi_v6_fragments.jsonl` with
`scripts/build_emi_v6_fragments.py`, then set `BACH_GEN_V6_EMI_FRAGMENTS` to that
JSONL path.

## Frontend

Run separately:

```bash
cd frontend
npm run dev
```

The frontend is not the current bottleneck. Keep it working while replacing the
model engine.

## Important Paths

- `frontend/src/App.tsx`: frontend shell.
- `frontend/src/components/ScoreViewer.tsx`: score viewer.
- `src/api/compose_app.py`: compose backend entrypoint.
- `src/api/compose_service.py`: compose service.
- `src/api/canonical/`: canonical score model and conversion utilities.
- `src/tokens/eventizer.py`: MusicXML eventization.
- `src/tokens/roundtrip.py`: token round-trip helpers.
- `src/music/counterpoint.py`: counterpoint metrics.
- `src/instrumental_v5/form_planner.py`: CAST-style form plans for v5
  conditioning.
- `src/instrumental_v5/ace_step.py`: ACE-Step 1.5 setup and downstream handoff
  metadata.
- `docs/ACE_STEP_15_INTEGRATION.md`: ACE-Step setup and handoff workflow.
- `scripts/train_v1.py`: legacy flat-token trainer; useful reference, not the
  desired final model.
- `src/models/notelm/`: legacy NoteLM model.

## Cleanup Policy

- `out/` is disposable experiment output unless a checkpoint is explicitly
  marked as kept.
- Keep raw source data under `data/tobis_xml` intact.
- Deduplicate during dataset building by SHA-256 instead of deleting source
  files.
- Prefer rebuilding processed datasets over preserving stale experimental
  derivatives.

## Development Rules

- Prefer small diffs.
- Do not refactor unrelated code.
- Do not weaken or remove tests to make them pass.
- If adding a feature, add focused tests in the same run.
- Do not hard-code tests; test real behavior.
- Prefer existing parsing, eventization, round-trip, canonical score, and export
  helpers over parallel implementations.
