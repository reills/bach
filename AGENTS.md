# AGENTS.md — bach-gen

## How to run
- Preferred runtime: `conda run -n bach python`
- Optional venv fallback: `python -m venv .venv && source .venv/bin/activate`
- Environment/test helper: `bash docs/skills/python-test-env/scripts/run_tests.sh --check`
- Tests: `bash docs/skills/python-test-env/scripts/run_tests.sh`

## Rules
- Do one task per run.
- Prefer small diffs; don't refactor unrelated code.
- Always run the test command from the "Test Command" section of TODO.md (targeted tests only, not the full suite).
- Don't weaken or remove tests to make them pass.
- If you add a new feature without tests, add tests in the same run.
- Do not hard-code tests; test real behavior.
- If tests fail, find and fix the underlying code issue before proceeding.
- Write summary of what changed to `finished.md` when finished (the task runner archives it automatically).
- Don't add heavy deps without asking.

## Current Baseline
These parts already exist and should usually be reused, not rewritten:
- Token schema and tokenizer in `src/tokens/schema.py`, `src/tokens/tokenizer.py`
- MusicXML eventization in `src/tokens/eventizer.py`
- Round-trip helpers in `src/tokens/roundtrip.py`
- Dataset builder and vocab builder in `scripts/make_dataset.py`, `scripts/build_vocab.py`
- Bar descriptor logic in `src/dataio/descriptors.py`
- Dataset loading and packing in `src/dataio/dataset.py`, `src/dataio/collate_miditok.py`
- NoteLM model skeleton in `src/models/notelm/model.py`
- Basic decoding helpers in `src/utils/decoding/sampler.py`, `src/utils/decoding/rules.py`, `src/utils/decoding/scg.py`
- Frontend shell and AlphaTab viewer in `frontend/src/App.tsx`, `frontend/src/components/ScoreViewer.tsx`
- Local frontend mock mode in `frontend/src/mock/localData.ts`
