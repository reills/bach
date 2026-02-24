# ExecPlan — Stage-1 Tokenizer & Dataset

## Rules
- One task per run.
- Run `bash docs/skills/python-test-env/scripts/run_tests.sh` and do not weaken tests.
- If you add a new feature without tests, add tests in the same run.
- Do not hard-code tests; test real behavior.
- If tests fail, fix the underlying code issue before proceeding.
- Update `TODO.md` and append to `PROGRESS.md`.

## Checklist (one item per loop)
- [x] Define EventSpec + DescriptorSpec in `src/tokens/schema.py` with version `remi_tab_v1` and `tpq=24`.
- [ ] Implement interval math helpers (melodic + harmonic ref) with QA vs production behavior.
- [ ] Implement event parsing/serialization in `src/tokens/tokenizer.py` with canonical ordering.
- [ ] Add round-trip tests for pitch reconstruction and `HARM_*` consistency.
- [ ] Implement `scripts/make_dataset.py` basic pipeline (parse, normalize, collapse doublings, track assignment, tokenize, stats).
- [ ] Produce dataset artifacts in `data/processed/` and document counts in `stats.json`.
- [ ] Verify `bash docs/skills/python-test-env/scripts/run_tests.sh` passes.
