# TODO - Post-MVP Follow-Up

The MVP is implemented. Remaining work should be limited to gaps that make the repo easier to onboard, safer to run, or closer to a production-quality release.

## Highest Value

- Add a checked-in Python environment definition or lockfile. The current docs can only point developers at the expected `conda run -n bach` environment and the import-based smoke check.
- Add a supported backend launcher that binds `/compose` to a real checkpoint and vocab pair. The route contract exists, but the default `src.api.app:app` still returns `503` on `/compose`.
- Either implement `repair` mode end to end or remove/disable the `Repair` option in the frontend until the backend supports it.
- Replace the in-memory score/draft repository with persistent storage and draft cleanup semantics suitable for longer-lived sessions.

## Model and Data

- Add a reproducible sample checkpoint or release artifact so a new developer can exercise browser compose without training first.
- Expand evaluation beyond `eval_basic.py` with musicality and playability regression fixtures.
- Document dataset provenance, expected corpus layout, and output artifact sizes more formally once the training corpus is stable.

## Product and UX

- Support multi-measure selection and clearer carry-in note visualization during inpaint review.
- Improve export coverage beyond MusicXML and MIDI if GPX/GP becomes a real release target.
- Add a single scripted smoke path that boots backend + frontend together for manual QA.

## Test Command

Run only these focused smoke tests for the current MVP surfaces:

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh \
  tests/test_train_v1_smoke.py \
  tests/test_compose_service.py \
  tests/test_api_scores.py \
  tests/test_api_fingering.py
```
