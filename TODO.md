# TODO — Active Task: P19

## P19 — Inpaint window-mode service

Implement an MVP inpaint service for window mode. It should accept a canonical score plus measureId, compute carry-in events active at the measure start, preserve those carry-in events, regenerate or replace only events whose startTick falls inside the selected measure, and return a draft result with changed measure IDs and lockedEventIds. Keep the first version deterministic if needed; correctness of measure boundaries and carry-in preservation matters more than model sophistication. Add tests for carry-in preservation and single-measure replacement. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_inpaint_service.py tests/test_canonical_ops.py 
```
