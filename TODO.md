# TODO — Active Task: P23

## P23 — Alternate positions API

Implement the /alt_positions endpoint. It should resolve the frontend eventHitKey to a canonical eventId using the stored hit map, compute alternate fingerings for that event, and return a compact response suitable for a picker UI. Add tests covering a valid request and a missing-event case. Update frontend/src/api/types.ts if a concrete response type is needed. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_fingering.py 
```
