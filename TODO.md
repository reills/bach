# TODO — Active Task: P05

## P05 — Canonical score utilities for measure and event lookup

Implement small canonical score utility functions for looking up measures by measureId, events by eventId, events starting inside a measure, and carry-in events active at a measure start. Keep the functions pure and easy to test because later API routes will depend on them. Add tests for measure lookup, carry-in detection, and event replacement inside one measure. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_ops.py 
```
