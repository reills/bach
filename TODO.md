# TODO — Active Task: P03

## P03 — Define canonical score types

Implement canonical score data structures under src/api/canonical/types.py using dataclasses or pydantic models. Include header fields, part info, measures, events, and optional fingering data needed for guitar output. Match the invariants in frontend-readme.md: quantized timing, durTick across barlines, stable measure IDs, stable event IDs, and voiceId per part. Add focused tests for construction and basic invariants. If frontend-readme.md needs a small clarification to match the final model shape, update it. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_types.py 
```
