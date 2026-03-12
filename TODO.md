# TODO — Active Task: P22

## P22 — Wire hit maps into compose and inpaint responses

Update the compose and inpaint service/route responses so they always include measureMap and eventHitMap when export succeeds. Keep the mapping generation centralized and avoid duplicating logic in the route layer. Extend the route tests to assert these fields are present and structurally valid. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_scores.py tests/test_hit_map.py 
```
