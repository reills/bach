# TODO — Active Task: P20

## P20 — FastAPI compose and draft routes

Add FastAPI routes for /compose, /inpaint_preview, /commit_draft, and /discard_draft. Wire them to the compose and draft services, keep request and response payloads aligned with frontend/src/api/types.ts, and return 409 on stale revision conflicts. Add route tests with TestClient that cover one compose call and one full preview -> commit flow. If the TypeScript payload types drift from the backend contract, update frontend/src/api/types.ts in the same change. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_scores.py tests/test_api_health.py 
```
