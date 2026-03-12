---
VERDICT: FAIL
REMAINING_WORK:
- Update `tests/test_api_health.py` to use `fastapi.testclient.TestClient` against `create_app()` and assert `/healthz` returns `200` with `{"status": "ok"}`, as required by `TODO.md`.
- Fix `docs/skills/python-test-env/scripts/run_tests.sh` so `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_health.py` forwards the positional test path to pytest and runs only the targeted API health test instead of the full suite.
---
