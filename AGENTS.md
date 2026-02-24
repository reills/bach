# AGENTS.md — bach-gen

## How to run
- Preferred runtime: `conda run -n bach python`
- Optional venv fallback: `python -m venv .venv && source .venv/bin/activate`
- Environment/test helper: `bash docs/skills/python-test-env/scripts/run_tests.sh --check`
- Tests: `bash docs/skills/python-test-env/scripts/run_tests.sh`

## Rules
- Do one task per run.
- Prefer small diffs; don't refactor unrelated code.
- Always run tests for touched areas (use `bash docs/skills/python-test-env/scripts/run_tests.sh`).
- Don't weaken or remove tests to make them pass.
- If you add a new feature without tests, add tests in the same run.
- Do not hard-code tests; test real behavior.
- If tests fail, find and fix the underlying code issue before proceeding.
- Update `TODO.md` when items are completed.
- Append a short entry to `PROGRESS.md` for each change set.
- Don't add heavy deps without asking.
