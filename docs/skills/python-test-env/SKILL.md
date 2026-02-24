---
name: python-test-env
description: Ensure deterministic Python test execution in this repo by selecting a valid runtime (`conda` env `bach` first, then `.venv`) and running pytest with dependency checks. Use when preparing to run tests, when a loop run needs a stable verify command, or when failures may be caused by environment mismatch.
---

# Python Test Env

## Workflow

1. Run `bash docs/skills/python-test-env/scripts/run_tests.sh --check`.
2. If check passes, use the printed verify command for all test runs.
3. If check fails, fix environment selection first before code changes.
4. Run `bash docs/skills/python-test-env/scripts/run_tests.sh` to execute tests.

## Environment Selection Rules

- Prefer `conda` env `bach` when available.
- Fall back to `.venv` only if `.venv/bin/python` is executable.
- Reject broken `.venv` layouts (for example zero-byte `python` launcher files).
- Verify imports: `pytest`, `torch`, `music21`, `pandas`, `pyarrow`.

## Commands

- Print selected verify command:
  - `bash docs/skills/python-test-env/scripts/run_tests.sh --print-cmd`
- Check environment only:
  - `bash docs/skills/python-test-env/scripts/run_tests.sh --check`
- Run tests:
  - `bash docs/skills/python-test-env/scripts/run_tests.sh`
