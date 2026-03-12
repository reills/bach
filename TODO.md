# TODO — Active Task: P14

## P14 — Inference checkpoint loading

Add a small inference helper that loads a NoteLM checkpoint plus vocab and returns a ready-to-run model on CPU by default. Keep the interface minimal and avoid introducing a heavy configuration system. Add a test that saves a tiny synthetic checkpoint and reloads it successfully. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_load_checkpoint.py 
```
