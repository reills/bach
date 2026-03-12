# TODO - Active Task: P28

## P28 - Training script hardening

Review scripts/train_v1.py and harden it for actual use. Add missing basics only if needed for a practical first training run: clearer logging, checkpoint save metadata, resume support, validation split support, or a small dry-run mode. Do not redesign the trainer. Add a smoke test if feasible with a tiny synthetic dataset. Update README.md only for concrete run instructions that match the script. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_train_v1_smoke.py
```
