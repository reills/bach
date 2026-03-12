# TODO — Active Task: P16

## P16 — Prompt/control token mapping

Implement a thin control-token layer that maps compose inputs like key, style, difficulty, and requested measure count into token prefixes expected by the training pipeline. Reuse the token conventions already present in scripts/train_v1.py and collate_miditok.py. Add small tests for normalization and token ordering. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_controls.py 
```
