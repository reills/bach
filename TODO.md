# TODO — Active Task: P15

## P15 — Autoregressive generation loop

Implement a minimal autoregressive generation loop for NoteLM in src/inference/generate_v1.py. It should load a checkpoint, accept seed tokens and decoding settings, step the model forward, sample next tokens, and stop on max length or EOS if present. Reuse existing sampler and SCG helper modules instead of rewriting them. Add tests using a tiny synthetic model or mocked logits so generation logic is exercised without expensive training. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_generate_v1.py tests/test_load_checkpoint.py 
```
