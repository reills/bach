VERDICT: FAIL
REMAINING_WORK:
- Resolve relative `vocab_path` values from checkpoint metadata against the checkpoint file location; the current loader fails for training-style paths like `out/notelm_v1/vocab.json` when inference runs from a different current working directory.
- Add a regression test that saves a checkpoint with a relative `vocab_path` containing directory components and reloads it after changing the current working directory, so the loader is verified against the format written by `scripts/train_v1.py`.
