
P28 complete. Here's what was done:

**`scripts/train_v1.py`** hardened with:
- `logging` module replacing bare `print()` — timestamped INFO lines
- `--resume PATH` — restores model/optimizer state and step counter from a checkpoint
- `--val-split F` + `--val-every N` — holds out a fraction of sequences and logs `val_loss` periodically
- `--dry-run-batches N` — exits after N batches for CI/smoke checks
- Checkpoint metadata enriched with `timestamp` (UTC ISO) and `args` dict

**`tests/test_train_v1_smoke.py`** — 6 tests using a tiny 6-bar synthetic parquet dataset:
1. Dry-run completes and logs "Dry-run complete"
2. Dry-run writes no final checkpoint
3. Full run produces one checkpoint with all required keys
4. Resume advances the step counter past the resume point
5. Val-split run logs `val_loss`
6. Checkpoint contains parseable `timestamp` and `args`

**`README.md`** — new "Stage-3: Train NoteLM v1" section with concrete commands for first run, dry-run, resume, val-split, and a flag reference table.