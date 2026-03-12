## Task ID
P14

## Files Changed
src/models/notelm/inference.py
src/models/notelm/__init__.py
src/models/__init__.py
tests/test_load_checkpoint.py
PROGRESS.md

## Behavior Implemented
Added a minimal NoteLM inference checkpoint loader that restores the saved model config and weights, resolves and loads the associated vocab, and returns an eval-mode model on CPU by default. Added a focused round-trip test that writes a tiny synthetic checkpoint plus vocab and verifies the reloaded model produces the same logits.
 

## Remaining Known Issues
None
