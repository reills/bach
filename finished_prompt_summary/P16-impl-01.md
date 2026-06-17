## Task ID
P16

## Files Changed
src/inference/controls.py
tests/test_controls.py
README.md
PROGRESS.md
finished.md

## Behavior Implemented
Added a small compose-control mapping layer in `src/inference/controls.py` that normalizes user-facing key/style/difficulty/measures inputs into the prefix tokens expected by the NoteLM training pipeline, while reusing the existing prefix token ordering from `collate_miditok.py`. Added focused tests for normalization, token ordering, and invalid measure counts, and documented the mapping with a short README example.
 

## Remaining Known Issues
None
