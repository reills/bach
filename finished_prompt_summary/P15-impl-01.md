## Task ID
P15

## Files Changed
src/inference/__init__.py
src/inference/generate_v1.py
tests/test_generate_v1.py
PROGRESS.md
finished.md

## Behavior Implemented
Added a minimal NoteLM autoregressive generation loop that loads a checkpoint, resolves seed tokens from ids or vocab strings, advances the model one step at a time, samples with the existing sampler or SCG helpers, and stops on `max_length` or EOS when available. Added focused tests covering EOS stopping, SCG-guided decoding, and context truncation to the model sequence window.
 

## Remaining Known Issues
None
