---
## Task ID
P04

## Files Changed
src/api/canonical/from_tokens.py
tests/test_tokens_to_canonical.py
PROGRESS.md
finished.md

## Behavior Implemented
Added canonical voice remapping in the token-stream bridge so sparse `VOICE_v` token indices are normalized to contiguous per-part `voice_id` values while preserving existing measure derivation, pitch reconstruction, stable event IDs, and cross-bar sustain behavior. Updated the targeted bridge test to cover sparse-voice normalization and verified the exact P04 test command passes.
 

## Remaining Known Issues
None
---
