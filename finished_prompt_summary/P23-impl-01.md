---
## Task ID
P23

## Files Changed
src/api/routes/scores.py
tests/test_api_fingering.py
frontend/src/api/types.ts
frontend/src/api/client.ts
PROGRESS.md
finished.md

## Behavior Implemented
Added the `/alt_positions` API endpoint. It validates the score and measure, resolves the frontend `eventHitKey` against the stored score's exported `eventHitMap`, finds the canonical event, computes alternate same-pitch guitar positions using the existing tabber helper, and returns compact picker options with `stringIndex`, `fret`, and `selected` for the current fingering. Added targeted API tests for a valid carry-note lookup and a missing-hit-key `404`, and updated the frontend API typings to use a concrete response type.
 

## Remaining Known Issues
None
---
