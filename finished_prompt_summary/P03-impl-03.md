---
## Task ID
P03

## Files Changed
src/api/canonical/types.py
tests/test_canonical_types.py
frontend-readme.md
PROGRESS.md
finished_prompt_summary/prompt5
finished.md

## Behavior Implemented
Tightened the canonical score dataclasses so invalid nested constructions fail immediately: `Part` now requires a real `PartInfo` plus `Event` entries, and `CanonicalScore` now requires `ScoreHeader`, `Measure`, and `Part` instances before applying the existing timing, ID, cross-bar `dur_tick`, and per-part `voice_id` invariants. Added focused tests for those construction checks, and clarified `frontend-readme.md` that backend `Part` values are stored as `Part { info: PartInfo, events[] }` while describing the same logical fields for the frontend.
 

## Remaining Known Issues
None
---
