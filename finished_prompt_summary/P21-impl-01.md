---
## Task ID
P21

## Files Changed
src/api/compose_service.py
tests/test_hit_map.py
PROGRESS.md
finished.md

## Behavior Implemented
Implemented deterministic `measureMap` and `eventHitMap` generation by traversing the exported MusicXML measure/note structure directly, so hit keys now stay aligned with the frontend `barIndex|voiceIndex|beatIndex|noteIndex` contract for the same rendered score layout. Added focused coverage for a small polyphonic export, cross-measure carry notes, rest-driven beat indexing, and chord-note `noteIndex` handling.
 

## Remaining Known Issues
None
---
