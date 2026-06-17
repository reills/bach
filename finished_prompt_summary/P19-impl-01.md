---
## Task ID
P19

## Files Changed
PROGRESS.md
finished.md
src/api/services/__init__.py
src/api/services/inpaint.py
tests/test_inpaint_service.py

## Behavior Implemented
Added a minimal window-mode inpaint preview service that creates a draft from a stored canonical score, computes carry-in events at the selected measure start, preserves and auto-locks those carry-in events, replaces only events whose `start_tick` lands inside the selected measure, and returns draft metadata with `changed_measure_ids`, `locked_event_ids`, rendered MusicXML, and the draft score.
 

## Remaining Known Issues
None
---
