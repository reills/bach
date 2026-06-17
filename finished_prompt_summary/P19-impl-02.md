---
## Task ID
P19

## Files Changed
PROGRESS.md
finished.md
src/api/services/inpaint.py
tests/test_inpaint_service.py

## Behavior Implemented
Updated the window-mode inpaint preview service so it still preserves and auto-locks carry-in events and replaces only events whose `start_tick` falls inside the selected measure, while now reporting `changed_measure_ids` from the actual replacement-event spans. Cross-bar replacement output now marks downstream measures it touches, and focused tests cover both carry-in preservation and single-measure replacement behavior plus the new downstream-measure reporting case.
 

## Remaining Known Issues
None
---
