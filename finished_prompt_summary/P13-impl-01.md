## Task ID
P13

## Files Changed
src/api/canonical/fingering.py
src/api/canonical/__init__.py
tests/test_canonical_fingering.py
PROGRESS.md
finished.md

## Behavior Implemented
Added a canonical fingering update helper that applies selected fingerings by `event_id` while preserving each event's existing pitch and timing, and rejecting selections that target unknown events or attempt to change non-fingering musical data. Exported the helper for later API use and added focused tests for the successful update path and invalid `event_id` rejection.
 

## Remaining Known Issues
None
