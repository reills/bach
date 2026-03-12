## Task ID
P05

## Files Changed
src/api/canonical/__init__.py
src/api/canonical/ops.py
tests/test_canonical_ops.py
PROGRESS.md
finished.md

## Behavior Implemented
Added pure canonical score utilities for measure lookup by `measureId`, event lookup by `eventId`, querying events that start within a measure, querying carry-in events active at a measure start, and replacing only the events that start inside one target measure. Exported the helpers from the canonical package and added focused tests covering lookup, carry-in detection, and single-measure replacement behavior.
 

## Remaining Known Issues
None
