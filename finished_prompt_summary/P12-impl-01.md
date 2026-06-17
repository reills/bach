## Task ID
P12

## Files Changed
src/tabber/heuristic.py
src/tabber/__init__.py
tests/test_tabber_alternates.py
PROGRESS.md
finished.md

## Behavior Implemented
Added a focused tabber helper that returns all valid alternate `GuitarFingering` positions for one pitched canonical event using standard tuning and the existing fret limit, ordered deterministically for UI selection. Added targeted tests covering both a pitch with several valid positions and a pitch constrained to a single in-range position.
 

## Remaining Known Issues
None
