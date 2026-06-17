---
## Task ID
P10

## Files Changed
PROGRESS.md
finished.md
src/tabber/__init__.py
src/tabber/heuristic.py
tests/test_tabber_heuristic.py

## Behavior Implemented
Implemented a first-pass heuristic guitar tabber in `src/tabber/heuristic.py` for standard six-string tuning. It tabs canonical `Event` sequences and a narrow `TabNote` structure by generating playable string/fret candidates within a fret limit, preferring lower-fret/open-string positions, and selecting same-onset voicings that never reuse one string. It raises a clear `ValueError` when a same-onset chord is not playable under those constraints, and the new targeted tests cover open-string preference, a basic chord assignment, and an impossible voicing case.
 

## Remaining Known Issues
None
---
