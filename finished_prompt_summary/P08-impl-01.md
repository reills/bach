---
## Task ID
P08

## Files Changed
src/tabber/ascii.py
src/tabber/__init__.py
tests/test_ascii_tab.py
PROGRESS.md
finished.md

## Behavior Implemented
Added a deterministic ASCII guitar tab renderer for canonical events that already carry string/fret assignments. It outputs a labeled six-line block with the high string first, aligns simultaneous notes by onset, widens each onset slot for multi-digit frets, and rejects pitched events that are missing fingering metadata.

## Remaining Known Issues
None
---
