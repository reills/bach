## Task ID
P06

## Files Changed
PROGRESS.md
finished.md
src/api/render/__init__.py
src/api/render/musicxml.py
tests/test_musicxml_export.py

## Behavior Implemented
Added a minimal canonical-score to MusicXML exporter for one part that emits measure and note `xml:id` values, writes `divisions` from `tpq`, and splits cross-bar notes into tied per-measure MusicXML notes while preserving the canonical event ID across the tied fragments.
 

## Remaining Known Issues
None
