## Task ID
P07

## Files Changed
src/api/render/midi.py
src/api/render/__init__.py
tests/test_midi_export.py
PROGRESS.md
finished.md

## Behavior Implemented
Added a canonical-score MIDI exporter in `src/api/render/midi.py` that converts canonical score metadata and note events into a `music21` score and returns MIDI bytes. The exporter preserves the canonical `tpq`, is exposed through `src/api/render/__init__.py`, and is covered by a focused test that confirms a simple exported score produces parseable MIDI data with valid header and track chunks.

## Remaining Known Issues
None
