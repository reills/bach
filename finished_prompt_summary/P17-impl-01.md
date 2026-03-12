## Task ID
P17

## Files Changed
src/api/compose_service.py
tests/test_compose_service.py
PROGRESS.md
finished.md

## Behavior Implemented
Added an internal baseline compose service that invokes the generation loop, converts generated tokens into a canonical guitar score, assigns tablature fingerings, renders MusicXML and MIDI, and returns measure and event hit maps for frontend score interactions. Added a targeted stubbed-generation test that covers the full transformation pipeline, including cross-bar event mapping and parseable MIDI output.
 

## Remaining Known Issues
None
