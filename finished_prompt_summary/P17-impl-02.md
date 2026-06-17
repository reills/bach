## Task ID
P17

## Files Changed
src/api/__init__.py
src/api/compose_service.py
tests/test_compose_service.py
PROGRESS.md
finished.md

## Behavior Implemented
Implemented the baseline internal compose pipeline in `src/api/compose_service.py`: it runs the generation loop, converts generated tokens into a canonical score, assigns guitar tab fingerings, renders MusicXML and MIDI, and returns the generated score plus frontend-ready `measure_map` and `event_hit_map`. Added coverage in `tests/test_compose_service.py` using a stubbed generator to exercise the transformation pipeline end to end, and exposed the service from `src/api/__init__.py`.
 

## Remaining Known Issues
None
