## Task ID
P22

## Files Changed
src/api/compose_service.py
src/api/services/inpaint.py
src/api/routes/scores.py
tests/test_api_scores.py
tests/test_hit_map.py
PROGRESS.md
finished.md

## Behavior Implemented
Centralized MusicXML export plus hit-map generation in a shared `export_score()` helper, then wired compose and inpaint preview service results to carry `measure_map` and `event_hit_map` so their route responses always include both fields after a successful export. Updated the draft commit route to reuse the same helper and extended the API tests to verify the returned maps are present and structurally valid.
 

## Remaining Known Issues
None
