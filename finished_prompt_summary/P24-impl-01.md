## Task ID
P24

## Files Changed
src/api/routes/scores.py
tests/test_api_fingering.py
PROGRESS.md
finished.md

## Behavior Implemented
Added the `/apply_fingering` API endpoint. It validates the requested score revision, applies one or more fingering selections by `eventId` using the existing canonical fingering helper, commits the updated score to produce a new revision, and returns re-exported MusicXML plus that revision. Added focused API tests proving fingering-related MusicXML technical tags update while note pitch content stays unchanged, and proving stale revisions return HTTP 409.
 

## Remaining Known Issues
None
