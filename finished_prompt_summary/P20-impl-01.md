## Task ID
P20

## Files Changed
src/api/app.py
src/api/compose_service.py
src/api/routes/scores.py
tests/test_api_scores.py
PROGRESS.md
finished.md

## Behavior Implemented
Added FastAPI score routes for `/compose`, `/inpaint_preview`, `/commit_draft`, and `/discard_draft`, aligned to the existing frontend camelCase payload contract. The routes now persist composed scores in the draft repository, return `measureMap` and `eventHitMap` metadata, render committed scores back to MusicXML, and translate stale revision conflicts into HTTP 409 responses. Added API tests covering one compose request, one full preview-to-commit flow, and one stale draft commit conflict.
 

## Remaining Known Issues
Default `create_app()` still needs a concrete compose service injected to make `/compose` succeed; without that configuration the route returns HTTP 503.
