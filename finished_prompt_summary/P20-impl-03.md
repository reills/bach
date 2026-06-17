## Task ID
P20

## Files Changed
PROGRESS.md
finished.md

## Behavior Implemented
Validated that the existing P20 FastAPI compose and draft routes already satisfy `TODO.md`: the backend exposes `/compose`, `/inpaint_preview`, `/commit_draft`, and `/discard_draft`, uses frontend-aligned camelCase payloads, returns HTTP 409 for stale draft conflicts, and has focused route coverage in `tests/test_api_scores.py`. Appended the required `PROGRESS.md` entry and reran the targeted task tests successfully.
 

## Remaining Known Issues
None
