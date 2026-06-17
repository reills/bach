---
## Task ID
P18

## Files Changed
PROGRESS.md
finished.md
src/api/__init__.py
src/api/store.py
tests/test_store.py

## Behavior Implemented
Added a small in-memory score and draft repository with replaceable interface types. It creates score and draft IDs, tracks integer score revisions, ties drafts to a base revision, supports draft save/commit/discard flows, and rejects stale draft creation or commit attempts when the underlying score revision has advanced.
 

## Remaining Known Issues
None
---
