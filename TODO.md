# TODO — Active Task: P18

## P18 — Draft/revision repository

Implement a small in-memory repository for scores and drafts. It should create score IDs, track integer revisions, create draft IDs tied to a base revision, commit drafts, discard drafts, and reject stale writes. Keep it in-memory for MVP and structure it so a later persistence layer can replace it. Add tests for happy path and stale revision rejection. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_store.py 
```
