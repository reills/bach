## Task ID
P27

## Files Changed
frontend/src/App.test.ts

## Behavior Implemented
Added 15 vitest integration tests in `frontend/src/App.test.ts` covering the complete compose/inpaint workflow. The tests mock the API layer with `vi.mock('./api/client')` so no live backend is required.

Tests cover:
- **Compose response loading**: `scoreId`, `revision`, `scoreXml`, `measureMap` are set from the mocked API response; all draft/selection fields are cleared.
- **Dirty-state reset on compose**: a second compose call clears any lingering draft, selection, and lock fields from a previous session.
- **Measure selection**: `selectedBarIndex` and `selectedMeasureId` are derived via `getMeasureId` from the current `measureMap`.
- **Inpaint preview**: `draftId`, `draftXml`, `draftBaseRevision`, `highlightMeasureId`, `lockedEventIds`, and `changedMeasureIds` are set from the preview response; existing `measureMap` is preserved when the response omits one.
- **Commit draft**: `scoreXml` and `revision` update from the commit response; all draft fields are cleared.
- **Discard draft**: all draft fields are cleared; `scoreXml` and `revision` are unchanged.
- **Full compose → select → preview → commit workflow** (end-to-end state machine).
- **Full compose → select → preview → discard workflow** (end-to-end state machine).
- **Status text preconditions**: three tests verify the state conditions that trigger the visible status messages "Score loaded", "Draft ready", "Draft committed", and "Draft discarded" in App.tsx.
- **EventHitMap resolution**: `getEventId` correctly resolves an event ID from the hit map attached via a compose response.

All 30 frontend tests (15 new + 15 existing) pass with `npx vitest run`.

## Remaining Known Issues
None
