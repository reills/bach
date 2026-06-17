## Task ID
P26

## Files Changed
frontend/src/state/types.ts
frontend/src/App.tsx
frontend/src/App.css
frontend/src/state/types.test.ts
PROGRESS.md

## Behavior Implemented
Added `changedMeasureIds: string[] | null` to `ScoreState`. The inpaint preview handler now stores `response.changedMeasureIds` (API mode) or derives it from the replaced measure ID (local mode). All state resets (compose, load local/demo, commit, discard) clear both `changedMeasureIds` and `lockedEventIds` to null.

The draft indicator now shows a sub-line beneath "Draft ready for review" with human-readable counts when either value is available, e.g. "2 measures changed · 3 events locked". The layout uses a new `draft-indicator__body` flex column wrapper with a `draft-indicator__meta` style for the muted secondary line. No existing panels were redesigned.

## Remaining Known Issues
None
