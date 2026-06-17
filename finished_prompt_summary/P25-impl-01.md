## Task ID
P25

## Files Changed
frontend/src/components/FingeringPicker.tsx
frontend/src/App.tsx
frontend/src/state/types.test.ts
frontend/vite.config.ts

## Behavior Implemented
When the user clicks a note in the score and the `eventHitMap` lookup resolves an `eventId`, `App.tsx` now calls `/alt_positions` (via the existing `altPositions` API client) and stores the returned options in a `fingeringPicker` state slice. A new `FingeringPicker` component renders inside the control panel showing each `{ stringIndex, fret }` option as a button (the currently-selected position is visually marked). Clicking an option calls `/apply_fingering` (single-note `fingeringSelections` list), then updates `scoreXml` and `revision` in state and dismisses the picker. The picker also exposes a close button to dismiss without applying. Twelve vitest pure-state tests cover the `toHitKey`, `getMeasureId`, and `getEventId` helpers including null/undefined guard paths.

## Remaining Known Issues
None
