# Bach Gen Frontend

React + Vite + AlphaTab UI for the MVP browser workflow.

This frontend can run in three ways:

- `API mode`: talk to the FastAPI backend for compose, inpaint preview, draft commit/discard, and fingering changes
- `Local test-data mode`: load a MusicXML base score plus measure snippets from `public/test-data/`
- `Demo mode`: load the built-in two-measure score from `App.tsx`

The contract between the frontend and backend lives in [frontend-readme.md](/mnt/c/Users/Admin/dev/bach_gen/frontend-readme.md).

## Prerequisites

- Node runtime compatible with the checked-in `package-lock.json`
- `frontend/package.json` pins Volta Node `25.3.0`
- `npm install` is the supported install path

Install once:

```bash
cd frontend
npm install
```

If `npm install` reports `ENOENT` for `/bach_gen/package.json`, you ran it from the repo root instead of `frontend/`.

## Run in API Mode

Start the backend separately, then run:

```bash
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Notes:

- The frontend calls `/compose`, `/inpaint_preview`, `/commit_draft`, `/discard_draft`, `/alt_positions`, and `/apply_fingering`.
- The checked-in backend app at `src.api.app:app` does not bind a default `compose_service`, so `/compose` returns `503` unless you start a custom app factory that wires one in.
- `Window` is the only backend-supported inpaint mode today. The `Repair` option is still present in the UI but the current backend rejects it.

## Run in Local Test-Data Mode

No backend is required:

```bash
cd frontend
VITE_USE_LOCAL_DATA=true npm run dev
```

Then in the browser:

1. Set `Data source` to `Local test-data`.
2. Click `Load Test Data`.
3. Click a measure in the rendered score.
4. Click `Generate Preview`.
5. Use `Keep` or `Discard`.

This mode loads:

- a full base score from `public/test-data/manifest.json`
- one or more snippet MusicXML files from the same manifest

Current manifest shape:

```json
{
  "baseScore": "base.musicxml",
  "snippets": [
    "measure-001.xml",
    "measure-002.xml"
  ]
}
```

Implementation details that matter:

- `baseScore` should be a full MusicXML score
- each snippet can be a full MusicXML file or a raw `<measure>` fragment
- local inpaint is a measure splice, not model inference
- local mode does not provide backend `eventHitMap` data, so alternate fingering selection is not available there
- MIDI export is only available when the backend returned MIDI bytes during compose

## Demo Mode

The `Demo` button loads the built-in two-measure score from [frontend/src/App.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/App.tsx). Use it for a quick render/playback smoke check.

## Browser Workflow

### Compose and Inpaint

In full API mode with a configured backend:

1. Leave `Data source` on `Backend API`.
2. Optionally enter a prompt.
3. Click `Generate Score`.
4. Click a measure in the score.
5. Toggle `Harmony`, `Rhythm`, or `Soprano` constraints if needed.
6. Leave mode on `Window`.
7. Click `Generate Preview`.
8. Use `Keep` or `Discard`.

### Fingering

The fingering picker opens after a note click only when the current score has an `eventHitMap` and the backend can answer `/alt_positions`.

Current backend-driven flow:

1. Click a rendered note.
2. Frontend sends `scoreId`, `measureId`, and the AlphaTab-derived hit key to `/alt_positions`.
3. Pick an alternate string/fret option.
4. Frontend sends the selected `eventId`, `stringIndex`, and `fret` to `/apply_fingering`.
5. The score rerenders from the returned MusicXML and revision.

## Local Test Data Files

Files under `public/test-data/` are the browser-only fixtures used by local mode.

- [frontend/public/test-data/manifest.json](/mnt/c/Users/Admin/dev/bach_gen/frontend/public/test-data/manifest.json): active local-mode fixture manifest
- [frontend/public/test-data/manifest.example.json](/mnt/c/Users/Admin/dev/bach_gen/frontend/public/test-data/manifest.example.json): example manifest shape
- [frontend/src/mock/localData.ts](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/mock/localData.ts): loader and measure replacement helpers

## Useful Files

- [frontend/src/App.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/App.tsx): state machine and user actions
- [frontend/src/components/ScoreViewer.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/components/ScoreViewer.tsx): AlphaTab integration
- [frontend/src/components/FingeringPicker.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/components/FingeringPicker.tsx): alternate position picker
- [frontend/src/api/client.ts](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/api/client.ts): API client
- [frontend/src/state/types.ts](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/state/types.ts): measure and hit-key helpers

## Test and Build

Targeted frontend tests:

```bash
cd frontend
npm test -- --run src/App.test.ts src/state/types.test.ts
```

Production build:

```bash
cd frontend
npm run build
```
