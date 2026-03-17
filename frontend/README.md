# Bach Gen Frontend

React + Vite + AlphaTab UI for the MVP browser workflow.

This file now serves as both the practical frontend guide and the live frontend/backend contract for the shipped MVP.

## 1. Modes

This frontend can run in three ways:

- `API mode`: talk to the FastAPI backend for compose, inpaint preview, draft commit/discard, and fingering changes
- `Local test-data mode`: load a MusicXML base score plus measure snippets from `public/test-data/`
- `Demo mode`: load the built-in two-measure score from `App.tsx`

## 2. Prerequisites

- Node runtime compatible with the checked-in `package-lock.json`
- `frontend/package.json` pins Volta Node `25.3.0`
- `npm install` is the supported install path

Install once:

```bash
cd frontend
npm install
```

If `npm install` reports `ENOENT` for `/bach_gen/package.json`, you ran it from the repo root instead of `frontend/`.

## 3. Run in API Mode

Start the backend separately, then run:

```bash
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Notes:

- The frontend calls `/compose`, `/inpaint_preview`, `/commit_draft`, `/discard_draft`, `/alt_positions`, and `/apply_fingering`.
- The checked-in backend app at `src.api.app:app` does not bind a default `compose_service`, so `/compose` returns `503` unless you start `src.api.compose_app:app` or wire a compose service into a custom app.
- `window` is the only backend-supported inpaint mode today. The `repair` option is still present in the UI but the current backend rejects it.

## 4. Run in Local Test-Data Mode

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

## 5. Demo Mode

The `Demo` button loads the built-in two-measure score from [frontend/src/App.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/App.tsx). Use it for a quick render/playback smoke check.

## 6. Browser Workflow

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

## 7. Frontend/Backend Contract

### Canonical Score

The backend owns the canonical score. MusicXML is a render/export format, not the source of truth.

Logical shape:

- `CanonicalScore { header, measures[], parts[] }`
- backend Python types use snake_case
- this document uses camelCase names for the same logical fields

Header:

- `tpq`
- `keySigMap`
- `timeSigMap`
- `tempoMap`
- optional `pickupTicks`

Measures:

- `id`
- `index`
- `startTick`
- `lengthTicks`

Parts:

- `id`
- `instrument`
- `tuning`
- `capo`
- `midiProgram`
- `events[]`

Events:

- `id`
- `startTick`
- `durTick`
- `pitchMidi | null`
- optional `velocity`
- `voiceId`
- optional `fingering`

Invariants:

- timing is quantized to `tpq`
- no tie objects exist in canonical state
- sustained notes may cross barlines via `durTick`
- chords are represented as multiple events sharing `startTick`
- `voiceId` is per-part and contiguous
- canonical pitch is sounding MIDI pitch
- fingering metadata is optional and stored as `{ stringIndex, fret }`
- generated output maps NoteLM `VOICE_v` directly to canonical `voiceId = v`

Event IDs are stable unless an event is explicitly replaced. Inpaint generates new IDs only for events replaced inside the regenerated span.

### Measure and Note Identity

Use `measureId` everywhere, not displayed measure numbers.

Returned mappings:

- `measureMap`: `barIndex -> measureId`
- `eventHitMap`: `barIndex|voiceIndex|beatIndex|noteIndex -> eventId`

Current implementation details:

- `measureMap` is built from exported MusicXML measure order
- `eventHitMap` is built from exported MusicXML note order
- click resolution is structural, not DOM-id based

### MusicXML Export Role

Frontend state renders from MusicXML, but MusicXML is not the canonical model.

Required export behavior:

- derive barlines from `timeSigMap`
- split sustained events at bar boundaries on export
- emit ties from those splits
- preserve the same canonical `event.id` across exported tie segments via `xml:id`
- emit `xml:id` on measures when possible so `measureMap` stays stable

### Rendering Assumptions

Current renderer: AlphaTab only.

The frontend expects:

- sheet + tab rendering from MusicXML
- playback from AlphaTab
- note click callbacks with enough hierarchy to recover bar, voice, beat, and note indices

AlphaTab assumptions used by the MVP:

- note clicks resolve to a hit key of `barIndex`, `voiceIndex`, `beatIndex`, `noteIndex`
- measure clicks resolve through AlphaTab bar selection, then map through `measureMap`
- tab display depends on exported MusicXML technical tags

One implementation detail from the original planning doc that is still useful: note hit detection expects AlphaTab note bounds to be enabled.

### MusicXML Tab Encoding

Fingering only survives render when exported as MusicXML technical notation:

```xml
<note>
  <notations>
    <technical>
      <string>1</string>
      <fret>3</fret>
    </technical>
  </notations>
</note>
```

String numbering contract:

- MusicXML `1` = high E
- MusicXML `6` = low E
- backend `stringIndex` is 0-based and must be converted on export
- backend index `5` -> MusicXML string `1`
- backend index `0` -> MusicXML string `6`

Also required:

- emit `<staff-details>` for tab staff
- emit `<clef><sign>TAB</sign>` for tab display
- omit `<technical>` for rests

### Inpaint Workflow

Current supported mode: `window`.

Request:

- `POST /inpaint_preview { scoreId, measureId, revision, constraints?, locks?, mode? }`

Constraints shape:

- `keepHarmony?: boolean`
- `keepRhythm?: boolean`
- `keepSoprano?: boolean`
- `fixedPitches?: string[]`
- `fixedOnsets?: number[]`

Locks shape:

- `lockedEventIds?: string[]`
- `lockedRanges?: [{ startTick, endTick, type: "pitch" | "onset" | "all" }]`

Behavior contract:

- selected measure is identified by `measureId`
- backend preserves carry-in events that started earlier and sustain into the target measure
- only events whose `startTick` falls inside the regenerated span may be replaced
- carry-in events may be reported back as locked event ids
- `changedMeasureIds` may include downstream measures when a replacement event sustains across a barline

Response:

- `draftId`
- `scoreXML`
- `baseRevision`
- optional `highlightMeasureId`
- optional `measureMap`
- optional `eventHitMap`
- optional `lockedEventIds`
- optional `changedMeasureIds`

Commit/discard:

- `POST /commit_draft { scoreId, draftId } -> { scoreXML, revision, measureMap?, eventHitMap? }`
- `POST /discard_draft { scoreId, draftId } -> { ok: true }`

Important current implementation note:

- the frontend still exposes a `repair` option, but the backend currently rejects it; `window` is the only valid runtime mode for the shipped MVP

### Fingering Workflow

Fingering is a same-pitch position edit, not a pitch edit.

Lookup:

- `POST /alt_positions { scoreId, measureId, eventHitKey }`

`eventHitKey` shape:

- `barIndex`
- optional `voiceIndex`
- optional `beatIndex`
- optional `noteIndex`

Response:

- `eventId`
- `options[]` where each option is `{ stringIndex, fret, selected }`

Apply:

- `POST /apply_fingering { scoreId, revision, fingeringSelections }`

`fingeringSelections` shape:

- `{ eventId, stringIndex, fret }[]`

Apply response:

- `{ scoreXML, revision }`

Contract guarantees:

- applying fingering updates only fingering metadata
- pitch and timing for the selected event remain unchanged
- frontend rerenders from returned MusicXML and revision

### Revisions and Drafts

Concurrency contract:

- stale writes return HTTP `409`
- drafts are tied to a `baseRevision`
- commit/discard operations validate score ownership

Current storage implementation:

- in-memory repository

The API contract does not assume in-memory storage, only revision-checked score/draft semantics.

## 8. Local Frontend Mode Contract

The frontend also supports a backend-free local mode for UI smoke testing.

Local-mode contract:

- base score comes from `public/test-data/manifest.json`
- snippet replacements are MusicXML measure swaps
- local mode does not provide backend `eventHitMap`
- local mode supports measure select, preview, keep, and discard
- local mode does not support backend fingering lookup/apply

## 9. Current Scope

Shipped:

- single-score render in AlphaTab
- measure selection
- window-mode inpaint preview
- draft keep/discard
- alternate fingering picker when backend hit-map data exists
- MusicXML export
- MIDI export when backend compose returned MIDI

Explicitly out of scope for this contract version:

- multi-part score editing
- drag editing
- repeat handling
- grace notes and tuplets
- repair-mode inpaint semantics
- persistent multi-user draft storage

## 10. Useful Files

- [frontend/src/App.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/App.tsx): state machine and user actions
- [frontend/src/components/ScoreViewer.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/components/ScoreViewer.tsx): AlphaTab integration
- [frontend/src/components/FingeringPicker.tsx](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/components/FingeringPicker.tsx): alternate position picker
- [frontend/src/api/client.ts](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/api/client.ts): API client
- [frontend/src/state/types.ts](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/state/types.ts): measure and hit-key helpers
- [frontend/public/test-data/manifest.json](/mnt/c/Users/Admin/dev/bach_gen/frontend/public/test-data/manifest.json): active local-mode fixture manifest
- [frontend/public/test-data/manifest.example.json](/mnt/c/Users/Admin/dev/bach_gen/frontend/public/test-data/manifest.example.json): example manifest shape
- [frontend/src/mock/localData.ts](/mnt/c/Users/Admin/dev/bach_gen/frontend/src/mock/localData.ts): loader and measure replacement helpers

## 11. Test and Build

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
