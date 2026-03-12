## Frontend Architecture Contract

This file is the current frontend/backend contract for the shipped MVP.

Design intent:

- backend owns the canonical score
- MusicXML is a render/export format, not the source of truth
- measure identity is stable through `measureId`
- note identity is stable through canonical `event.id`
- AlphaTab is the only renderer in the current implementation

---

## Canonical Score

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

---

## MusicXML Role

Frontend state renders from MusicXML, but MusicXML is not the canonical model.

Required export behavior:

- derive barlines from `timeSigMap`
- split sustained events at bar boundaries on export
- emit ties from those splits
- preserve the same canonical `event.id` across exported tie segments via `xml:id`
- emit `xml:id` on measures when possible so `measureMap` stays stable

---

## Measure and Note Identity

Use `measureId` everywhere, not displayed measure numbers.

Returned mappings:

- `measureMap`: `barIndex -> measureId`
- `eventHitMap`: `barIndex|voiceIndex|beatIndex|noteIndex -> eventId`

Current implementation details:

- `measureMap` is built from exported MusicXML measure order
- `eventHitMap` is built from exported MusicXML note order
- click resolution is structural, not DOM-id based

---

## Rendering

Current renderer: AlphaTab only.

The frontend expects:

- sheet + tab rendering from MusicXML
- playback from AlphaTab
- note click callbacks with enough hierarchy to recover bar, voice, beat, and note indices

AlphaTab assumptions used by the MVP:

- note clicks resolve to a hit key of `barIndex`, `voiceIndex`, `beatIndex`, `noteIndex`
- measure clicks resolve through AlphaTab bar selection, then map through `measureMap`
- tab display depends on exported MusicXML technical tags

---

## MusicXML Tab Encoding

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

---

## Inpaint Workflow

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

---

## Fingering Workflow

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

---

## Revisions and Drafts

Concurrency contract:

- stale writes return HTTP `409`
- drafts are tied to a `baseRevision`
- commit/discard operations validate score ownership

Current storage implementation:

- in-memory repository

The API contract does not assume in-memory storage, only revision-checked score/draft semantics.

---

## Local Frontend Mode

The frontend also supports a backend-free local mode for UI smoke testing.

Local-mode contract:

- base score comes from `public/test-data/manifest.json`
- snippet replacements are MusicXML measure swaps
- local mode does not provide backend `eventHitMap`
- local mode supports measure select, preview, keep, and discard
- local mode does not support backend fingering lookup/apply

---

## MVP Scope

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
