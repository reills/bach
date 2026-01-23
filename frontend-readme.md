## Frontend plan (short)

Build around a server-side Canonical Score (one source of truth). Use MusicXML only as the wire/render format. MVP loop: generate → click measure → inpaint preview → accept/revert → export.

---

## Canonical Score (backend truth)

Minimal schema (v0):

- Header: `tpq`, `keySigMap`, `timeSigMap`, `tempoMap`, optional `pickupTicks`
- Parts: `instrument`, `tuning`, `capo`, `midiProgram`
- Measures: `measures[]` with `id` (UUID), `index` (0-based), `startTick`, `lengthTicks`
- Events: `{ id, startTick, durTick, pitchMidi | null, velocity?, voiceId }`

Invariants:
- Timing quantized to `tpq`
- No tie objects in canonical score; sustained notes may cross barlines via `durTick`
- Chords = multiple events sharing `startTick` (may share or differ by `voiceId`)
- Canonical pitch is MIDI; MusicXML spelling is derived from key/context
- `voiceId` is per-part (`0..N-1`) for polyphonic separation
- Canonical score is stored in sounding pitch (concert pitch)
- MusicXML ties are derived on export by splitting sustained events at barlines
- For generated output, `voiceId` defaults to the NoteLM track index (`VOICE_v`); imported scores may map part/voice indices into `VOICE_v` during eventization

Event IDs are stable unless replaced. Inpainting a measure creates new IDs only inside that measure.
`index` is derived from order; `measure.id` is the stable identity.

---

## MusicXML wire format (render cache only)

Frontend stores `currentScoreXML` just for rendering. Real state is `{scoreId, revision}` on the server.

Do not make MusicXML the canonical model: it is verbose and not suited for ML constraints or fast edits.

**Implementation warning: exporter is heavy**
- Grid time: compute barlines from `timeSigMap`
- Slice events: if `startTick + durTick` crosses a barline, split into multiple `<note>`s
- Inject ties: `start` on the first segment, `stop` on the next (and `start` again if it continues)
- Preserve IDs: sliced notes should reference the same canonical `event.id` (e.g., shared `xml:id`) so clicks map back to one event

---

## Measure identity

Use `measureId` everywhere (not measure numbers). Map clicks to `measureId`.

- Emit `xml:id` on `<measure>` and `<note>` when possible (Verovio can use these directly)
- AlphaTab does not guarantee `xml:id` propagation, so map clicks via a structural hit key
- Return deterministic mappings: `measureMap` (`barIndex` → `measureId`) and `eventHitMap` (`barIndex, voiceIndex, beatIndex, noteIndex` → `eventId`)

---

## Rendering choices

### Option A: AlphaTab-only (recommended MVP)

Use AlphaTab for sheet + tab + playback.

Pros: one renderer, simple state, built-in playback, supports MusicXML.
Cons: engraving fidelity is lower than Verovio.

AlphaTab notes:
- `noteMouseDown` needs `CoreSettings.includeNoteBounds = true`
- Measure clicks come from the hierarchy (`staff → bar → voice → beat → note`)
- Click mapping is via bounds/events, not DOM IDs
- Stave profile example:

```typescript
const api = new alphaTab.AlphaTabApi(el, {
  layout: { staveProfile: alphaTab.LayoutStaveProfile.ScoreTab }
});
```

### Option B: Verovio (sheet) + AlphaTab (tab)

Use Verovio for high-quality SVG sheet rendering and reliable IDs (`xml:id`).
Use AlphaTab for tab + playback.

Note: Verovio measure indices are not stable; prefer `xml:id` mapped from `measure.id`.

---

## MusicXML tab encoding (required)

AlphaTab will ignore your fingering unless you emit string/fret:

```xml
<note>
  <pitch><step>E</step><octave>2</octave></pitch>
  <duration>24</duration>
  <notations>
    <technical>
      <string>6</string>
      <fret>0</fret>
    </technical>
  </notations>
</note>
```

**String numbering (definitive):**
- **MusicXML standard:** 1 = High E (highest pitch), 6 = Low E (per W3C MusicXML 4.0 spec)
- **AlphaTab convention:** 1 = High E (highest pitch), 6 = Low E (matches MusicXML)
- **No flip needed:** Both systems use identical 1-based pitch-height ordering

**Implementation note:**
If  backend uses 0-based arrays (e.g., `tuning = [E2, A2, D3, G3, B3, E4]` where index 5 = High E), must convert to 1-based when emitting MusicXML:
- Backend index 5 (High E) → `<string>1</string>` in MusicXML
- Backend index 0 (Low E) → `<string>6</string>` in MusicXML

Also include `<staff-details>` + `<clef><sign>TAB</sign>` for the tab staff.

For rests, omit `<technical>`.
Set `<divisions>` so MusicXML durations map deterministically to `durTick`.

---

## Inpaint workflow (drafts)

1. User clicks a measure → resolve `measureId`
2. `POST /inpaint_preview { scoreId, measureId, revision, constraints?, locks?, mode? }`
   - `constraints` (minimal): `{ keepHarmony?: bool, keepRhythm?: bool, keepSoprano?: bool, fixedPitches?: [eventId], fixedOnsets?: [startTick] }`
   - Backend computes `carryIn` = events with `startTick < measureStartTick` and `startTick + durTick > measureStartTick`
   - Inpaint replaces only events whose `startTick` is inside the target measure
   - Constraint: regeneration must not modify any `carryIn` events (pitch or endTick). It may only add/replace events whose `startTick` is inside the target measure.
   - UI: show carry-in notes as "held from previous measure" (ghosted/locked) and allow regeneration of new onsets/rests within the measure
   - Measure-click regeneration does not expand selection; expansion happens only when the user targets a carry-in note for change
3. Backend returns `{ draftId, scoreXML, baseRevision, highlightMeasureId, lockedEventIds? }`
4. Frontend renders draft XML and highlights the measure
5. User action:
   - Keep → `POST /commit_draft { scoreId, draftId }` → returns `{ scoreXML, revision }`
   - Revert → `POST /discard_draft { scoreId, draftId }`

---

## Tab “edits” (Phase 2, picker only)

No dragging. Position choices only (same pitch).

1. `noteMouseDown` → resolve a hit key `{ barIndex, voiceIndex, beatIndex, noteIndex }`
2. `POST /alt_positions { scoreId, measureId, eventHitKey }`
3. Backend resolves to `eventId` and returns alternate fingerings
4. User selects → `POST /apply_fingering { scoreId, revision, fingeringSelections }` → returns `scoreXML`

`eventId` is the canonical `event.id` value.

---

## Minimal API

- `POST /compose` → `{ scoreId, revision, scoreXML, measureMap?, eventHitMap?, midi }`
- `POST /inpaint_preview` → `{ scoreId, measureId, revision, constraints?, locks?, mode? }` → `{ draftId, scoreXML, highlightMeasureId, baseRevision, measureMap?, eventHitMap?, lockedEventIds?, changedMeasureIds? }`
- `POST /commit_draft` → `{ scoreId, draftId }` → `{ scoreXML, revision, measureMap?, eventHitMap? }`
- `POST /discard_draft` → `{ scoreId, draftId }` → `{ ok: true }`
- `POST /alt_positions` → `{ scoreId, measureId, eventHitKey? }` → alternate fingerings
- `POST /apply_fingering` → `{ scoreId, revision, fingeringSelections }` → `{ scoreXML, revision }`

`constraints`:
- `{ keepHarmony?, keepRhythm?, keepSoprano?, fixedPitches?, fixedOnsets? }`

`locks` may include:
- `lockedEventIds?: [eventId]` // includes carry-in events (backend may auto-add)
- `lockedRanges?: [{ startTick, endTick, type: "pitch" | "onset" | "all" }]`

`mode` options:
- `"window"` (default MVP): regenerate only selected measure(s); suffix compatibility is approximate
- `"repair"` (optional): blocked-Gibbs refinement over masked measures to better match frozen prefix/suffix constraints

`changedMeasureIds`:
- `"window"`: `[measureId]` always
- `"repair"`: may include neighbors

---

## Revisions and drafts (optimistic concurrency)

- Server rejects stale writes with `409 Conflict`
- Drafts are tied to `baseRevision`
- Storage: `scores` + `drafts` tables, TTL/GC for drafts

---

## MVP scope + ignored edge cases

MVP: read-only sheet/tab, inpaint per measure, keep/revert, playback, exports.

Ignore in Phase 1:
- editing the onset of a carry-in note without expanding selection (UI expands selection instead)
- repeats/DS-DC/voltas
- grace notes/tuplets
- multi-part scores beyond single guitar staff
- multi-rest compression

---

## Inpaint guarantees (important)

Guaranteed:
- Canonical sustained events are never split in the model; ties are derived at export
- Carry-in notes (already sounding at measure start) are preserved during inpaint
- Inpaint never creates orphan ties because ties are generated only during MusicXML export from sustained events

Approximate (depends on mode):
- If using left-to-right decode ("window" mode), compatibility with the frozen suffix is not guaranteed
- If using blocked-Gibbs repair ("repair" mode), suffix compatibility improves but remains approximate

---

## Golden tests (keep it small)

- CanonicalScore → MusicXML → AlphaTab render smoke test
- Tabber output includes `<string>`/`<fret>` for all fretted notes
- Event crossing barline: create an event that starts on beat 4 of measure 1 and lasts 2 beats; MusicXML emits two `<note>` tags with `tie start` then `tie stop`, both mapping to the same `eventId`
- A few regression fixtures with known fingerings + measure IDs
- Click note → hit key → resolve `eventId` → `apply_fingering` updates only `<technical>`, not pitch

---

## Performance (MVP)

- Keep pieces ~16–24 measures
- Use `renderStarted` / `renderFinished` to show loading
- Target `POST /inpaint_preview` < 3s; add a queue only if needed
