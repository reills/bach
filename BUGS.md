# Known Bugs — Bach Gen

Tracked issues with root-cause analysis and fix plan.

---

## Bug 1 — Playback produces no sound

**Symptom:** Clicking Play does nothing audible.

**Root cause (two parts):**

1. `ScoreViewer.tsx` initialises the AlphaTab player with `enablePlayer: true` but never passes a `soundFont` URL. AlphaTab requires an explicit path to a `.sf2`/`.sf3` file before it can synthesise audio. The files exist at `frontend/public/soundfont/sonivox.sf2` and `sonivox.sf3` but are never referenced.

2. `App.tsx` `handlePlay/Pause/Stop` guesses the player sub-object path (`alphaTabApi.player ?? alphaTabApi.Player`). In recent AlphaTab builds the player is accessed via `api.player` (lowercase), but the underlying play/pause/stop calls must only happen **after** the `playerReady` event fires. The current code calls play immediately, before the soundfont has loaded.

**Fix plan:**

- Add `soundFont: '/soundfont/sonivox.sf2'` (and optionally `soundFontSampleRate`) to the `player` block in the `settings` object in `ScoreViewer.tsx`.
- Subscribe to `api.playerReady.on(...)` and expose a ready flag through `onApiReady` or a new `onPlayerReady` prop so the Play button in `App.tsx` is only enabled once the soundfont is loaded.
- In `handlePlay`, call `alphaTabApi.playPause()` (the unified toggle that AlphaTab exposes) rather than navigating the sub-object hierarchy.

---

## Bug 2 — No playback cursor on the score

**Symptom:** Music plays (once Bug 1 is fixed) but there is no moving cursor on the staff showing the current beat position.

**Root cause:** AlphaTab's cursor overlay (`alphaTab-cursor` elements) is injected automatically when the player is active, but requires the container element to have `position: relative` or `position: absolute` and the CSS class `alphaTab` applied. The current `score-viewer__canvas` div has neither. Additionally, AlphaTab needs the `player.cursor` setting enabled (it defaults to `true` but may be overridden by the `display` settings or a missing DOM layout).

**Fix plan:**

- Ensure the wrapper `div` that is passed to `AlphaTabApi` has `position: relative` in its CSS so the cursor overlay renders inside it, not at the document root.
- Add `cursor: true` explicitly to the `player` settings block to be safe.
- Wire `api.playerPositionChanged.on(...)` to update a visible time-position indicator in the UI as a fallback for non-visual playback.

---

## Bug 3 — Sheet music and tab shown together; should be separate tabs

**Symptom:** The viewer label says "Score + Tab" and both staves are always rendered together. There is no way to view only sheet music or only tab.

**Root cause:** `ScoreViewer.tsx` sets `staveProfile: StaveProfile.ScoreTab` (value `0`), which renders both staves in a single combined view. There is no tab UI to switch between score-only and tab-only views.

**Fix plan — instrument mode + two-tab viewer:**

The viewer and backend should both understand an instrument mode. This drives what stave profiles are available and whether the tabber runs.

**Backend (`compose_service.py`):**
- Add `render_mode: Literal["guitar", "piano"] = "guitar"` to the compose call and the API route.
- In `"piano"` mode: skip `_tab_score` entirely; set `PartInfo.instrument = "piano"` so the MusicXML exports with treble+bass grand stave, no tab staff.
- In `"guitar"` mode: run `_normalize_to_guitar_range` (see Bug 5) then `_tab_score` as now. The MusicXML will contain fret/string `<technical>` annotations on every note.

**Frontend (`ScoreViewer.tsx` + `App.tsx`):**
- Add `viewTab: 'score' | 'tab'` state to `App`.
- Pass it as a prop to `ScoreViewer` and map it to `StaveProfile`:
  - `'score'` → `StaveProfile.Score` (sheet music only, treble+bass or treble clef)
  - `'tab'` → `StaveProfile.Tab` (guitar tablature only)
- When `viewTab` changes, call `api.settings.display.staveProfile = newValue; api.updateSettings(); api.render()` — no need to destroy and recreate the API instance.
- In piano mode only expose the "Sheet Music" tab. In guitar mode expose both "Sheet Music" and "Guitar Tab".
- AlphaTab reads fret/string data from the MusicXML `<technical>` elements automatically, so tab rendering is free once the backend includes that data.

**Why client-side tab transcription is not the right approach:**
AlphaTab cannot auto-assign fret/string from pitch alone when reading MusicXML — it needs explicit `<string>/<fret>` annotations. The backend tabber (`heuristic.py`) already solves this correctly. The client should just trust the annotations in the XML and switch stave profiles.

---

## Future: mini score editor

Realistic scope, ordered by complexity:

| Feature | Difficulty | Approach |
|---|---|---|
| Key transposition | Low | Parse MusicXML, shift `<fifths>` and all pitch elements by N semitones, reload into AlphaTab |
| Note pitch up/down (semitone/octave) | Medium | AlphaTab `noteMouseDown` gives bar/beat/voice/note indices; find the note in the XML, adjust `<step>/<octave>/<alter>`, reload |
| Measure re-generation (inpaint) | Already wired | Exists in `handleInpaintPreview` |
| Add/remove notes | Hard | Requires tracking beat subdivisions, duration accounting, beaming — full notation editor scope |

The note pitch editing path uses what's already half-built: `onNoteClick` → `HitKey` → find in XML → mutate → `api.load()`. The key blocker is a reliable XML round-trip function that can locate a note by its AlphaTab hit key (bar/beat/voice/note indices map to `<measure number>` / beat position / `<voice>` / note ordinal in MusicXML). That mapping function is the core piece to build.

---

## Bug 4 — Score is not interactive (cannot click measures or notes)

**Symptom:** Clicking anywhere on the rendered score does nothing. Measure selection never fires; fingering picker never opens.

**Root cause (likely):** AlphaTab's `noteMouseDown` / `barMouseDown` events are registered correctly in `ScoreViewer.tsx`, but AlphaTab only fires these events when `core.includeNoteBounds` is `true` **and** the container element has received the pointer events. The `score-viewer__canvas` div is a plain div with no explicit CSS `cursor` or `pointer-events` style, but the real problem is that the AlphaTab canvas element (`canvas`) may be receiving the events while the wrapper div does not forward them. Additionally, if the `staveProfile` resolves to `0` via the fallback chain and AlphaTab internally maps that to a mode where bounds are not computed, no events fire.

**Fix plan:**

- Add `display: { interactionMode: 'GestureAndMouse' }` (or the equivalent enum value) to the AlphaTab settings to ensure the library listens for mouse events.
- Confirm that `StaveProfile.Score` (value `1`) or `StaveProfile.ScoreTab` (value `3`) is being set — not the default `0` which may be `Default` (no staves) in some builds.
- Log the resolved `staveProfile` value and `alphaTab.StaveProfile` enum on init to verify the fallback chain picks the right constant.
- Test clicking a note after confirming the above; if still broken, subscribe to the raw `pointerdown` event on the canvas element as a workaround and map screen coordinates back via `api.noteBoundsLookup`.

---

## Bug 5 — Generated music has notes at impossible pitch extremes (two symptoms)

### Symptom A — Rendering
Rendered score shows notes many ledger lines above or below the staff, far outside the range of a real guitar.

### Symptom B — Tabber crash (new failure type: `stage: "tab"`)
Compose fails with `"no playable guitar voicing at onset N: note is outside the fret range"`. Example: MIDI pitch 39 (Eb2) is generated, which is one semitone below the lowest open string on a standard guitar (MIDI 40 = E2). `_candidate_fingerings` returns an empty list and `_assign_fingerings` raises.

**Root cause:** The model was trained on classical SATB / Bach-chorale data where bass voices routinely descend to MIDI 36 (C2) or lower. These are valid pitches for piano or organ but not for a 6-string guitar in standard tuning (range MIDI 40–88). The parser (`_events_from_bar`) accumulates `pitch_midi = previous_pitch + mel_int` with no clamping, and the tabber (`heuristic.py`) treats an unplayable note as a hard error. The pipeline has no concept of instrument modes.

**Proposed fix — instrument mode switch:**

The cleanest solution is a `render_mode` parameter (`"guitar"` | `"piano"`) passed into `compose_service`:

**Piano mode (`render_mode="piano"`):**
- Skip `_tab_score` entirely — no fingering assignment, no guitar range check.
- Export MusicXML with a standard grand-staff (treble + bass) clef instead of a guitar TAB stave. The existing `canonical_score_to_musicxml` already supports this if the `PartInfo.instrument` is set to something other than `"classical_guitar"`.
- No pitch constraints beyond MIDI 0–127. The model's SATB output is rendered as-is.

**Guitar mode (`render_mode="guitar"`, current default):**
- Before calling `_tab_score`, compute the minimum pitch across all events.
- If `min_pitch < 40`, shift the entire score up by `ceil((40 - min_pitch) / 12) * 12` semitones (nearest whole octave) so the lowest note lands at or above MIDI 40.
- Then run the tabber as normal.
- This keeps all intervals intact and avoids re-training; it just moves the key up one octave when needed.

```python
# In compose_service.py, before _tab_score:
def _normalize_to_guitar_range(score: CanonicalScore) -> CanonicalScore:
    import math
    events = [e for p in score.parts for e in p.events if e.pitch_midi is not None]
    if not events:
        return score
    min_pitch = min(e.pitch_midi for e in events)
    if min_pitch >= 40:
        return score
    shift = math.ceil((40 - min_pitch) / 12) * 12
    new_parts = []
    for part in score.parts:
        new_events = [
            replace(e, pitch_midi=e.pitch_midi + shift) if e.pitch_midi is not None else e
            for e in part.events
        ]
        new_parts.append(replace(part, events=new_events))
    return replace(score, parts=new_parts)
```

**Short-term mitigation (while mode switch is being built):**
Apply `_normalize_to_guitar_range` unconditionally before `_tab_score` in the current pipeline. This stops crashes immediately with no API changes.

---

## Bug 6 — Compose fails: "part events must be sorted by start_tick"

**Symptom:** Most compose attempts write a failure report with `"stage": "parse"` and `"message": "part events must be sorted by start_tick"`.

**Root cause:** In `tokens_to_canonical_score`, events are collected bar-by-bar and appended in token-stream order. Within a single bar the model is free to emit `POS_*` tokens in any order it learned — it may emit `POS_48` followed by `POS_12` within the same bar. Because `_events_from_bar` simply appends each event as its `POS_*` is encountered, the resulting list is not guaranteed to be sorted by `start_tick`. The `CanonicalScore` constructor then raises `ValueError` when it detects the unsorted sequence (line 210–211 of `types.py`).

**Fix plan (minimal, safe):**

In `tokens_to_canonical_score` (or just before constructing `Part`), sort the full events list:

```python
events.sort(key=lambda e: e.start_tick)
```

This matches what the validator requires and is harmless because the event IDs are already stable (derived from voice, tick, and ordinal). The ordinal within a tick may shift if two events land at the same tick in a different order, but that is already non-deterministic when a model emits two voices at the same position.

**Alternative (stricter):** sort within each bar's output in `_events_from_bar` immediately before returning, so the sort is local and the root cause is contained:

```python
return sorted(events, key=lambda e: e.start_tick)
```

Apply both for belt-and-suspenders correctness.
