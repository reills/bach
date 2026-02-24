# SPEC — Stage-1 Tokenizer & Dataset (Condensed)

## Goal
Implement the Stage-1 symbolic event stream and dataset builder for Bach-style polyphony on a fixed 24 TPQ grid.

## Core Rules
- TPQ is fixed at 24.
- Events are sparse: emit `POS_t` only where at least one voice event starts.
- Per-voice continuity is tracked across barlines by `prev_pitch` and `active_until`.
- Absolute pitch truth is `ABS_VOICE_*` + `MEL_INT12_*`.
- `HARM_*` tokens are derived from the lowest active reference pitch at time t.

## Event Schema (v1)
Structural:
- `BAR`, `TIME_SIG_*`, `KEY_*`, `TEMPO_*`, `POS_*`

Voice events:
- `VOICE_{0..K-1}`, `DUR_{ticks}`, optional `REST_{ticks}`

Anchors:
- `ABS_VOICE_{v}_{MIDI}` (per-voice)
- Optional `ABS_LOW_{MIDI}`, `ABS_HIGH_{MIDI}`

Intervals (pitched onsets only):
- `MEL_INT12_{-24..+24}`
- `HARM_OCT_{-2..4 | NA}`
- `HARM_CLASS_{0..11 | NA}`

Canonical ordering for pitched onsets:
`VOICE_v, DUR_{ticks}, (optional DUP_{n}), MEL_INT12_*, HARM_OCT_*, HARM_CLASS_*`

## Harmonic Reference
At each `POS_t` slice:
- `ref_pitch_t` is the lowest active pitch considering onsets at `POS_t` and sustaining notes.
- `HARM_*` is computed from `pitch - ref_pitch_t` if a reference exists; otherwise emit `NA`.

## Anchors
- Emit `ABS_VOICE_{v}_{MIDI}` at bar start if a track is sounding at the boundary.
- Re-entry or large leaps should emit anchors deterministically.

## Invariants (QA mode)
- Reconstruct pitch from anchors + `MEL_INT12` and verify `HARM_*` matches `pitch - ref_pitch_t` when reference exists.
- Out-of-range `HARM_OCT` raises in QA; in production mode, repair and log counts.

## Dataset Outputs
`scripts/make_dataset.py` produces:
- `data/processed/events.parquet`
- `data/processed/barplans.parquet`
- `data/processed/stats.json`

## Tests
- Round-trip decode of random bars preserves pitch (tolerance ±1 tick for tie joins).
- Interval range and reference-pitch edge cases are covered.
