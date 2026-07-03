from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Sequence

from src.api.canonical import CanonicalScore, Event, GuitarFingering, Part, PartInfo, ScoreHeader
from src.arrangers.guitar.constraints import GuitarArrangementSettings
from src.arrangers.guitar.diagnostics import (
    DroppedNoteDiagnostic,
    GuitarConversionDiagnostics,
    HandPositionCompromiseDiagnostic,
    ImpossibleChordDiagnostic,
    OctaveShiftDiagnostic,
    RangeChangeDiagnostic,
)
from src.arrangers.guitar.source_map import PianoToGuitarNoteMap, PianoToGuitarSourceMap
from src.tabber import tab_events


@dataclass(frozen=True)
class GuitarArrangement:
    score: CanonicalScore
    diagnostics: GuitarConversionDiagnostics
    source_map: PianoToGuitarSourceMap
    settings: GuitarArrangementSettings

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostics": self.diagnostics.to_dict(),
            "sourceMap": self.source_map.to_dict(),
            "settings": {
                "tuning": list(self.settings.tuning),
                "difficulty": self.settings.difficulty,
                "maxFret": self.settings.max_fret,
                "octaveShiftPolicy": self.settings.octave_shift_policy,
                "allowDropNotes": self.settings.allow_drop_notes,
                "preserveMelody": self.settings.preserve_melody,
                "preserveBass": self.settings.preserve_bass,
                "maxHandSpanFrets": self.settings.resolved_max_hand_span_frets,
                "maxNotesPerOnset": self.settings.resolved_max_notes_per_onset,
                "preferredPosition": self.settings.preferred_position,
                "targetInstrument": self.settings.target_instrument,
                "midiProgram": self.settings.midi_program,
            },
        }


@dataclass(frozen=True)
class _PreparedEvent:
    source_event: Event
    target_event: Event
    semitone_shift: int = 0


def convert_piano_score_to_guitar(
    score: CanonicalScore,
    *,
    settings: GuitarArrangementSettings | None = None,
) -> GuitarArrangement:
    resolved_settings = settings or GuitarArrangementSettings()
    if len(score.parts) != 1:
        raise ValueError("guitar arranger supports exactly one source part")

    source_part = score.parts[0]
    dropped_notes: list[DroppedNoteDiagnostic] = []
    octave_shifts: list[OctaveShiftDiagnostic] = []
    impossible_chords: list[ImpossibleChordDiagnostic] = []
    range_changes: list[RangeChangeDiagnostic] = []
    source_maps_by_id: dict[str, PianoToGuitarNoteMap] = {}
    tonic_pc = _tonic_pc_from_header(score.header)

    prepared_events = _prepare_events(
        source_part.events,
        settings=resolved_settings,
        dropped_notes=dropped_notes,
        octave_shifts=octave_shifts,
        range_changes=range_changes,
        source_maps_by_id=source_maps_by_id,
    )

    playable_events = _fit_onset_voicings(
        prepared_events,
        settings=resolved_settings,
        tonic_pc=tonic_pc,
        dropped_notes=dropped_notes,
        octave_shifts=octave_shifts,
        range_changes=range_changes,
        impossible_chords=impossible_chords,
        source_maps_by_id=source_maps_by_id,
    )

    try:
        tabbed_events = tab_events(
            [prepared.target_event for prepared in playable_events],
            tuning=resolved_settings.tuning,
            max_fret=resolved_settings.max_fret,
        )
    except ValueError:
        if resolved_settings.allow_drop_notes:
            playable_events = _fit_onset_voicings(
                playable_events,
                settings=resolved_settings,
                tonic_pc=tonic_pc,
                dropped_notes=dropped_notes,
                octave_shifts=octave_shifts,
                range_changes=range_changes,
                impossible_chords=impossible_chords,
                source_maps_by_id=source_maps_by_id,
                force_drop_until_tabbed=True,
            )
            tabbed_events = tab_events(
                [prepared.target_event for prepared in playable_events],
                tuning=resolved_settings.tuning,
                max_fret=resolved_settings.max_fret,
            )
        else:
            raise

    tabbed_events = _renumber_voice_ids(tabbed_events)
    prepared_by_target_id = {prepared.target_event.id: prepared for prepared in playable_events}
    note_maps: list[PianoToGuitarNoteMap] = []
    tabbed_events_by_id = {event.id: event for event in tabbed_events}
    for source_event in source_part.events:
        if source_event.pitch_midi is None:
            continue
        dropped_map = source_maps_by_id.get(source_event.id)
        if dropped_map is not None:
            note_maps.append(dropped_map)
            continue

        target_id = _guitar_event_id(source_event.id)
        tabbed_event = tabbed_events_by_id[target_id]
        prepared = prepared_by_target_id[target_id]
        note_maps.append(
            PianoToGuitarNoteMap(
                source_event_id=source_event.id,
                target_event_id=tabbed_event.id,
                start_tick=source_event.start_tick,
                dur_tick=source_event.dur_tick,
                voice_id=source_event.voice_id,
                source_pitch_midi=source_event.pitch_midi,
                target_pitch_midi=tabbed_event.pitch_midi,
                semitone_shift=prepared.semitone_shift,
                dropped=False,
                fingering=tabbed_event.fingering,
            )
        )

    hand_position_compromises = _find_hand_position_compromises(
        tabbed_events,
        source_part.events,
        settings=resolved_settings,
    )
    diagnostics = GuitarConversionDiagnostics(
        dropped_notes=dropped_notes,
        octave_shifted_notes=octave_shifts,
        impossible_chords=impossible_chords,
        range_changes=range_changes,
        hand_position_compromises=hand_position_compromises,
        warnings=_build_warnings(
            dropped_notes=dropped_notes,
            octave_shifts=octave_shifts,
            impossible_chords=impossible_chords,
            hand_position_compromises=hand_position_compromises,
        ),
    )
    guitar_part = Part(
        info=PartInfo(
            id=source_part.info.id,
            instrument=resolved_settings.target_instrument,
            tuning=list(resolved_settings.tuning),
            midi_program=resolved_settings.midi_program,
        ),
        events=tabbed_events,
    )
    guitar_score = CanonicalScore(
        header=_copy_header(score.header),
        measures=list(score.measures),
        parts=[guitar_part],
    )
    return GuitarArrangement(
        score=guitar_score,
        diagnostics=diagnostics,
        source_map=PianoToGuitarSourceMap(notes=note_maps),
        settings=resolved_settings,
    )


def _prepare_events(
    source_events: Sequence[Event],
    *,
    settings: GuitarArrangementSettings,
    dropped_notes: list[DroppedNoteDiagnostic],
    octave_shifts: list[OctaveShiftDiagnostic],
    range_changes: list[RangeChangeDiagnostic],
    source_maps_by_id: dict[str, PianoToGuitarNoteMap],
) -> list[_PreparedEvent]:
    prepared_events: list[_PreparedEvent] = []
    for onset_tick in sorted({event.start_tick for event in source_events}):
        onset_events = [event for event in source_events if event.start_tick == onset_tick]
        pitched_events = [event for event in onset_events if event.pitch_midi is not None]

        for source_event in onset_events:
            if source_event.pitch_midi is None:
                prepared_events.append(
                    _PreparedEvent(
                        source_event=source_event,
                        target_event=_to_guitar_event(source_event, source_event.pitch_midi),
                    )
                )
                continue

            arranged_pitch, range_reason = _arrange_pitch_for_context(
                source_event,
                onset_pitched_events=pitched_events,
                settings=settings,
            )
            if arranged_pitch is None:
                if not settings.allow_drop_notes:
                    raise ValueError("no playable guitar voicing at onset "
                                     f"{source_event.start_tick}: note is outside the fret range")
                _record_dropped_source_event(
                    source_event,
                    reason=range_reason or "note is outside the configured guitar range",
                    dropped_notes=dropped_notes,
                    source_maps_by_id=source_maps_by_id,
                )
                continue

            semitone_shift = arranged_pitch - source_event.pitch_midi
            if semitone_shift != 0:
                octave_shifts.append(
                    OctaveShiftDiagnostic(
                        source_event_id=source_event.id,
                        start_tick=source_event.start_tick,
                        voice_id=source_event.voice_id,
                        original_pitch_midi=source_event.pitch_midi,
                        arranged_pitch_midi=arranged_pitch,
                        semitone_shift=semitone_shift,
                    )
                )
                range_changes.append(
                    RangeChangeDiagnostic(
                        source_event_id=source_event.id,
                        start_tick=source_event.start_tick,
                        original_pitch_midi=source_event.pitch_midi,
                        arranged_pitch_midi=arranged_pitch,
                        reason=range_reason or "octave-shifted into configured guitar range",
                    )
                )

            prepared_events.append(
                _PreparedEvent(
                    source_event=source_event,
                    target_event=_to_guitar_event(source_event, arranged_pitch),
                    semitone_shift=semitone_shift,
                )
            )

    return prepared_events


def _arrange_pitch_for_context(
    source_event: Event,
    *,
    onset_pitched_events: Sequence[Event],
    settings: GuitarArrangementSettings,
) -> tuple[int | None, str | None]:
    pitch_midi = source_event.pitch_midi
    if pitch_midi is None:
        return None, None
    if _has_fingering_candidate(pitch_midi, settings=settings):
        return pitch_midi, None
    if settings.octave_shift_policy == "none":
        return None, "note is outside the configured guitar range and octave shifting is disabled"

    if pitch_midi < settings.lowest_pitch and settings.octave_shift_policy in {"below_range", "outside_range"}:
        shifted_pitch = _nearest_octave_candidate(pitch_midi, direction=12, settings=settings)
        if shifted_pitch is None:
            return None, "note is below the configured guitar range"
        if _octave_shift_would_invert_bass(source_event, shifted_pitch, onset_pitched_events, settings=settings):
            return None, "below-range bass note was dropped because octave shifting would invert the bass line"
        if _below_range_inner_note_should_drop(source_event, onset_pitched_events, settings=settings):
            return None, "below-range inner note was dropped instead of octave-shifted"
        return shifted_pitch, "octave-shifted upward into configured guitar range"

    if pitch_midi > settings.highest_pitch and settings.octave_shift_policy == "outside_range":
        shifted_pitch = _nearest_octave_candidate(pitch_midi, direction=-12, settings=settings)
        if shifted_pitch is None:
            return None, "note is above the configured guitar range"
        if _above_range_inner_note_should_drop(source_event, onset_pitched_events, settings=settings):
            return None, "above-range inner note was dropped instead of octave-shifted"
        return shifted_pitch, "octave-shifted downward into configured guitar range"

    return None, "note is outside the configured guitar range"


def _nearest_octave_candidate(
    pitch_midi: int,
    *,
    direction: int,
    settings: GuitarArrangementSettings,
) -> int | None:
    shifted_pitch = pitch_midi
    while 0 <= shifted_pitch <= 127:
        shifted_pitch += direction
        if not 0 <= shifted_pitch <= 127:
            return None
        if _has_fingering_candidate(shifted_pitch, settings=settings):
            return shifted_pitch
    return None


def _octave_shift_would_invert_bass(
    source_event: Event,
    shifted_pitch: int,
    onset_pitched_events: Sequence[Event],
    *,
    settings: GuitarArrangementSettings,
) -> bool:
    if not settings.allow_drop_notes or source_event.pitch_midi is None:
        return False
    source_pitches = [event.pitch_midi for event in onset_pitched_events if event.pitch_midi is not None]
    if not source_pitches or source_event.pitch_midi != min(source_pitches):
        return False
    other_playable_pitches = [
        event.pitch_midi
        for event in onset_pitched_events
        if event.id != source_event.id
        and event.pitch_midi is not None
        and _has_fingering_candidate(event.pitch_midi, settings=settings)
    ]
    return bool(other_playable_pitches and shifted_pitch > min(other_playable_pitches))


def _below_range_inner_note_should_drop(
    source_event: Event,
    onset_pitched_events: Sequence[Event],
    *,
    settings: GuitarArrangementSettings,
) -> bool:
    if not settings.allow_drop_notes or source_event.pitch_midi is None:
        return False
    source_pitches = [event.pitch_midi for event in onset_pitched_events if event.pitch_midi is not None]
    return bool(source_pitches and source_event.pitch_midi != min(source_pitches) and len(source_pitches) > 1)


def _above_range_inner_note_should_drop(
    source_event: Event,
    onset_pitched_events: Sequence[Event],
    *,
    settings: GuitarArrangementSettings,
) -> bool:
    if not settings.allow_drop_notes or source_event.pitch_midi is None:
        return False
    source_pitches = [event.pitch_midi for event in onset_pitched_events if event.pitch_midi is not None]
    return bool(source_pitches and source_event.pitch_midi != max(source_pitches) and len(source_pitches) > 1)


def _fit_onset_voicings(
    prepared_events: Sequence[_PreparedEvent],
    *,
    settings: GuitarArrangementSettings,
    tonic_pc: int,
    dropped_notes: list[DroppedNoteDiagnostic],
    octave_shifts: list[OctaveShiftDiagnostic],
    range_changes: list[RangeChangeDiagnostic],
    impossible_chords: list[ImpossibleChordDiagnostic],
    source_maps_by_id: dict[str, PianoToGuitarNoteMap],
    force_drop_until_tabbed: bool = False,
) -> list[_PreparedEvent]:
    kept_events: list[_PreparedEvent] = []
    previous_by_voice: dict[int, GuitarFingering] = {}
    previous_position = settings.preferred_position
    for onset_tick in sorted({prepared.target_event.start_tick for prepared in prepared_events}):
        onset_events = [
            prepared
            for prepared in prepared_events
            if prepared.target_event.start_tick == onset_tick
        ]
        rests = [prepared for prepared in onset_events if prepared.target_event.pitch_midi is None]
        pitched = [prepared for prepared in onset_events if prepared.target_event.pitch_midi is not None]
        playable = list(pitched)
        max_notes = settings.resolved_max_notes_per_onset

        while len(playable) > max_notes:
            if not settings.allow_drop_notes:
                raise ValueError(f"no playable guitar voicing at onset {onset_tick}: too many notes for difficulty")
            dropped = _least_important_note(playable, settings=settings, tonic_pc=tonic_pc)
            playable.remove(dropped)
            _record_dropped_note(
                dropped,
                reason="dense piano voicing was thinned for configured guitar difficulty",
                dropped_notes=dropped_notes,
                source_maps_by_id=source_maps_by_id,
            )

        while playable:
            fingerings = _choose_onset_fingerings(
                playable,
                previous_by_voice=previous_by_voice,
                previous_position=previous_position,
                settings=settings,
            )
            if fingerings is not None:
                break

            wide_fingerings = _choose_onset_fingerings(
                playable,
                previous_by_voice=previous_by_voice,
                previous_position=previous_position,
                settings=settings,
                enforce_hand_span=False,
            )
            rewritten = _try_octave_rewrite_for_hand_span(
                playable,
                wide_fingerings=wide_fingerings,
                previous_by_voice=previous_by_voice,
                previous_position=previous_position,
                settings=settings,
            )
            if rewritten is not None:
                _record_octave_rewrites(
                    playable,
                    rewritten,
                    octave_shifts=octave_shifts,
                    range_changes=range_changes,
                )
                playable = rewritten
                continue

            if not settings.allow_drop_notes:
                raise ValueError(
                    f"no playable guitar voicing at onset {onset_tick}: "
                    "duplicate string use or impractical hand span required"
                )
            dropped = _least_important_note(playable, settings=settings, tonic_pc=tonic_pc)
            playable.remove(dropped)
            _record_dropped_note(
                dropped,
                reason="voicing required duplicate string use or an impractical fret span",
                dropped_notes=dropped_notes,
                source_maps_by_id=source_maps_by_id,
            )

        if force_drop_until_tabbed:
            while playable:
                try:
                    tab_events(
                        [prepared.target_event for prepared in playable],
                        tuning=settings.tuning,
                        max_fret=settings.max_fret,
                    )
                    break
                except ValueError as exc:
                    if not settings.allow_drop_notes:
                        raise
                    dropped = _least_important_note(playable, settings=settings, tonic_pc=tonic_pc)
                    playable.remove(dropped)
                    _record_dropped_note(
                        dropped,
                        reason=str(exc),
                        dropped_notes=dropped_notes,
                        source_maps_by_id=source_maps_by_id,
                    )

        fingerings = _choose_onset_fingerings(
            playable,
            previous_by_voice=previous_by_voice,
            previous_position=previous_position,
            settings=settings,
        )
        if fingerings is None:
            playable_with_fingerings: list[_PreparedEvent] = []
        else:
            playable_with_fingerings = [
                replace(
                    prepared,
                    target_event=replace(
                        prepared.target_event,
                        fingering=fingerings[prepared.target_event.id],
                    ),
                )
                for prepared in playable
            ]
            for prepared in playable_with_fingerings:
                fingering = prepared.target_event.fingering
                if fingering is not None:
                    previous_by_voice[prepared.target_event.voice_id] = fingering
            if fingerings:
                previous_position = _position_from_fingerings(fingerings.values())

        if pitched and not playable:
            impossible_chords.append(
                ImpossibleChordDiagnostic(
                    onset_tick=onset_tick,
                    source_event_ids=[prepared.source_event.id for prepared in pitched],
                    reason="all pitched notes at this onset were unplayable",
                )
            )

        if len(playable) < len(pitched):
            impossible_chords.append(
                ImpossibleChordDiagnostic(
                    onset_tick=onset_tick,
                    source_event_ids=[prepared.source_event.id for prepared in pitched],
                    reason="voicing required dropping one or more notes",
                )
            )

        kept_events.extend(rests)
        kept_events.extend(playable_with_fingerings)

    return sorted(kept_events, key=lambda prepared: (prepared.target_event.start_tick, prepared.target_event.voice_id))


def _least_important_note(
    notes: Sequence[_PreparedEvent],
    *,
    settings: GuitarArrangementSettings,
    tonic_pc: int,
) -> _PreparedEvent:
    source_pitches = [prepared.source_event.pitch_midi for prepared in notes if prepared.source_event.pitch_midi is not None]
    highest_pitch = max(source_pitches)
    lowest_pitch = min(source_pitches)

    def drop_priority(prepared: _PreparedEvent) -> tuple[int, int, int, int]:
        source_event = prepared.source_event
        pitch = source_event.pitch_midi or 0
        middle_distance = min(abs(pitch - lowest_pitch), abs(pitch - highest_pitch))
        return (
            _note_importance(prepared, notes, settings=settings, tonic_pc=tonic_pc),
            -middle_distance,
            pitch,
            -source_event.voice_id,
        )

    return min(notes, key=drop_priority)


def _try_octave_rewrite_for_hand_span(
    playable: Sequence[_PreparedEvent],
    *,
    wide_fingerings: dict[str, GuitarFingering] | None,
    previous_by_voice: dict[int, GuitarFingering],
    previous_position: int | None,
    settings: GuitarArrangementSettings,
) -> list[_PreparedEvent] | None:
    if wide_fingerings is None or settings.octave_shift_policy == "none":
        return None

    attempts = _octave_rewrite_attempts(playable, wide_fingerings, settings=settings)
    for prepared, semitone_delta in attempts:
        rewritten_event = _rewrite_prepared_pitch(
            prepared,
            semitone_delta=semitone_delta,
            settings=settings,
        )
        if rewritten_event is None:
            continue

        rewritten = [
            rewritten_event if candidate.target_event.id == prepared.target_event.id else candidate
            for candidate in playable
        ]
        if not _preserves_source_pitch_order(rewritten):
            continue
        if _choose_onset_fingerings(
            rewritten,
            previous_by_voice=previous_by_voice,
            previous_position=previous_position,
            settings=settings,
        ) is None:
            continue
        return rewritten

    return None


def _octave_rewrite_attempts(
    playable: Sequence[_PreparedEvent],
    wide_fingerings: dict[str, GuitarFingering],
    *,
    settings: GuitarArrangementSettings,
) -> list[tuple[_PreparedEvent, int]]:
    fretted = [
        (prepared, wide_fingerings[prepared.target_event.id])
        for prepared in playable
        if prepared.target_event.id in wide_fingerings
        and wide_fingerings[prepared.target_event.id].fret > 0
    ]
    if not fretted:
        return []

    max_fret = max(fingering.fret for _, fingering in fretted)
    min_fret = min(fingering.fret for _, fingering in fretted)
    attempts: list[tuple[_PreparedEvent, int]] = []
    seen: set[tuple[str, int]] = set()
    source_pitches = [
        prepared.source_event.pitch_midi
        for prepared in playable
        if prepared.source_event.pitch_midi is not None
    ]
    highest_source_pitch = max(source_pitches) if source_pitches else None

    def add(prepared: _PreparedEvent, semitone_delta: int) -> None:
        key = (prepared.target_event.id, semitone_delta)
        if key in seen:
            return
        seen.add(key)
        attempts.append((prepared, semitone_delta))

    if settings.preserve_melody:
        for prepared, fingering in sorted(
            fretted,
            key=lambda item: item[0].source_event.pitch_midi or 128,
        ):
            if fingering.fret == min_fret and prepared.source_event.pitch_midi != highest_source_pitch:
                add(prepared, 12)

    for prepared, fingering in sorted(
        fretted,
        key=lambda item: item[0].source_event.pitch_midi or -1,
        reverse=True,
    ):
        if fingering.fret == max_fret and prepared.source_event.pitch_midi != highest_source_pitch:
            add(prepared, -12)

    for prepared, fingering in sorted(
        fretted,
        key=lambda item: item[0].source_event.pitch_midi or 128,
    ):
        if fingering.fret == min_fret:
            add(prepared, 12)

    for prepared in sorted(
        playable,
        key=lambda item: item.source_event.pitch_midi or -1,
        reverse=True,
    ):
        add(prepared, -12)

    return attempts


def _rewrite_prepared_pitch(
    prepared: _PreparedEvent,
    *,
    semitone_delta: int,
    settings: GuitarArrangementSettings,
) -> _PreparedEvent | None:
    source_pitch = prepared.source_event.pitch_midi
    target_pitch = prepared.target_event.pitch_midi
    if source_pitch is None or target_pitch is None:
        return None

    rewritten_pitch = target_pitch + semitone_delta
    if not 0 <= rewritten_pitch <= 127:
        return None
    if not _has_fingering_candidate(rewritten_pitch, settings=settings):
        return None

    return replace(
        prepared,
        target_event=replace(prepared.target_event, pitch_midi=rewritten_pitch, fingering=None),
        semitone_shift=rewritten_pitch - source_pitch,
    )


def _preserves_source_pitch_order(prepared_events: Sequence[_PreparedEvent]) -> bool:
    pitched = [
        prepared
        for prepared in prepared_events
        if prepared.source_event.pitch_midi is not None
        and prepared.target_event.pitch_midi is not None
    ]
    for left_index, left in enumerate(pitched):
        for right in pitched[left_index + 1 :]:
            left_source = left.source_event.pitch_midi
            right_source = right.source_event.pitch_midi
            left_target = left.target_event.pitch_midi
            right_target = right.target_event.pitch_midi
            if left_source is None or right_source is None or left_target is None or right_target is None:
                continue
            if left_source < right_source and left_target > right_target:
                return False
            if left_source > right_source and left_target < right_target:
                return False
    return True


def _record_octave_rewrites(
    before_events: Sequence[_PreparedEvent],
    after_events: Sequence[_PreparedEvent],
    *,
    octave_shifts: list[OctaveShiftDiagnostic],
    range_changes: list[RangeChangeDiagnostic],
) -> None:
    before_by_id = {
        prepared.target_event.id: prepared
        for prepared in before_events
    }
    for after in after_events:
        before = before_by_id.get(after.target_event.id)
        if before is None:
            continue
        if before.target_event.pitch_midi == after.target_event.pitch_midi:
            continue
        _replace_octave_shift_diagnostic(
            after,
            octave_shifts=octave_shifts,
            range_changes=range_changes,
            reason="octave-shifted to keep the guitar fingering within hand span",
        )


def _replace_octave_shift_diagnostic(
    prepared: _PreparedEvent,
    *,
    octave_shifts: list[OctaveShiftDiagnostic],
    range_changes: list[RangeChangeDiagnostic],
    reason: str,
) -> None:
    source_pitch = prepared.source_event.pitch_midi
    target_pitch = prepared.target_event.pitch_midi
    if source_pitch is None or target_pitch is None:
        return

    octave_shifts[:] = [
        diagnostic
        for diagnostic in octave_shifts
        if diagnostic.source_event_id != prepared.source_event.id
    ]
    range_changes[:] = [
        diagnostic
        for diagnostic in range_changes
        if diagnostic.source_event_id != prepared.source_event.id
    ]
    if target_pitch == source_pitch:
        return

    octave_shifts.append(
        OctaveShiftDiagnostic(
            source_event_id=prepared.source_event.id,
            start_tick=prepared.source_event.start_tick,
            voice_id=prepared.source_event.voice_id,
            original_pitch_midi=source_pitch,
            arranged_pitch_midi=target_pitch,
            semitone_shift=target_pitch - source_pitch,
        )
    )
    range_changes.append(
        RangeChangeDiagnostic(
            source_event_id=prepared.source_event.id,
            start_tick=prepared.source_event.start_tick,
            original_pitch_midi=source_pitch,
            arranged_pitch_midi=target_pitch,
            reason=reason,
        )
    )


def _choose_onset_fingerings(
    prepared_events: Sequence[_PreparedEvent],
    *,
    previous_by_voice: dict[int, GuitarFingering],
    previous_position: int | None,
    settings: GuitarArrangementSettings,
    enforce_hand_span: bool = True,
) -> dict[str, GuitarFingering] | None:
    if not prepared_events:
        return {}

    candidate_lists: list[list[GuitarFingering]] = []
    for prepared in prepared_events:
        pitch = prepared.target_event.pitch_midi
        if pitch is None:
            continue
        candidates = _candidate_fingerings(pitch, settings=settings)
        if not candidates:
            return None
        candidates = sorted(
            candidates,
            key=lambda fingering: _single_fingering_cost(
                prepared,
                fingering,
                previous_by_voice=previous_by_voice,
                previous_position=previous_position,
                settings=settings,
            ),
        )
        candidate_lists.append(candidates[:_candidate_limit(settings)])

    best_combo: tuple[GuitarFingering, ...] | None = None
    best_cost: tuple[int, int, int, int, int, int, int, tuple[int, ...]] | None = None
    combo_iter = product(*candidate_lists) if candidate_lists else [()]
    for combo in combo_iter:
        strings = {fingering.string_index for fingering in combo}
        if len(strings) != len(combo):
            continue
        if enforce_hand_span and _fret_span(combo) > settings.resolved_max_hand_span_frets:
            continue
        cost = _combo_fingering_cost(
            prepared_events,
            combo,
            previous_by_voice=previous_by_voice,
            previous_position=previous_position,
            settings=settings,
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_combo = combo

    if best_combo is None:
        return None
    return {
        prepared.target_event.id: fingering
        for prepared, fingering in zip(prepared_events, best_combo)
    }


def _single_fingering_cost(
    prepared: _PreparedEvent,
    fingering: GuitarFingering,
    *,
    previous_by_voice: dict[int, GuitarFingering],
    previous_position: int | None,
    settings: GuitarArrangementSettings,
) -> tuple[int, int, int, int]:
    previous = previous_by_voice.get(prepared.target_event.voice_id)
    voice_movement = 0
    if previous is not None:
        voice_movement = abs(previous.fret - fingering.fret) * 2 + abs(previous.string_index - fingering.string_index)
    position_shift = 0 if previous_position is None else abs(previous_position - fingering.fret)
    preferred_shift = 0 if settings.preferred_position is None else abs(settings.preferred_position - fingering.fret)
    return (
        voice_movement,
        position_shift,
        preferred_shift,
        _open_string_cost(prepared, fingering, settings=settings),
    )


def _combo_fingering_cost(
    prepared_events: Sequence[_PreparedEvent],
    combo: Sequence[GuitarFingering],
    *,
    previous_by_voice: dict[int, GuitarFingering],
    previous_position: int | None,
    settings: GuitarArrangementSettings,
) -> tuple[int, int, int, int, int, int, int, tuple[int, ...]]:
    span = _fret_span(combo)
    span_overage = max(0, span - settings.resolved_max_hand_span_frets)
    max_fret_gap = _max_fret_gap(combo)
    position = _position_from_fingerings(combo)
    position_shift = 0 if previous_position is None else abs(position - previous_position)
    preferred_shift = 0 if settings.preferred_position is None else abs(position - settings.preferred_position)
    voice_movement = 0
    open_string_cost = 0
    total_fret = 0

    for prepared, fingering in zip(prepared_events, combo):
        total_fret += fingering.fret
        open_string_cost += _open_string_cost(prepared, fingering, settings=settings)
        previous = previous_by_voice.get(prepared.target_event.voice_id)
        if previous is not None:
            voice_movement += (
                abs(previous.fret - fingering.fret) * 2
                + abs(previous.string_index - fingering.string_index)
            )

    position_weight = {
        "easy": 10,
        "medium": 8,
        "hard": 5,
    }[settings.difficulty]
    span_weight = {
        "easy": 20,
        "medium": 16,
        "hard": 10,
    }[settings.difficulty]
    return (
        span_overage * 1000,
        span * span_weight,
        max_fret_gap * 8,
        position_shift * position_weight,
        voice_movement * 8,
        preferred_shift * 5,
        open_string_cost + total_fret,
        tuple(fingering.string_index for fingering in combo),
    )


def _open_string_cost(
    prepared: _PreparedEvent,
    fingering: GuitarFingering,
    *,
    settings: GuitarArrangementSettings,
) -> int:
    if fingering.fret != 0:
        return 0
    source_pitch = prepared.source_event.pitch_midi or prepared.target_event.pitch_midi or 0
    is_low_bass = source_pitch <= settings.lowest_pitch + 12
    is_long = prepared.source_event.dur_tick >= 48
    if settings.difficulty == "easy":
        return -8 if is_low_bass or is_long else -3
    return -2 if is_low_bass or is_long else 4


def _candidate_limit(settings: GuitarArrangementSettings) -> int:
    return {
        "easy": 5,
        "medium": 7,
        "hard": 9,
    }[settings.difficulty]


def _candidate_fingerings(
    pitch_midi: int,
    *,
    settings: GuitarArrangementSettings,
) -> list[GuitarFingering]:
    return [
        GuitarFingering(string_index=string_index, fret=pitch_midi - open_pitch)
        for string_index, open_pitch in enumerate(settings.tuning)
        if 0 <= pitch_midi - open_pitch <= settings.max_fret
    ]


def _fret_span(fingerings: Sequence[GuitarFingering]) -> int:
    fretted = [fingering.fret for fingering in fingerings if fingering.fret > 0]
    if len(fretted) < 2:
        return 0
    return max(fretted) - min(fretted)


def _max_fret_gap(fingerings: Sequence[GuitarFingering]) -> int:
    fretted = sorted(
        (fingering.string_index, fingering.fret)
        for fingering in fingerings
        if fingering.fret > 0
    )
    if len(fretted) < 2:
        return 0
    return max(
        abs(right_fret - left_fret)
        for (_, left_fret), (_, right_fret) in zip(fretted, fretted[1:])
    )


def _position_from_fingerings(fingerings: Sequence[GuitarFingering]) -> int:
    fretted = [fingering.fret for fingering in fingerings if fingering.fret > 0]
    if not fretted:
        return 0
    return round(sum(fretted) / len(fretted))


def _note_importance(
    prepared: _PreparedEvent,
    notes: Sequence[_PreparedEvent],
    *,
    settings: GuitarArrangementSettings,
    tonic_pc: int,
) -> int:
    source_pitch = prepared.source_event.pitch_midi or 0
    target_pitch = prepared.target_event.pitch_midi or source_pitch
    source_pitches = [note.source_event.pitch_midi for note in notes if note.source_event.pitch_midi is not None]
    target_pitches = [note.target_event.pitch_midi for note in notes if note.target_event.pitch_midi is not None]
    highest_pitch = max(source_pitches)
    lowest_pitch = min(source_pitches)
    pitch_class_counts: dict[int, int] = {}
    for pitch in target_pitches:
        pitch_class_counts[pitch % 12] = pitch_class_counts.get(pitch % 12, 0) + 1

    important_harmony_pcs = _important_harmony_pitch_classes(target_pitches, tonic_pc=tonic_pc)
    importance = prepared.source_event.dur_tick
    if settings.preserve_melody and source_pitch == highest_pitch:
        importance += 10000
    if settings.preserve_bass and source_pitch == lowest_pitch:
        importance += 8500
    if source_pitch in (lowest_pitch, highest_pitch):
        importance += 1000
    if target_pitch % 12 in important_harmony_pcs:
        importance += 1800
    if pitch_class_counts.get(target_pitch % 12, 0) > 1 and source_pitch not in (lowest_pitch, highest_pitch):
        importance -= 1500
    if lowest_pitch < source_pitch < highest_pitch:
        importance -= 100
    return importance


def _important_harmony_pitch_classes(pitches: Sequence[int], *, tonic_pc: int) -> set[int]:
    pitch_classes = {pitch % 12 for pitch in pitches}
    if not pitch_classes:
        return set()

    leading_tone = (tonic_pc - 1) % 12
    best_root = next(iter(pitch_classes))
    best_third = (best_root + 4) % 12
    best_seventh = (best_root + 10) % 12
    best_score = -1
    quality_intervals = (
        (4, 7, 10),
        (3, 7, 10),
        (4, 7, 11),
        (3, 7, 10),
    )
    bass_pc = min(pitches) % 12
    for root in range(12):
        for third_interval, fifth_interval, seventh_interval in quality_intervals:
            third = (root + third_interval) % 12
            fifth = (root + fifth_interval) % 12
            seventh = (root + seventh_interval) % 12
            score = 0
            if root in pitch_classes:
                score += 2
            if third in pitch_classes:
                score += 6
            if fifth in pitch_classes:
                score += 1
            if seventh in pitch_classes:
                score += 4
            if bass_pc == root:
                score += 3
            if score > best_score:
                best_score = score
                best_root = root
                best_third = third
                best_seventh = seventh

    important = {
        pitch_class
        for pitch_class in (best_third, best_seventh, leading_tone, (best_root + 5) % 12)
        if pitch_class in pitch_classes
    }
    return important


def _record_dropped_source_event(
    source_event: Event,
    *,
    reason: str,
    dropped_notes: list[DroppedNoteDiagnostic],
    source_maps_by_id: dict[str, PianoToGuitarNoteMap],
) -> None:
    if source_event.pitch_midi is None:
        return
    dropped_notes.append(
        DroppedNoteDiagnostic(
            source_event_id=source_event.id,
            start_tick=source_event.start_tick,
            voice_id=source_event.voice_id,
            pitch_midi=source_event.pitch_midi,
            reason=reason,
        )
    )
    source_maps_by_id[source_event.id] = _dropped_note_map(source_event, reason_pitch=None)


def _record_dropped_note(
    prepared: _PreparedEvent,
    *,
    reason: str,
    dropped_notes: list[DroppedNoteDiagnostic],
    source_maps_by_id: dict[str, PianoToGuitarNoteMap],
) -> None:
    source_event = prepared.source_event
    if source_event.pitch_midi is None:
        return
    dropped_notes.append(
        DroppedNoteDiagnostic(
            source_event_id=source_event.id,
            start_tick=source_event.start_tick,
            voice_id=source_event.voice_id,
            pitch_midi=source_event.pitch_midi,
            reason=reason,
        )
    )
    source_maps_by_id[source_event.id] = _dropped_note_map(source_event, reason_pitch=prepared.target_event.pitch_midi)


def _find_hand_position_compromises(
    tabbed_events: Sequence[Event],
    source_events: Sequence[Event],
    *,
    settings: GuitarArrangementSettings,
) -> list[HandPositionCompromiseDiagnostic]:
    source_by_target_id = {
        _guitar_event_id(source_event.id): source_event
        for source_event in source_events
    }
    compromises: list[HandPositionCompromiseDiagnostic] = []
    for onset_tick in sorted({event.start_tick for event in tabbed_events}):
        onset_events = [
            event
            for event in tabbed_events
            if event.start_tick == onset_tick and event.fingering is not None and event.fingering.fret > 0
        ]
        if len(onset_events) < 2:
            continue
        frets = [event.fingering.fret for event in onset_events if event.fingering is not None]
        min_fret = min(frets)
        max_fret = max(frets)
        span = max_fret - min_fret
        if span <= settings.resolved_max_hand_span_frets:
            continue
        compromises.append(
            HandPositionCompromiseDiagnostic(
                onset_tick=onset_tick,
                source_event_ids=[
                    source_by_target_id[event.id].id
                    for event in onset_events
                    if event.id in source_by_target_id
                ],
                min_fret=min_fret,
                max_fret=max_fret,
                span_frets=span,
                reason="assigned fingering exceeds configured hand-span preference",
            )
        )
    return compromises


def _build_warnings(
    *,
    dropped_notes: Sequence[DroppedNoteDiagnostic],
    octave_shifts: Sequence[OctaveShiftDiagnostic],
    impossible_chords: Sequence[ImpossibleChordDiagnostic],
    hand_position_compromises: Sequence[HandPositionCompromiseDiagnostic],
) -> list[str]:
    warnings: list[str] = []
    if dropped_notes:
        warnings.append(f"dropped {len(dropped_notes)} note(s) while thinning piano material for guitar")
    if octave_shifts:
        warnings.append(f"octave-shifted {len(octave_shifts)} note(s) for guitar range or playability")
    if impossible_chords:
        warnings.append(f"rewrote {len(impossible_chords)} onset(s) that were not directly playable")
    if hand_position_compromises:
        warnings.append(f"kept {len(hand_position_compromises)} wide voicing(s) that exceed the hand-span preference")
    return warnings


def _to_guitar_event(source_event: Event, pitch_midi: int | None) -> Event:
    return replace(
        source_event,
        id=_guitar_event_id(source_event.id),
        pitch_midi=pitch_midi,
        fingering=None,
    )


def _guitar_event_id(source_event_id: str) -> str:
    return f"gtr-{source_event_id}"


def _renumber_voice_ids(events: Sequence[Event]) -> list[Event]:
    voice_ids = sorted({event.voice_id for event in events})
    voice_id_map = {
        voice_id: index
        for index, voice_id in enumerate(voice_ids)
    }
    return [
        replace(event, voice_id=voice_id_map[event.voice_id])
        for event in events
    ]


def _dropped_note_map(source_event: Event, *, reason_pitch: int | None) -> PianoToGuitarNoteMap:
    if source_event.pitch_midi is None:
        raise ValueError("dropped source map requires a pitched source event")
    target_pitch_midi = reason_pitch
    semitone_shift = 0 if target_pitch_midi is None else target_pitch_midi - source_event.pitch_midi
    return PianoToGuitarNoteMap(
        source_event_id=source_event.id,
        target_event_id=None,
        start_tick=source_event.start_tick,
        dur_tick=source_event.dur_tick,
        voice_id=source_event.voice_id,
        source_pitch_midi=source_event.pitch_midi,
        target_pitch_midi=None,
        semitone_shift=semitone_shift,
        dropped=True,
    )


def _has_fingering_candidate(
    pitch_midi: int,
    *,
    settings: GuitarArrangementSettings,
) -> bool:
    return any(
        0 <= pitch_midi - open_pitch <= settings.max_fret
        for open_pitch in settings.tuning
    )


def _tonic_pc_from_header(header: ScoreHeader) -> int:
    if not header.key_sig_map:
        return 0
    key_name = header.key_sig_map[min(header.key_sig_map)]
    normalized = key_name.strip()
    if normalized.endswith("m"):
        normalized = normalized[:-1]
    note_name = normalized[:2] if len(normalized) >= 2 and normalized[1] in {"#", "-"} else normalized[:1]
    return {
        "C": 0,
        "C#": 1,
        "D-": 1,
        "D": 2,
        "D#": 3,
        "E-": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G-": 6,
        "G": 7,
        "G#": 8,
        "A-": 8,
        "A": 9,
        "A#": 10,
        "B-": 10,
        "B": 11,
    }.get(note_name, 0)


def _copy_header(header: ScoreHeader) -> ScoreHeader:
    return ScoreHeader(
        tpq=header.tpq,
        key_sig_map=dict(header.key_sig_map),
        time_sig_map=dict(header.time_sig_map),
        tempo_map=dict(header.tempo_map),
        pickup_ticks=header.pickup_ticks,
    )
