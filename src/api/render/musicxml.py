from dataclasses import dataclass
import xml.etree.ElementTree as ET

from src.api.canonical.types import CanonicalScore, Event, Measure, Part

XML_NS = "http://www.w3.org/XML/1998/namespace"

_PITCH_CLASS_TO_XML = {
    0: ("C", None),
    1: ("C", 1),
    2: ("D", None),
    3: ("D", 1),
    4: ("E", None),
    5: ("F", None),
    6: ("F", 1),
    7: ("G", None),
    8: ("G", 1),
    9: ("A", None),
    10: ("A", 1),
    11: ("B", None),
}

# MIDI 60 is middle C (C4) — piano staff split threshold
_PIANO_SPLIT_MIDI = 60


@dataclass(frozen=True)
class _RenderedSlice:
    event: Event | None
    voice_id: int
    start_tick: int
    dur_tick: int
    tie_start: bool = False
    tie_stop: bool = False


def canonical_score_to_musicxml(score: CanonicalScore) -> str:
    if len(score.parts) != 1:
        raise ValueError("musicxml exporter supports exactly one part")

    part = score.parts[0]
    root = ET.Element("score-partwise", version="4.0")

    part_list_el = ET.SubElement(root, "part-list")
    score_part_el = ET.SubElement(part_list_el, "score-part", id=part.info.id)
    ET.SubElement(score_part_el, "part-name").text = part.info.instrument

    part_el = ET.SubElement(root, "part", id=part.info.id)
    for measure in score.measures:
        measure_el = ET.SubElement(part_el, "measure", number=str(measure.index + 1))
        measure_el.set(f"{{{XML_NS}}}id", measure.id)
        _append_measure_attributes(measure_el, score, measure)
        _append_measure_content(measure_el, score, measure)

    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def score_to_musicxml(score: CanonicalScore) -> str:
    return canonical_score_to_musicxml(score)


def _is_piano(part: Part) -> bool:
    return part.info.instrument == "piano"


def _append_measure_attributes(measure_el: ET.Element, score: CanonicalScore, measure: Measure) -> None:
    has_time_sig = measure.start_tick in score.header.time_sig_map
    has_first_measure = measure.index == 0
    if not has_first_measure and not has_time_sig:
        return

    part = score.parts[0]
    attributes_el = ET.SubElement(measure_el, "attributes")
    ET.SubElement(attributes_el, "divisions").text = str(score.header.tpq)

    if has_time_sig:
        beats, beat_type = score.header.time_sig_map[measure.start_tick].split("/", 1)
        time_el = ET.SubElement(attributes_el, "time")
        ET.SubElement(time_el, "beats").text = beats
        ET.SubElement(time_el, "beat-type").text = beat_type

    if has_first_measure:
        if _is_piano(part):
            _append_piano_attributes(attributes_el)
        else:
            _append_guitar_attributes(attributes_el, part)


def _append_guitar_attributes(attributes_el: ET.Element, part: "Part") -> None:
    """Emit guitar treble-clef (8vb) and staff-details with tuning."""
    clef_el = ET.SubElement(attributes_el, "clef")
    ET.SubElement(clef_el, "sign").text = "G"
    ET.SubElement(clef_el, "line").text = "2"
    ET.SubElement(clef_el, "clef-octave-change").text = "-1"

    tuning = part.info.tuning
    if not tuning:
        return

    staff_details_el = ET.SubElement(attributes_el, "staff-details")
    ET.SubElement(staff_details_el, "staff-lines").text = str(len(tuning))

    string_count = len(tuning)
    for line_num in range(1, string_count + 1):
        # line 1 = highest string = tuning[-1]; line N = lowest = tuning[0]
        midi = tuning[string_count - line_num]
        step, alter, octave = _pitch_to_musicxml(midi)
        tuning_el = ET.SubElement(staff_details_el, "staff-tuning", line=str(line_num))
        ET.SubElement(tuning_el, "tuning-step").text = step
        if alter is not None:
            ET.SubElement(tuning_el, "tuning-alter").text = str(alter)
        ET.SubElement(tuning_el, "tuning-octave").text = str(octave)

    if part.info.capo > 0:
        ET.SubElement(staff_details_el, "capo").text = str(part.info.capo)


def _append_piano_attributes(attributes_el: ET.Element) -> None:
    """Emit grand-staff attributes: staves=2 and treble+bass clefs."""
    ET.SubElement(attributes_el, "staves").text = "2"

    clef1_el = ET.SubElement(attributes_el, "clef", number="1")
    ET.SubElement(clef1_el, "sign").text = "G"
    ET.SubElement(clef1_el, "line").text = "2"

    clef2_el = ET.SubElement(attributes_el, "clef", number="2")
    ET.SubElement(clef2_el, "sign").text = "F"
    ET.SubElement(clef2_el, "line").text = "4"


def _append_measure_content(measure_el: ET.Element, score: CanonicalScore, measure: Measure) -> None:
    part = score.parts[0]
    if _is_piano(part):
        _append_measure_content_piano(measure_el, part, measure)
    else:
        _append_measure_content_guitar(measure_el, part, measure)


def _append_measure_content_guitar(measure_el: ET.Element, part: "Part", measure: Measure) -> None:
    slices_by_voice: dict[int, list[_RenderedSlice]] = {}

    for event in part.events:
        slice_ = _slice_for_measure(event, measure)
        if slice_ is None:
            continue
        slices_by_voice.setdefault(event.voice_id, []).append(slice_)

    string_count = len(part.info.tuning)
    for voice_index, voice_id in enumerate(sorted(slices_by_voice)):
        if voice_index > 0:
            backup_el = ET.SubElement(measure_el, "backup")
            ET.SubElement(backup_el, "duration").text = str(measure.length_ticks)

        last_pitched_onset: int | None = None
        cursor = measure.start_tick
        for slice_ in slices_by_voice[voice_id]:
            if slice_.start_tick > cursor:
                _append_note(
                    measure_el,
                    _RenderedSlice(
                        event=None,
                        voice_id=voice_id,
                        start_tick=cursor,
                        dur_tick=slice_.start_tick - cursor,
                    ),
                    string_count=string_count,
                )
            is_chord = (
                slice_.event is not None
                and slice_.event.pitch_midi is not None
                and last_pitched_onset == slice_.start_tick
            )
            _append_note(measure_el, slice_, string_count=string_count, is_chord=is_chord)
            if slice_.event is not None and slice_.event.pitch_midi is not None:
                last_pitched_onset = slice_.start_tick
            cursor = max(cursor, slice_.start_tick + slice_.dur_tick)

        if cursor < measure.end_tick:
            _append_note(
                measure_el,
                _RenderedSlice(
                    event=None,
                    voice_id=voice_id,
                    start_tick=cursor,
                    dur_tick=measure.end_tick - cursor,
                ),
                string_count=string_count,
            )


def _append_measure_content_piano(measure_el: ET.Element, part: "Part", measure: Measure) -> None:
    """Emit piano grand-staff content with per-note <staff> tags."""
    slices_by_voice: dict[int, list[_RenderedSlice]] = {}

    for event in part.events:
        slice_ = _slice_for_measure(event, measure)
        if slice_ is None:
            continue
        slices_by_voice.setdefault(event.voice_id, []).append(slice_)

    # Compute staff assignment per (voice_id, onset): lowest pitch >= 60 → staff 1 else staff 2
    staff_for_onset: dict[tuple[int, int], int] = {}
    for voice_id, slices in slices_by_voice.items():
        by_onset: dict[int, list[_RenderedSlice]] = {}
        for s in slices:
            by_onset.setdefault(s.start_tick, []).append(s)
        for onset, group in sorted(by_onset.items()):
            pitches = [
                s.event.pitch_midi
                for s in group
                if s.event is not None and s.event.pitch_midi is not None
            ]
            if pitches:
                staff_for_onset[(voice_id, onset)] = 1 if min(pitches) >= _PIANO_SPLIT_MIDI else 2

    for voice_index, voice_id in enumerate(sorted(slices_by_voice)):
        if voice_index > 0:
            backup_el = ET.SubElement(measure_el, "backup")
            ET.SubElement(backup_el, "duration").text = str(measure.length_ticks)

        current_staff = _initial_piano_staff_for_voice(voice_id, slices_by_voice[voice_id], staff_for_onset)
        last_pitched_onset: int | None = None
        cursor = measure.start_tick
        for slice_ in slices_by_voice[voice_id]:
            if slice_.start_tick > cursor:
                _append_note(
                    measure_el,
                    _RenderedSlice(
                        event=None,
                        voice_id=voice_id,
                        start_tick=cursor,
                        dur_tick=slice_.start_tick - cursor,
                    ),
                    string_count=0,
                    staff=current_staff,
                )
            staff = staff_for_onset.get((voice_id, slice_.start_tick), current_staff)
            if slice_.event is not None and slice_.event.pitch_midi is not None:
                current_staff = staff
            is_chord = (
                slice_.event is not None
                and slice_.event.pitch_midi is not None
                and last_pitched_onset == slice_.start_tick
            )
            _append_note(measure_el, slice_, string_count=0, staff=staff, is_chord=is_chord)
            if slice_.event is not None and slice_.event.pitch_midi is not None:
                last_pitched_onset = slice_.start_tick
            cursor = max(cursor, slice_.start_tick + slice_.dur_tick)

        if cursor < measure.end_tick:
            _append_note(
                measure_el,
                _RenderedSlice(
                    event=None,
                    voice_id=voice_id,
                    start_tick=cursor,
                    dur_tick=measure.end_tick - cursor,
                ),
                string_count=0,
                staff=current_staff,
            )


def _initial_piano_staff_for_voice(
    voice_id: int,
    slices: list[_RenderedSlice],
    staff_for_onset: dict[tuple[int, int], int],
) -> int:
    for slice_ in slices:
        if slice_.event is None or slice_.event.pitch_midi is None:
            continue
        return staff_for_onset.get((voice_id, slice_.start_tick), 1)
    return 1


def _slice_for_measure(event: Event, measure: Measure) -> _RenderedSlice | None:
    start_tick = max(event.start_tick, measure.start_tick)
    end_tick = min(event.end_tick, measure.end_tick)
    if start_tick >= end_tick:
        return None

    return _RenderedSlice(
        event=event,
        voice_id=event.voice_id,
        start_tick=start_tick,
        dur_tick=end_tick - start_tick,
        tie_start=event.pitch_midi is not None and end_tick < event.end_tick,
        tie_stop=event.pitch_midi is not None and start_tick > event.start_tick,
    )


def _append_note(
    measure_el: ET.Element,
    slice_: _RenderedSlice,
    string_count: int,
    staff: int | None = None,
    is_chord: bool = False,
) -> None:
    note_el = ET.SubElement(measure_el, "note")

    if is_chord:
        ET.SubElement(note_el, "chord")

    if slice_.event is None or slice_.event.pitch_midi is None:
        ET.SubElement(note_el, "rest")
    else:
        note_el.set(f"{{{XML_NS}}}id", slice_.event.id)
        pitch_el = ET.SubElement(note_el, "pitch")
        step, alter, octave = _pitch_to_musicxml(slice_.event.pitch_midi)
        ET.SubElement(pitch_el, "step").text = step
        if alter is not None:
            ET.SubElement(pitch_el, "alter").text = str(alter)
        ET.SubElement(pitch_el, "octave").text = str(octave)

    ET.SubElement(note_el, "duration").text = str(slice_.dur_tick)
    ET.SubElement(note_el, "voice").text = str(slice_.voice_id + 1)

    if staff is not None:
        ET.SubElement(note_el, "staff").text = str(staff)

    if slice_.tie_stop or slice_.tie_start or (
        slice_.event is not None and slice_.event.pitch_midi is not None and slice_.event.fingering is not None
    ):
        notations_el = ET.SubElement(note_el, "notations")
        if slice_.tie_stop:
            ET.SubElement(note_el, "tie", type="stop")
            ET.SubElement(notations_el, "tied", type="stop")
        if slice_.tie_start:
            ET.SubElement(note_el, "tie", type="start")
            ET.SubElement(notations_el, "tied", type="start")
        if slice_.event is not None and slice_.event.pitch_midi is not None and slice_.event.fingering is not None:
            technical_el = ET.SubElement(notations_el, "technical")
            ET.SubElement(technical_el, "string").text = str(_musicxml_string_number(slice_.event, string_count))
            ET.SubElement(technical_el, "fret").text = str(slice_.event.fingering.fret)


def _musicxml_string_number(event: Event, string_count: int) -> int:
    if event.fingering is None:
        raise ValueError("musicxml string conversion requires fingering data")
    if string_count <= 0:
        raise ValueError("musicxml exporter requires part tuning when fingering data is present")
    string_number = string_count - event.fingering.string_index
    if not 1 <= string_number <= string_count:
        raise ValueError(f"fingering string_index {event.fingering.string_index} out of range for tuning size {string_count}")
    return string_number


def _pitch_to_musicxml(midi_pitch: int) -> tuple[str, int | None, int]:
    step, alter = _PITCH_CLASS_TO_XML[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1
    return step, alter, octave
