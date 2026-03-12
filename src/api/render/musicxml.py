from dataclasses import dataclass
import xml.etree.ElementTree as ET

from src.api.canonical.types import CanonicalScore, Event, Measure

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


def _append_measure_attributes(measure_el: ET.Element, score: CanonicalScore, measure: Measure) -> None:
    has_time_sig = measure.start_tick in score.header.time_sig_map
    has_first_measure = measure.index == 0
    if not has_first_measure and not has_time_sig:
        return

    attributes_el = ET.SubElement(measure_el, "attributes")
    ET.SubElement(attributes_el, "divisions").text = str(score.header.tpq)

    if has_time_sig:
        beats, beat_type = score.header.time_sig_map[measure.start_tick].split("/", 1)
        time_el = ET.SubElement(attributes_el, "time")
        ET.SubElement(time_el, "beats").text = beats
        ET.SubElement(time_el, "beat-type").text = beat_type


def _append_measure_content(measure_el: ET.Element, score: CanonicalScore, measure: Measure) -> None:
    part = score.parts[0]
    slices_by_voice: dict[int, list[_RenderedSlice]] = {}

    for event in part.events:
        slice_ = _slice_for_measure(event, measure)
        if slice_ is None:
            continue
        slices_by_voice.setdefault(event.voice_id, []).append(slice_)

    for voice_index, voice_id in enumerate(sorted(slices_by_voice)):
        if voice_index > 0:
            backup_el = ET.SubElement(measure_el, "backup")
            ET.SubElement(backup_el, "duration").text = str(measure.length_ticks)

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
                    string_count=len(part.info.tuning),
                )
            _append_note(measure_el, slice_, string_count=len(part.info.tuning))
            cursor = slice_.start_tick + slice_.dur_tick

        if cursor < measure.end_tick:
            _append_note(
                measure_el,
                _RenderedSlice(
                    event=None,
                    voice_id=voice_id,
                    start_tick=cursor,
                    dur_tick=measure.end_tick - cursor,
                ),
                string_count=len(part.info.tuning),
            )


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


def _append_note(measure_el: ET.Element, slice_: _RenderedSlice, string_count: int) -> None:
    note_el = ET.SubElement(measure_el, "note")

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
