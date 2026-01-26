from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Event:
    offset: float
    duration: float
    midi: int


def _quantize(value: float, places: int = 6) -> float:
    return round(float(value), places)


def _part_name(part) -> str:
    return part.partName or part.id or ""


def _voice_label(voice, index: int) -> str:
    label = voice.id or ""
    return f"voice{index + 1}({label})" if label else f"voice{index + 1}"


def _extract_events(score) -> dict[tuple[int, str], Counter[Event]]:
    from music21 import chord, stream

    events: dict[tuple[int, str], Counter[Event]] = {}

    for p_index, part in enumerate(score.parts):
        voice_ids = {}
        for v_index, voice in enumerate(part.recurse().getElementsByClass(stream.Voice)):
            voice_ids[id(voice)] = _voice_label(voice, v_index)

        for element in part.recurse().notes:
            voice = element.getContextByClass(stream.Voice)
            voice_key = voice_ids.get(id(voice)) if voice is not None else "default"
            if voice_key is None:
                voice_key = "default"

            offset = _quantize(element.getOffsetInHierarchy(part))
            duration = _quantize(element.duration.quarterLength)

            if isinstance(element, chord.Chord):
                for pitch in element.pitches:
                    ev = Event(offset=offset, duration=duration, midi=int(pitch.midi))
                    events.setdefault((p_index, voice_key), Counter())[ev] += 1
            else:
                ev = Event(offset=offset, duration=duration, midi=int(element.pitch.midi))
                events.setdefault((p_index, voice_key), Counter())[ev] += 1

    return events


def _extract_measure_summary(score) -> list[dict[str, object]]:
    from music21 import meter, stream

    summary: list[dict[str, object]] = []
    for p_index, part in enumerate(score.parts):
        measures = part.getElementsByClass(stream.Measure)
        time_sigs = []
        for ts in part.recurse().getElementsByClass(meter.TimeSignature):
            time_sigs.append((_quantize(ts.getOffsetInHierarchy(part)), ts.ratioString))

        summary.append({
            "name": _part_name(part),
            "measures": len(measures),
            "time_sigs": time_sigs,
            "duration": _quantize(part.highestTime),
        })
    return summary


def _diff_counts(a: Counter[Event], b: Counter[Event]) -> tuple[int, int, Counter[Event], Counter[Event]]:
    missing = a - b
    extra = b - a
    return sum(missing.values()), sum(extra.values()), missing, extra


def _format_examples(counter: Counter[Event], limit: int = 5) -> Iterable[str]:
    for event in sorted(counter.keys(), key=lambda e: (e.offset, e.duration, e.midi))[:limit]:
        yield f"(offset={event.offset}, dur={event.duration}, midi={event.midi})"


def main() -> int:
    try:
        from music21 import converter
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"music21 import failed: {exc}")
        return 1

    root = Path(__file__).resolve().parents[1]
    xml_path = root / "data/tobis_xml/instrumental-works/Art of fugue/BWV_1080_05/BWV_1080_05.xml"
    orig_path = root / "data/tobis_xml/instrumental-works/Art of fugue/artfugue-005.krn"

    if not xml_path.exists():
        print(f"missing xml: {xml_path}")
        return 1
    if not orig_path.exists():
        print(f"missing original: {orig_path}")
        return 1

    print(f"xml path: {xml_path}")
    print(f"original path: {orig_path}")

    orig_score = converter.parse(orig_path)
    xml_score = converter.parse(xml_path)

    orig_summary = _extract_measure_summary(orig_score)
    xml_summary = _extract_measure_summary(xml_score)
    orig_names = [entry["name"] for entry in orig_summary]
    xml_names = [entry["name"] for entry in xml_summary]

    print("\nMeasure/time signature summary:")
    max_parts = max(len(orig_summary), len(xml_summary))
    for index in range(max_parts):
        part_label = f"part{index + 1}"
        o = orig_summary[index] if index < len(orig_summary) else None
        x = xml_summary[index] if index < len(xml_summary) else None
        o_name = o["name"] if o else ""
        x_name = x["name"] if x else ""
        name_info = f"(orig: {o_name or '-'}, xml: {x_name or '-'})"
        if o is None or x is None:
            print(f"- {part_label} {name_info}: missing in one score")
            continue
        print(
            f"- {part_label} {name_info}: measures {o['measures']} vs {x['measures']}, "
            f"duration {o['duration']} vs {x['duration']}"
        )
        o_ts = o["time_sigs"]
        x_ts = x["time_sigs"]
        if o_ts != x_ts:
            print(f"  time sigs: {len(o_ts)} vs {len(x_ts)}")
            print("  example orig:", o_ts[:3])
            print("  example xml :", x_ts[:3])

    orig_events = _extract_events(orig_score)
    xml_events = _extract_events(xml_score)

    print("\nNote event comparison (pitch/offset/duration):")
    for key in sorted(set(orig_events) | set(xml_events)):
        part_index, voice_key = key
        part_label = f"part{part_index + 1}"
        o_name = orig_names[part_index] if part_index < len(orig_names) else ""
        x_name = xml_names[part_index] if part_index < len(xml_names) else ""
        name_info = f"(orig: {o_name or '-'}, xml: {x_name or '-'})"
        o_counter = orig_events.get(key, Counter())
        x_counter = xml_events.get(key, Counter())
        missing_count, extra_count, missing, extra = _diff_counts(o_counter, x_counter)
        if missing_count == 0 and extra_count == 0:
            print(f"- {part_label} {name_info} / {voice_key}: events match ({sum(o_counter.values())})")
            continue
        print(
            f"- {part_label} {name_info} / {voice_key}: orig {sum(o_counter.values())}, "
            f"xml {sum(x_counter.values())}, missing {missing_count}, extra {extra_count}"
        )
        if missing_count:
            print("  missing examples:", "; ".join(_format_examples(missing)))
        if extra_count:
            print("  extra examples:", "; ".join(_format_examples(extra)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
