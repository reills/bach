from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import music21

SUSPICIOUS_METERS = {"23/16", "33/32"}


@dataclass(frozen=True)
class MeterOverride:
    path: str
    movement_index: int
    time_signature: str
    start_measure: int = 0
    end_measure: int | None = None
    source: str = ""
    note: str = ""


@dataclass(frozen=True)
class MeterChange:
    movement_index: int
    start_measure: int
    end_measure: int
    before: str
    after: str
    reason: str
    confidence: float


@dataclass
class CleanReport:
    source_path: str
    output_path: str | None
    source_sha256: str
    output_sha256: str | None
    status: str
    movement_count: int
    changes: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    meter_distribution: dict[str, int]
    training_approved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MeasureData:
    effective_meter: str | None
    actual_quarter_length: float
    final_barline: bool


def load_meter_overrides(path: str | Path | None) -> list[MeterOverride]:
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    values = data.get("overrides", data) if isinstance(data, dict) else data
    if not isinstance(values, list):
        raise ValueError("meter overrides must be a list or contain an 'overrides' list")
    return [
        MeterOverride(
            path=str(item["path"]).replace("\\", "/"),
            movement_index=int(item["movement_index"]),
            time_signature=_normalize_meter(str(item["time_signature"])),
            start_measure=int(item.get("start_measure", 0)),
            end_measure=None if item.get("end_measure") is None else int(item["end_measure"]),
            source=str(item.get("source", "")),
            note=str(item.get("note", "")),
        )
        for item in values
    ]


def clean_musicxml_file(
    source_path: str | Path,
    output_path: str | Path | None,
    *,
    relative_path: str,
    overrides: Iterable[MeterOverride] = (),
    auto_repair: bool = True,
    max_auto_run: int = 2,
    dominant_support: float = 0.7,
) -> CleanReport:
    source_path = Path(source_path)
    destination = None if output_path is None else Path(output_path)
    source_sha256 = sha256_file(source_path)
    changes: list[MeterChange] = []
    issues: list[dict[str, Any]] = []
    try:
        score = music21.converter.parse(source_path)
        tree = ET.parse(source_path)
    except Exception as exc:
        return _error_report(source_path, destination, source_sha256, "parse_error", str(exc))

    root = tree.getroot()
    xml_parts = _children(root, "part")
    score_parts = list(score.parts)
    if not score_parts or not xml_parts:
        issues.append({"kind": "no_parts"})
    if len(score_parts) != len(xml_parts):
        issues.append(
            {
                "kind": "stream_part_count_diff",
                "music21_parts": len(score_parts),
                "xml_parts": len(xml_parts),
            }
        )
    part_data = [_measure_data(part) for part in score_parts]
    movement_ranges = _movement_ranges(part_data[0] if part_data else [])
    meter_distribution = Counter(
        measure.effective_meter
        for measures in part_data
        for measure in measures
        if measure.effective_meter is not None
    )

    normalized_path = relative_path.replace("\\", "/")
    path_overrides = [
        override
        for override in overrides
        if override.path.replace("\\", "/") == normalized_path
    ]
    overridden_movements: set[int] = set()
    for override in path_overrides:
        if not 0 <= override.movement_index < len(movement_ranges):
            issues.append(
                {
                    "kind": "invalid_override_movement",
                    "movement_index": override.movement_index,
                }
            )
            continue
        start, end = movement_ranges[override.movement_index]
        local_end = end - start if override.end_measure is None else override.end_measure
        absolute_start = start + override.start_measure
        absolute_end = min(end, start + local_end)
        if absolute_start >= absolute_end:
            issues.append(
                {
                    "kind": "invalid_override_range",
                    "movement_index": override.movement_index,
                }
            )
            continue
        before = _consensus_meter(part_data, absolute_start) or "UNKNOWN"
        _apply_meter_range(
            xml_parts,
            start_measure=absolute_start,
            end_measure=absolute_end,
            meter=override.time_signature,
            ensure_at_start=True,
        )
        override_mismatches = [
            {
                "measure": index - start,
                "actual_quarter_length": actual,
            }
            for index in range(absolute_start, absolute_end)
            if (
                (actual := _actual_measure_length(part_data, index)) > 0
                and not _close(actual, _meter_quarter_length(override.time_signature) or 0)
                and not (
                    index == start
                    or index == end - 1
                )
            )
        ]
        if override_mismatches:
            issues.append(
                {
                    "kind": "override_content_mismatch",
                    "movement_index": override.movement_index,
                    "time_signature": override.time_signature,
                    "measures": override_mismatches,
                }
            )
        changes.append(
            MeterChange(
                movement_index=override.movement_index,
                start_measure=override.start_measure,
                end_measure=local_end,
                before=before,
                after=override.time_signature,
                reason="reviewed_override",
                confidence=1.0,
            )
        )
        overridden_movements.add(override.movement_index)

    for movement_index, (start, end) in enumerate(movement_ranges):
        if movement_index in overridden_movements:
            continue
        movement_changes, movement_issues = _auto_meter_changes(
            part_data,
            movement_index=movement_index,
            start=start,
            end=end,
            max_auto_run=max_auto_run if auto_repair else 0,
            dominant_support=dominant_support,
        )
        issues.extend(movement_issues)
        for change in movement_changes:
            _apply_meter_range(
                xml_parts,
                start_measure=start + change.start_measure,
                end_measure=start + change.end_measure,
                meter=change.after,
                ensure_at_start=True,
            )
            changes.append(change)

    repaired_meters = {change.before for change in changes}
    unresolved_suspicious = (SUSPICIOUS_METERS & set(meter_distribution)) - repaired_meters
    for meter in sorted(unresolved_suspicious):
        issues.append(
            {
                "kind": "suspicious_meter",
                "meter": meter,
                "occurrences": meter_distribution[meter],
            }
        )

    blocking_kinds = {
        "parse_error",
        "no_parts",
        "ambiguous_meter",
        "content_meter_mismatch",
        "equivalent_meter_change",
        "invalid_override_movement",
        "invalid_override_range",
        "override_content_mismatch",
        "suspicious_meter",
    }
    unresolved = [issue for issue in issues if issue["kind"] in blocking_kinds]
    output_sha256: str | None = None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if changes:
            _write_validated_tree(tree, destination)
        else:
            destination.write_bytes(source_path.read_bytes())
        output_sha256 = sha256_file(destination)

    status = "repaired" if changes else "clean"
    if unresolved:
        status = "review_required"
    return CleanReport(
        source_path=str(source_path),
        output_path=None if destination is None else str(destination),
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        status=status,
        movement_count=len(movement_ranges),
        changes=[asdict(change) for change in changes],
        issues=issues,
        meter_distribution=dict(sorted(meter_distribution.items())),
        training_approved=not unresolved,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _auto_meter_changes(
    part_data: list[list[_MeasureData]],
    *,
    movement_index: int,
    start: int,
    end: int,
    max_auto_run: int,
    dominant_support: float,
) -> tuple[list[MeterChange], list[dict[str, Any]]]:
    changes: list[MeterChange] = []
    issues: list[dict[str, Any]] = []
    length = end - start
    consensus = [_consensus_meter(part_data, index) for index in range(start, end)]
    internal = consensus[1:-1] if length > 2 else consensus
    usable = [meter for meter in internal if meter is not None]
    if not usable:
        return changes, [{"kind": "missing_meter", "movement_index": movement_index}]
    counts = Counter(usable)
    dominant, dominant_count = counts.most_common(1)[0]
    support = dominant_count / len(usable)
    if support < dominant_support:
        return changes, [
            {
                "kind": "ambiguous_meter",
                "movement_index": movement_index,
                "distribution": dict(counts),
                "dominant_support": round(support, 4),
            }
        ]
    dominant_length = _meter_quarter_length(dominant)
    if dominant_length is None:
        return changes, [
            {
                "kind": "unsupported_meter",
                "movement_index": movement_index,
                "meter": dominant,
            }
        ]

    index = 0
    while index < length:
        if consensus[index] in {None, dominant}:
            index += 1
            continue
        run_start = index
        before = consensus[index]
        while index < length and consensus[index] == before:
            index += 1
        run_end = index
        actual_lengths = [
            _actual_measure_length(part_data, absolute)
            for absolute in range(start + run_start, start + run_end)
        ]
        actual_lengths = [value for value in actual_lengths if value > 0]
        before_length = _meter_quarter_length(before)
        matches_dominant = bool(actual_lengths) and all(
            _close(value, dominant_length) for value in actual_lengths
        )
        mismatches_declared = (
            before_length is not None
            and bool(actual_lengths)
            and any(not _close(value, before_length) for value in actual_lengths)
        )
        surrounded = (
            run_start > 0
            and run_end < length
            and consensus[run_start - 1] == dominant
            and consensus[run_end] == dominant
        )
        if (
            max_auto_run > 0
            and run_end - run_start <= max_auto_run
            and matches_dominant
            and mismatches_declared
        ):
            changes.append(
                MeterChange(
                    movement_index=movement_index,
                    start_measure=run_start,
                    end_measure=run_end,
                    before=str(before),
                    after=dominant,
                    reason="isolated_meter_conflicts_with_bar_duration",
                    confidence=0.98 if mismatches_declared else 0.92,
                )
            )
        elif (
            max_auto_run > 0
            and run_end - run_start <= max_auto_run
            and surrounded
            and before_length is not None
            and _close(before_length, dominant_length)
        ):
            issues.append(
                {
                    "kind": "equivalent_meter_change",
                    "movement_index": movement_index,
                    "start_measure": run_start,
                    "end_measure": run_end,
                    "declared_meter": before,
                    "surrounding_meter": dominant,
                    "actual_quarter_lengths": actual_lengths,
                }
            )
        elif mismatches_declared:
            issues.append(
                {
                    "kind": "content_meter_mismatch",
                    "movement_index": movement_index,
                    "start_measure": run_start,
                    "end_measure": run_end,
                    "declared_meter": before,
                    "dominant_meter": dominant,
                    "actual_quarter_lengths": actual_lengths,
                }
            )
    return changes, issues


def _measure_data(part: music21.stream.Part) -> list[_MeasureData]:
    current_meter: str | None = None
    result: list[_MeasureData] = []
    for measure in part.getElementsByClass(music21.stream.Measure):
        signatures = [
            signature
            for signature in measure.getTimeSignatures(returnDefault=False)
            if abs(float(signature.offset)) < 1e-6
        ]
        if signatures:
            current_meter = _normalize_meter(signatures[0].ratioString)
        result.append(
            _MeasureData(
                effective_meter=current_meter,
                actual_quarter_length=round(float(measure.duration.quarterLength), 6),
                final_barline=(
                    measure.rightBarline is not None
                    and measure.rightBarline.type == "final"
                ),
            )
        )
    return result


def _movement_ranges(measures: list[_MeasureData]) -> list[tuple[int, int]]:
    if not measures:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index, measure in enumerate(measures):
        if measure.final_barline or index == len(measures) - 1:
            ranges.append((start, index + 1))
            start = index + 1
    return ranges


def _consensus_meter(part_data: list[list[_MeasureData]], index: int) -> str | None:
    values = [
        measures[index].effective_meter
        for measures in part_data
        if index < len(measures) and measures[index].effective_meter is not None
    ]
    return None if not values else Counter(values).most_common(1)[0][0]


def _actual_measure_length(part_data: list[list[_MeasureData]], index: int) -> float:
    values = [
        measures[index].actual_quarter_length
        for measures in part_data
        if index < len(measures) and measures[index].actual_quarter_length > 0
    ]
    return 0.0 if not values else float(median(values))


def _apply_meter_range(
    xml_parts: list[ET.Element],
    *,
    start_measure: int,
    end_measure: int,
    meter: str,
    ensure_at_start: bool,
) -> None:
    for part in xml_parts:
        measures = _children(part, "measure")
        if start_measure >= len(measures):
            continue
        for index in range(start_measure, min(end_measure, len(measures))):
            for element in _time_elements(measures[index]):
                _set_time_element(element, meter)
        if ensure_at_start and not _time_elements(measures[start_measure]):
            _insert_time_element(measures[start_measure], meter)


def _time_elements(measure: ET.Element) -> list[ET.Element]:
    return [
        child
        for attributes in _children(measure, "attributes")
        for child in list(attributes)
        if _local_name(child.tag) == "time"
    ]


def _set_time_element(element: ET.Element, meter: str) -> None:
    beats, beat_type = meter.split("/", 1)
    namespace = _namespace(element.tag)
    for child in list(element):
        if _local_name(child.tag) in {"beats", "beat-type", "senza-misura"}:
            element.remove(child)
    element.attrib.pop("symbol", None)
    ET.SubElement(element, f"{namespace}beats").text = beats
    ET.SubElement(element, f"{namespace}beat-type").text = beat_type


def _insert_time_element(measure: ET.Element, meter: str) -> None:
    attributes = next(iter(_children(measure, "attributes")), None)
    namespace = _namespace(measure.tag)
    if attributes is None:
        attributes = ET.Element(f"{namespace}attributes")
        measure.insert(0, attributes)
    time_element = ET.Element(f"{namespace}time")
    insert_at = len(attributes)
    for index, child in enumerate(list(attributes)):
        if _local_name(child.tag) in {"staves", "clef", "staff-details", "transpose"}:
            insert_at = index
            break
    attributes.insert(insert_at, time_element)
    _set_time_element(time_element, meter)


def _write_validated_tree(tree: ET.ElementTree, destination: Path) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        music21.converter.parse(temporary, format="musicxml")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _meter_quarter_length(meter: str | None) -> float | None:
    if meter is None or "/" not in meter:
        return None
    beats, beat_type = meter.split("/", 1)
    if not beats.isdigit() or not beat_type.isdigit() or int(beat_type) <= 0:
        return None
    return int(beats) * 4.0 / int(beat_type)


def _normalize_meter(value: str) -> str:
    return value.strip().replace(" ", "")


def _close(left: float, right: float, tolerance: float = 1.0 / 32.0) -> bool:
    return abs(left - right) <= tolerance


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str:
    return tag[: tag.index("}") + 1] if tag.startswith("{") else ""


def _error_report(
    source_path: Path,
    destination: Path | None,
    source_sha256: str,
    kind: str,
    message: str,
) -> CleanReport:
    return CleanReport(
        source_path=str(source_path),
        output_path=None if destination is None else str(destination),
        source_sha256=source_sha256,
        output_sha256=None,
        status=kind,
        movement_count=0,
        changes=[],
        issues=[{"kind": kind, "message": message}],
        meter_distribution={},
        training_approved=False,
    )
