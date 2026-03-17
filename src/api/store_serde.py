"""JSON serialization and deserialization for CanonicalScore."""
from __future__ import annotations

import dataclasses
import json

from src.api.canonical.types import (
    CanonicalScore,
    Event,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
)


def score_to_json(score: CanonicalScore) -> str:
    return json.dumps(dataclasses.asdict(score))


def score_from_json(data: str) -> CanonicalScore:
    return _score_from_dict(json.loads(data))


def _score_from_dict(d: dict) -> CanonicalScore:
    return CanonicalScore(
        header=_header_from_dict(d["header"]),
        measures=[_measure_from_dict(m) for m in d["measures"]],
        parts=[_part_from_dict(p) for p in d["parts"]],
    )


def _header_from_dict(d: dict) -> ScoreHeader:
    # JSON serializes integer keys as strings; restore them here.
    return ScoreHeader(
        tpq=d["tpq"],
        key_sig_map={int(k): v for k, v in d.get("key_sig_map", {}).items()},
        time_sig_map={int(k): v for k, v in d.get("time_sig_map", {}).items()},
        tempo_map={int(k): v for k, v in d.get("tempo_map", {}).items()},
        pickup_ticks=d.get("pickup_ticks", 0),
    )


def _measure_from_dict(d: dict) -> Measure:
    return Measure(
        id=d["id"],
        index=d["index"],
        start_tick=d["start_tick"],
        length_ticks=d["length_ticks"],
    )


def _part_from_dict(d: dict) -> Part:
    info = d["info"]
    return Part(
        info=PartInfo(
            id=info["id"],
            instrument=info["instrument"],
            tuning=info.get("tuning", []),
            capo=info.get("capo", 0),
            midi_program=info.get("midi_program"),
        ),
        events=[_event_from_dict(e) for e in d.get("events", [])],
    )


def _event_from_dict(d: dict) -> Event:
    f = d.get("fingering")
    fingering = GuitarFingering(string_index=f["string_index"], fret=f["fret"]) if f else None
    return Event(
        id=d["id"],
        start_tick=d["start_tick"],
        dur_tick=d["dur_tick"],
        voice_id=d["voice_id"],
        pitch_midi=d["pitch_midi"],
        velocity=d.get("velocity"),
        fingering=fingering,
    )
