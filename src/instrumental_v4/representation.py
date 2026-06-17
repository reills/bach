from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.instrumental_v3.representation import (
    FEATURE_SPECS as V3_FEATURE_SPECS,
    FIELD_NAMES as V3_FIELD_NAMES,
    InstrumentalV3Piece,
    SliceEvent,
    STATE_NOTE,
    STATE_HOLD,
)

PLAN_FIELD_NAMES = [
    "plan_phrase_pos",
    "plan_cadence_zone",
    "plan_bass_pc",
    "plan_top_pc",
    "plan_bass_oct",
    "plan_top_oct",
    "plan_v0_density",
    "plan_v1_density",
    "plan_mean_vertical",
    "plan_final_interval_class",
]

PLAN_FEATURE_SPECS: dict[str, int] = {
    "plan_phrase_pos": 8,
    "plan_cadence_zone": 2,
    "plan_bass_pc": 13,
    "plan_top_pc": 13,
    "plan_bass_oct": 11,
    "plan_top_oct": 11,
    "plan_v0_density": 17,
    "plan_v1_density": 17,
    "plan_mean_vertical": 50,
    "plan_final_interval_class": 13,
}

V4_FIELD_NAMES = list(V3_FIELD_NAMES) + PLAN_FIELD_NAMES
V4_FEATURE_SPECS = {**V3_FEATURE_SPECS, **PLAN_FEATURE_SPECS}


@dataclass(frozen=True)
class MeasurePlan:
    values: list[int]

    def field(self, name: str) -> int:
        return self.values[PLAN_FIELD_NAMES.index(name)]


@dataclass(frozen=True)
class V4Piece:
    piece_id: str
    source_path: str
    tpq: int
    grid_ticks: int
    time_signature: str
    key: str | None
    key_pc: int
    mode: int
    bar_len_ticks: int
    steps_per_bar: int
    plans: list[MeasurePlan]
    rows: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["plans"] = [plan.values for plan in self.plans]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "V4Piece":
        return cls(
            piece_id=str(data["piece_id"]),
            source_path=str(data["source_path"]),
            tpq=int(data["tpq"]),
            grid_ticks=int(data["grid_ticks"]),
            time_signature=str(data["time_signature"]),
            key=data.get("key"),
            key_pc=int(data["key_pc"]),
            mode=int(data["mode"]),
            bar_len_ticks=int(data["bar_len_ticks"]),
            steps_per_bar=int(data["steps_per_bar"]),
            plans=[MeasurePlan([int(v) for v in row]) for row in data["plans"]],
            rows=[[int(v) for v in row] for row in data["rows"]],
        )


def build_v4_piece(piece: InstrumentalV3Piece) -> V4Piece:
    v3_rows = [slice_.values[:] for slice_ in piece.slices]
    plans = _plans_for_piece(piece, v3_rows)
    rows: list[list[int]] = []
    for idx, row in enumerate(v3_rows):
        bar = min(len(plans) - 1, idx // piece.steps_per_bar)
        rows.append(row + plans[bar].values)
    return V4Piece(
        piece_id=piece.piece_id,
        source_path=piece.source_path,
        tpq=piece.tpq,
        grid_ticks=piece.grid_ticks,
        time_signature=piece.time_signature,
        key=piece.key,
        key_pc=piece.key_pc,
        mode=piece.mode,
        bar_len_ticks=piece.bar_len_ticks,
        steps_per_bar=piece.steps_per_bar,
        plans=plans,
        rows=rows,
    )


def _plans_for_piece(piece: InstrumentalV3Piece, rows: list[list[int]]) -> list[MeasurePlan]:
    if not rows:
        return []
    bar_count = max(1, (len(rows) + piece.steps_per_bar - 1) // piece.steps_per_bar)
    plans = []
    for bar in range(bar_count):
        measure_rows = rows[bar * piece.steps_per_bar : (bar + 1) * piece.steps_per_bar]
        if not measure_rows:
            continue
        plans.append(_measure_plan(piece, bar, measure_rows))
    return plans


def _measure_plan(piece: InstrumentalV3Piece, bar: int, rows: list[list[int]]) -> MeasurePlan:
    def col(name: str) -> int:
        return V3_FIELD_NAMES.index(name)

    phrase_pos = bar % PLAN_FEATURE_SPECS["plan_phrase_pos"]
    cadence_zone = 1 if phrase_pos in {6, 7} else 0
    active_by_voice: list[list[int]] = [[], []]
    note_counts = [0, 0]
    verticals = []
    final_low = None
    final_high = None
    for row in rows:
        low = None
        high = None
        for voice in range(2):
            state = row[col(f"v{voice}_state")]
            pitch = row[col(f"v{voice}_pitch")]
            if state in {STATE_NOTE, STATE_HOLD} and pitch > 0:
                active_by_voice[voice].append(pitch)
                if voice == 0:
                    low = pitch
                else:
                    high = pitch
            if state == STATE_NOTE and pitch > 0:
                note_counts[voice] += 1
        if low is not None and high is not None:
            verticals.append(abs(high - low))
            final_low = low
            final_high = high

    bass_pitch = _last_or_median(active_by_voice[0])
    top_pitch = _last_or_median(active_by_voice[1])
    mean_vertical = int(round(sum(verticals) / len(verticals))) if verticals else 0
    final_interval_class = 12
    if final_low is not None and final_high is not None:
        final_interval_class = abs(final_high - final_low) % 12

    values = [
        phrase_pos,
        cadence_zone,
        _pitch_pc(bass_pitch),
        _pitch_pc(top_pitch),
        _pitch_oct(bass_pitch),
        _pitch_oct(top_pitch),
        min(16, note_counts[0]),
        min(16, note_counts[1]),
        min(49, mean_vertical),
        final_interval_class,
    ]
    return MeasurePlan(values)


def _last_or_median(pitches: list[int]) -> int | None:
    if not pitches:
        return None
    return pitches[-1]


def _pitch_pc(pitch: int | None) -> int:
    if pitch is None or pitch <= 0:
        return 12
    return pitch % 12


def _pitch_oct(pitch: int | None) -> int:
    if pitch is None or pitch <= 0:
        return 10
    return max(0, min(10, pitch // 12))
