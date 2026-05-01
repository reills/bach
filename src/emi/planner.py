from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from src.emi.buckets import CADENCE_TYPE_NAMES, HARMONIC_FUNCTION_NAMES, SPEAC_LABEL_NAMES
from src.inference.controls import normalize_compose_key, normalize_texture

PhraseRole = Literal[
    "OPENING",
    "SUBJECT_ENTRY",
    "ANSWER_ENTRY",
    "COUNTERSUBJECT",
    "EPISODE",
    "SEQUENCE",
    "CADENTIAL_PREPARATION",
    "CADENCE",
    "CLOSING",
]

SPEACLabel = Literal["S", "P", "E", "A", "C"]

PHRASE_ROLE_NAMES = [
    "UNKNOWN",
    "OPENING",
    "SUBJECT_ENTRY",
    "ANSWER_ENTRY",
    "COUNTERSUBJECT",
    "EPISODE",
    "SEQUENCE",
    "CADENTIAL_PREP",
    "CADENCE",
    "CLOSING",
]

_KEY_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


@dataclass(frozen=True)
class PhrasePlanStep:
    index: int
    phrase_role: PhraseRole
    speac_label: SPEACLabel
    cadence_target: str
    local_key_pc: int
    mode: int
    harmonic_function: str
    texture: int
    target_beats: float = 4.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_phrase_plan(
    *,
    measures: int,
    key: str | None = None,
    key_pc: int | None = None,
    mode: int | None = None,
    texture: int = 2,
) -> list[PhrasePlanStep]:
    if measures <= 0:
        raise ValueError("measures must be positive")
    resolved_key_pc, resolved_mode = _key_context(key=key, key_pc=key_pc, mode=mode)
    resolved_texture = normalize_texture(texture)
    roles = _role_plan(measures)
    return [
        PhrasePlanStep(
            index=index,
            phrase_role=role,
            speac_label=phrase_role_to_speac(role),
            cadence_target=cadence_type_for_role(role),
            local_key_pc=_local_key_for_role(resolved_key_pc, role=role, index=index),
            mode=resolved_mode,
            harmonic_function=harmonic_function_for_role(role),
            texture=resolved_texture,
        )
        for index, role in enumerate(roles)
    ]


def plan_step_for_row(row_index: int, *, steps_per_bar: int, plan: Sequence[PhrasePlanStep]) -> PhrasePlanStep:
    if not plan:
        raise ValueError("plan must not be empty")
    if steps_per_bar <= 0:
        raise ValueError("steps_per_bar must be positive")
    bar = max(0, row_index // steps_per_bar)
    return plan[min(bar, len(plan) - 1)]


def phrase_role_to_speac(role: str) -> SPEACLabel:
    return {
        "OPENING": "S",
        "SUBJECT_ENTRY": "S",
        "ANSWER_ENTRY": "A",
        "COUNTERSUBJECT": "E",
        "EPISODE": "E",
        "SEQUENCE": "P",
        "CADENTIAL_PREPARATION": "P",
        "CADENTIAL_PREP": "P",
        "CADENCE": "C",
        "CLOSING": "C",
    }.get(role, "E")  # type: ignore[return-value]


def cadence_type_for_role(role: str) -> str:
    if role == "CADENCE":
        return "AUTHENTIC"
    if role in {"CADENTIAL_PREPARATION", "CADENTIAL_PREP"}:
        return "HALF"
    return "NONE"


def harmonic_function_for_role(role: str) -> str:
    if role in {"OPENING", "SUBJECT_ENTRY", "CLOSING"}:
        return "TONIC"
    if role in {"ANSWER_ENTRY", "CADENCE"}:
        return "DOMINANT" if role == "ANSWER_ENTRY" else "CADENTIAL"
    if role in {"CADENTIAL_PREPARATION", "CADENTIAL_PREP"}:
        return "PREDOMINANT"
    if role == "SEQUENCE":
        return "SEQUENTIAL"
    if role in {"COUNTERSUBJECT", "EPISODE"}:
        return "OTHER"
    return "UNKNOWN"


def phrase_role_id(role: str | None) -> int:
    if role == "CADENTIAL_PREPARATION":
        role = "CADENTIAL_PREP"
    return _lookup(PHRASE_ROLE_NAMES, role)


def speac_label_id(label: str | None) -> int:
    return _lookup(SPEAC_LABEL_NAMES, label)


def cadence_type_id(cadence_type: str | None) -> int:
    return _lookup(CADENCE_TYPE_NAMES, cadence_type)


def harmonic_function_id(function: str | None) -> int:
    return _lookup(HARMONIC_FUNCTION_NAMES, function)


def _role_plan(measures: int) -> list[PhraseRole]:
    if measures == 1:
        return ["CADENCE"]
    plan: list[PhraseRole] = []
    for bar in range(measures):
        if bar == 0:
            plan.append("SUBJECT_ENTRY")
        elif bar == 1:
            plan.append("ANSWER_ENTRY")
        elif bar == measures - 1:
            plan.append("CADENCE")
        elif bar == measures - 2:
            plan.append("CADENTIAL_PREPARATION")
        elif bar % 2 == 0:
            plan.append("SEQUENCE")
        else:
            plan.append("EPISODE")
    return plan


def _local_key_for_role(key_pc: int, *, role: str, index: int) -> int:
    if key_pc >= 12:
        return 12
    if role == "ANSWER_ENTRY":
        return (key_pc + 7) % 12
    if role == "SEQUENCE" and index % 4 == 2:
        return (key_pc + 2) % 12
    return key_pc


def _key_context(
    *,
    key: str | None,
    key_pc: int | None,
    mode: int | None,
) -> tuple[int, int]:
    if key is not None:
        normalized = normalize_compose_key(key)
        resolved_mode = 1 if normalized.endswith("m") else 0
        tonic = normalized[:-1] if resolved_mode else normalized
        return _KEY_PC.get(tonic, 12), resolved_mode
    return (
        12 if key_pc is None else max(0, min(12, int(key_pc))),
        0 if mode is None else max(0, min(2, int(mode))),
    )


def _lookup(names: Sequence[str], value: str | None) -> int:
    if value is None:
        return 0
    normalized = value if value != "CADENTIAL_PREPARATION" else "CADENTIAL_PREP"
    try:
        return list(names).index(normalized)
    except ValueError:
        return 0
