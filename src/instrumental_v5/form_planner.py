from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from src.emi.cmmc import cmmc_function_for_role
from src.emi.planner import (
    PhrasePlanStep,
    cadence_type_for_role,
    harmonic_function_for_role,
    phrase_role_to_speac,
)
from src.inference.controls import normalize_compose_key, normalize_texture

V5FormName = Literal["invention", "sinfonia", "fugue", "suite", "partita", "prelude"]

_FORM_NAMES = {"invention", "sinfonia", "fugue", "suite", "partita", "prelude"}
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
_NOTE_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


@dataclass(frozen=True)
class FormEntry:
    bar: int
    role: str
    voice: int | None
    local_key_pc: int
    label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5FormPlan:
    form: V5FormName
    key: str
    key_pc: int
    mode: int
    texture: int
    measures: int
    subject_pitches: tuple[int, ...]
    steps: list[PhrasePlanStep]
    entries: list[FormEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "form": self.form,
            "key": self.key,
            "key_pc": self.key_pc,
            "mode": self.mode,
            "texture": self.texture,
            "measures": self.measures,
            "subject_pitches": list(self.subject_pitches),
            "roles": [step.phrase_role for step in self.steps],
            "local_key_pcs": [step.local_key_pc for step in self.steps],
            "entries": [entry.to_dict() for entry in self.entries],
        }


def build_v5_form_plan(
    *,
    form: str,
    measures: int,
    key: str | None = None,
    texture: int = 2,
    subject: str | Sequence[int] | None = None,
) -> V5FormPlan:
    """Build a CAST-style macro plan for v5 conditioning.

    This is deliberately symbolic: it schedules subject/answer/episode/cadence
    regions and leaves exact notes to the v5 compound generator and verifier.
    """

    normalized_form = _normalize_form(form)
    if measures <= 0:
        raise ValueError("measures must be positive")
    normalized_key = normalize_compose_key(key or "C")
    key_pc, mode = _key_context(normalized_key)
    resolved_texture = normalize_texture(texture)
    subject_pitches = _parse_subject(subject)

    roles = _roles_for_form(normalized_form, measures=measures, texture=resolved_texture)
    steps: list[PhrasePlanStep] = []
    entries: list[FormEntry] = []
    for bar, role in enumerate(roles):
        local_key_pc = _local_key_pc(
            key_pc,
            mode=mode,
            form=normalized_form,
            role=role,
            bar=bar,
            measures=measures,
        )
        steps.append(_step(bar, role=role, local_key_pc=local_key_pc, mode=mode, texture=resolved_texture))
        if role in {"SUBJECT_ENTRY", "ANSWER_ENTRY", "COUNTERSUBJECT"}:
            entries.append(
                FormEntry(
                    bar=bar,
                    role=role,
                    voice=_entry_voice(normalized_form, role=role, bar=bar, texture=resolved_texture),
                    local_key_pc=local_key_pc,
                    label=_entry_label(normalized_form, role=role, bar=bar),
                )
            )

    return V5FormPlan(
        form=normalized_form,
        key=normalized_key,
        key_pc=key_pc,
        mode=mode,
        texture=resolved_texture,
        measures=measures,
        subject_pitches=subject_pitches,
        steps=steps,
        entries=entries,
    )


def _normalize_form(form: str) -> V5FormName:
    normalized = form.strip().lower().replace("-", "_")
    if normalized not in _FORM_NAMES:
        raise ValueError(f"unsupported v5 form: {form!r}")
    return normalized  # type: ignore[return-value]


def _key_context(key: str) -> tuple[int, int]:
    mode = 1 if key.endswith("m") else 0
    tonic = key[:-1] if mode else key
    return _KEY_PC.get(tonic, 12), mode


def _parse_subject(subject: str | Sequence[int] | None) -> tuple[int, ...]:
    if subject is None:
        return ()
    if isinstance(subject, str):
        stripped = subject.strip()
        if not stripped:
            return ()
        return tuple(_parse_pitch_token(token) for token in stripped.split())
    return tuple(_clip_midi(int(value)) for value in subject)


def _parse_pitch_token(token: str) -> int:
    match = _PITCH_RE.match(token.strip())
    if match is None:
        raise ValueError(f"unsupported subject pitch token: {token!r}")
    name = match.group(1).upper()
    accidental = match.group(2)
    octave = int(match.group(3))
    pc = _NOTE_PC[name]
    if accidental == "#":
        pc += 1
    elif accidental == "b":
        pc -= 1
    return _clip_midi((octave + 1) * 12 + (pc % 12))


def _roles_for_form(form: V5FormName, *, measures: int, texture: int) -> list[str]:
    if form == "invention":
        base = [
            "SUBJECT_ENTRY",
            "ANSWER_ENTRY",
            "EPISODE",
            "SEQUENCE",
            "SUBJECT_ENTRY",
            "EPISODE",
            "CADENTIAL_PREPARATION",
            "CADENCE",
        ]
    elif form == "sinfonia":
        base = [
            "SUBJECT_ENTRY",
            "ANSWER_ENTRY",
            "COUNTERSUBJECT",
            "EPISODE",
            "SEQUENCE",
            "EPISODE",
            "CADENTIAL_PREPARATION",
            "CADENCE",
            "CLOSING",
        ]
    elif form == "fugue":
        exposition = []
        for idx in range(max(2, texture)):
            exposition.append("SUBJECT_ENTRY" if idx % 2 == 0 else "ANSWER_ENTRY")
            if idx >= 1:
                exposition.append("COUNTERSUBJECT")
        base = [*exposition, "EPISODE", "SEQUENCE", "CADENTIAL_PREPARATION", "CADENCE", "CLOSING"]
    elif form in {"suite", "partita"}:
        base = ["OPENING", "SUBJECT_ENTRY", "EPISODE", "SEQUENCE", "EPISODE", "CADENTIAL_PREPARATION", "CADENCE", "CLOSING"]
    else:
        base = ["OPENING", "SEQUENCE", "EPISODE", "SEQUENCE", "EPISODE", "CADENTIAL_PREPARATION", "CADENCE"]

    if measures <= len(base):
        if measures == 1:
            return ["CADENCE"]
        return _force_cadential_close(base[:measures])

    extension = ["EPISODE", "SEQUENCE"] * ((measures - len(base) + 1) // 2)
    return _force_cadential_close([*base[:-2], *extension[: measures - len(base)], *base[-2:]])


def _force_cadential_close(roles: list[str]) -> list[str]:
    if len(roles) >= 2:
        roles[-2] = "CADENTIAL_PREPARATION"
    roles[-1] = "CADENCE"
    return roles


def _step(index: int, *, role: str, local_key_pc: int, mode: int, texture: int) -> PhrasePlanStep:
    return PhrasePlanStep(
        index=index,
        phrase_role=role,  # type: ignore[arg-type]
        speac_label=phrase_role_to_speac(role),
        cmmc_function=cmmc_function_for_role(role),
        cadence_target=cadence_type_for_role(role),
        local_key_pc=local_key_pc,
        mode=mode,
        harmonic_function=harmonic_function_for_role(role),
        texture=texture,
    )


def _local_key_pc(key_pc: int, *, mode: int, form: V5FormName, role: str, bar: int, measures: int) -> int:
    if key_pc >= 12:
        return 12
    if role == "ANSWER_ENTRY":
        return (key_pc + 7) % 12
    if role == "SEQUENCE":
        return (key_pc + (3 if mode == 1 and form in {"invention", "sinfonia"} else 2)) % 12
    if role in {"CADENTIAL_PREPARATION", "CADENCE", "CLOSING"} and bar >= max(0, measures - 3):
        return key_pc
    if form == "fugue" and role == "EPISODE" and bar % 4 == 0:
        return (key_pc + 7) % 12
    return key_pc


def _entry_voice(form: V5FormName, *, role: str, bar: int, texture: int) -> int | None:
    if role == "COUNTERSUBJECT":
        return None
    if form == "fugue":
        return bar % max(1, texture)
    if role == "ANSWER_ENTRY":
        return 0
    return 1 if bar == 0 else bar % max(1, min(texture, 2))


def _entry_label(form: V5FormName, *, role: str, bar: int) -> str:
    if form == "fugue" and role in {"SUBJECT_ENTRY", "ANSWER_ENTRY"}:
        return "exposition_entry"
    if role == "ANSWER_ENTRY":
        return "dominant_answer"
    if role == "SUBJECT_ENTRY" and bar > 0:
        return "subject_return"
    return role.lower()


def _clip_midi(value: int) -> int:
    return max(0, min(127, int(value)))
