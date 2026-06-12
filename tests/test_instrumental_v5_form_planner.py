from __future__ import annotations

from src.inference.controls import ComposeControls
from src.inference.hybrid import build_hybrid_context
from src.instrumental_v5.form_planner import build_v5_form_plan


def test_invention_form_plan_schedules_subject_answer_and_cadence() -> None:
    plan = build_v5_form_plan(
        form="invention",
        measures=8,
        key="D minor",
        texture=2,
        subject="D4 E4 F4 A4 G4 F4 E4 D4",
    )

    assert plan.key == "Dm"
    assert plan.key_pc == 2
    assert plan.mode == 1
    assert plan.subject_pitches == (62, 64, 65, 69, 67, 65, 64, 62)
    assert [step.phrase_role for step in plan.steps[:2]] == ["SUBJECT_ENTRY", "ANSWER_ENTRY"]
    assert plan.steps[1].local_key_pc == 9
    assert plan.steps[-2].phrase_role == "CADENTIAL_PREPARATION"
    assert plan.steps[-1].phrase_role == "CADENCE"
    assert plan.entries[0].label == "subject_entry"
    assert plan.entries[1].label == "dominant_answer"


def test_fugue_form_plan_expands_exposition_by_texture() -> None:
    plan = build_v5_form_plan(form="fugue", measures=10, key="C", texture=3)

    roles = [step.phrase_role for step in plan.steps]
    assert roles[:5] == ["SUBJECT_ENTRY", "ANSWER_ENTRY", "COUNTERSUBJECT", "SUBJECT_ENTRY", "COUNTERSUBJECT"]
    assert [entry.voice for entry in plan.entries[:2]] == [0, 1]
    assert roles[-1] == "CADENCE"


def test_hybrid_context_accepts_explicit_v5_form_plan() -> None:
    form_plan = build_v5_form_plan(form="invention", measures=4, key="G", texture=2)

    context = build_hybrid_context(
        ComposeControls(key="G", measures=4, texture=2),
        plan=form_plan.steps,
        planning_metadata=form_plan.to_dict(),
    )

    diagnostics = context.diagnostics()
    assert diagnostics["rolePlan"] == ["SUBJECT_ENTRY", "ANSWER_ENTRY", "CADENTIAL_PREPARATION", "CADENCE"]
    assert diagnostics["planningMetadata"]["form"] == "invention"  # type: ignore[index]
    assert context.conditioning_rows[0]["phrase_role"] > 0
