from __future__ import annotations

from src.emi.planner import build_phrase_plan, plan_step_for_row


def test_phrase_plan_outputs_bounded_musical_labels() -> None:
    plan = build_phrase_plan(measures=4, key="D minor", texture=4)

    assert [step.phrase_role for step in plan] == [
        "SUBJECT_ENTRY",
        "ANSWER_ENTRY",
        "CADENTIAL_PREPARATION",
        "CADENCE",
    ]
    assert [step.speac_label for step in plan] == ["S", "A", "P", "C"]
    assert [step.cadence_target for step in plan] == ["NONE", "NONE", "HALF", "AUTHENTIC"]
    assert all(0 <= step.local_key_pc <= 12 for step in plan)
    assert all(step.harmonic_function for step in plan)
    assert all(step.texture == 4 for step in plan)


def test_plan_step_for_row_maps_grid_rows_to_measure_plan() -> None:
    plan = build_phrase_plan(measures=3, key_pc=0, mode=0, texture=2)

    assert plan_step_for_row(0, steps_per_bar=4, plan=plan).phrase_role == "SUBJECT_ENTRY"
    assert plan_step_for_row(5, steps_per_bar=4, plan=plan).phrase_role == "ANSWER_ENTRY"
    assert plan_step_for_row(99, steps_per_bar=4, plan=plan).phrase_role == "CADENCE"
