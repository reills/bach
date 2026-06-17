from __future__ import annotations

from pathlib import Path

from src.emi.fragments import Fragment, fragment_to_jsonl
from src.inference.controls import ComposeControls
from src.inference.hybrid import (
    apply_conditioning_to_v5_row,
    apply_conditioning_to_v5_rows,
    build_hybrid_context,
    conditioning_has_raw_fragment_ids,
)
from src.instrumental_v5.representation import CONTOUR_BUCKET_TO_ID, RHYTHM_BUCKET_TO_ID, V5_FIELD_NAMES


def _fragment() -> Fragment:
    return Fragment(
        id="toy_v0_s0_l8",
        piece_id="toy",
        source_path="toy.musicxml",
        voice=0,
        start_slice=0,
        length_slices=8,
        start_bar=0,
        start_pos=0,
        beats=2.0,
        phrase_role="SUBJECT_ENTRY",
        key="C",
        key_pc=0,
        mode=0,
        start_pitch=60,
        end_pitch=67,
        start_degree=1,
        end_degree=5,
        melodic_intervals=[2, 2, 1],
        rhythm_steps=[2, 2, 2],
        vertical_intervals=[12, 10],
        state_pattern=[2, 1, 2, 1],
        contour_hash="contour",
        fingerprint="fingerprint",
        speac_label="S",
        cadence_type="NONE",
        contour_bucket="ASCENDING_STEPWISE",
        rhythm_bucket="EVEN_8THS",
        local_key_pc=0,
        harmonic_function="TONIC",
        entry_degree=1,
        exit_degree=5,
        min_pitch=60,
        max_pitch=67,
        copy_hash="copyhash",
        transposition_hash="transhash",
    )


def test_hybrid_context_retrieves_fragments_and_exposes_only_bounded_fields(tmp_path: Path) -> None:
    path = tmp_path / "fragments.jsonl"
    path.write_text(fragment_to_jsonl(_fragment()) + "\n", encoding="utf-8")

    context = build_hybrid_context(
        ComposeControls(key="C", measures=2, texture=2),
        fragment_path=path,
    )
    conditioning = context.model_conditioning()

    assert context.fragment_count == 1
    assert context.retrieved_matches
    assert conditioning["field_names"] == [
        "phrase_role",
        "speac_label",
        "cmmc_function",
        "cadence_target",
        "harmonic_function",
        "local_key_pc",
        "retrieved_contour_bucket",
        "retrieved_rhythm_bucket",
    ]
    assert conditioning["rows"][0]["retrieved_contour_bucket"] == CONTOUR_BUCKET_TO_ID["ASCENDING_STEPWISE"]
    assert conditioning["rows"][0]["retrieved_rhythm_bucket"] == RHYTHM_BUCKET_TO_ID["EVEN_8THS"]
    assert not conditioning_has_raw_fragment_ids(conditioning)


def test_hybrid_context_uses_unknown_retrieval_buckets_without_memory() -> None:
    context = build_hybrid_context(ComposeControls(key="C", measures=1, texture=2))

    row = context.conditioning_rows[0]
    assert context.fragment_count == 0
    assert context.retrieved_matches == []
    assert row["retrieved_contour_bucket"] == CONTOUR_BUCKET_TO_ID["UNKNOWN"]
    assert row["retrieved_rhythm_bucket"] == RHYTHM_BUCKET_TO_ID["UNKNOWN"]


def test_hybrid_conditioning_is_written_into_actual_v5_rows(tmp_path: Path) -> None:
    path = tmp_path / "fragments.jsonl"
    path.write_text(fragment_to_jsonl(_fragment()) + "\n", encoding="utf-8")
    context = build_hybrid_context(
        ComposeControls(key="C", measures=2, texture=2),
        fragment_path=path,
    )
    rows = [[0] * len(V5_FIELD_NAMES) for _ in range(8)]

    conditioned = apply_conditioning_to_v5_rows(rows, context, steps_per_bar=4)

    first_plan_row = context.conditioning_rows[0]
    second_plan_row = context.conditioning_rows[1]
    assert conditioned[0][V5_FIELD_NAMES.index("phrase_role")] == first_plan_row["phrase_role"]
    assert conditioned[0][V5_FIELD_NAMES.index("cadence_target")] == first_plan_row["cadence_target"]
    assert conditioned[0][V5_FIELD_NAMES.index("retrieved_contour_bucket")] == first_plan_row["retrieved_contour_bucket"]
    assert conditioned[4][V5_FIELD_NAMES.index("phrase_role")] == second_plan_row["phrase_role"]
    assert conditioned[4][V5_FIELD_NAMES.index("harmonic_function")] == second_plan_row["harmonic_function"]


def test_single_generated_v5_row_can_be_conditioned_by_absolute_row_index(tmp_path: Path) -> None:
    path = tmp_path / "fragments.jsonl"
    path.write_text(fragment_to_jsonl(_fragment()) + "\n", encoding="utf-8")
    context = build_hybrid_context(
        ComposeControls(key="C", measures=2, texture=2),
        fragment_path=path,
    )

    conditioned = apply_conditioning_to_v5_row(
        [0] * len(V5_FIELD_NAMES),
        context,
        row_index=4,
        steps_per_bar=4,
    )

    second_plan_row = context.conditioning_rows[1]
    assert conditioned[V5_FIELD_NAMES.index("phrase_role")] == second_plan_row["phrase_role"]
    assert conditioned[V5_FIELD_NAMES.index("cadence_target")] == second_plan_row["cadence_target"]
