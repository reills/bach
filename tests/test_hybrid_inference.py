from __future__ import annotations

from pathlib import Path

from src.emi.fragments import Fragment, fragment_to_jsonl
from src.inference.controls import ComposeControls
from src.inference.hybrid import build_hybrid_context, conditioning_has_raw_fragment_ids
from src.instrumental_v5.representation import CONTOUR_BUCKET_TO_ID, RHYTHM_BUCKET_TO_ID


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
