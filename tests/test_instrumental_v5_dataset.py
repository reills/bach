from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.make_instrumental_v5_dataset import build_v5_outputs
from src.instrumental_v3.representation import FIELD_NAMES as V3_FIELD_NAMES, InstrumentalV3Piece, SliceEvent
from src.instrumental_v4.representation import build_v4_piece
from src.instrumental_v5.data import summarize_conditioning_coverage
from src.instrumental_v5.representation import (
    CADENCE_TYPE_TO_ID,
    CONTOUR_BUCKET_TO_ID,
    CP_MOTION_TYPE_TO_ID,
    HARMONIC_FUNCTION_TO_ID,
    PHRASE_ROLE_TO_ID,
    RHYTHM_BUCKET_TO_ID,
    SPEAC_LABEL_TO_ID,
    V5_EMI_FIELD_NAMES,
    V5_COUNTERPOINT_FIELD_NAMES,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
    build_v5_piece,
    classify_contour_bucket,
    classify_rhythm_bucket,
)


def _toy_v3_piece(piece_id: str) -> InstrumentalV3Piece:
    rows = []
    for idx in range(16):
        bar = idx // 4
        pos = idx % 4
        note = pos in {0, 2}
        state0 = 2 if note else 1
        state1 = 2 if note else 1
        p0 = 48 + idx // 2
        p1 = 60 + (idx // 2 if bar < 2 else 4 - idx // 4)
        row = [
            bar,
            pos,
            bar % 8,
            1 if bar >= 2 else 0,
            0,
            0,
            2,
            state0,
            p0,
            25,
            2,
            0 if note else 1,
            1,
            state1,
            p1,
            25,
            2,
            0 if note else 1,
            1,
            abs(p1 - p0) + 1,
            2,
            abs(p1 - p0) + 1,
        ]
        assert len(row) == len(V3_FIELD_NAMES)
        rows.append(row)
    return InstrumentalV3Piece(
        piece_id=piece_id,
        source_path=f"{piece_id}.musicxml",
        tpq=24,
        grid_ticks=6,
        time_signature="1/1",
        key="C",
        key_pc=0,
        mode=0,
        bar_len_ticks=24,
        steps_per_bar=4,
        slices=[SliceEvent(row) for row in rows],
    )


def test_v5_bucket_classifiers_are_bounded_and_abstract() -> None:
    assert classify_contour_bucket([1, 1, 2]) == "ASCENDING_STEPWISE"
    assert classify_contour_bucket([-5, -2]) == "DESCENDING_LEAPY"
    assert classify_contour_bucket([2, 1, -1]) == "ARCH"
    assert classify_contour_bucket([-2, 3, -1]) == "ZIGZAG"
    assert classify_rhythm_bucket([1, 1, 1]) == "EVEN_16THS"
    assert classify_rhythm_bucket([2, 2]) == "EVEN_8THS"
    assert classify_rhythm_bucket([3, 1]) == "DOTTED"

    assert max(CONTOUR_BUCKET_TO_ID.values()) < V5_FEATURE_SPECS["retrieved_contour_bucket"]
    assert max(RHYTHM_BUCKET_TO_ID.values()) < V5_FEATURE_SPECS["retrieved_rhythm_bucket"]


def test_build_v5_piece_adds_bounded_plan_and_retrieval_fields() -> None:
    piece = build_v5_piece(build_v4_piece(_toy_v3_piece("toy")), length_slices=4, hop_slices=2)

    assert V5_FIELD_NAMES[-8:] == V5_EMI_FIELD_NAMES
    assert set(V5_COUNTERPOINT_FIELD_NAMES).issubset(V5_FIELD_NAMES)
    assert len(piece.rows[0]) == len(V5_FIELD_NAMES)
    assert len(V5_EMI_FIELD_NAMES) == 8
    assert "fragment_id" not in V5_FIELD_NAMES
    assert piece.rows[0][V5_FIELD_NAMES.index("phrase_role")] < V5_FEATURE_SPECS["phrase_role"]
    assert piece.rows[0][V5_FIELD_NAMES.index("speac_label")] < V5_FEATURE_SPECS["speac_label"]
    assert piece.rows[0][V5_FIELD_NAMES.index("cmmc_function")] < V5_FEATURE_SPECS["cmmc_function"]
    assert piece.rows[0][V5_FIELD_NAMES.index("cp_motion_type")] < V5_FEATURE_SPECS["cp_motion_type"]
    assert piece.rows[0][V5_FIELD_NAMES.index("cp_curr_interval_class")] < V5_FEATURE_SPECS["cp_curr_interval_class"]
    assert piece.rows[0][V5_FIELD_NAMES.index("cadence_target")] < V5_FEATURE_SPECS["cadence_target"]
    assert piece.rows[0][V5_FIELD_NAMES.index("harmonic_function")] < V5_FEATURE_SPECS["harmonic_function"]
    assert piece.rows[0][V5_FIELD_NAMES.index("local_key_pc")] < V5_FEATURE_SPECS["local_key_pc"]
    assert (
        piece.rows[0][V5_FIELD_NAMES.index("retrieved_contour_bucket")]
        < V5_FEATURE_SPECS["retrieved_contour_bucket"]
    )
    assert (
        piece.rows[0][V5_FIELD_NAMES.index("retrieved_rhythm_bucket")]
        < V5_FEATURE_SPECS["retrieved_rhythm_bucket"]
    )
    assert piece.rows[2][V5_FIELD_NAMES.index("cp_motion_type")] == CP_MOTION_TYPE_TO_ID["PARALLEL"]
    assert piece.rows[2][V5_FIELD_NAMES.index("cp_parallel_perfect")] == 1


def test_build_v5_piece_keeps_plan_fields_when_no_fragment_covers_rows() -> None:
    piece = build_v5_piece(build_v4_piece(_toy_v3_piece("toy")), length_slices=64, hop_slices=8)

    assert piece.rows[0][V5_FIELD_NAMES.index("phrase_role")] == PHRASE_ROLE_TO_ID["SUBJECT_ENTRY"]
    assert piece.rows[0][V5_FIELD_NAMES.index("speac_label")] == SPEAC_LABEL_TO_ID["S"]
    assert piece.rows[0][V5_FIELD_NAMES.index("cmmc_function")] > 0
    assert piece.rows[0][V5_FIELD_NAMES.index("cadence_target")] == CADENCE_TYPE_TO_ID["NONE"]
    assert piece.rows[0][V5_FIELD_NAMES.index("harmonic_function")] == HARMONIC_FUNCTION_TO_ID["TONIC"]
    assert piece.rows[0][V5_FIELD_NAMES.index("local_key_pc")] == 0
    assert piece.rows[0][V5_FIELD_NAMES.index("cp_motion_type")] == CP_MOTION_TYPE_TO_ID["UNKNOWN"]
    assert {row[V5_FIELD_NAMES.index("retrieved_contour_bucket")] for row in piece.rows} == {
        CONTOUR_BUCKET_TO_ID["UNKNOWN"]
    }
    assert {row[V5_FIELD_NAMES.index("retrieved_rhythm_bucket")] for row in piece.rows} == {
        RHYTHM_BUCKET_TO_ID["UNKNOWN"]
    }


def test_v5_builder_writes_parquet_vocab_metadata_and_split_fragment_files(tmp_path: Path) -> None:
    pieces = [build_v4_piece(_toy_v3_piece("toy_a")), build_v4_piece(_toy_v3_piece("toy_b"))]

    summary = build_v5_outputs(
        pieces,
        output_dir=tmp_path,
        source_dataset="toy_v4.json",
        length_slices=4,
        hop_slices=2,
        val_split=0.5,
        seed=7,
    )

    events = pd.read_parquet(tmp_path / "events.parquet")
    assert set(V5_EMI_FIELD_NAMES).issubset(events.columns)
    assert set(V5_COUNTERPOINT_FIELD_NAMES).issubset(events.columns)
    for field in V5_FIELD_NAMES:
        assert pd.api.types.is_integer_dtype(events[field])
        assert int(events[field].min()) >= 0
        assert int(events[field].max()) < V5_FEATURE_SPECS[field]

    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    assert vocab["phrase_role"]["UNKNOWN"] == 0
    assert vocab["speac_label"]["UNKNOWN"] == 0
    assert vocab["cmmc_function"]["UNKNOWN"] == 0
    assert vocab["cp_motion_type"]["UNKNOWN"] == 0
    assert vocab["cadence_target"]["UNKNOWN"] == 0
    assert vocab["harmonic_function"]["UNKNOWN"] == 0
    assert vocab["retrieved_contour_bucket"]["UNKNOWN"] == 0
    assert vocab["retrieved_rhythm_bucket"]["UNKNOWN"] == 0
    assert "fragment_id" not in json.dumps(vocab)
    assert "toy_a_v0" not in json.dumps(vocab)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["conditioning_coverage"]["phrase_role"]["non_default_rate"] > 0
    train_piece_ids = set(metadata["train_piece_ids"])
    val_piece_ids = set(metadata["val_piece_ids"])
    assert train_piece_ids
    assert val_piece_ids
    assert train_piece_ids.isdisjoint(val_piece_ids)
    assert set(events.loc[events["split"] == "train", "piece_id"]).isdisjoint(
        set(events.loc[events["split"] == "val", "piece_id"])
    )

    train_fragment_lines = (tmp_path / "train_emi_fragments.jsonl").read_text(encoding="utf-8").splitlines()
    assert train_fragment_lines
    train_fragment_piece_ids = {json.loads(line)["piece_id"] for line in train_fragment_lines}
    assert train_fragment_piece_ids.issubset(train_piece_ids)
    assert train_fragment_piece_ids.isdisjoint(val_piece_ids)

    assert Path(summary["events_path"]).name == "events.parquet"


def test_conditioning_coverage_reports_non_default_label_rates(tmp_path: Path) -> None:
    pieces = [build_v4_piece(_toy_v3_piece("toy_a")), build_v4_piece(_toy_v3_piece("toy_b"))]
    build_v5_outputs(
        pieces,
        output_dir=tmp_path,
        source_dataset="toy_v4.json",
        length_slices=4,
        hop_slices=2,
        val_split=0.5,
        seed=7,
    )
    events = pd.read_parquet(tmp_path / "events.parquet")

    coverage = summarize_conditioning_coverage(events)

    assert coverage["phrase_role"]["non_default_rate"] > 0
    assert coverage["speac_label"]["non_default_rate"] > 0
    assert coverage["harmonic_function"]["non_default_rate"] > 0
    assert coverage["local_key_pc"]["non_default_rate"] > 0
    assert coverage["retrieved_contour_bucket"]["non_default_rate"] > 0
    assert coverage["retrieved_rhythm_bucket"]["non_default_rate"] > 0
    assert coverage["cadence_target"]["default_id"] == CADENCE_TYPE_TO_ID["NONE"]
