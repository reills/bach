from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scripts.generate_instrumental_v6 import (
    _cadence_target_pitches,
    _candidate_score,
    _development_interval,
    _duration_log_prior,
    _fits_tonal_context,
    _motif_report,
    _repetition_penalty,
    generate_rows,
    _sample_duration,
    _select_template,
)
from src.instrumental_v6.decoding import (
    PitchOption,
    creates_parallel_perfect,
    select_counterpoint_pitches,
)
from src.instrumental_v6.model import FactorizedConfig, build_generator, multihead_loss
from src.instrumental_v6.metrics import evaluate_piece_rows
from src.instrumental_v6.representation import (
    DEVELOPMENT_TO_ID,
    GLOBAL_FIELD_NAMES,
    PAIR_FIELD_NAMES,
    VOICE_FIELD_NAMES,
    _select_movement_indices,
    build_development_plan,
    parse_musicxml_movements,
)


def test_factorized_model_supports_six_voice_axis() -> None:
    config = FactorizedConfig(
        max_voices=6,
        d_model=48,
        n_heads=6,
        n_layers=1,
        dropout=0.0,
        max_seq_len=16,
    )
    model = build_generator(config)
    global_values = torch.zeros((2, 8, len(GLOBAL_FIELD_NAMES)), dtype=torch.long)
    global_values[..., GLOBAL_FIELD_NAMES.index("voice_count")] = 6
    voice_values = torch.zeros((2, 8, 6, len(VOICE_FIELD_NAMES)), dtype=torch.long)
    pair_values = torch.zeros((2, 8, 6, 6, len(PAIR_FIELD_NAMES)), dtype=torch.long)

    logits = model(global_values, voice_values, pair_values)

    assert logits["voice"]["pitch"].shape == (2, 8, 6, 129)
    assert logits["pair"]["motion"].shape == (2, 8, 6, 6, 6)


def test_voice_aware_model_does_not_pool_distinct_voice_histories() -> None:
    config = FactorizedConfig(
        max_voices=6,
        d_model=48,
        n_heads=6,
        n_layers=1,
        n_cross_layers=1,
        dropout=0.0,
        max_seq_len=8,
        architecture="voice_aware_v2",
    )
    model = build_generator(config).eval()
    global_values = torch.zeros((1, 4, len(GLOBAL_FIELD_NAMES)), dtype=torch.long)
    global_values[..., GLOBAL_FIELD_NAMES.index("voice_count")] = 2
    voice_values = torch.zeros((1, 4, 6, len(VOICE_FIELD_NAMES)), dtype=torch.long)
    voice_values[..., :2, VOICE_FIELD_NAMES.index("state")] = 2
    voice_values[..., :2, VOICE_FIELD_NAMES.index("pitch")] = 60
    pair_values = torch.zeros((1, 4, 6, 6, len(PAIR_FIELD_NAMES)), dtype=torch.long)

    baseline = build_generator(config)
    baseline.load_state_dict(model.state_dict())
    changed = voice_values.clone()
    changed[:, 0, 0, VOICE_FIELD_NAMES.index("pitch")] = 48

    with torch.no_grad():
        original_logits = baseline(global_values, voice_values, pair_values)["voice"]["pitch"]
        changed_logits = model(global_values, changed, pair_values)["voice"]["pitch"]

    assert not torch.allclose(original_logits[:, -1, 0], changed_logits[:, -1, 0])


def test_factorized_loss_masks_inactive_voice_slots() -> None:
    config = FactorizedConfig(
        max_voices=6,
        d_model=48,
        n_heads=6,
        n_layers=1,
        dropout=0.0,
        max_seq_len=8,
    )
    model = build_generator(config)
    global_values = torch.zeros((1, 5, len(GLOBAL_FIELD_NAMES)), dtype=torch.long)
    global_values[..., GLOBAL_FIELD_NAMES.index("voice_count")] = 4
    voice_values = torch.zeros((1, 5, 6, len(VOICE_FIELD_NAMES)), dtype=torch.long)
    voice_values[..., :4, VOICE_FIELD_NAMES.index("state")] = 2
    voice_values[..., :4, VOICE_FIELD_NAMES.index("pitch")] = 60
    pair_values = torch.zeros((1, 5, 6, 6, len(PAIR_FIELD_NAMES)), dtype=torch.long)
    logits = model(global_values[:, :-1], voice_values[:, :-1], pair_values[:, :-1])

    loss, metrics = multihead_loss(
        logits,
        global_values[:, 1:],
        voice_values[:, 1:],
        pair_values[:, 1:],
        torch.ones((1, 4), dtype=torch.bool),
    )

    assert loss.item() > 0
    assert metrics["voice.state_count"] == 16
    assert metrics["voice.state.v3_count"] == 4
    assert metrics["voice.state.v4_count"] == 0
    assert metrics["pair.motion_count"] == 24


def test_six_part_ricercar_preserves_six_voices() -> None:
    pieces = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/Musical offering/BWV_1079_02/BWV_1079_02.xml",
        form="fugue",
        target_voices=6,
        max_voices=6,
        max_bars=32,
    )

    assert pieces
    assert pieces[0].voice_count == 6
    assert len(pieces[0].voice_rows[0]) == 6


def test_auto_voice_count_handles_two_three_and_six_voice_sources() -> None:
    cases = [
        (
            "data/tobis_xml/instrumental-works/keyboard-works/"
            "BWV 772-786 Inventions/BWV_0772/BWV_0772.xml",
            "invention",
            2,
            4,
        ),
        (
            "data/tobis_xml/instrumental-works/keyboard-works/"
            "BWV 787-801 Sinfonias/BWV_0787/BWV_0787.xml",
            "sinfonia",
            3,
            4,
        ),
        (
            "data/tobis_xml/instrumental-works/Musical offering/"
            "BWV_1079_02/BWV_1079_02.xml",
            "fugue",
            6,
            32,
        ),
    ]

    for path, form, expected, max_bars in cases:
        pieces = parse_musicxml_movements(
            path,
            form=form,
            target_voices=None,
            max_voices=6,
            max_bars=max_bars,
            max_movements=1,
        )
        assert pieces[0].voice_count == expected


def test_multi_part_scores_use_explicit_part_count_for_voice_count() -> None:
    pieces = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/Art of fugue/"
        "BWV_1080_01/BWV_1080_01.xml",
        form="fugue",
        target_voices=None,
        max_voices=6,
        max_bars=32,
        max_movements=1,
    )

    assert pieces[0].voice_count == 4


def test_movement_limit_spreads_across_the_work() -> None:
    assert _select_movement_indices(19, 2) == [0, 18]
    assert _select_movement_indices(19, 3) == [0, 9, 18]


def test_wtc_fugue_uses_verified_voice_count_metadata() -> None:
    cases = [
        ("BWV_0849", 5),
        ("BWV_0850", 4),
        ("BWV_0853", 3),
        ("BWV_0869", 4),
    ]
    root = (
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 846-869 The Well-Tempered Clavier Book I"
    )

    for work, expected in cases:
        pieces = parse_musicxml_movements(
            f"{root}/{work}/{work}.xml",
            form="wtc",
            target_voices=None,
            max_voices=6,
            max_bars=32,
        )
        assert pieces[1].voice_count == expected


def test_wtc_movement_limit_keeps_prelude_and_fugue() -> None:
    pieces = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 846-869 The Well-Tempered Clavier Book I/BWV_0869/BWV_0869.xml",
        form="wtc",
        target_voices=None,
        max_voices=6,
        max_bars=8,
        max_movements=2,
    )

    assert [piece.movement_index for piece in pieces] == [0, 1]
    assert [piece.form for piece in pieces] == ["PRELUDE", "FUGUE"]


def test_generation_template_can_be_selected_by_exact_piece_id() -> None:
    first = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 772-786 Inventions/BWV_0772/BWV_0772.xml",
        form="invention",
        target_voices=2,
        max_voices=6,
        max_bars=1,
    )[0]
    second = replace(first, piece_id="alternate")

    selected = _select_template(
        [first, second],
        voices=2,
        requested_form="INVENTION",
        piece_id="alternate",
        piece_index=0,
    )

    assert selected.piece_id == "alternate"
    with pytest.raises(SystemExit, match="not requested 3"):
        _select_template(
            [first],
            voices=3,
            requested_form="INVENTION",
            piece_id=first.piece_id,
            piece_index=0,
        )


def test_duration_sampling_can_use_a_corpus_prior() -> None:
    piece = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 772-786 Inventions/BWV_0772/BWV_0772.xml",
        form="invention",
        target_voices=2,
        max_voices=6,
        max_bars=4,
    )[0]
    prior = _duration_log_prior(
        [piece],
        voices=2,
        form="INVENTION",
        device=torch.device("cpu"),
    )

    selected = _sample_duration(
        torch.zeros_like(prior),
        temperature=0.0,
        top_k=12,
        max_duration=32,
        log_prior=prior,
        prior_strength=1.0,
    )

    assert prior[1] > prior[8]
    assert selected == 1


def test_tonal_context_accepts_global_or_planned_local_scale() -> None:
    row = [0] * len(GLOBAL_FIELD_NAMES)
    row[GLOBAL_FIELD_NAMES.index("key_pc")] = 0
    row[GLOBAL_FIELD_NAMES.index("local_key_pc")] = 7

    assert _fits_tonal_context(60, row, mode=0)
    assert _fits_tonal_context(66, row, mode=0)
    assert not _fits_tonal_context(61, row, mode=0)


def test_stretto_without_planned_entry_rotates_subject_voice() -> None:
    row = [0] * len(GLOBAL_FIELD_NAMES)
    row[GLOBAL_FIELD_NAMES.index("bar")] = 6
    row[GLOBAL_FIELD_NAMES.index("voice_count")] = 4
    row[GLOBAL_FIELD_NAMES.index("entry_voice")] = 6
    row[GLOBAL_FIELD_NAMES.index("development")] = DEVELOPMENT_TO_ID["STRETTO"]
    previous_rows = [
        [[0] * len(VOICE_FIELD_NAMES) for _ in range(6)]
        for _ in range(16)
    ]

    assert _development_interval(
        row,
        previous_rows,
        voice=2,
        subject=[2, -1, 2],
        steps_per_bar=16,
    ) == 2
    assert _development_interval(
        row,
        previous_rows,
        voice=1,
        subject=[2, -1, 2],
        steps_per_bar=16,
    ) is None


def test_motif_report_finds_transposed_interval_subject() -> None:
    rows = [
        [[2, pitch, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]]
        for pitch in [60, 62, 63, 62, 67]
    ]

    report = _motif_report(rows, subject=[2, 1, -1, 5], voice_count=2)

    assert report["subject_head_hits"] == 1
    assert report["max_subject_prefix"] == 4
    assert report["voice_subject_head_hits"] == [1, 0]
    assert report["section_subject_head_hits"] == {
        "opening": 1,
        "middle": 0,
        "closing": 0,
    }


def test_motif_report_tracks_recurrence_across_the_piece_arch() -> None:
    rows = [
        [[0, 0, 0, 0, 0, 0] for _ in range(3)]
        for _ in range(12)
    ]
    for voice, start in enumerate((0, 4, 8)):
        for offset, pitch in enumerate((60, 62, 63, 62)):
            rows[start + offset][voice] = [2, pitch, 0, 1, 0, 1]

    report = _motif_report(rows, subject=[2, 1, -1], voice_count=3)

    assert report["section_subject_head_hits"] == {
        "opening": 1,
        "middle": 1,
        "closing": 1,
    }


def test_repetition_penalty_escalates_for_same_note_and_short_loop() -> None:
    same_note_rows = [
        [[2, 60, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]]
        for _ in range(5)
    ]
    loop_rows = [
        [[2, pitch, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]]
        for pitch in [60, 62, 60, 62, 60, 62]
    ]

    assert _repetition_penalty(
        same_note_rows,
        voice=0,
        candidate_pitch=60,
        protected_interval=None,
    ) > 20.0
    assert _repetition_penalty(
        loop_rows,
        voice=0,
        candidate_pitch=60,
        protected_interval=None,
    ) > 0.0
    assert _repetition_penalty(
        loop_rows,
        voice=0,
        candidate_pitch=64,
        protected_interval=None,
    ) == 0.0


def test_generation_metrics_expose_retrigger_and_short_loop_collapse() -> None:
    global_rows = [[0] * len(GLOBAL_FIELD_NAMES) for _ in range(8)]
    for index, row in enumerate(global_rows):
        row[GLOBAL_FIELD_NAMES.index("pos")] = index
        row[GLOBAL_FIELD_NAMES.index("key_pc")] = 0
        row[GLOBAL_FIELD_NAMES.index("local_key_pc")] = 0
        row[GLOBAL_FIELD_NAMES.index("voice_count")] = 2
    pitches = [60, 60, 60, 60, 62, 64, 62, 64]
    voice_rows = [
        [
            [2, pitch, 0, 1, 0, 1],
            [2, 72 + (index % 2) * 2, 0, 1, 0, 1],
        ]
        for index, pitch in enumerate(pitches)
    ]
    pair_rows = [
        [[[0] * len(PAIR_FIELD_NAMES) for _ in range(2)] for _ in range(2)]
        for _ in range(8)
    ]

    report = evaluate_piece_rows(
        global_rows,
        voice_rows,
        pair_rows,
        voice_count=2,
    )

    assert report["voice_max_repeated_note_attacks"][0] == 4
    assert report["voice_repeated_note_attack_rates"][0] > 0.4
    assert report["voice_short_loop_rates"][1] > 0.5


def test_generation_replans_prompt_and_continuation_consistently() -> None:
    piece = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 772-786 Inventions/BWV_0772/BWV_0772.xml",
        form="invention",
        target_voices=2,
        max_voices=6,
        max_bars=4,
    )[0]
    config = FactorizedConfig(
        max_voices=6,
        d_model=48,
        n_heads=6,
        n_layers=1,
        n_cross_layers=1,
        dropout=0.0,
        max_seq_len=32,
    )
    generated = generate_rows(
        build_generator(config).eval(),
        prompt=(
            [row[:] for row in piece.global_rows[:8]],
            [[voice[:] for voice in row] for row in piece.voice_rows[:8]],
            [[[pair[:] for pair in left] for left in row] for row in piece.pair_rows[:8]],
        ),
        template=piece,
        form="INVENTION",
        voice_count=2,
        max_new_rows=1,
        device=torch.device("cpu"),
        max_context=32,
        temperature=0.0,
        duration_temperature=0.0,
        top_k=4,
        beam_size=8,
    )

    plan = build_development_plan(
        form="INVENTION",
        measures=1,
        voice_count=2,
        key_pc=piece.key_pc,
        mode=piece.mode,
    )
    assert all(
        row[GLOBAL_FIELD_NAMES.index("development")]
        == DEVELOPMENT_TO_ID[plan[0].operation]
        for row in generated[0]
    )


def test_candidate_reranking_rejects_hyperactive_duration_correction() -> None:
    baseline = {
        "slice_count": 128,
        "voice_note_rates": [0.4, 0.5, 0.6],
        "voice_active_rates": [0.8, 0.7, 0.6],
        "repeated_sonority_rate": 0.25,
    }
    common = {
        "invalid_pitch_state_rate": 0.0,
        "voice_crossing_rate": 0.0,
        "parallel_fifth_octave_rate": 0.0,
        "strong_beat_dissonance_rate": 0.0,
        "tonal_outlier_rate": 0.0,
        "strong_beat_tonal_outlier_rate": 0.0,
        "empty_slice_rate": 0.0,
        "voice_stuck_rates": [0.01, 0.01, 0.01],
    }
    balanced = {
        **common,
        "voice_note_rates": [0.42, 0.48, 0.58],
        "voice_active_rates": [0.82, 0.72, 0.62],
        "repeated_sonority_rate": 0.28,
    }
    hyperactive = {
        **common,
        "voice_note_rates": [0.9, 0.9, 0.9],
        "voice_active_rates": [1.0, 1.0, 1.0],
        "repeated_sonority_rate": 0.02,
    }
    overlap = {"source_ngram_overlap_rate": 0.0, "max_contiguous_source_match": 2}

    assert _candidate_score(
        balanced,
        overlap,
        source_baseline=baseline,
    ) > _candidate_score(
        hyperactive,
        overlap,
        source_baseline=baseline,
    )


def test_partita_plan_has_binary_development_and_recap() -> None:
    plan = build_development_plan(
        form="partita",
        measures=12,
        voice_count=4,
        key_pc=0,
        mode=0,
    )

    operations = [step.operation for step in plan]
    assert "BINARY_A" in operations
    assert "BINARY_B" in operations
    assert "RECAP" in operations


@pytest.mark.parametrize(("form", "return_operation"), [("invention", "RECAP"), ("fugue", "STRETTO")])
def test_stretched_plan_keeps_thematic_return_near_the_cadence(
    form: str,
    return_operation: str,
) -> None:
    plan = build_development_plan(
        form=form,
        measures=16,
        voice_count=4,
        key_pc=0,
        mode=0,
    )

    operations = [step.operation for step in plan]
    assert operations[-3:] == [return_operation, "EPISODE", "CADENCE"]


@pytest.mark.parametrize("voice_count", [2, 3, 4, 6])
def test_cadence_targets_scale_to_variable_voice_counts(voice_count: int) -> None:
    previous = [40 + voice * 7 for voice in range(voice_count)]
    dominant = _cadence_target_pitches(
        previous,
        voice_count=voice_count,
        key_pc=0,
        mode=0,
        tonic_stage=False,
        beam_size=96,
    )
    tonic = _cadence_target_pitches(
        dominant,
        voice_count=voice_count,
        key_pc=0,
        mode=0,
        tonic_stage=True,
        beam_size=96,
    )

    assert all(pitch is not None for pitch in dominant)
    assert all(pitch is not None for pitch in tonic)
    assert dominant[0] is not None and dominant[0] % 12 == 7
    assert tonic[0] is not None and tonic[0] % 12 == 0
    assert all(left < right for left, right in zip(dominant, dominant[1:]))
    assert all(left < right for left, right in zip(tonic, tonic[1:]))
    assert {pitch % 12 for pitch in dominant if pitch is not None} <= {2, 7, 11}
    assert {pitch % 12 for pitch in tonic if pitch is not None} <= {0, 4, 7}


def test_generation_metrics_detect_authentic_final_cadence() -> None:
    global_rows = [[0] * len(GLOBAL_FIELD_NAMES) for _ in range(8)]
    for index, row in enumerate(global_rows):
        row[GLOBAL_FIELD_NAMES.index("pos")] = index
        row[GLOBAL_FIELD_NAMES.index("key_pc")] = 0
        row[GLOBAL_FIELD_NAMES.index("local_key_pc")] = 0
        row[GLOBAL_FIELD_NAMES.index("voice_count")] = 3
    voice_rows = []
    for index in range(8):
        pitches = (43, 50, 59) if index < 4 else (48, 55, 64)
        state = 2 if index in {0, 4} else 1
        voice_rows.append(
            [[state, pitch, 0, 4 - (index % 4), int(state == 1), 1] for pitch in pitches]
        )
    pair_rows = [
        [[[0] * len(PAIR_FIELD_NAMES) for _ in range(3)] for _ in range(3)]
        for _ in range(8)
    ]

    report = evaluate_piece_rows(
        global_rows,
        voice_rows,
        pair_rows,
        voice_count=3,
    )

    assert report["penultimate_dominant_sonority"] is True
    assert report["final_tonic_sonority"] is True
    assert report["authentic_cadence_proxy"] is True
    assert report["final_sonority_pitch_classes"] == [0, 4, 7]


def test_wtc_movements_are_labeled_prelude_then_fugue() -> None:
    pieces = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 846-869 The Well-Tempered Clavier Book I/BWV_0846/BWV_0846.xml",
        form="wtc",
        target_voices=4,
        max_voices=6,
        max_bars=2,
    )

    assert [piece.form for piece in pieces[:2]] == ["PRELUDE", "FUGUE"]


def test_dynamic_pitch_beam_rejects_parallel_perfects_across_six_voices() -> None:
    previous = [36, 43, 48, 55, 60, 67]
    options = [
        [PitchOption(pitch + 2, 2.0), PitchOption(pitch + (1 if voice % 2 else -1), 1.0)]
        for voice, pitch in enumerate(previous)
    ]

    selected, _ = select_counterpoint_pitches(
        options,
        previous,
        strong_beat=True,
        strict=True,
    )

    assert all(
        left is None or right is None or left < right
        for left, right in zip(selected, selected[1:])
    )
    assert not any(
        creates_parallel_perfect(previous[left], previous[right], selected[left], selected[right])
        for left in range(6)
        for right in range(left + 1, 6)
    )


def test_two_part_invention_is_register_canonical() -> None:
    piece = parse_musicxml_movements(
        "data/tobis_xml/instrumental-works/keyboard-works/"
        "BWV 772-786 Inventions/BWV_0772/BWV_0772.xml",
        form="invention",
        target_voices=2,
        max_voices=6,
        max_bars=4,
    )[0]
    active_pairs = [
        (row[0][1], row[1][1])
        for row in piece.voice_rows
        if row[0][1] > 0 and row[1][1] > 0
    ]

    assert active_pairs
    assert sum(lower >= upper for lower, upper in active_pairs) / len(active_pairs) < 0.05
