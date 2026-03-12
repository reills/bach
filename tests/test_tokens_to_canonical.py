from src.api.canonical import PartInfo
from src.api.canonical.from_tokens import tokens_to_canonical_score


def test_tokens_to_canonical_score_reconstructs_simple_monophonic_bar():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_0_60",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "POS_24",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_2",
    ]

    first = tokens_to_canonical_score(tokens)
    second = tokens_to_canonical_score(tokens)

    assert first.header.time_sig_map == {0: "4/4"}
    assert first.header.key_sig_map == {0: "C"}
    assert len(first.measures) == 1
    assert first.measures[0].length_ticks == 96
    assert len(first.parts) == 1
    assert [event.start_tick for event in first.parts[0].events] == [0, 24]
    assert [event.pitch_midi for event in first.parts[0].events] == [60, 62]
    assert [event.voice_id for event in first.parts[0].events] == [0, 0]
    assert [event.id for event in first.parts[0].events] == [event.id for event in second.parts[0].events]
    assert first.measures[0].id == second.measures[0].id


def test_tokens_to_canonical_score_preserves_cross_bar_sustain():
    tokens = [
        "BAR",
        "TIME_SIG_1_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_0_60",
        "VOICE_0",
        "DUR_48",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "BAR",
        "TIME_SIG_1_4",
        "KEY_C",
    ]

    score = tokens_to_canonical_score(
        tokens,
        tpq=24,
        part_info=PartInfo(
            id="guitar",
            instrument="classical_guitar",
            tuning=[40, 45, 50, 55, 59, 64],
            midi_program=24,
        ),
    )

    assert [measure.start_tick for measure in score.measures] == [0, 24]
    assert [measure.length_ticks for measure in score.measures] == [24, 24]
    assert len(score.parts[0].events) == 1

    event = score.parts[0].events[0]
    assert event.start_tick == 0
    assert event.dur_tick == 48
    assert event.end_tick == 48
    assert event.pitch_midi == 60
    assert score.measure_for_tick(24).id == score.measures[1].id
