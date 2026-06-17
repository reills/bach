from src.music.counterpoint import evaluate_counterpoint_tokens, pitched_events_from_tokens


def test_pitched_events_reconstruct_actual_voice_pitches_from_tokens():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_48",
        "ABS_VOICE_1_55",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_-1",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]

    events = pitched_events_from_tokens(tokens)

    assert [(event.voice, event.start_tick, event.dur_tick, event.pitch) for event in events] == [
        (0, 0, 24, 50),
        (1, 0, 24, 54),
    ]


def test_counterpoint_counts_parallel_fifths_and_static_voice_rate():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_48",
        "ABS_VOICE_1_55",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "POS_24",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "POS_48",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]

    metrics = evaluate_counterpoint_tokens(tokens)

    assert metrics.parallel_fifths == 1
    assert metrics.static_voice_rate == 0.5
    assert metrics.avg_active_voices == 2.0
    assert metrics.monophonic_position_rate == 0.0


def test_counterpoint_counts_direct_octave_and_dissonance():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_60",
        "ABS_VOICE_1_61",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "POS_24",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+12",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_+23",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]

    metrics = evaluate_counterpoint_tokens(tokens)

    assert metrics.dissonance_on_strong_beat == 1
    assert metrics.direct_octaves == 1
    assert metrics.unresolved_dissonances == 0


def test_counterpoint_counts_spacing_violation():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_60",
        "ABS_VOICE_1_84",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]

    metrics = evaluate_counterpoint_tokens(tokens)

    assert metrics.spacing_violations == 1


def test_counterpoint_counts_voice_crossing():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_60",
        "ABS_VOICE_1_72",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+20",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_-2",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]

    metrics = evaluate_counterpoint_tokens(tokens)

    assert metrics.voice_crossings == 1


def test_counterpoint_exposes_harmonic_metadata_mismatches():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_60",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_4",
    ]

    metrics = evaluate_counterpoint_tokens(tokens)

    assert metrics.harmonic_metadata_mismatches is not None
    assert metrics.harmonic_metadata_mismatches >= 1
