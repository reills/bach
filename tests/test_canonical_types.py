import pytest

from src.api.canonical import (
    CanonicalScore,
    Event,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
)


def make_header() -> ScoreHeader:
    return ScoreHeader(
        tpq=24,
        key_sig_map={0: "C"},
        time_sig_map={0: "4/4"},
        tempo_map={0: 90},
    )


def make_measures() -> list[Measure]:
    return [
        Measure(id="m1", index=0, start_tick=0, length_ticks=96),
        Measure(id="m2", index=1, start_tick=96, length_ticks=96),
    ]


def test_canonical_score_allows_cross_barline_events_and_per_part_voice_ids():
    score = CanonicalScore(
        header=make_header(),
        measures=make_measures(),
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=[40, 45, 50, 55, 59, 64],
                    midi_program=24,
                ),
                events=[
                    Event(
                        id="ev-1",
                        start_tick=72,
                        dur_tick=48,
                        pitch_midi=52,
                        velocity=72,
                        voice_id=0,
                        fingering=GuitarFingering(string_index=0, fret=12),
                    ),
                    Event(
                        id="ev-2",
                        start_tick=96,
                        dur_tick=24,
                        pitch_midi=55,
                        voice_id=1,
                    ),
                ],
            ),
            Part(
                info=PartInfo(id="bass", instrument="continuo"),
                events=[
                    Event(
                        id="ev-3",
                        start_tick=0,
                        dur_tick=96,
                        pitch_midi=36,
                        voice_id=0,
                    )
                ],
            ),
        ],
    )

    assert score.total_ticks == 192
    assert score.measure_for_tick(72).id == "m1"
    assert score.measure_for_tick(96).id == "m2"
    assert score.parts[0].events[0].end_tick == 120
    assert score.parts[0].events[0].fingering == GuitarFingering(string_index=0, fret=12)
    assert score.parts[1].events[0].voice_id == 0


def test_canonical_score_rejects_duplicate_event_ids():
    with pytest.raises(ValueError, match="duplicate event id"):
        CanonicalScore(
            header=make_header(),
            measures=make_measures(),
            parts=[
                Part(
                    info=PartInfo(id="guitar", instrument="classical_guitar"),
                    events=[
                        Event(id="ev-1", start_tick=0, dur_tick=24, pitch_midi=60, voice_id=0),
                        Event(id="ev-1", start_tick=24, dur_tick=24, pitch_midi=62, voice_id=0),
                    ],
                )
            ],
        )


def test_canonical_score_rejects_non_contiguous_measures():
    with pytest.raises(ValueError, match="measures must be contiguous"):
        CanonicalScore(
            header=make_header(),
            measures=[
                Measure(id="m1", index=0, start_tick=0, length_ticks=96),
                Measure(id="m2", index=1, start_tick=120, length_ticks=96),
            ],
            parts=[
                Part(
                    info=PartInfo(id="guitar", instrument="classical_guitar"),
                    events=[Event(id="ev-1", start_tick=0, dur_tick=24, pitch_midi=60, voice_id=0)],
                )
            ],
        )


def test_event_rejects_fingering_on_rest():
    with pytest.raises(ValueError, match="rests cannot carry fingering data"):
        Event(
            id="rest-1",
            start_tick=0,
            dur_tick=24,
            pitch_midi=None,
            voice_id=0,
            fingering=GuitarFingering(string_index=2, fret=3),
        )
