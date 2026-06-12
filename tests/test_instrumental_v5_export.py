from __future__ import annotations

from scripts.generate_instrumental_v5 import _arrange_score_for_instrument, _score_with_tempo
from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.render.musicxml import canonical_score_to_musicxml


def _score() -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}, tempo_map={0: 92}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(id="P1", instrument="piano", midi_program=0),
                events=[
                    Event(id="n0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=52),
                    Event(id="n1", start_tick=0, dur_tick=24, voice_id=1, pitch_midi=64),
                    Event(id="n2", start_tick=24, dur_tick=24, voice_id=0, pitch_midi=55),
                    Event(id="n3", start_tick=24, dur_tick=24, voice_id=1, pitch_midi=67),
                ],
            )
        ],
    )


def test_arrange_score_for_classical_guitar_adds_tuning_and_fingerings() -> None:
    arranged = _arrange_score_for_instrument(_score(), instrument="classical_guitar")

    part = arranged.parts[0]
    assert part.info.instrument == "classical_guitar"
    assert part.info.tuning == [40, 45, 50, 55, 59, 64]
    assert all(event.fingering is not None for event in part.events if event.pitch_midi is not None)
    xml = canonical_score_to_musicxml(arranged)
    assert "<staff-tuning" in xml
    assert "<fret>" in xml


def test_score_with_tempo_updates_header_without_changing_events() -> None:
    score = _score()
    updated = _score_with_tempo(score, tempo=76)

    assert updated.header.tempo_map == {0: 76}
    assert updated.parts[0].events == score.parts[0].events
