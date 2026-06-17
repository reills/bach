from __future__ import annotations

from src.emi.structured_invention import StructuredInventionConfig, compose_structured_invention


def test_structured_invention_has_subject_answer_return_and_arch() -> None:
    composition = compose_structured_invention(StructuredInventionConfig(key="D minor", measures=16, seed=5))
    diagnostics = composition.diagnostics

    assert diagnostics["engine"] == "structured_invention_v1"
    assert diagnostics["form"][:2] == ["subject", "answer"]
    assert diagnostics["form"][-1] == "final_cadence"
    assert diagnostics["subjectBars"] == [0, 1, 12, 13]
    assert diagnostics["highestUpperBar"] in {9, 10, 11}
    assert len(composition.score.measures) == 16
    assert composition.score.parts[0].info.instrument == "piano"
    assert {event.voice_id for event in composition.score.parts[0].events} == {0, 1}


def test_structured_invention_keeps_voices_ordered_on_sixteenth_grid() -> None:
    composition = compose_structured_invention(StructuredInventionConfig(key="C", measures=12, seed=9))
    score = composition.score
    events = score.parts[0].events

    for tick in range(0, score.total_ticks, 6):
        sounding = {
            event.voice_id: event.pitch_midi
            for event in events
            if event.pitch_midi is not None and event.start_tick <= tick < event.end_tick
        }
        if 0 in sounding and 1 in sounding:
            assert sounding[0] < sounding[1]
            assert sounding[1] - sounding[0] <= 31
