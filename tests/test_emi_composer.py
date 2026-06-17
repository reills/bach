from __future__ import annotations

from pathlib import Path

from src.emi.composer import EMI_ENGINE_VERSION, EmiComposerConfig, compose_emi
from src.emi.fragments import Fragment, fragment_to_jsonl


def test_compose_emi_returns_ordered_polyphonic_canonical_score() -> None:
    composition = compose_emi(
        EmiComposerConfig(
            key="D minor",
            measures=6,
            texture=4,
            seed=7,
        )
    )

    score = composition.score
    assert len(score.measures) == 6
    assert score.header.key_sig_map == {0: "Dm"}
    assert {event.voice_id for event in score.parts[0].events} == {0, 1, 2, 3}
    assert composition.diagnostics["emiVersion"] == EMI_ENGINE_VERSION
    assert composition.diagnostics["rolePlan"][-1] == "CADENCE"
    assert composition.diagnostics["speacLabels"][-1] == "C"

    for tick in range(0, score.total_ticks, 12):
        sounding = [
            (event.voice_id, event.pitch_midi)
            for event in score.parts[0].events
            if event.pitch_midi is not None and event.start_tick <= tick < event.end_tick
        ]
        pitches = [pitch for _, pitch in sorted(sounding)]
        assert pitches == sorted(pitches)


def test_compose_emi_can_use_fragment_database(tmp_path: Path) -> None:
    fragment = Fragment(
        id="frag-subject",
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
        melodic_intervals=[2, 2, 1, 2, -2, -1, -2],
        rhythm_steps=[2, 2, 2, 2],
        vertical_intervals=[12, 10],
        state_pattern=[2, 1, 2, 1],
        contour_hash="abc123",
        fingerprint="fp123",
    )
    fragment_path = tmp_path / "fragments.jsonl"
    fragment_path.write_text(fragment_to_jsonl(fragment) + "\n", encoding="utf-8")

    composition = compose_emi(
        EmiComposerConfig(
            key="C",
            measures=3,
            texture=2,
            seed=1,
            fragment_path=fragment_path,
        )
    )

    assert composition.diagnostics["fragmentCount"] == 1
    assert "frag-subject" in composition.diagnostics["usedFragmentIds"]
