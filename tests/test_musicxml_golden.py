from pathlib import Path

from src.api.canonical import (
    CanonicalScore,
    Event,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
)
from src.api.render.musicxml import canonical_score_to_musicxml


def test_canonical_score_to_musicxml_matches_backend_contract_golden():
    score = CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "1/4", 24: "1/4"}),
        measures=[
            Measure(id="measure-intro", index=0, start_tick=0, length_ticks=24),
            Measure(id="measure-outro", index=1, start_tick=24, length_ticks=24),
        ],
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
                        id="note-open-high-e",
                        start_tick=0,
                        dur_tick=12,
                        voice_id=0,
                        pitch_midi=64,
                        fingering=GuitarFingering(string_index=5, fret=0),
                    ),
                    Event(
                        id="note-tied-low-e",
                        start_tick=12,
                        dur_tick=24,
                        voice_id=0,
                        pitch_midi=43,
                        fingering=GuitarFingering(string_index=0, fret=3),
                    ),
                ],
            )
        ],
    )

    xml_text = canonical_score_to_musicxml(score)
    expected_path = Path(__file__).parent / "fixtures" / "musicxml" / "canonical_bridge.xml"

    assert xml_text.strip() == expected_path.read_text().strip()
