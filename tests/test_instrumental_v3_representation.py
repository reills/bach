from pathlib import Path

from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
from src.instrumental_v3.metrics import evaluate_slices
from src.instrumental_v3.representation import FIELD_NAMES, MAX_VOICES, parse_musicxml_to_piece, piece_to_canonical_score

INVENTION_772 = Path("data/tobis_xml/instrumental-works/keyboard-works/BWV 772-786 Inventions/BWV_0772/BWV_0772.xml")


def test_parse_invention_to_compound_slices():
    piece = parse_musicxml_to_piece(INVENTION_772, max_bars=4)
    assert piece.piece_id == "BWV_0772"
    assert piece.steps_per_bar > 0
    assert len(piece.slices) == piece.steps_per_bar * 4
    assert len(piece.slices[0].values) == len(FIELD_NAMES)
    assert {s.field("voice_count") for s in piece.slices} == {MAX_VOICES}
    assert any(s.field("v0_state") == 2 for s in piece.slices)
    assert any(s.field("v1_state") == 2 for s in piece.slices)
    assert any(s.field("vertical_interval") > 0 for s in piece.slices)


def test_compound_slices_export_musicxml_and_midi():
    piece = parse_musicxml_to_piece(INVENTION_772, max_bars=2)
    score = piece_to_canonical_score(piece)
    xml_text = canonical_score_to_musicxml(score)
    midi_bytes = canonical_score_to_midi(score)
    assert "score-partwise" in xml_text
    assert "<part-name>piano</part-name>" in xml_text
    assert len(midi_bytes) > 100


def test_metrics_report_counterpoint_fields():
    piece = parse_musicxml_to_piece(INVENTION_772, max_bars=4)
    report = evaluate_slices(piece.slices)
    assert report.slice_count == len(piece.slices)
    assert 0 <= report.repeated_sonority_rate <= 1
    assert 0 <= report.parallel_fifth_octave_rate <= 1
    assert report.vertical_interval_distribution
