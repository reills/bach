import sys
from pathlib import Path

import pytest
from music21 import meter, note, stream

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tokens.eventizer import eventize_musicxml
from src.tokens.intervals import (
    IntervalRepairStats,
    compute_harmonic_interval,
    compute_melodic_interval,
    compute_reference_pitch,
    harmonic_tokens_for_pitch,
)


def test_compute_reference_pitch():
    assert compute_reference_pitch([67, 60, 64]) == 60
    assert compute_reference_pitch([]) is None


def test_melodic_interval_qa_raises_out_of_range():
    with pytest.raises(ValueError, match="MEL_INT12 out of range"):
        compute_melodic_interval(96, 60, qa_mode=True)


def test_melodic_interval_production_clamps_and_counts():
    stats = IntervalRepairStats()
    mel_int = compute_melodic_interval(96, 60, qa_mode=False, stats=stats)
    assert mel_int == 24
    assert stats.mel_clamped == 1


def test_harmonic_interval_qa_raises_out_of_range():
    with pytest.raises(ValueError, match="HARM_OCT out of range"):
        compute_harmonic_interval(96, 24, qa_mode=True)


def test_harmonic_interval_production_clamps_and_counts():
    stats = IntervalRepairStats()
    octv, klass = compute_harmonic_interval(96, 24, qa_mode=False, stats=stats)
    assert octv == 4
    assert klass == 0
    assert stats.harm_oct_clamped == 1

    harm_oct_tok, harm_cls_tok = harmonic_tokens_for_pitch(96, 24, qa_mode=False, stats=stats)
    assert harm_oct_tok == "HARM_OCT_4"
    assert harm_cls_tok == "HARM_CLASS_0"
    assert stats.harm_oct_clamped == 2


def _write_large_leap_xml(tmp_path: Path) -> Path:
    score = stream.Score()
    part = stream.Part()
    part.append(meter.TimeSignature("4/4"))
    part.append(note.Note("C4", quarterLength=1.0))
    part.append(note.Note("C7", quarterLength=1.0))
    score.insert(0, part)
    out_path = tmp_path / "large_leap.xml"
    score.write("musicxml", fp=str(out_path))
    return out_path


def test_eventizer_production_repairs_large_mel_interval(tmp_path):
    xml_path = _write_large_leap_xml(tmp_path)
    tokens, meta = eventize_musicxml(str(xml_path), voice_mode="parts")
    assert "ABS_VOICE_0_96" in tokens
    assert "MEL_INT12_0" in tokens
    assert "interval_repairs=mel:1,harm_oct:0" in meta.mapping_note


def test_eventizer_qa_mode_raises_on_large_mel_interval(tmp_path):
    xml_path = _write_large_leap_xml(tmp_path)
    with pytest.raises(ValueError, match="MEL_INT12 out of range"):
        eventize_musicxml(str(xml_path), voice_mode="parts", qa_mode=True)
