import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from music21 import meter, note, stream

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tokens.eventizer import eventize_musicxml
from src.tokens.roundtrip import tokens_to_score
from src.tokens.tokenizer import parse_voice_event
from src.tokens.validator import validate_harm_tokens

TPQ = 24


def _note_events_by_voice(score: stream.Score) -> Dict[int, List[Tuple[int, int, int]]]:
    by_voice: Dict[int, List[Tuple[int, int, int]]] = {}
    for voice_idx, part in enumerate(score.parts):
        events: List[Tuple[int, int, int]] = []
        flat_part = part.flatten()
        for el in flat_part.notesAndRests:
            if el.isRest:
                continue
            if el.isChord:
                pitch = max(p.midi for p in el.pitches)
            else:
                pitch = el.pitch.midi
            onset = int(round(el.offset * TPQ))
            duration = int(round(el.duration.quarterLength * TPQ))
            events.append((onset, duration, pitch))
        by_voice[voice_idx] = events
    return by_voice


def _assert_events_match_with_tick_tolerance(
    expected: Dict[int, List[Tuple[int, int, int]]],
    actual: Dict[int, List[Tuple[int, int, int]]],
    *,
    tick_tolerance: int = 1,
) -> None:
    assert set(expected.keys()) == set(actual.keys())
    for voice_idx in expected:
        exp_events = expected[voice_idx]
        got_events = actual[voice_idx]
        assert len(exp_events) == len(got_events)
        for (exp_onset, exp_dur, exp_pitch), (got_onset, got_dur, got_pitch) in zip(
            exp_events, got_events
        ):
            assert got_pitch == exp_pitch
            assert abs(got_onset - exp_onset) <= tick_tolerance
            assert abs(got_dur - exp_dur) <= tick_tolerance


def _build_random_score(seed: int, *, bars: int = 3, voices: int = 3) -> stream.Score:
    rng = random.Random(seed)
    score = stream.Score()
    for voice_idx in range(voices):
        part = stream.Part(id=f"Voice{voice_idx}")
        part.append(meter.TimeSignature("4/4"))
        base_pitch = 48 + voice_idx * 7
        for _ in range(bars):
            remaining = 4.0
            while remaining > 1e-9:
                options = [dur for dur in (0.5, 1.0, 1.5, 2.0) if dur <= remaining + 1e-9]
                quarter_len = rng.choice(options)
                if rng.random() < 0.3:
                    part.append(note.Rest(quarterLength=quarter_len))
                else:
                    pitch = base_pitch + rng.randint(0, 12)
                    part.append(note.Note(midi=pitch, quarterLength=quarter_len))
                remaining = round(remaining - quarter_len, 6)
        score.insert(0, part)
    return score


def _write_sustained_reference_xml(tmp_path: Path) -> Path:
    score = stream.Score()

    bass = stream.Part(id="Bass")
    bass.append(meter.TimeSignature("4/4"))
    bass.append(note.Note("C3", quarterLength=2.0))
    bass.append(note.Rest(quarterLength=2.0))

    upper = stream.Part(id="Upper")
    upper.append(meter.TimeSignature("4/4"))
    upper.append(note.Note("G3", quarterLength=1.0))
    upper.append(note.Note("A3", quarterLength=1.0))
    upper.append(note.Rest(quarterLength=2.0))

    score.insert(0, bass)
    score.insert(0, upper)

    out_path = tmp_path / "sustained_reference.xml"
    score.write("musicxml", fp=str(out_path))
    return out_path


@pytest.mark.parametrize("seed", [7, 17, 27])
def test_roundtrip_pitch_reconstruction_on_random_bars(seed: int, tmp_path: Path) -> None:
    score = _build_random_score(seed)
    xml_path = tmp_path / f"random_seed_{seed}.xml"
    score.write("musicxml", fp=str(xml_path))

    tokens, _ = eventize_musicxml(str(xml_path), voice_mode="parts")
    harm_errors = validate_harm_tokens(tokens)
    assert not harm_errors

    reconstructed = tokens_to_score(tokens, tpq=TPQ)
    _assert_events_match_with_tick_tolerance(
        _note_events_by_voice(score),
        _note_events_by_voice(reconstructed),
        tick_tolerance=1,
    )


def test_harm_consistency_uses_lowest_active_reference_pitch(tmp_path: Path) -> None:
    xml_path = _write_sustained_reference_xml(tmp_path)
    tokens, _ = eventize_musicxml(str(xml_path), voice_mode="parts")

    harm_errors = validate_harm_tokens(tokens)
    assert not harm_errors

    pos_idx = tokens.index("POS_24")
    found_upper = False
    idx = pos_idx + 1
    while idx < len(tokens) and tokens[idx] != "BAR" and not tokens[idx].startswith("POS_"):
        if not tokens[idx].startswith("VOICE_"):
            idx += 1
            continue
        event, next_idx = parse_voice_event(tokens, idx)
        if event.voice == 1:
            found_upper = True
            assert event.mel_int == 2
            assert event.harm_oct == 0
            assert event.harm_class == 9
        idx = next_idx

    assert found_upper
