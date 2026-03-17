import xml.etree.ElementTree as ET
from pathlib import Path

from music21.midi import MidiFile

from src.api.canonical import Event
from src.api.canonical.from_tokens import _ensure_unique_event_ids
from src.api.compose_service import compose_baseline
from src.inference.generate_v1 import GenerationConfig, GenerationResult

XML_NS = "http://www.w3.org/XML/1998/namespace"


def test_compose_baseline_runs_transformation_pipeline_with_stubbed_generation():
    generated_tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_3_60",
        "VOICE_3",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "POS_72",
        "VOICE_3",
        "DUR_48",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_2",
        "BAR",
        "TIME_SIG_1_4",
        "KEY_C",
    ]
    captured: dict[str, object] = {}

    def fake_generator(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
        device="cpu",
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["seed_tokens"] = list(seed_tokens)
        captured["generation_config"] = generation_config
        captured["vocab_path"] = vocab_path
        captured["device"] = device
        return GenerationResult(
            ids=[11, 12, 13],
            tokens=generated_tokens,
            stopped_on_eos=False,
        )

    config = GenerationConfig(max_length=32, top_p=1.0)
    result = compose_baseline(
        Path("/tmp/fake-checkpoint.pt"),
        seed_tokens=["KEY_C", "STYLE_baroque"],
        generation_config=config,
        vocab_path=Path("/tmp/fake-vocab.json"),
        generator=fake_generator,
    )

    assert captured == {
        "checkpoint_path": Path("/tmp/fake-checkpoint.pt"),
        "seed_tokens": ["KEY_C", "STYLE_baroque"],
        "generation_config": config,
        "vocab_path": Path("/tmp/fake-vocab.json"),
        "device": "cpu",
    }
    assert result.generation.tokens == generated_tokens

    score = result.score
    assert [measure.index for measure in score.measures] == [0, 1]
    assert result.measure_map == {
        "0": score.measures[0].id,
        "1": score.measures[1].id,
    }

    first_event, second_event = score.parts[0].events
    assert first_event.fingering is not None
    assert second_event.fingering is not None
    assert result.event_hit_map == {
        "0|0|0|0": first_event.id,
        "0|0|2|0": second_event.id,
        "1|0|0|0": second_event.id,
    }

    root = ET.fromstring(result.score_xml)
    measures = root.findall("./part/measure")
    assert [measure.attrib[f"{{{XML_NS}}}id"] for measure in measures] == [
        score.measures[0].id,
        score.measures[1].id,
    ]

    note_els = root.findall(".//note")
    pitched_note_els = [note for note in note_els if note.find("./pitch") is not None]
    assert [note.attrib[f"{{{XML_NS}}}id"] for note in pitched_note_els] == [
        first_event.id,
        second_event.id,
        second_event.id,
    ]
    assert all(note.findtext("./notations/technical/string") is not None for note in pitched_note_els)
    assert all(note.findtext("./notations/technical/fret") is not None for note in pitched_note_els)

    midi_file = MidiFile()
    midi_file.readstr(result.midi)
    assert result.midi.startswith(b"MThd")
    assert midi_file.ticksPerQuarterNote == score.header.tpq


def test_compose_baseline_trims_incomplete_generated_voice_event_suffix():
    generated_tokens = [
        "KEY_C",
        "MEAS_2",
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "ABS_VOICE_0_60",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "ABS_VOICE_0_62",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
    ]

    def fake_generator(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
        device="cpu",
    ):
        return GenerationResult(
            ids=list(range(len(generated_tokens))),
            tokens=generated_tokens,
            stopped_on_eos=False,
        )

    result = compose_baseline(
        Path("/tmp/fake-checkpoint.pt"),
        seed_tokens=["KEY_C"],
        generation_config=GenerationConfig(max_length=32),
        generator=fake_generator,
    )

    assert result.generation.tokens == generated_tokens[:-4]
    assert [measure.index for measure in result.score.measures] == [0, 1]
    assert len(result.score.parts[0].events) == 1


def test_compose_baseline_skips_generated_events_without_pos_or_anchor():
    generated_tokens = [
        "KEY_C",
        "MEAS_4",
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "POS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "ABS_VOICE_0_60",
        "POS_24",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]

    def fake_generator(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
        device="cpu",
    ):
        return GenerationResult(
            ids=list(range(len(generated_tokens))),
            tokens=generated_tokens,
            stopped_on_eos=False,
        )

    result = compose_baseline(
        Path("/tmp/fake-checkpoint.pt"),
        seed_tokens=["KEY_C", "MEAS_4"],
        generation_config=GenerationConfig(max_length=32),
        generator=fake_generator,
    )

    assert [measure.index for measure in result.score.measures] == [0]
    assert len(result.score.parts[0].events) == 1
    assert result.score.parts[0].events[0].start_tick == 24


def test_ensure_unique_event_ids_rewrites_duplicates():
    events = [
        Event(id="dup", start_tick=0, dur_tick=24, pitch_midi=60, voice_id=0),
        Event(id="dup", start_tick=24, dur_tick=24, pitch_midi=62, voice_id=0),
    ]

    deduped = _ensure_unique_event_ids(events, "part-0")

    assert deduped[0].id == "dup"
    assert deduped[1].id != "dup"
    assert deduped[0].id != deduped[1].id
