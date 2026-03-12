import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import torch

from src.api.canonical import CanonicalScore, PartInfo, tokens_to_canonical_score
from src.api.render import canonical_score_to_midi, canonical_score_to_musicxml
from src.inference.generate_v1 import GenerationConfig, GenerationResult, generate_v1
from src.tabber import DEFAULT_MAX_FRET, tab_events

XML_NS = "http://www.w3.org/XML/1998/namespace"


@dataclass(frozen=True)
class ComposeServiceResult:
    generation: GenerationResult
    score: CanonicalScore
    score_xml: str
    midi: bytes
    measure_map: dict[str, str]
    event_hit_map: dict[str, str]


def compose_baseline(
    checkpoint_path: str | Path,
    *,
    seed_tokens: Sequence[str | int],
    generation_config: GenerationConfig,
    vocab_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    tpq: int = 24,
    part_info: PartInfo | None = None,
    max_fret: int = DEFAULT_MAX_FRET,
    generator: Callable[..., GenerationResult] = generate_v1,
) -> ComposeServiceResult:
    generation = generator(
        checkpoint_path,
        seed_tokens=seed_tokens,
        generation_config=generation_config,
        vocab_path=vocab_path,
        device=device,
    )
    score = tokens_to_canonical_score(generation.tokens, tpq=tpq, part_info=part_info)
    score = _tab_score(score, max_fret=max_fret)
    score_xml = canonical_score_to_musicxml(score)
    return ComposeServiceResult(
        generation=generation,
        score=score,
        score_xml=score_xml,
        midi=canonical_score_to_midi(score),
        measure_map=build_measure_map(score_xml),
        event_hit_map=build_event_hit_map(score_xml),
    )


def _tab_score(score: CanonicalScore, *, max_fret: int) -> CanonicalScore:
    if len(score.parts) != 1:
        raise ValueError("compose service supports exactly one part")

    part = score.parts[0]
    tabbed_part = replace(
        part,
        events=tab_events(
            part.events,
            tuning=part.info.tuning,
            max_fret=max_fret,
        ),
    )
    return replace(score, parts=[tabbed_part])


def build_measure_map(score: CanonicalScore | str) -> dict[str, str]:
    root = _musicxml_root(score)
    measure_map: dict[str, str] = {}

    for bar_index, measure_el in enumerate(_measure_elements(root)):
        measure_id = measure_el.attrib.get(f"{{{XML_NS}}}id")
        if measure_id is None:
            measure_id = f"measure-{bar_index + 1}"
        measure_map[str(bar_index)] = measure_id

    return measure_map


@dataclass
class _VoiceBeatState:
    beat_index: int = -1
    note_index: int = 0


def build_event_hit_map(score: CanonicalScore | str) -> dict[str, str]:
    root = _musicxml_root(score)
    event_hit_map: dict[str, str] = {}

    for bar_index, measure_el in enumerate(_measure_elements(root)):
        states_by_voice: dict[int, _VoiceBeatState] = {}
        for note_el in measure_el.findall("./note"):
            voice_index = _voice_index_for_note(note_el)
            state = states_by_voice.setdefault(voice_index, _VoiceBeatState())
            if note_el.find("./chord") is None:
                state.beat_index += 1
                state.note_index = 0
            else:
                if state.beat_index < 0:
                    state.beat_index = 0
                    state.note_index = 0
                else:
                    state.note_index += 1

            event_id = note_el.attrib.get(f"{{{XML_NS}}}id")
            if event_id is None:
                continue
            event_hit_map[_to_hit_key(bar_index, voice_index, state.beat_index, state.note_index)] = event_id

    return event_hit_map


def _musicxml_root(score: CanonicalScore | str) -> ET.Element:
    if isinstance(score, CanonicalScore):
        if len(score.parts) != 1:
            raise ValueError("compose service supports exactly one part")
        score_xml = canonical_score_to_musicxml(score)
    else:
        score_xml = score
    return ET.fromstring(score_xml)


def _measure_elements(root: ET.Element) -> list[ET.Element]:
    return root.findall("./part/measure")


def _voice_index_for_note(note_el: ET.Element) -> int:
    voice_text = note_el.findtext("./voice")
    if voice_text is None:
        return 0
    return max(int(voice_text) - 1, 0)


def _to_hit_key(bar_index: int, voice_index: int, beat_index: int, note_index: int) -> str:
    return f"{bar_index}|{voice_index}|{beat_index}|{note_index}"
