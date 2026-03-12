from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import torch

from src.api.canonical import CanonicalScore, PartInfo, tokens_to_canonical_score
from src.api.render import canonical_score_to_midi, canonical_score_to_musicxml
from src.inference.generate_v1 import GenerationConfig, GenerationResult, generate_v1
from src.tabber import DEFAULT_MAX_FRET, tab_events


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
    return ComposeServiceResult(
        generation=generation,
        score=score,
        score_xml=canonical_score_to_musicxml(score),
        midi=canonical_score_to_midi(score),
        measure_map=_build_measure_map(score),
        event_hit_map=_build_event_hit_map(score),
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


def _build_measure_map(score: CanonicalScore) -> dict[str, str]:
    return {str(measure.index): measure.id for measure in score.measures}


@dataclass(frozen=True)
class _EventSlice:
    event_id: str
    voice_id: int
    start_tick: int
    dur_tick: int
    ordinal: int


def _build_event_hit_map(score: CanonicalScore) -> dict[str, str]:
    if len(score.parts) != 1:
        raise ValueError("compose service supports exactly one part")

    event_hit_map: dict[str, str] = {}
    part = score.parts[0]

    for measure in score.measures:
        slices_by_voice: dict[int, list[_EventSlice]] = {}
        for ordinal, event in enumerate(part.events):
            if event.pitch_midi is None:
                continue

            start_tick = max(event.start_tick, measure.start_tick)
            end_tick = min(event.end_tick, measure.end_tick)
            if start_tick >= end_tick:
                continue

            slices_by_voice.setdefault(event.voice_id, []).append(
                _EventSlice(
                    event_id=event.id,
                    voice_id=event.voice_id,
                    start_tick=start_tick,
                    dur_tick=end_tick - start_tick,
                    ordinal=ordinal,
                )
            )

        for voice_id, slices in slices_by_voice.items():
            cursor = measure.start_tick
            beat_index = 0

            for slice_ in sorted(slices, key=lambda item: (item.start_tick, item.ordinal)):
                if slice_.start_tick > cursor:
                    beat_index += 1
                    cursor = slice_.start_tick

                event_hit_map[_to_hit_key(measure.index, voice_id, beat_index, 0)] = slice_.event_id
                beat_index += 1
                cursor = max(cursor, slice_.start_tick + slice_.dur_tick)

    return event_hit_map


def _to_hit_key(bar_index: int, voice_index: int, beat_index: int, note_index: int) -> str:
    return f"{bar_index}|{voice_index}|{beat_index}|{note_index}"
