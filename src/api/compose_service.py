from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence
from uuid import uuid4

import torch

from src.api.canonical import CanonicalScore, PartInfo, tokens_to_canonical_score
from src.api.canonical.from_tokens import ParseDiagnostics
from src.api.render import (
    canonical_score_to_midi,
    canonical_score_to_musicxml,
    canonical_score_to_standard_musicxml,
    canonical_score_to_tab_musicxml,
)
from src.inference.generate_v1 import GenerationConfig, GenerationResult, generate_v1
from src.tabber import DEFAULT_MAX_FRET, tab_events
from src.tokens.roundtrip import parse_time_sig_token, parse_token_int
from src.tokens.tokenizer import parse_voice_event
from src.tokens.validator import validate_harm_tokens

# Lowest open string on a standard-tuned guitar (low E2)
_GUITAR_LOWEST_MIDI = 40

XML_NS = "http://www.w3.org/XML/1998/namespace"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAILURE_DIR = REPO_ROOT / "out" / "compose_failures"


@dataclass(frozen=True, init=False)
class ComposeServiceResult:
    generation: GenerationResult
    score: CanonicalScore
    document: "ScoreDocumentBundleExport"
    midi: bytes
    render_mode: Literal["guitar", "piano"] = "guitar"

    def __init__(
        self,
        *,
        generation: GenerationResult,
        score: CanonicalScore,
        document: "ScoreDocumentBundleExport" | None = None,
        midi: bytes,
        render_mode: Literal["guitar", "piano"] = "guitar",
        score_xml: str | None = None,
        measure_map: dict[str, str] | None = None,
        event_hit_map: dict[str, str] | None = None,
    ) -> None:
        if document is None:
            if score_xml is None or measure_map is None or event_hit_map is None:
                raise TypeError(
                    "ComposeServiceResult requires either document or score_xml/measure_map/event_hit_map"
                )
            legacy_view = RenderViewExport(
                xml=score_xml,
                measure_map=measure_map,
                event_hit_map=event_hit_map,
            )
            views = {"score": legacy_view}
            if render_mode == "guitar":
                views["tab"] = legacy_view
            document = ScoreDocumentBundleExport(
                instrument_mode=render_mode,
                views=views,
            )

        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "midi", midi)
        object.__setattr__(self, "render_mode", render_mode)

    @property
    def score_xml(self) -> str:
        return self.document.score_xml

    @property
    def measure_map(self) -> dict[str, str]:
        return self.document.measure_map

    @property
    def event_hit_map(self) -> dict[str, str]:
        return self.document.event_hit_map


@dataclass(frozen=True)
class RenderViewExport:
    xml: str
    measure_map: dict[str, str]
    event_hit_map: dict[str, str]


@dataclass(frozen=True)
class ScoreDocumentBundleExport:
    instrument_mode: Literal["guitar", "piano"]
    views: dict[str, RenderViewExport]

    @property
    def primary_view(self) -> RenderViewExport:
        return self.views.get("tab") or self.views["score"]

    @property
    def score_xml(self) -> str:
        return self.primary_view.xml

    @property
    def measure_map(self) -> dict[str, str]:
        return self.primary_view.measure_map

    @property
    def event_hit_map(self) -> dict[str, str]:
        return self.primary_view.event_hit_map


class ComposeDiagnosticsError(ValueError):
    def __init__(
        self,
        *,
        stage: str,
        message: str,
        report_path: Path | None = None,
    ) -> None:
        self.stage = stage
        self.message = message
        self.report_path = report_path
        detail = f"compose failed during {stage}: {message}"
        if report_path is not None:
            detail += f" [report: {_display_path(report_path)}]"
        super().__init__(detail)


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
    render_mode: Literal["guitar", "piano"] = "guitar",
    generator: Callable[..., GenerationResult] = generate_v1,
) -> ComposeServiceResult:
    generation: GenerationResult | None = None
    trimmed_generation: GenerationResult | None = None
    parse_diagnostics = ParseDiagnostics()
    score: CanonicalScore | None = None
    exported: ScoreDocumentBundleExport | None = None
    resolved_render_mode = render_mode

    try:
        generation = generator(
            checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
            vocab_path=vocab_path,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - exercised via route path
        raise _compose_failure(
            stage="generation",
            exc=exc,
            checkpoint_path=checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
        ) from exc

    try:
        trimmed_generation = _trim_incomplete_generation(generation, tpq=tpq)
    except Exception as exc:
        raise _compose_failure(
            stage="trim",
            exc=exc,
            checkpoint_path=checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
            generation=generation,
        ) from exc

    try:
        score = tokens_to_canonical_score(
            trimmed_generation.tokens,
            tpq=tpq,
            part_info=part_info,
            ignore_invalid_events=True,
            diagnostics=parse_diagnostics,
        )
    except Exception as exc:
        raise _compose_failure(
            stage="parse",
            exc=exc,
            checkpoint_path=checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
            generation=generation,
            effective_tokens=trimmed_generation.tokens,
            parse_diagnostics=parse_diagnostics,
        ) from exc

    if render_mode == "guitar":
        try:
            score = _normalize_to_guitar_range(score)
            score = _tab_score(score, max_fret=max_fret)
        except ValueError as exc:
            if _should_fallback_to_piano_score(exc):
                score = _to_piano_score(score)
                resolved_render_mode = "piano"
            else:
                raise _compose_failure(
                    stage="tab",
                    exc=exc,
                    checkpoint_path=checkpoint_path,
                    seed_tokens=seed_tokens,
                    generation_config=generation_config,
                    generation=generation,
                    effective_tokens=trimmed_generation.tokens,
                    parse_diagnostics=parse_diagnostics,
                    score=score,
                ) from exc
        except Exception as exc:
            raise _compose_failure(
                stage="tab",
                exc=exc,
                checkpoint_path=checkpoint_path,
                seed_tokens=seed_tokens,
                generation_config=generation_config,
                generation=generation,
                effective_tokens=trimmed_generation.tokens,
                parse_diagnostics=parse_diagnostics,
                score=score,
            ) from exc
    else:
        try:
            score = _to_piano_score(score)
        except Exception as exc:
            raise _compose_failure(
                stage="tab",
                exc=exc,
                checkpoint_path=checkpoint_path,
                seed_tokens=seed_tokens,
                generation_config=generation_config,
                generation=generation,
                effective_tokens=trimmed_generation.tokens,
                parse_diagnostics=parse_diagnostics,
                score=score,
            ) from exc

    try:
        exported = export_score(score)
        midi = canonical_score_to_midi(score)
    except Exception as exc:
        raise _compose_failure(
            stage="export",
            exc=exc,
            checkpoint_path=checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
            generation=generation,
            effective_tokens=trimmed_generation.tokens,
            parse_diagnostics=parse_diagnostics,
            score=score,
        ) from exc

    return ComposeServiceResult(
        generation=trimmed_generation,
        score=score,
        document=exported,
        midi=midi,
        render_mode=resolved_render_mode,
    )


def _compose_failure(
    *,
    stage: str,
    exc: Exception,
    checkpoint_path: str | Path,
    seed_tokens: Sequence[str | int],
    generation_config: GenerationConfig,
    generation: GenerationResult | None = None,
    effective_tokens: Sequence[str] | None = None,
    parse_diagnostics: ParseDiagnostics | None = None,
    score: CanonicalScore | None = None,
) -> ComposeDiagnosticsError:
    if isinstance(exc, ComposeDiagnosticsError):
        return exc

    report_path = _safe_write_failure_report(
        stage=stage,
        message=str(exc),
        checkpoint_path=checkpoint_path,
        seed_tokens=seed_tokens,
        generation_config=generation_config,
        generation=generation,
        effective_tokens=effective_tokens,
        parse_diagnostics=parse_diagnostics,
        score=score,
        exception_type=type(exc).__name__,
    )
    return ComposeDiagnosticsError(
        stage=stage,
        message=str(exc),
        report_path=report_path,
    )


def _safe_write_failure_report(
    *,
    stage: str,
    message: str,
    checkpoint_path: str | Path,
    seed_tokens: Sequence[str | int],
    generation_config: GenerationConfig,
    generation: GenerationResult | None,
    effective_tokens: Sequence[str] | None,
    parse_diagnostics: ParseDiagnostics | None,
    score: CanonicalScore | None,
    exception_type: str,
) -> Path | None:
    try:
        return _write_failure_report(
            stage=stage,
            message=message,
            checkpoint_path=checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
            generation=generation,
            effective_tokens=effective_tokens,
            parse_diagnostics=parse_diagnostics,
            score=score,
            exception_type=exception_type,
        )
    except Exception:
        return None


def _display_path(path: Path) -> str:
    return path.as_posix()


def _write_failure_report(
    *,
    stage: str,
    message: str,
    checkpoint_path: str | Path,
    seed_tokens: Sequence[str | int],
    generation_config: GenerationConfig,
    generation: GenerationResult | None,
    effective_tokens: Sequence[str] | None,
    parse_diagnostics: ParseDiagnostics | None,
    score: CanonicalScore | None,
    exception_type: str,
) -> Path:
    failure_dir = _failure_dir()
    failure_dir.mkdir(parents=True, exist_ok=True)

    report_id = _new_report_id()
    report_path = failure_dir / f"{report_id}_{stage}.json"

    generated_tokens = list(generation.tokens) if generation is not None else None
    current_tokens = list(effective_tokens) if effective_tokens is not None else generated_tokens
    generated_tokens_path = _write_tokens_file(
        failure_dir / f"{report_id}_{stage}.generated.tokens.txt",
        generated_tokens,
    )
    effective_tokens_path = None
    if current_tokens is not None and current_tokens != generated_tokens:
        effective_tokens_path = _write_tokens_file(
            failure_dir / f"{report_id}_{stage}.effective.tokens.txt",
            current_tokens,
        )

    report = {
        "report_id": report_id,
        "stage": stage,
        "message": message,
        "exception_type": exception_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(Path(checkpoint_path)),
        "seed_tokens": [str(token) for token in seed_tokens],
        "generation_config": asdict(generation_config),
        "generation": _generation_summary(generation, generated_tokens_path, effective_tokens_path, current_tokens),
        "parse_diagnostics": (parse_diagnostics.to_dict() if parse_diagnostics is not None else None),
        "score_summary": _score_summary(score),
        "token_summary": _token_summary(current_tokens),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _generation_summary(
    generation: GenerationResult | None,
    generated_tokens_path: Path | None,
    effective_tokens_path: Path | None,
    current_tokens: Sequence[str] | None,
) -> dict[str, object]:
    return {
        "available": generation is not None,
        "stopped_on_eos": generation.stopped_on_eos if generation is not None else None,
        "generated_token_count": (len(generation.tokens) if generation is not None else 0),
        "effective_token_count": (len(current_tokens) if current_tokens is not None else 0),
        "generated_tokens_path": (str(generated_tokens_path) if generated_tokens_path is not None else None),
        "effective_tokens_path": (str(effective_tokens_path) if effective_tokens_path is not None else None),
    }


def _score_summary(score: CanonicalScore | None) -> dict[str, object] | None:
    if score is None:
        return None
    event_count = sum(len(part.events) for part in score.parts)
    return {
        "measure_count": len(score.measures),
        "part_count": len(score.parts),
        "event_count": event_count,
        "last_tick": score.total_ticks,
    }


def _token_summary(tokens: Sequence[str] | None) -> dict[str, object] | None:
    if tokens is None:
        return None
    summary = {
        "token_count": len(tokens),
        "bar_count": sum(1 for token in tokens if token == "BAR"),
        "pos_token_count": sum(1 for token in tokens if token.startswith("POS_")),
        "voice_token_count": sum(1 for token in tokens if token.startswith("VOICE_")),
        "string_token_count": sum(1 for token in tokens if token.startswith("STR_")),
        "fret_token_count": sum(1 for token in tokens if token.startswith("FRET_")),
        "preview_head": list(tokens[:32]),
        "preview_tail": list(tokens[-32:]) if len(tokens) > 32 else list(tokens),
    }
    try:
        harm_errors = validate_harm_tokens(list(tokens))
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        summary["harmonic_validation_error"] = str(exc)
    else:
        summary["harmonic_validation_error_count"] = len(harm_errors)
        summary["harmonic_validation_errors_preview"] = harm_errors[:8]
    return summary


def _failure_dir() -> Path:
    configured = os.environ.get("BACH_GEN_COMPOSE_FAILURE_DIR")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_FAILURE_DIR


def _new_report_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _write_tokens_file(path: Path, tokens: Sequence[str] | None) -> Path | None:
    if tokens is None:
        return None
    path.write_text(" ".join(tokens), encoding="utf-8")
    return path


def _trim_incomplete_generation(generation: GenerationResult, *, tpq: int) -> GenerationResult:
    complete_tokens = _trim_incomplete_suffix(generation.tokens)
    if not generation.stopped_on_eos:
        complete_tokens = _trim_trailing_bar_overflow(complete_tokens, tpq=tpq)
    if len(complete_tokens) == len(generation.tokens):
        return generation
    if not complete_tokens:
        raise ValueError("generated token stream does not contain a complete event prefix")
    return replace(
        generation,
        ids=generation.ids[: len(complete_tokens)],
        tokens=complete_tokens,
    )


def _trim_incomplete_suffix(tokens: Sequence[str]) -> list[str]:
    safe_end = 0
    idx = 0
    while idx < len(tokens):
        next_idx = _next_complete_token_index(tokens, idx)
        if next_idx is None:
            break
        safe_end = next_idx
        idx = next_idx
    return list(tokens[:safe_end])


def _next_complete_token_index(tokens: Sequence[str], idx: int) -> int | None:
    token = tokens[idx]
    if token.startswith("VOICE_"):
        return _next_complete_voice_event_index(tokens, idx)
    if token.startswith("STR_") or token.startswith("FRET_"):
        return None
    return idx + 1


def _next_complete_voice_event_index(tokens: Sequence[str], idx: int) -> int | None:
    next_idx = idx + 1
    if next_idx >= len(tokens):
        return None

    first_token = tokens[next_idx]
    if first_token.startswith("REST_"):
        return next_idx + 1
    if not first_token.startswith("DUR_"):
        return idx + 1

    next_idx += 1
    if next_idx < len(tokens) and tokens[next_idx].startswith("DUP_"):
        next_idx += 1

    required_prefixes = ("MEL_INT12_", "HARM_OCT_", "HARM_CLASS_")
    for prefix in required_prefixes:
        if next_idx >= len(tokens):
            return None
        if not tokens[next_idx].startswith(prefix):
            return idx + 1
        next_idx += 1

    if next_idx < len(tokens) and tokens[next_idx].startswith("STR_"):
        next_idx += 1
        if next_idx >= len(tokens):
            return None
        if not tokens[next_idx].startswith("FRET_"):
            return idx + 1
        next_idx += 1
    elif next_idx < len(tokens) and tokens[next_idx].startswith("FRET_"):
        return idx + 1

    return next_idx


def _trim_trailing_bar_overflow(tokens: Sequence[str], tpq: int = 24) -> list[str]:
    if not tokens:
        return []

    last_bar_idx = -1
    current_time_sig: tuple[int, int] | None = None
    for idx, token in enumerate(tokens):
        if token == "BAR":
            last_bar_idx = idx
        elif token.startswith("TIME_SIG_"):
            current_time_sig = parse_time_sig_token(token)

    if last_bar_idx < 0 or current_time_sig is None:
        return list(tokens)

    numerator, denominator = current_time_sig
    bar_len_ticks = int(round(numerator * (4.0 / denominator) * tpq))

    safe_end = last_bar_idx + 1
    current_pos_tick: int | None = None
    overflow_detected = False
    idx = last_bar_idx + 1

    while idx < len(tokens):
        token = tokens[idx]

        if token.startswith("POS_"):
            pos_tick = parse_token_int(token)
            if pos_tick >= bar_len_ticks:
                overflow_detected = True
                break
            current_pos_tick = pos_tick
            idx += 1
            continue

        if token.startswith("VOICE_"):
            try:
                voice_event, next_idx = parse_voice_event(tokens, idx)
            except ValueError:
                break

            if current_pos_tick is not None:
                duration_ticks = voice_event.rest_ticks if voice_event.is_rest else voice_event.duration_ticks
                if current_pos_tick + duration_ticks > bar_len_ticks:
                    overflow_detected = True
                    break

            safe_end = next_idx
            idx = next_idx
            continue

        safe_end = idx + 1
        idx += 1

    if not overflow_detected:
        return list(tokens)
    return list(tokens[:safe_end])


def _normalize_to_guitar_range(score: CanonicalScore) -> CanonicalScore:
    """Shift pitches below the lowest guitar string up by whole octaves."""
    if len(score.parts) != 1:
        raise ValueError("compose service supports exactly one part")
    part = score.parts[0]
    normalized: list = []
    for event in part.events:
        if event.pitch_midi is not None and event.pitch_midi < _GUITAR_LOWEST_MIDI:
            new_pitch = event.pitch_midi
            while new_pitch < _GUITAR_LOWEST_MIDI:
                new_pitch += 12
            event = replace(event, pitch_midi=new_pitch)
        normalized.append(event)
    return replace(score, parts=[replace(part, events=normalized)])


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


def _to_piano_score(score: CanonicalScore) -> CanonicalScore:
    """Remove string-instrument semantics so the score can render without tablature."""
    if len(score.parts) != 1:
        raise ValueError("compose service supports exactly one part")
    part = score.parts[0]
    piano_info = replace(part.info, instrument="piano", tuning=[], capo=0)
    piano_events = [replace(event, fingering=None) for event in part.events]
    return replace(score, parts=[replace(part, info=piano_info, events=piano_events)])


def _should_fallback_to_piano_score(exc: ValueError) -> bool:
    message = str(exc)
    return "duplicate string use required" in message


def export_render_view(score_xml: str) -> RenderViewExport:
    root = ET.fromstring(score_xml)
    return RenderViewExport(
        xml=score_xml,
        measure_map=_build_measure_map(root),
        event_hit_map=_build_event_hit_map(root),
    )


def export_score(score: CanonicalScore | str) -> ScoreDocumentBundleExport:
    if isinstance(score, CanonicalScore):
        if len(score.parts) != 1:
            raise ValueError("compose service supports exactly one part")
        part = score.parts[0]
        instrument_mode: Literal["guitar", "piano"] = (
            "piano" if part.info.instrument == "piano" else "guitar"
        )
        views = {
            "score": export_render_view(canonical_score_to_standard_musicxml(score)),
        }
        if instrument_mode == "guitar":
            views["tab"] = export_render_view(canonical_score_to_tab_musicxml(score))
        return ScoreDocumentBundleExport(
            instrument_mode=instrument_mode,
            views=views,
        )

    return ScoreDocumentBundleExport(
        instrument_mode="guitar",
        views={"score": export_render_view(score)},
    )


def build_measure_map(score: CanonicalScore | str) -> dict[str, str]:
    return _build_measure_map(_musicxml_root(score))


def _build_measure_map(root: ET.Element) -> dict[str, str]:
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
    return _build_event_hit_map(_musicxml_root(score))


def _build_event_hit_map(root: ET.Element) -> dict[str, str]:
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
