#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.canonical import CanonicalScore
from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
from src.instrumental_v6.data import load_dataset
from src.instrumental_v6.decoding import PitchOption, select_counterpoint_pitches, voice_range
from src.instrumental_v6.metrics import evaluate_piece_rows, source_overlap_report
from src.instrumental_v6.model import build_generator, config_from_checkpoint
from src.instrumental_v6.representation import (
    DEVELOPMENT_TO_ID,
    FORM_TO_ID,
    GLOBAL_FEATURE_SPECS,
    GLOBAL_FIELD_NAMES,
    MAX_DURATION_STEPS,
    MAX_INTERVAL,
    ROLE_TO_ID,
    STATE_HOLD,
    STATE_NOTE,
    STATE_REST,
    VOICE_FIELD_NAMES,
    InstrumentalV6Piece,
    build_development_plan,
    decode_interval,
    meter_id,
    piece_to_canonical_score,
    recompute_pair_rows,
    rows_to_piece,
    scale_degree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate constrained variable-voice v6 counterpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/instrumental_v6/tiny_mixed_gate")
    parser.add_argument("--out-dir", default="out/instrumental_v6/generated")
    parser.add_argument("--voices", type=int, default=3)
    parser.add_argument(
        "--form",
        choices=["auto", "invention", "sinfonia", "fugue", "suite", "partita", "prelude"],
        default="auto",
    )
    parser.add_argument(
        "--piece-id",
        default=None,
        help="Use an exact dataset piece as the prompt template instead of --piece-index.",
    )
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--prompt-rows", type=int, default=64)
    parser.add_argument("--max-new-rows", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument(
        "--duration-temperature",
        type=float,
        default=0.7,
        help="Sample rhythm separately so conservative pitches do not collapse to one duration.",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--beam-size", type=int, default=96)
    parser.add_argument(
        "--duration-prior-strength",
        type=float,
        default=0.35,
        help="Blend a corpus-derived form/voice-count duration prior into duration sampling.",
    )
    parser.add_argument("--tempo", type=int, default=88)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2604)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_checkpoint(checkpoint["config"])
    if not 2 <= args.voices <= config.max_voices:
        raise SystemExit(f"--voices must be between 2 and checkpoint max_voices={config.max_voices}")
    model = build_generator(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pieces, _ = load_dataset(Path(args.data_dir) / "pieces.json")
    requested_form = None if args.form == "auto" else args.form.upper()
    template = _select_template(
        pieces,
        voices=args.voices,
        requested_form=requested_form,
        piece_id=args.piece_id,
        piece_index=args.piece_index,
    )
    form = requested_form or template.form
    prompt_count = min(max(2, args.prompt_rows), len(template.global_rows))
    prompt = (
        [row[:] for row in template.global_rows[:prompt_count]],
        [[voice[:] for voice in row] for row in template.voice_rows[:prompt_count]],
        [
            [[pair[:] for pair in left] for left in row]
            for row in template.pair_rows[:prompt_count]
        ],
    )
    source_rows = [
        piece.voice_rows
        for piece in pieces
        if piece.voice_count == args.voices
    ]
    duration_log_prior = _duration_log_prior(
        pieces,
        voices=args.voices,
        form=form,
        device=device,
    )
    source_start = prompt_count
    source_end = min(len(template.global_rows), source_start + args.max_new_rows)
    source_baseline = evaluate_piece_rows(
        template.global_rows[source_start:source_end],
        template.voice_rows[source_start:source_end],
        template.pair_rows[source_start:source_end],
        voice_count=args.voices,
    )
    subject = _subject_contour(prompt[1], args.voices)

    candidates: list[tuple[float, InstrumentalV6Piece, dict[str, object]]] = []
    for candidate_index in range(max(1, args.candidates)):
        generated = generate_rows(
            model,
            prompt=prompt,
            template=template,
            form=form,
            voice_count=args.voices,
            max_new_rows=args.max_new_rows,
            device=device,
            max_context=config.max_seq_len,
            temperature=args.temperature,
            duration_temperature=args.duration_temperature,
            top_k=args.top_k,
            beam_size=args.beam_size,
            duration_log_prior=duration_log_prior,
            duration_prior_strength=max(0.0, args.duration_prior_strength),
        )
        piece = rows_to_piece(
            global_rows=generated[0],
            voice_rows=generated[1],
            pair_rows=generated[2],
            template=replace(template, form=form, voice_count=args.voices),
            piece_id=f"instrumental_v6_{args.voices}v_candidate{candidate_index:02d}",
        )
        continuation = _continuation_piece(piece, prompt_count)
        report = evaluate_piece_rows(
            continuation.global_rows,
            continuation.voice_rows,
            continuation.pair_rows,
            voice_count=args.voices,
        )
        overlap = source_overlap_report(
            continuation.voice_rows,
            source_rows,
            voice_count=args.voices,
            ngram=16,
        )
        motif = _motif_report(
            piece.voice_rows,
            subject=subject,
            voice_count=args.voices,
        )
        score = _candidate_score(
            report,
            overlap,
            source_baseline=source_baseline,
            motif_report=motif,
        )
        diagnostics = {
            "candidate_index": candidate_index,
            "score": score,
            "counterpoint": report,
            "motif": motif,
            "source_overlap": overlap,
        }
        candidates.append((score, piece, diagnostics))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best, selected = candidates[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    piece_id = f"instrumental_v6_{args.voices}v_{form.lower()}"
    best = replace(best, piece_id=piece_id)
    score = _with_tempo(piece_to_canonical_score(best), args.tempo)
    midi_path = out_dir / f"{piece_id}.mid"
    xml_path = out_dir / f"{piece_id}.musicxml"
    rows_path = out_dir / f"{piece_id}.rows.json"
    metrics_path = out_dir / f"{piece_id}.metrics.json"
    midi_path.write_bytes(canonical_score_to_midi(score))
    xml_path.write_text(canonical_score_to_musicxml(score), encoding="utf-8")
    rows_path.write_text(json.dumps(best.to_dict()), encoding="utf-8")
    manifest = {
        "piece_id": piece_id,
        "checkpoint": args.checkpoint,
        "voices": args.voices,
        "form": form,
        "template_piece_id": template.piece_id,
        "prompt_rows": prompt_count,
        "generated_rows": args.max_new_rows,
        "duration_temperature": args.duration_temperature,
        "duration_prior_strength": max(0.0, args.duration_prior_strength),
        "source_baseline": source_baseline,
        "midi": str(midi_path),
        "musicxml": str(xml_path),
        "rows": str(rows_path),
        "selected": selected,
        "candidates": [item[2] for item in candidates],
    }
    metrics_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**manifest, "metrics": str(metrics_path)}, indent=2, sort_keys=True))


def _select_template(
    pieces: list[InstrumentalV6Piece],
    *,
    voices: int,
    requested_form: str | None,
    piece_id: str | None,
    piece_index: int,
) -> InstrumentalV6Piece:
    if piece_id is not None:
        exact = [piece for piece in pieces if piece.piece_id == piece_id]
        if not exact:
            raise SystemExit(f"template piece not found: {piece_id}")
        template = exact[0]
        if template.voice_count != voices:
            raise SystemExit(
                f"template {piece_id} has {template.voice_count} voices, not requested {voices}"
            )
        if requested_form is not None and template.form != requested_form:
            raise SystemExit(
                f"template {piece_id} has form {template.form}, not requested {requested_form}"
            )
        return template

    templates = [piece for piece in pieces if piece.voice_count == voices]
    if requested_form is not None:
        matching = [piece for piece in templates if piece.form == requested_form]
        if matching:
            templates = matching
    if not templates:
        available = sorted({piece.voice_count for piece in pieces})
        raise SystemExit(f"no template with {voices} voices; available voice counts: {available}")
    return templates[piece_index % len(templates)]


def _duration_log_prior(
    pieces: list[InstrumentalV6Piece],
    *,
    voices: int,
    form: str,
    device: torch.device,
) -> torch.Tensor:
    matching = [
        piece
        for piece in pieces
        if piece.voice_count == voices and piece.form == form
    ]
    if not matching:
        matching = [piece for piece in pieces if piece.voice_count == voices]
    counts = torch.ones(MAX_DURATION_STEPS + 1, dtype=torch.float32, device=device)
    state_col = VOICE_FIELD_NAMES.index("state")
    dur_col = VOICE_FIELD_NAMES.index("dur")
    for piece in matching:
        for row in piece.voice_rows:
            for voice in row[:voices]:
                if voice[state_col] == STATE_NOTE:
                    counts[min(MAX_DURATION_STEPS, max(1, voice[dur_col]))] += 1.0
    counts[0] = 0.0
    return torch.log(counts / counts.sum().clamp_min(1.0))


def generate_rows(
    model: torch.nn.Module,
    *,
    prompt: tuple[
        list[list[int]],
        list[list[list[int]]],
        list[list[list[list[int]]]],
    ],
    template: InstrumentalV6Piece,
    form: str,
    voice_count: int,
    max_new_rows: int,
    device: torch.device,
    max_context: int,
    temperature: float,
    top_k: int,
    beam_size: int,
    duration_temperature: float | None = None,
    duration_log_prior: torch.Tensor | None = None,
    duration_prior_strength: float = 0.0,
) -> tuple[list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]:
    prompt_length = len(prompt[0])
    total_rows = prompt_length + max_new_rows
    measures = math.ceil(total_rows / template.steps_per_bar)
    plan = build_development_plan(
        form=form,
        measures=measures,
        voice_count=voice_count,
        key_pc=template.key_pc,
        mode=template.mode,
    )
    global_rows = [
        _planned_global_row(
            row_index,
            template=template,
            form=form,
            voice_count=voice_count,
            plan=plan,
        )
        for row_index in range(prompt_length)
    ]
    voice_rows = [[voice[:] for voice in row] for row in prompt[1]]
    pair_rows = [[[pair[:] for pair in left] for left in row] for row in prompt[2]]
    subject = _subject_contour(voice_rows, voice_count)
    while len(global_rows) < total_rows:
        global_row = _planned_global_row(
            len(global_rows),
            template=template,
            form=form,
            voice_count=voice_count,
            plan=plan,
        )
        voice_row = _planned_cadence_voice_row(
            row_index=len(global_rows),
            total_rows=total_rows,
            previous_rows=voice_rows,
            template=template,
            voice_count=voice_count,
            beam_size=beam_size,
        )
        if voice_row is None:
            context_start = max(0, len(global_rows) - max_context)
            global_tensor = torch.tensor(
                global_rows[context_start:],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            voice_tensor = torch.tensor(
                voice_rows[context_start:],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            pair_tensor = torch.tensor(
                pair_rows[context_start:],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            with torch.inference_mode():
                logits = model(global_tensor, voice_tensor, pair_tensor)
            voice_row = _decode_voice_row(
                logits["voice"],
                global_row=global_row,
                previous_rows=voice_rows,
                template=template,
                voice_count=voice_count,
                subject=subject,
                temperature=temperature,
                duration_temperature=(
                    temperature if duration_temperature is None else duration_temperature
                ),
                top_k=top_k,
                beam_size=beam_size,
                duration_log_prior=duration_log_prior,
                duration_prior_strength=duration_prior_strength,
            )
        pair_row = recompute_pair_rows(
            voice_row,
            voice_rows[-1],
            max_voices=template.max_voices,
        )
        global_rows.append(global_row)
        voice_rows.append(voice_row)
        pair_rows.append(pair_row)
    return global_rows, voice_rows, pair_rows


def _planned_global_row(
    row_index: int,
    *,
    template: InstrumentalV6Piece,
    form: str,
    voice_count: int,
    plan: list[object],
) -> list[int]:
    bar = min(GLOBAL_FEATURE_SPECS["bar"] - 1, row_index // template.steps_per_bar)
    pos = row_index % template.steps_per_bar
    step = plan[min(bar, len(plan) - 1)]
    entry_voice = template.max_voices if step.entry_voice is None else step.entry_voice
    return [
        bar,
        pos,
        bar % GLOBAL_FEATURE_SPECS["phrase_pos"],
        int(step.role in {"CADENTIAL_PREP", "CADENCE"}),
        template.key_pc,
        template.mode,
        voice_count,
        FORM_TO_ID.get(form.upper(), 0),
        meter_id(template.time_signature),
        ROLE_TO_ID.get(step.role, 0),
        DEVELOPMENT_TO_ID.get(step.operation, 0),
        entry_voice,
        step.local_key_pc,
    ]


def _decode_voice_row(
    logits: dict[str, torch.Tensor],
    *,
    global_row: list[int],
    previous_rows: list[list[list[int]]],
    template: InstrumentalV6Piece,
    voice_count: int,
    subject: list[int],
    temperature: float,
    duration_temperature: float,
    top_k: int,
    beam_size: int,
    duration_log_prior: torch.Tensor | None,
    duration_prior_strength: float,
) -> list[list[int]]:
    previous = previous_rows[-1]
    state_col = VOICE_FIELD_NAMES.index("state")
    pitch_col = VOICE_FIELD_NAMES.index("pitch")
    dur_col = VOICE_FIELD_NAMES.index("dur")
    sampled_states: list[int] = []
    sampled_durations: list[int] = []
    silence_runs = [_trailing_rest_run(previous_rows, voice) for voice in range(voice_count)]

    def sample_duration(voice: int) -> int:
        return _sample_duration(
            logits["dur"][0, -1, voice],
            duration_temperature,
            top_k,
            max_duration=16 if voice_count >= 5 else MAX_DURATION_STEPS,
            log_prior=duration_log_prior,
            prior_strength=duration_prior_strength,
        )

    for voice in range(voice_count):
        previous_state = previous[voice][state_col]
        previous_duration = previous[voice][dur_col]
        if previous_state in {STATE_NOTE, STATE_HOLD} and previous_duration > 1:
            sampled_states.append(STATE_HOLD)
            sampled_durations.append(previous_duration - 1)
            continue
        state = _sample(logits["state"][0, -1, voice], temperature, top_k)
        if state == STATE_HOLD:
            state = STATE_NOTE
        sampled_states.append(state)
        sampled_durations.append(
            sample_duration(voice)
            if state == STATE_NOTE
            else 0
        )
    for voice in range(voice_count):
        if (
            sampled_states[voice] == STATE_REST
            and silence_runs[voice] >= template.steps_per_bar
        ):
            sampled_states[voice] = STATE_NOTE
            sampled_durations[voice] = sample_duration(voice)
    minimum_active = min(voice_count, max(2, (voice_count + 1) // 2))
    for voice in sorted(range(voice_count), key=lambda index: silence_runs[index], reverse=True):
        if sum(state != STATE_REST for state in sampled_states) >= minimum_active:
            break
        if sampled_states[voice] == STATE_REST:
            sampled_states[voice] = STATE_NOTE
            sampled_durations[voice] = sample_duration(voice)
    if all(state == STATE_REST for state in sampled_states):
        entry = global_row[GLOBAL_FIELD_NAMES.index("entry_voice")]
        forced_voice = entry if entry < voice_count else len(previous_rows) % voice_count
        sampled_states[forced_voice] = STATE_NOTE
        sampled_durations[forced_voice] = sample_duration(forced_voice)

    previous_active = [_active_pitch(row) for row in previous[:voice_count]]
    options_by_voice: list[list[PitchOption]] = []
    for voice in range(voice_count):
        state = sampled_states[voice]
        if state == STATE_REST:
            options_by_voice.append([PitchOption(None, 0.0)])
        elif state == STATE_HOLD and previous_active[voice] is not None:
            options_by_voice.append([PitchOption(previous_active[voice], 0.0)])
        else:
            options_by_voice.append(
                _pitch_options(
                    logits,
                    voice=voice,
                    voice_count=voice_count,
                    previous_rows=previous_rows,
                    global_row=global_row,
                    subject=subject,
                    steps_per_bar=template.steps_per_bar,
                    mode=template.mode,
                    top_k=top_k,
                )
            )
    pitches, _ = select_counterpoint_pitches(
        options_by_voice,
        previous_active,
        strong_beat=global_row[GLOBAL_FIELD_NAMES.index("pos")] % 4 == 0,
        beam_size=beam_size,
        strict=True,
    )
    voice_row: list[list[int]] = []
    for voice in range(template.max_voices):
        if voice >= voice_count:
            voice_row.append([STATE_REST, 0, 0, 0, 0, 0])
            continue
        state = sampled_states[voice]
        pitch = pitches[voice]
        if pitch is None:
            voice_row.append([STATE_REST, 0, 0, 0, 0, 0])
            continue
        if state == STATE_HOLD:
            voice_row.append(
                [
                    STATE_HOLD,
                    pitch,
                    0,
                    sampled_durations[voice],
                    1,
                    scale_degree(pitch, template.key_pc, template.mode),
                ]
            )
            continue
        previous_note = _previous_note_pitch(previous_rows, voice)
        melodic = 0 if previous_note is None else _encode_interval(pitch - previous_note)
        voice_row.append(
            [
                STATE_NOTE,
                pitch,
                melodic,
                sampled_durations[voice],
                0,
                scale_degree(pitch, template.key_pc, template.mode),
            ]
        )
    return voice_row


def _planned_cadence_voice_row(
    *,
    row_index: int,
    total_rows: int,
    previous_rows: list[list[list[int]]],
    template: InstrumentalV6Piece,
    voice_count: int,
    beam_size: int,
) -> list[list[int]] | None:
    pulse_rows = max(2, min(4, template.steps_per_bar // 4))
    rows_remaining = total_rows - row_index
    if rows_remaining > pulse_rows * 2:
        return None

    tonic_stage = rows_remaining <= pulse_rows
    stage_rows_remaining = rows_remaining if tonic_stage else rows_remaining - pulse_rows
    stage_onset = rows_remaining in {pulse_rows * 2, pulse_rows}
    previous = previous_rows[-1]
    previous_active = [_active_pitch(row) for row in previous[:voice_count]]
    if stage_onset:
        pitches = _cadence_target_pitches(
            previous_active,
            voice_count=voice_count,
            key_pc=template.key_pc,
            mode=template.mode,
            tonic_stage=tonic_stage,
            beam_size=beam_size,
        )
    else:
        pitches = previous_active

    voice_row: list[list[int]] = []
    for voice in range(template.max_voices):
        if voice >= voice_count:
            voice_row.append([STATE_REST, 0, 0, 0, 0, 0])
            continue
        pitch = pitches[voice]
        if pitch is None:
            voice_row.append([STATE_REST, 0, 0, 0, 0, 0])
            continue
        if not stage_onset:
            voice_row.append(
                [
                    STATE_HOLD,
                    pitch,
                    0,
                    stage_rows_remaining,
                    1,
                    scale_degree(pitch, template.key_pc, template.mode),
                ]
            )
            continue
        previous_note = _previous_note_pitch(previous_rows, voice)
        melodic = 0 if previous_note is None else _encode_interval(pitch - previous_note)
        voice_row.append(
            [
                STATE_NOTE,
                pitch,
                melodic,
                stage_rows_remaining,
                0,
                scale_degree(pitch, template.key_pc, template.mode),
            ]
        )
    return voice_row


def _cadence_target_pitches(
    previous_pitches: list[int | None],
    *,
    voice_count: int,
    key_pc: int,
    mode: int,
    tonic_stage: bool,
    beam_size: int,
) -> list[int | None]:
    third = 3 if mode == 1 else 4
    relative_classes = [0, third, 7] if tonic_stage else [7, 11, 2]
    root_class = relative_classes[0]
    options_by_voice: list[list[PitchOption]] = []
    for voice in range(voice_count):
        low, high = voice_range(voice, voice_count)
        if voice == 0:
            preferred_classes = [root_class]
        elif voice == voice_count - 1:
            preferred_classes = [0, third, 7] if tonic_stage else [11, 2, 7]
        else:
            rotation = (voice - 1) % len(relative_classes)
            preferred_classes = [
                *relative_classes[rotation:],
                *relative_classes[:rotation],
            ]
        options: list[PitchOption] = []
        for preference, relative_class in enumerate(preferred_classes):
            pitch_class = (key_pc + relative_class) % 12
            for pitch in _pitches_with_class(low, high, pitch_class):
                previous = previous_pitches[voice]
                motion_cost = 0.0 if previous is None else abs(pitch - previous) * 0.08
                options.append(PitchOption(pitch, 4.0 - preference - motion_cost))
        options_by_voice.append(sorted(options, key=lambda option: option.score, reverse=True))
    pitches, _ = select_counterpoint_pitches(
        options_by_voice,
        previous_pitches,
        strong_beat=True,
        beam_size=max(beam_size, 192),
        strict=True,
    )
    if all(pitch is not None for pitch in pitches):
        return pitches
    return _ordered_cadence_fallback(options_by_voice)


def _ordered_cadence_fallback(
    options_by_voice: list[list[PitchOption]],
) -> list[int | None]:
    selected: list[int | None] = []
    floor: int | None = None
    for options in options_by_voice:
        valid = [
            option
            for option in options
            if option.pitch is not None and (floor is None or option.pitch > floor)
        ]
        if not valid:
            selected.append(None)
            continue
        choice = max(valid, key=lambda option: option.score)
        selected.append(choice.pitch)
        floor = choice.pitch
    return selected


def _pitch_options(
    logits: dict[str, torch.Tensor],
    *,
    voice: int,
    voice_count: int,
    previous_rows: list[list[list[int]]],
    global_row: list[int],
    subject: list[int],
    steps_per_bar: int,
    mode: int,
    top_k: int,
) -> list[PitchOption]:
    pitch_log_probs = torch.log_softmax(logits["pitch"][0, -1, voice], dim=-1)
    mel_log_probs = torch.log_softmax(logits["mel"][0, -1, voice], dim=-1)
    degree_log_probs = torch.log_softmax(logits["degree"][0, -1, voice], dim=-1)
    low, high = voice_range(voice, voice_count)
    previous_note = _previous_note_pitch(previous_rows, voice)
    desired: int | None = None
    scores: dict[int, float] = {}
    for pitch_id in torch.topk(pitch_log_probs, k=min(top_k, pitch_log_probs.numel())).indices.tolist():
        if low <= pitch_id <= high:
            scores[pitch_id] = max(
                scores.get(pitch_id, -1e9),
                float(pitch_log_probs[pitch_id]),
            )
    if previous_note is not None:
        for mel_id in torch.topk(mel_log_probs, k=min(top_k, mel_log_probs.numel())).indices.tolist():
            interval = decode_interval(mel_id)
            if interval is None:
                continue
            pitch = previous_note + interval
            if low <= pitch <= high:
                score = float(mel_log_probs[mel_id]) * 1.7
                score += float(pitch_log_probs[pitch]) * 0.35
                scores[pitch] = max(scores.get(pitch, -1e9), score)
        desired = _development_interval(
            global_row,
            previous_rows,
            voice,
            subject,
            steps_per_bar=steps_per_bar,
        )
        if desired is not None:
            pitch = previous_note + desired
            if low <= pitch <= high:
                scores[pitch] = max(scores.get(pitch, -1e9), 1.5)
        for delta in (-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7):
            pitch = previous_note + delta
            if low <= pitch <= high and _fits_tonal_context(pitch, global_row, mode=mode):
                scores.setdefault(pitch, -5.5 - abs(delta) * 0.08)
    if global_row[GLOBAL_FIELD_NAMES.index("cadence_zone")]:
        local_key = global_row[GLOBAL_FIELD_NAMES.index("local_key_pc")]
        if local_key < 12:
            third = 3 if mode == 1 else 4
            role = global_row[GLOBAL_FIELD_NAMES.index("section_role")]
            if role == ROLE_TO_ID["CADENTIAL_PREP"]:
                root = (local_key + 7) % 12
                chord_classes = [root, (local_key + 11) % 12, (local_key + 2) % 12]
            else:
                root = local_key
                chord_classes = [root, (local_key + third) % 12, (local_key + 7) % 12]
            pitch_classes = [root] if voice == 0 else chord_classes
            for pitch_class in pitch_classes:
                for pitch in _pitches_with_class(low, high, pitch_class):
                    cadence_score = -0.2 if voice == 0 and pitch_class == root else -0.8
                    scores[pitch] = max(scores.get(pitch, -1e9), cadence_score)
    strong_beat = global_row[GLOBAL_FIELD_NAMES.index("pos")] % 4 == 0
    global_key = global_row[GLOBAL_FIELD_NAMES.index("key_pc")]
    for pitch in list(scores):
        degree = scale_degree(pitch, global_key, mode)
        scores[pitch] += float(degree_log_probs[degree]) * 0.8
        if not _fits_tonal_context(pitch, global_row, mode=mode):
            scores[pitch] -= 12.0 if strong_beat else 4.0
        scores[pitch] -= _repetition_penalty(
            previous_rows,
            voice=voice,
            candidate_pitch=pitch,
            protected_interval=desired,
        )
    if not scores:
        tonal = [
            pitch
            for pitch in range(low, high + 1)
            if _fits_tonal_context(pitch, global_row, mode=mode)
        ]
        center = (low + high) // 2
        fallback = min(tonal, key=lambda pitch: abs(pitch - center)) if tonal else center
        scores[fallback] = -8.0
    return [
        PitchOption(pitch, score)
        for pitch, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:24]
    ]


def _development_interval(
    global_row: list[int],
    previous_rows: list[list[list[int]]],
    voice: int,
    subject: list[int],
    steps_per_bar: int = 16,
) -> int | None:
    if not subject:
        return None
    operation = global_row[GLOBAL_FIELD_NAMES.index("development")]
    entry_voice = global_row[GLOBAL_FIELD_NAMES.index("entry_voice")]
    voice_count = global_row[GLOBAL_FIELD_NAMES.index("voice_count")]
    if operation == DEVELOPMENT_TO_ID["STRETTO"] and entry_voice >= voice_count:
        entry_voice = global_row[GLOBAL_FIELD_NAMES.index("bar")] % voice_count
    if entry_voice != voice:
        return None
    allowed = {
        DEVELOPMENT_TO_ID["SUBJECT"],
        DEVELOPMENT_TO_ID["ANSWER"],
        DEVELOPMENT_TO_ID["INVERSION"],
        DEVELOPMENT_TO_ID["STRETTO"],
        DEVELOPMENT_TO_ID["RECAP"],
    }
    if operation not in allowed:
        return None
    bar = global_row[GLOBAL_FIELD_NAMES.index("bar")]
    note_index = sum(
        row[voice][VOICE_FIELD_NAMES.index("state")] == STATE_NOTE
        for index, row in enumerate(previous_rows)
        if index // max(1, steps_per_bar) == bar
    )
    interval = subject[note_index % len(subject)]
    return -interval if operation == DEVELOPMENT_TO_ID["INVERSION"] else interval


def _pitches_with_class(low: int, high: int, pitch_class: int) -> list[int]:
    first = low + ((pitch_class - low) % 12)
    return list(range(first, high + 1, 12))


def _fits_tonal_context(pitch: int, global_row: list[int], *, mode: int) -> bool:
    scale = {0, 2, 3, 5, 7, 8, 10, 11} if mode == 1 else {0, 2, 4, 5, 7, 9, 11}
    pitch_class = pitch % 12
    keys = {
        global_row[GLOBAL_FIELD_NAMES.index("key_pc")],
        global_row[GLOBAL_FIELD_NAMES.index("local_key_pc")],
    }
    return any(
        key_pc < 12 and (pitch_class - key_pc) % 12 in scale
        for key_pc in keys
    )


def _subject_contour(rows: list[list[list[int]]], voice_count: int) -> list[int]:
    for voice in reversed(range(voice_count)):
        pitches = [
            row[voice][VOICE_FIELD_NAMES.index("pitch")]
            for row in rows
            if row[voice][VOICE_FIELD_NAMES.index("state")] == STATE_NOTE
            and row[voice][VOICE_FIELD_NAMES.index("pitch")] > 0
        ][:9]
        if len(pitches) >= 4:
            return [right - left for left, right in zip(pitches, pitches[1:])]
    return []


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(logits.argmax().item())
    values, indices = torch.topk(logits / max(0.05, temperature), k=min(top_k, logits.numel()))
    probabilities = torch.softmax(values, dim=-1)
    selected = torch.multinomial(probabilities, 1)
    return int(indices[selected].item())


def _sample_duration(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    *,
    max_duration: int,
    log_prior: torch.Tensor | None = None,
    prior_strength: float = 0.0,
) -> int:
    durations = [
        duration
        for duration in [1, 2, 3, 4, 6, 8, 12, 16, 24, MAX_DURATION_STEPS]
        if duration <= max_duration
    ]
    allowed = torch.tensor(durations, device=logits.device)
    values = logits[allowed]
    if log_prior is not None and prior_strength > 0.0:
        values = values + log_prior[allowed] * prior_strength
    selected = _sample(values, temperature, min(top_k, allowed.numel()))
    return int(allowed[selected].item())


def _active_pitch(row: list[int]) -> int | None:
    state = row[VOICE_FIELD_NAMES.index("state")]
    pitch = row[VOICE_FIELD_NAMES.index("pitch")]
    return pitch if state in {STATE_NOTE, STATE_HOLD} and pitch > 0 else None


def _previous_note_pitch(rows: list[list[list[int]]], voice: int) -> int | None:
    for row in reversed(rows):
        if row[voice][VOICE_FIELD_NAMES.index("state")] == STATE_NOTE:
            pitch = row[voice][VOICE_FIELD_NAMES.index("pitch")]
            if pitch > 0:
                return pitch
    return None


def _recent_note_pitches(
    rows: list[list[list[int]]],
    voice: int,
    *,
    limit: int = 20,
) -> list[int]:
    pitches = [
        row[voice][VOICE_FIELD_NAMES.index("pitch")]
        for row in reversed(rows)
        if row[voice][VOICE_FIELD_NAMES.index("state")] == STATE_NOTE
        and row[voice][VOICE_FIELD_NAMES.index("pitch")] > 0
    ][:limit]
    return list(reversed(pitches))


def _repetition_penalty(
    rows: list[list[list[int]]],
    *,
    voice: int,
    candidate_pitch: int,
    protected_interval: int | None,
) -> float:
    recent = _recent_note_pitches(rows, voice)
    if not recent:
        return 0.0
    pitches = [*recent, candidate_pitch]
    same_run = 1
    for pitch in reversed(pitches[:-1]):
        if pitch != candidate_pitch:
            break
        same_run += 1
    penalty = 0.0
    if same_run >= 2:
        penalty += 1.25 * (same_run - 1) ** 2
        if same_run >= 4:
            penalty += 8.0 * (same_run - 3)

    intervals = [right - left for left, right in zip(pitches, pitches[1:])]
    for period in range(2, min(4, len(intervals) // 2) + 1):
        repeated_blocks = 1
        tail = intervals[-period:]
        cursor = len(intervals) - period
        while cursor >= period and intervals[cursor - period : cursor] == tail:
            repeated_blocks += 1
            cursor -= period
        if repeated_blocks >= 3:
            penalty += (repeated_blocks - 2) * (4.0 + period)

    candidate_interval = candidate_pitch - recent[-1]
    if protected_interval is not None and candidate_interval == protected_interval:
        return max(0.0, penalty - 6.0)
    return penalty


def _trailing_rest_run(rows: list[list[list[int]]], voice: int) -> int:
    run = 0
    for row in reversed(rows):
        if row[voice][VOICE_FIELD_NAMES.index("state")] != STATE_REST:
            break
        run += 1
    return run


def _encode_interval(delta: int) -> int:
    return max(-MAX_INTERVAL, min(MAX_INTERVAL, delta)) + MAX_INTERVAL + 1


def _candidate_score(
    report: dict[str, object],
    overlap: dict[str, float | int],
    *,
    source_baseline: dict[str, object] | None = None,
    motif_report: dict[str, object] | None = None,
) -> float:
    note_rates = [float(value) for value in report["voice_note_rates"]]
    active_rates = [float(value) for value in report["voice_active_rates"]]
    stuck_rates = [float(value) for value in report["voice_stuck_rates"]]
    repeated_note_rates = [
        float(value) for value in report.get("voice_repeated_note_attack_rates", [])
    ]
    short_loop_rates = [
        float(value) for value in report.get("voice_short_loop_rates", [])
    ]
    max_repeated_attacks = [
        int(value) for value in report.get("voice_max_repeated_note_attacks", [])
    ]
    score = 100.0
    score -= 400.0 * float(report["invalid_pitch_state_rate"])
    score -= 350.0 * float(report["voice_crossing_rate"])
    score -= 350.0 * float(report["parallel_fifth_octave_rate"])
    score -= 140.0 * float(report["strong_beat_dissonance_rate"])
    score -= 600.0 * float(report.get("tonal_outlier_rate", 0.0))
    score -= 300.0 * float(report.get("strong_beat_tonal_outlier_rate", 0.0))
    if source_baseline and int(source_baseline.get("slice_count", 0)) > 0:
        baseline_note_rates = [float(value) for value in source_baseline["voice_note_rates"]]
        baseline_active_rates = [float(value) for value in source_baseline["voice_active_rates"]]
        baseline_repeated_note_rates = [
            float(value)
            for value in source_baseline.get("voice_repeated_note_attack_rates", [])
        ]
        baseline_short_loop_rates = [
            float(value) for value in source_baseline.get("voice_short_loop_rates", [])
        ]
        note_mean = sum(note_rates) / max(1, len(note_rates))
        active_mean = sum(active_rates) / max(1, len(active_rates))
        baseline_note_mean = sum(baseline_note_rates) / max(1, len(baseline_note_rates))
        baseline_active_mean = sum(baseline_active_rates) / max(1, len(baseline_active_rates))
        score -= 75.0 * sum(
            abs(rate - baseline_rate)
            for rate, baseline_rate in zip(note_rates, baseline_note_rates)
        ) / max(1, len(note_rates))
        score -= 90.0 * sum(
            abs(rate - baseline_rate)
            for rate, baseline_rate in zip(active_rates, baseline_active_rates)
        ) / max(1, len(active_rates))
        score -= 220.0 * sum(
            max(0.0, rate - baseline_rate - 0.02)
            for rate, baseline_rate in zip(
                repeated_note_rates,
                baseline_repeated_note_rates,
            )
        )
        score -= 160.0 * sum(
            max(0.0, rate - baseline_rate - 0.04)
            for rate, baseline_rate in zip(
                short_loop_rates,
                baseline_short_loop_rates,
            )
        )
        score -= 90.0 * abs(
            float(report["repeated_sonority_rate"])
            - float(source_baseline["repeated_sonority_rate"])
        )
        score -= 100.0 * abs(note_mean - baseline_note_mean)
        score -= 60.0 * abs(active_mean - baseline_active_mean)
        score -= 250.0 * max(0.0, note_mean - baseline_note_mean - 0.15)
        score -= 180.0 * max(0.0, active_mean - baseline_active_mean - 0.15)
        score -= 600.0 * max(
            0.0,
            float(report.get("tonal_outlier_rate", 0.0))
            - float(source_baseline.get("tonal_outlier_rate", 0.0)),
        )
        score -= 320.0 * max(
            0.0,
            float(report.get("strong_beat_tonal_outlier_rate", 0.0))
            - float(source_baseline.get("strong_beat_tonal_outlier_rate", 0.0)),
        )
    else:
        score -= 100.0 * float(report["repeated_sonority_rate"])
    score -= 120.0 * float(report["empty_slice_rate"])
    score -= 70.0 * max(stuck_rates, default=0.0)
    score -= 30.0 * sum(max(0, run - 3) for run in max_repeated_attacks)
    score -= 50.0 * sum(max(0.0, 0.08 - rate) for rate in note_rates)
    score -= 80.0 * sum(max(0.0, 0.25 - rate) for rate in active_rates)
    score -= 140.0 * float(overlap["source_ngram_overlap_rate"])
    score -= max(0.0, float(overlap["max_contiguous_source_match"]) - 12.0) * 2.0
    if motif_report:
        score += min(4, int(motif_report["subject_head_hits"])) * 5.0
        score += min(8, int(motif_report["max_subject_prefix"])) * 1.5
        section_hits = motif_report.get("section_subject_head_hits", {})
        if isinstance(section_hits, dict):
            score += 6.0 if int(section_hits.get("opening", 0)) > 0 else -4.0
            score += 10.0 if int(section_hits.get("middle", 0)) > 0 else -10.0
            score += 18.0 if int(section_hits.get("closing", 0)) > 0 else -24.0
    score += 24.0 if bool(report.get("final_tonic_sonority")) else -80.0
    score += 30.0 if bool(report.get("authentic_cadence_proxy")) else -40.0
    return score


def _motif_report(
    voice_rows: list[list[list[int]]],
    *,
    subject: list[int],
    voice_count: int,
) -> dict[str, object]:
    if not subject:
        return {
            "subject": [],
            "subject_head_hits": 0,
            "max_subject_prefix": 0,
            "voice_subject_head_hits": [0] * voice_count,
            "section_subject_head_hits": {"opening": 0, "middle": 0, "closing": 0},
        }
    head_length = min(3, len(subject))
    head = subject[:head_length]
    voice_hits: list[int] = []
    section_hits = [0, 0, 0]
    max_prefix = 0
    for voice in range(voice_count):
        attacks = [
            (row_index, row[voice][VOICE_FIELD_NAMES.index("pitch")])
            for row_index, row in enumerate(voice_rows)
            if row[voice][VOICE_FIELD_NAMES.index("state")] == STATE_NOTE
            and row[voice][VOICE_FIELD_NAMES.index("pitch")] > 0
        ]
        pitches = [pitch for _, pitch in attacks]
        intervals = [right - left for left, right in zip(pitches, pitches[1:])]
        hits = 0
        for index in range(max(0, len(intervals) - head_length + 1)):
            if intervals[index : index + head_length] != head:
                continue
            hits += 1
            row_index = attacks[index][0]
            section = min(2, row_index * 3 // max(1, len(voice_rows)))
            section_hits[section] += 1
        voice_hits.append(hits)
        for index in range(len(intervals)):
            prefix = 0
            while (
                prefix < len(subject)
                and index + prefix < len(intervals)
                and intervals[index + prefix] == subject[prefix]
            ):
                prefix += 1
            max_prefix = max(max_prefix, prefix)
    return {
        "subject": subject,
        "subject_head_hits": sum(voice_hits),
        "max_subject_prefix": max_prefix,
        "voice_subject_head_hits": voice_hits,
        "section_subject_head_hits": dict(
            zip(("opening", "middle", "closing"), section_hits)
        ),
    }


def _continuation_piece(piece: InstrumentalV6Piece, start: int) -> InstrumentalV6Piece:
    return replace(
        piece,
        global_rows=piece.global_rows[start:],
        voice_rows=piece.voice_rows[start:],
        pair_rows=piece.pair_rows[start:],
    )


def _with_tempo(score: CanonicalScore, tempo: int) -> CanonicalScore:
    return replace(score, header=replace(score.header, tempo_map={0: max(1, tempo)}))


if __name__ == "__main__":
    main()
