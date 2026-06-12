#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
from src.api.canonical import CanonicalScore, Part, PartInfo
from src.inference.controls import ComposeControls
from src.inference.hybrid import (
    HybridContext,
    apply_conditioning_to_v5_row,
    apply_conditioning_to_v5_rows,
    build_hybrid_context,
)
from src.instrumental_v3.metrics import evaluate_slices, source_overlap_report
from src.instrumental_v3.representation import (
    FEATURE_SPECS as V3_FEATURE_SPECS,
    FIELD_NAMES as V3_FIELD_NAMES,
    InstrumentalV3Piece,
    STATE_HOLD,
    STATE_NOTE,
    STATE_REST,
    SliceEvent,
    piece_to_canonical_score,
    slice_rows_to_piece,
)
from src.instrumental_v4.model import CompoundConfig
from src.instrumental_v4.representation import PLAN_FIELD_NAMES, V4_FIELD_NAMES
from src.instrumental_v5.model import build_generator
from src.instrumental_v5.ace_step import (
    ACE_STEP_DEFAULT_MODEL,
    ACE_STEP_DEFAULT_TAG,
    build_ace_step_setup_plan,
    write_ace_step_handoff,
    write_ace_step_manifest,
)
from src.instrumental_v5.form_planner import build_v5_form_plan
from src.instrumental_v5.representation import (
    V5_COUNTERPOINT_FIELD_NAMES,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
    counterpoint_features_for_transition,
)
from src.tabber.heuristic import STANDARD_GUITAR_TUNING, tab_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/export from a v5 checkpoint.")
    parser.add_argument("--checkpoint", default="out/instrumental_v5_onset_strict_ft/checkpoint_latest.pt")
    parser.add_argument("--data-dir", default="data/instrumental_v5/keyboard_overture_cnorm_outer2_v5")
    parser.add_argument("--out-dir", default="out/instrumental_v5_onset_strict_ft/generated")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--prompt-rows", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--pitch-strategy",
        choices=["sampled", "interval", "blend"],
        default="interval",
        help=(
            "How NOTE pitches are repaired. 'sampled' keeps the old absolute-pitch path; "
            "'interval' derives pitch primarily from the melodic-interval head; "
            "'blend' considers both interval and absolute pitch candidates."
        ),
    )
    parser.add_argument(
        "--counterpoint-policy",
        choices=["strict", "soft", "off"],
        default="strict",
        help=(
            "Strict rejects crossings, parallel perfect intervals, and extreme spacing; "
            "exposed/direct perfects remain strongly penalized."
        ),
    )
    parser.add_argument("--candidates", type=int, default=4, help="Generate N candidates and keep the best objective score.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument(
        "--hybrid-conditioning",
        action="store_true",
        help="Force planned/retrieved EMI conditioning fields during v5 generation.",
    )
    parser.add_argument(
        "--fragment-path",
        default=None,
        help="EMI fragment memory JSONL. Defaults to <data-dir>/train_emi_fragments.jsonl.",
    )
    parser.add_argument("--key", default=None, help="Planning key for hybrid conditioning. Defaults to the prompt key.")
    parser.add_argument("--measures", type=int, default=0, help="Planning length in measures. Defaults from prompt+generation rows.")
    parser.add_argument("--texture", type=int, default=2, help="Planning texture/voice count.")
    parser.add_argument("--retrieval-limit", type=int, default=1, help="Fragments to consider per plan step.")
    parser.add_argument(
        "--form",
        choices=["auto", "invention", "sinfonia", "fugue", "suite", "partita", "prelude"],
        default="auto",
        help="CAST-style symbolic form plan to condition v5 generation.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help='Optional subject pitch sequence, e.g. "D4 E4 F4 A4 G4 F4 E4 D4".',
    )
    parser.add_argument(
        "--instrument",
        choices=["piano", "classical_guitar", "nylon_guitar", "lute", "harpsichord"],
        default="piano",
        help="Export arrangement target. Guitar/lute targets require playable tab fingerings.",
    )
    parser.add_argument("--tempo", type=int, default=92, help="Tempo metadata for ACE-Step handoff.")
    parser.add_argument(
        "--ace-step-handoff",
        action="store_true",
        help="Write ACE-Step 1.5 prompt/lyrics/metadata sidecars without making ACE-Step the notation generator.",
    )
    parser.add_argument("--ace-step-model", default=ACE_STEP_DEFAULT_MODEL)
    parser.add_argument("--ace-step-tag", default=ACE_STEP_DEFAULT_TAG)
    parser.add_argument("--ace-step-thinking", action="store_true", help="Set thinking=true in ACE-Step API request JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if list(ckpt.get("field_names", [])) != V5_FIELD_NAMES:
        raise SystemExit("checkpoint field_names do not match v5 representation")
    config = CompoundConfig(**ckpt["config"])
    model = build_generator(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    events = pd.read_parquet(Path(args.data_dir) / "events.parquet")
    pieces = [group.copy() for _, group in events.groupby("piece_id", sort=False)]
    if not pieces:
        raise SystemExit("events.parquet has no pieces")
    template_df = pieces[min(args.piece_index, len(pieces) - 1)].sort_values("row_index")
    template = _template_piece(template_df)
    source_pieces = [_template_piece(group.sort_values("row_index")) for group in pieces]
    prompt = template_df[V5_FIELD_NAMES].to_numpy(dtype="int64").tolist()[: args.prompt_rows]
    if len(prompt) < 2:
        raise SystemExit("prompt must contain at least two rows")
    hybrid_context = _hybrid_context_from_args(
        args,
        template=template,
        data_dir=Path(args.data_dir),
        total_rows=len(prompt) + args.max_new_tokens,
    )
    if hybrid_context is not None:
        prompt = apply_conditioning_to_v5_rows(prompt, hybrid_context, steps_per_bar=template.steps_per_bar)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    ace_handoffs = []
    for sample_idx in range(args.samples):
        rows, rerank_diagnostics = _generate_best_rows(
            model,
            prompt_rows=[row[:] for row in prompt],
            template=template,
            max_new_rows=args.max_new_tokens,
            device=device,
            max_context=config.max_seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pitch_strategy=args.pitch_strategy,
            counterpoint_policy=args.counterpoint_policy,
            hybrid_context=hybrid_context,
            candidate_count=args.candidates,
            source_pieces=source_pieces,
        )
        piece_id = f"instrumental_v5_sample{sample_idx:02d}"
        report = _write_exports(
            out_dir,
            rows=rows,
            template=template,
            piece_id=piece_id,
            checkpoint=str(args.checkpoint),
            prompt_rows=len(prompt),
            generated_rows=args.max_new_tokens,
            source_pieces=source_pieces,
            hybrid_diagnostics=hybrid_context.diagnostics() if hybrid_context is not None else None,
            rerank_diagnostics=rerank_diagnostics,
            instrument=args.instrument,
            tempo=args.tempo,
            ace_step_handoff=args.ace_step_handoff,
            ace_step_model=args.ace_step_model,
            ace_step_thinking=args.ace_step_thinking,
            form=args.form if args.form != "auto" else None,
            subject=args.subject,
        )
        reports.append(report)
        if report.get("ace_step") is not None:
            ace_handoffs.append(report["ace_step"])
    manifest_path = None
    if ace_handoffs:
        manifest_path = write_ace_step_manifest(
            out_dir,
            ace_handoffs,
            setup_plan=build_ace_step_setup_plan(recommended_tag=args.ace_step_tag),
        )
    print(json.dumps({"samples": reports, "ace_step_manifest": None if manifest_path is None else str(manifest_path)}, indent=2, sort_keys=True))


def _generate_rows(
    model: torch.nn.Module,
    *,
    prompt_rows: list[list[int]],
    template: InstrumentalV3Piece,
    max_new_rows: int,
    device: torch.device,
    max_context: int,
    temperature: float,
    top_p: float,
    top_k: int,
    pitch_strategy: str = "interval",
    counterpoint_policy: str = "strict",
    hybrid_context: HybridContext | None = None,
) -> list[list[int]]:
    rows = [row[:] for row in prompt_rows]
    total = len(rows) + max_new_rows
    while len(rows) < total:
        context = rows[-max_context:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        next_row = []
        for name in V5_FIELD_NAMES:
            value = _sample(logits[name][0, -1], temperature=temperature, top_p=top_p, top_k=top_k)
            next_row.append(max(0, min(V5_FEATURE_SPECS[name] - 1, value)))
        _repair_generated_row(
            next_row,
            rows,
            template,
            pitch_strategy=pitch_strategy,
            counterpoint_policy=counterpoint_policy,
        )
        if hybrid_context is not None:
            next_row = apply_conditioning_to_v5_row(
                next_row,
                hybrid_context,
                row_index=len(rows),
                steps_per_bar=template.steps_per_bar,
            )
        rows.append(next_row)
    return rows


def _generate_best_rows(
    model: torch.nn.Module,
    *,
    prompt_rows: list[list[int]],
    template: InstrumentalV3Piece,
    max_new_rows: int,
    device: torch.device,
    max_context: int,
    temperature: float,
    top_p: float,
    top_k: int,
    pitch_strategy: str = "interval",
    counterpoint_policy: str = "strict",
    hybrid_context: HybridContext | None = None,
    candidate_count: int = 1,
    source_pieces: list[InstrumentalV3Piece] | None = None,
) -> tuple[list[list[int]], dict[str, object]]:
    candidate_count = max(1, int(candidate_count))
    source_pieces = source_pieces or []
    candidates: list[tuple[float, list[list[int]], dict[str, object]]] = []
    batched_rows = _generate_rows_batch(
        model,
        prompt_rows=prompt_rows,
        template=template,
        max_new_rows=max_new_rows,
        device=device,
        max_context=max_context,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pitch_strategy=pitch_strategy,
        counterpoint_policy=counterpoint_policy,
        hybrid_context=hybrid_context,
        candidate_count=candidate_count,
    )
    for candidate_idx, rows in enumerate(batched_rows):
        score, diagnostics = _score_candidate_rows(
            rows,
            template=template,
            prompt_row_count=len(prompt_rows),
            source_pieces=source_pieces,
            counterpoint_policy=counterpoint_policy,
        )
        diagnostics["candidate_index"] = candidate_idx
        candidates.append((score, rows, diagnostics))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_rows, best_diagnostics = candidates[0]
    return best_rows, {
        "candidate_count": candidate_count,
        "selected_candidate_index": best_diagnostics["candidate_index"],
        "selected_score": best_score,
        "candidates": [diagnostics for _, _, diagnostics in candidates],
    }


def _generate_rows_batch(
    model: torch.nn.Module,
    *,
    prompt_rows: list[list[int]],
    template: InstrumentalV3Piece,
    max_new_rows: int,
    device: torch.device,
    max_context: int,
    temperature: float,
    top_p: float,
    top_k: int,
    pitch_strategy: str = "interval",
    counterpoint_policy: str = "strict",
    hybrid_context: HybridContext | None = None,
    candidate_count: int = 1,
) -> list[list[list[int]]]:
    candidate_count = max(1, int(candidate_count))
    rows_by_candidate = [[row[:] for row in prompt_rows] for _ in range(candidate_count)]
    total = len(prompt_rows) + max_new_rows
    while len(rows_by_candidate[0]) < total:
        context_rows = [rows[-max_context:] for rows in rows_by_candidate]
        x = torch.tensor(context_rows, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(x)
        sampled_by_field = {
            name: _sample_many(logits[name][:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            for name in V5_FIELD_NAMES
        }
        row_index = len(rows_by_candidate[0])
        for candidate_idx, rows in enumerate(rows_by_candidate):
            next_row = [
                max(0, min(V5_FEATURE_SPECS[name] - 1, int(sampled_by_field[name][candidate_idx].item())))
                for name in V5_FIELD_NAMES
            ]
            _repair_generated_row(
                next_row,
                rows,
                template,
                pitch_strategy=pitch_strategy,
                counterpoint_policy=counterpoint_policy,
            )
            if hybrid_context is not None:
                next_row = apply_conditioning_to_v5_row(
                    next_row,
                    hybrid_context,
                    row_index=row_index,
                    steps_per_bar=template.steps_per_bar,
                )
            rows.append(next_row)
    return rows_by_candidate


def _score_candidate_rows(
    rows: list[list[int]],
    *,
    template: InstrumentalV3Piece,
    prompt_row_count: int,
    source_pieces: list[InstrumentalV3Piece],
    counterpoint_policy: str = "soft",
) -> tuple[float, dict[str, object]]:
    continuation = rows[prompt_row_count:] or rows
    v3_rows = [row[: len(V3_FIELD_NAMES)] for row in continuation]
    piece = slice_rows_to_piece(
        v3_rows,
        template=template,
        piece_id="candidate",
        source_path="candidate",
    )
    report = evaluate_slices(piece.slices).to_dict()
    novelty = source_overlap_report(piece.slices, [source.slices for source in source_pieces], ngram=16)
    v0_note_rate = float(report["v0_note_rate"])
    v1_note_rate = float(report["v1_note_rate"])
    stuck_rate = max(float(report["v0_stuck_rate"]), float(report["v1_stuck_rate"]))
    same_pitch_rate = max(float(report["v0_same_pitch_run_rate"]), float(report["v1_same_pitch_run_rate"]))
    activity_floor_penalty = max(0.0, 0.20 - v0_note_rate) + max(0.0, 0.20 - v1_note_rate)
    activity_ceiling_penalty = max(0.0, v0_note_rate - 0.90) + max(0.0, v1_note_rate - 0.90)
    balance_penalty = abs(v0_note_rate - v1_note_rate)
    overlap_rate = float(novelty.get("source_ngram_overlap_rate", 0.0))
    contiguous = float(novelty.get("max_contiguous_source_match", 0.0))

    score = 100.0
    score -= 260.0 * float(report["invalid_pitch_state_rate"])
    score -= 220.0 * float(report["voice_crossing_rate"])
    score -= 200.0 * float(report["parallel_fifth_octave_rate"])
    score -= 120.0 * float(report["empty_slice_rate"])
    score -= 100.0 * float(report["repeated_sonority_rate"])
    score -= 90.0 * stuck_rate
    score -= 80.0 * same_pitch_rate
    score -= 70.0 * activity_floor_penalty
    score -= 55.0 * activity_ceiling_penalty
    score -= 35.0 * balance_penalty
    score -= 120.0 * overlap_rate
    score -= 1.5 * max(0.0, contiguous - 8.0)
    if counterpoint_policy == "strict" and (
        float(report["voice_crossing_rate"]) > 0.0
        or float(report["parallel_fifth_octave_rate"]) > 0.0
    ):
        score = -1_000_000_000.0

    diagnostics = {
        "score": round(score, 4),
        "invalid_pitch_state_rate": float(report["invalid_pitch_state_rate"]),
        "voice_crossing_rate": float(report["voice_crossing_rate"]),
        "parallel_fifth_octave_rate": float(report["parallel_fifth_octave_rate"]),
        "empty_slice_rate": float(report["empty_slice_rate"]),
        "repeated_sonority_rate": float(report["repeated_sonority_rate"]),
        "stuck_voice_rate": stuck_rate,
        "same_pitch_run_rate": same_pitch_rate,
        "voice_note_balance": balance_penalty,
        "source_overlap_rate": overlap_rate,
        "max_contiguous_source_match": int(contiguous),
    }
    return score, diagnostics


def _repair_generated_row(
    row: list[int],
    rows: list[list[int]],
    template: InstrumentalV3Piece,
    *,
    pitch_strategy: str = "interval",
    counterpoint_policy: str = "soft",
) -> None:
    prev = rows[-1]
    idx = len(rows)
    bar = min(V3_FEATURE_SPECS["bar"] - 1, idx // template.steps_per_bar)
    pos = min(V3_FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
    phrase_pos = bar % V3_FEATURE_SPECS["phrase_pos"]
    row[V5_FIELD_NAMES.index("bar")] = bar
    row[V5_FIELD_NAMES.index("pos")] = pos
    row[V5_FIELD_NAMES.index("phrase_pos")] = phrase_pos
    row[V5_FIELD_NAMES.index("cadence_zone")] = 1 if phrase_pos in {6, 7} else 0
    row[V5_FIELD_NAMES.index("key_pc")] = template.key_pc
    row[V5_FIELD_NAMES.index("mode")] = template.mode
    row[V5_FIELD_NAMES.index("voice_count")] = 2

    active: list[int | None] = []
    pitch_candidates: dict[int, list[int]] = {}
    for voice in (0, 1):
        state_i = V5_FIELD_NAMES.index(f"v{voice}_state")
        pitch_i = V5_FIELD_NAMES.index(f"v{voice}_pitch")
        mel_i = V5_FIELD_NAMES.index(f"v{voice}_mel")
        dur_i = V5_FIELD_NAMES.index(f"v{voice}_dur")
        tie_i = V5_FIELD_NAMES.index(f"v{voice}_tie")
        degree_i = V5_FIELD_NAMES.index(f"v{voice}_degree")
        state = row[state_i]
        prev_state = prev[state_i]
        prev_pitch = _clip_midi_pitch(prev[pitch_i]) if prev[pitch_i] > 0 else 0
        if state == STATE_HOLD and not (prev_state in {STATE_NOTE, STATE_HOLD} and prev_pitch > 0):
            state = STATE_NOTE
            row[state_i] = state
        if state == STATE_REST:
            row[pitch_i] = row[mel_i] = row[dur_i] = row[tie_i] = row[degree_i] = 0
            active.append(None)
            continue
        if state == STATE_HOLD:
            row[pitch_i] = prev_pitch
            row[mel_i] = 0
            row[dur_i] = max(1, row[dur_i])
            row[tie_i] = 1
            row[degree_i] = _scale_degree_id(prev_pitch, template.key_pc, template.mode)
            active.append(prev_pitch)
            continue
        row[state_i] = STATE_NOTE
        row[dur_i] = max(1, row[dur_i])
        row[tie_i] = 0
        if pitch_strategy == "sampled":
            if row[pitch_i] <= 0:
                row[pitch_i] = prev_pitch if prev_pitch > 0 else (48 if voice == 0 else 60)
            row[pitch_i] = _clip_midi_pitch(row[pitch_i])
            previous_note = _previous_note_pitch(rows, voice)
            row[mel_i] = _encode_melody_delta(None if previous_note is None else row[pitch_i] - previous_note)
            row[degree_i] = _scale_degree_id(row[pitch_i], template.key_pc, template.mode)
            active.append(row[pitch_i])
            continue
        candidates = _note_pitch_candidates(
            row,
            rows,
            voice,
            template,
            pitch_strategy=pitch_strategy,
            counterpoint_policy=counterpoint_policy,
        )
        pitch_candidates[voice] = candidates
        active.append(candidates[0])

    if pitch_candidates:
        active = _choose_active_pitches(
            row,
            rows,
            template,
            active,
            pitch_candidates,
            pitch_strategy=pitch_strategy,
            counterpoint_policy=counterpoint_policy,
        )
        for voice, pitch in enumerate(active):
            state_i = V5_FIELD_NAMES.index(f"v{voice}_state")
            pitch_i = V5_FIELD_NAMES.index(f"v{voice}_pitch")
            mel_i = V5_FIELD_NAMES.index(f"v{voice}_mel")
            degree_i = V5_FIELD_NAMES.index(f"v{voice}_degree")
            if row[state_i] != STATE_NOTE or pitch is None:
                continue
            previous_note = _previous_note_pitch(rows, voice)
            row[pitch_i] = pitch
            row[mel_i] = _encode_melody_delta(None if previous_note is None else pitch - previous_note)
            row[degree_i] = _scale_degree_id(pitch, template.key_pc, template.mode)

    _derive_vertical(row, active)
    _derive_counterpoint_transition(row, prev)
    # Plan fields are conditioning summaries. Keeping sampled values is fine, but clamp defensively.
    for name in PLAN_FIELD_NAMES:
        idx = V5_FIELD_NAMES.index(name)
        row[idx] = max(0, min(V5_FEATURE_SPECS[name] - 1, row[idx]))


def _note_pitch_candidates(
    row: list[int],
    rows: list[list[int]],
    voice: int,
    template: InstrumentalV3Piece,
    *,
    pitch_strategy: str,
    counterpoint_policy: str,
) -> list[int]:
    pitch_i = V5_FIELD_NAMES.index(f"v{voice}_pitch")
    mel_i = V5_FIELD_NAMES.index(f"v{voice}_mel")
    sampled_pitch = _valid_sampled_pitch(row[pitch_i])
    previous_note = _previous_note_pitch(rows, voice)
    previous_active = _active_pitch_from_row(rows[-1], voice)
    anchor = previous_note or previous_active
    default_pitch = 48 if voice == 0 else 60
    decoded_interval = _decode_melody_id(row[mel_i])
    candidates: list[int] = []

    if pitch_strategy in {"interval", "blend"} and anchor is not None and decoded_interval is not None:
        interval_pitch = anchor + decoded_interval
        candidates.extend([interval_pitch, interval_pitch - 12, interval_pitch + 12])

    if pitch_strategy in {"sampled", "blend"} and sampled_pitch is not None:
        candidates.append(sampled_pitch)

    plan_pitch = _plan_pitch(row, voice)
    if plan_pitch is not None:
        candidates.append(plan_pitch)

    if sampled_pitch is not None:
        candidates.append(sampled_pitch)
    if previous_active is not None:
        candidates.append(previous_active)
    if counterpoint_policy == "strict" and anchor is not None:
        candidates.extend(anchor + delta for delta in (-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7))
    candidates.append(default_pitch)

    low, high = _voice_range(voice)
    bounded: list[int] = []
    for pitch in candidates:
        clipped = _clip_midi_pitch(pitch)
        if clipped < low:
            clipped = low
        elif clipped > high:
            clipped = high
        if clipped not in bounded:
            bounded.append(clipped)
    return bounded or [default_pitch]


def _choose_active_pitches(
    row: list[int],
    rows: list[list[int]],
    template: InstrumentalV3Piece,
    active: list[int | None],
    pitch_candidates: dict[int, list[int]],
    *,
    pitch_strategy: str,
    counterpoint_policy: str,
) -> list[int | None]:
    options_by_voice = [
        pitch_candidates.get(0, [active[0]] if active[0] is not None else [None]),
        pitch_candidates.get(1, [active[1]] if active[1] is not None else [None]),
    ]
    best_score: float | None = None
    best_pair: tuple[int | None, int | None] = (active[0], active[1])
    for pitch0, pitch1 in itertools.product(options_by_voice[0], options_by_voice[1]):
        pair = (pitch0, pitch1)
        score = _score_pitch_pair(
            row,
            rows,
            template,
            pair,
            pitch_strategy=pitch_strategy,
            counterpoint_policy=counterpoint_policy,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_pair = pair
    return [best_pair[0], best_pair[1]]


def _score_pitch_pair(
    row: list[int],
    rows: list[list[int]],
    template: InstrumentalV3Piece,
    pair: tuple[int | None, int | None],
    *,
    pitch_strategy: str,
    counterpoint_policy: str,
) -> float:
    score = 0.0
    prev_pair = (_active_pitch_from_row(rows[-1], 0), _active_pitch_from_row(rows[-1], 1))
    pos = row[V5_FIELD_NAMES.index("pos")]
    cadence_zone = row[V5_FIELD_NAMES.index("cadence_zone")]

    for voice, pitch in enumerate(pair):
        if pitch is None:
            continue
        sampled = _valid_sampled_pitch(row[V5_FIELD_NAMES.index(f"v{voice}_pitch")])
        previous_note = _previous_note_pitch(rows, voice)
        decoded_interval = _decode_melody_id(row[V5_FIELD_NAMES.index(f"v{voice}_mel")])
        if previous_note is not None:
            actual_delta = pitch - previous_note
            if decoded_interval is not None:
                interval_weight = 2.2 if pitch_strategy == "interval" else 1.2 if pitch_strategy == "blend" else 0.25
                score -= interval_weight * abs(actual_delta - decoded_interval)
            leap = abs(actual_delta)
            if leap > 12:
                score -= 18.0 + (leap - 12) * 2.0
            elif leap > 7:
                score -= 6.0 + (leap - 7) * 1.2
            elif leap <= 2:
                score += 1.4
            if actual_delta == 0:
                score -= 0.4 * _same_active_pitch_run(rows, voice, pitch)
        elif decoded_interval is None:
            score += 0.5

        if sampled is not None:
            sampled_weight = 1.1 if pitch_strategy == "sampled" else 0.45 if pitch_strategy == "blend" else 0.15
            score -= sampled_weight * min(24, abs(pitch - sampled))

        low, high = _voice_range(voice)
        if pitch < low:
            score -= 4.0 * (low - pitch)
        elif pitch > high:
            score -= 4.0 * (pitch - high)
        degree = _scale_degree_id(pitch, template.key_pc, template.mode)
        if degree >= 8:
            score -= 1.25

    low_pitch, high_pitch = pair
    if low_pitch is None or high_pitch is None:
        return score

    if counterpoint_policy == "strict" and low_pitch >= high_pitch:
        return float("-inf")
    if low_pitch > high_pitch:
        score -= 80.0 + (low_pitch - high_pitch) * 3.0
    elif low_pitch == high_pitch:
        score -= 10.0

    spacing = abs(high_pitch - low_pitch)
    if counterpoint_policy == "strict" and spacing > 28:
        return float("-inf")
    interval_pc = spacing % 12
    if spacing > 24:
        score -= 3.0 * (spacing - 24)
    if spacing > 19:
        score -= 1.2 * (spacing - 19)
    if spacing < 3:
        score -= 5.0
    if interval_pc in {3, 4, 8, 9}:
        score += 2.5
    elif interval_pc == 7:
        score += 1.5
    elif interval_pc == 0:
        score -= 1.5 if not cadence_zone else -1.0
    elif interval_pc in {1, 2, 6, 10, 11}:
        weak_position = bool(pos % 2)
        score -= 1.0 if weak_position else 4.0

    if prev_pair[0] is not None and prev_pair[1] is not None:
        if pair == prev_pair:
            score -= 8.0
        prev_spacing = abs(prev_pair[1] - prev_pair[0])
        prev_pc = prev_spacing % 12
        motion0 = low_pitch - prev_pair[0]
        motion1 = high_pitch - prev_pair[1]
        same_direction = motion0 * motion1 > 0
        parallel_perfect = (
            same_direction
            and prev_pc in {0, 7}
            and interval_pc == prev_pc
            and motion0 != 0
            and motion1 != 0
        )
        direct_perfect = (
            same_direction
            and interval_pc in {0, 7}
            and (abs(motion0) > 2 or abs(motion1) > 2)
        )
        if counterpoint_policy == "strict" and parallel_perfect:
            return float("-inf")
        if motion0 * motion1 < 0:
            score += 1.5
        elif motion0 == 0 or motion1 == 0:
            score += 0.6
        elif motion0 * motion1 > 0:
            score -= 0.8
        if counterpoint_policy != "off" and parallel_perfect:
            score -= 55.0
        if counterpoint_policy != "off" and direct_perfect:
            score -= 36.0 if counterpoint_policy == "strict" else 24.0

    return score


def _sample(logits: torch.Tensor, *, temperature: float, top_p: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(0.05, temperature)
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, k=top_k)
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, num_samples=1)].item())
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        kept_probs = sorted_probs[keep]
        kept_indices = sorted_indices[keep]
        kept_probs = kept_probs / kept_probs.sum()
        return int(kept_indices[torch.multinomial(kept_probs, num_samples=1)].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def _sample_many(logits: torch.Tensor, *, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("expected batched logits with shape (batch, vocab)")
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / max(0.05, temperature)
    if top_k > 0 and top_k < logits.size(-1):
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        selected = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return indices.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[:, 0] = True
        filtered = sorted_probs.masked_fill(~keep, 0.0)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        selected = torch.multinomial(filtered, num_samples=1).squeeze(-1)
        return sorted_indices.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _write_exports(
    out_dir: Path,
    *,
    rows: list[list[int]],
    template: InstrumentalV3Piece,
    piece_id: str,
    checkpoint: str,
    prompt_rows: int,
    generated_rows: int,
    source_pieces: list[InstrumentalV3Piece],
    hybrid_diagnostics: dict[str, object] | None = None,
    rerank_diagnostics: dict[str, object] | None = None,
    instrument: str = "piano",
    tempo: int = 92,
    ace_step_handoff: bool = False,
    ace_step_model: str = ACE_STEP_DEFAULT_MODEL,
    ace_step_thinking: bool = False,
    form: str | None = None,
    subject: str | None = None,
) -> dict[str, object]:
    v3_rows = [row[: len(V3_FIELD_NAMES)] for row in rows]
    piece = slice_rows_to_piece(v3_rows, template=template, piece_id=piece_id, source_path=checkpoint)
    score = _score_with_tempo(piece_to_canonical_score(piece), tempo=tempo)
    score = _arrange_score_for_instrument(score, instrument=instrument)
    report = evaluate_slices(piece.slices)
    novelty = source_overlap_report(piece.slices, [source.slices for source in source_pieces], ngram=16)

    xml_path = out_dir / f"{piece_id}.musicxml"
    midi_path = out_dir / f"{piece_id}.mid"
    metrics_path = out_dir / f"{piece_id}.metrics.json"
    rows_path = out_dir / f"{piece_id}.v5_rows.json"
    xml_path.write_text(canonical_score_to_musicxml(score), encoding="utf-8")
    midi_path.write_bytes(canonical_score_to_midi(score))
    metrics = {**report.to_dict(), "source_overlap": novelty}
    if hybrid_diagnostics is not None:
        metrics["hybrid"] = hybrid_diagnostics
    if rerank_diagnostics is not None:
        metrics["candidate_rerank"] = rerank_diagnostics
    metrics["instrument"] = instrument
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    rows_path.write_text(
        json.dumps(
            {
                "piece_id": piece_id,
                "checkpoint": checkpoint,
                "prompt_rows": prompt_rows,
                "generated_rows": generated_rows,
                "field_names": V5_FIELD_NAMES,
                "hybrid": hybrid_diagnostics,
                "candidate_rerank": rerank_diagnostics,
                "instrument": instrument,
                "form": form,
                "subject": subject,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ace_step = None
    if ace_step_handoff:
        duration_seconds = max(10.0, score.total_ticks / score.header.tpq * 60.0 / max(1, tempo))
        handoff = write_ace_step_handoff(
            out_dir,
            sample_id=piece_id,
            musicxml_path=xml_path,
            midi_path=midi_path,
            key=template.key,
            time_signature=template.time_signature,
            bpm=tempo,
            duration_seconds=duration_seconds,
            form=form,
            instrument=instrument,
            voices=2,
            bars=max(1, len(rows) // max(1, template.steps_per_bar)),
            subject=subject,
            model=ace_step_model,
            thinking=ace_step_thinking,
        )
        ace_step = handoff.to_dict()
    return {
        "piece_id": piece_id,
        "musicxml": str(xml_path),
        "midi": str(midi_path),
        "metrics": str(metrics_path),
        "rows": str(rows_path),
        "instrument": instrument,
        "counterpoint": report.to_dict(),
        "source_overlap": novelty,
        "hybrid": hybrid_diagnostics,
        "candidate_rerank": rerank_diagnostics,
        "ace_step": ace_step,
    }


def _hybrid_context_from_args(
    args: argparse.Namespace,
    *,
    template: InstrumentalV3Piece,
    data_dir: Path,
    total_rows: int,
) -> HybridContext | None:
    form = getattr(args, "form", "auto")
    subject = getattr(args, "subject", None)
    form_requested = form != "auto" or bool(subject)
    if not getattr(args, "hybrid_conditioning", False) and not form_requested:
        return None
    fragment_path = _resolve_fragment_path(getattr(args, "fragment_path", None), data_dir=data_dir)
    measures = int(getattr(args, "measures", 0) or _rows_to_measures(total_rows, template.steps_per_bar))
    key = getattr(args, "key", None) or template.key or "C"
    form_plan = None
    planning_metadata = None
    if form_requested:
        form_plan = build_v5_form_plan(
            form="invention" if form == "auto" else form,
            measures=measures,
            key=key,
            texture=int(getattr(args, "texture", 2) or 2),
            subject=subject,
        )
        planning_metadata = form_plan.to_dict()
    return build_hybrid_context(
        ComposeControls(
            key=key,
            measures=measures,
            texture=int(getattr(args, "texture", 2) or 2),
        ),
        fragment_path=fragment_path,
        retrieval_limit=int(getattr(args, "retrieval_limit", 1) or 1),
        plan=None if form_plan is None else form_plan.steps,
        planning_metadata=planning_metadata,
    )


def _resolve_fragment_path(fragment_path: str | None, *, data_dir: Path) -> Path:
    if fragment_path:
        return Path(fragment_path)
    return data_dir / "train_emi_fragments.jsonl"


def _score_with_tempo(score: CanonicalScore, *, tempo: int) -> CanonicalScore:
    return CanonicalScore(
        header=replace(score.header, tempo_map={0: max(1, int(tempo))}),
        measures=score.measures,
        parts=score.parts,
    )


def _arrange_score_for_instrument(score: CanonicalScore, *, instrument: str) -> CanonicalScore:
    if instrument == "piano":
        return score
    part = score.parts[0]
    if instrument == "harpsichord":
        arranged_part = Part(
            info=PartInfo(id=part.info.id, instrument="harpsichord", midi_program=6),
            events=part.events,
        )
    elif instrument in {"classical_guitar", "nylon_guitar", "lute"}:
        midi_program = 24 if instrument in {"classical_guitar", "nylon_guitar"} else 24
        arranged_part = Part(
            info=PartInfo(
                id=part.info.id,
                instrument="classical_guitar" if instrument == "nylon_guitar" else instrument,
                tuning=list(STANDARD_GUITAR_TUNING),
                midi_program=midi_program,
            ),
            events=tab_events(part.events, tuning=STANDARD_GUITAR_TUNING),
        )
    else:
        raise ValueError(f"unsupported export instrument: {instrument!r}")
    return CanonicalScore(header=score.header, measures=score.measures, parts=[arranged_part])


def _rows_to_measures(row_count: int, steps_per_bar: int) -> int:
    if steps_per_bar <= 0:
        return 1
    return max(1, (max(1, row_count) + steps_per_bar - 1) // steps_per_bar)


def _template_piece(df: pd.DataFrame) -> InstrumentalV3Piece:
    rows = df[V5_FIELD_NAMES].to_numpy(dtype="int64").tolist()
    first = df.iloc[0]
    key = first.get("key")
    if pd.isna(key):
        key = None
    return InstrumentalV3Piece(
        piece_id=str(first["piece_id"]),
        source_path=str(first["source_path"]),
        tpq=int(first["tpq"]),
        grid_ticks=int(first["grid_ticks"]),
        time_signature=str(first["time_signature"]),
        key=None if key is None else str(key),
        key_pc=int(first["key_pc"]),
        mode=int(first["mode"]),
        bar_len_ticks=int(first["bar_len_ticks"]),
        steps_per_bar=int(first["steps_per_bar"]),
        slices=[SliceEvent(row[: len(V3_FIELD_NAMES)]) for row in rows],
    )


def _derive_vertical(row: list[int], active: list[int | None]) -> None:
    vi = V5_FIELD_NAMES.index("vertical_interval")
    ci = V5_FIELD_NAMES.index("consonance")
    si = V5_FIELD_NAMES.index("spacing")
    if active[0] is None or active[1] is None:
        row[vi] = row[ci] = row[si] = 0
        return
    spacing = min(48, abs(active[1] - active[0])) + 1
    row[vi] = row[si] = spacing
    pc = (spacing - 1) % 12
    row[ci] = 1 if pc in {0, 7} else 2 if pc in {3, 4, 8, 9} else 3


def _derive_counterpoint_transition(row: list[int], previous_row: list[int] | None) -> None:
    previous_v4 = None if previous_row is None else previous_row[: len(V4_FIELD_NAMES)]
    features = counterpoint_features_for_transition(previous_v4, row[: len(V4_FIELD_NAMES)])
    for idx, name in enumerate(V5_COUNTERPOINT_FIELD_NAMES):
        row[V5_FIELD_NAMES.index(name)] = features[idx]


def _valid_sampled_pitch(value: int) -> int | None:
    value = int(value)
    if value <= 0:
        return None
    return _clip_midi_pitch(value)


def _decode_melody_id(value: int) -> int | None:
    value = int(value)
    if value <= 0:
        return None
    return max(-24, min(24, value - 25))


def _encode_melody_delta(delta: int | None) -> int:
    if delta is None:
        return 0
    return max(-24, min(24, int(delta))) + 25


def _active_pitch_from_row(row: list[int], voice: int) -> int | None:
    state = row[V5_FIELD_NAMES.index(f"v{voice}_state")]
    pitch = row[V5_FIELD_NAMES.index(f"v{voice}_pitch")]
    if state in {STATE_NOTE, STATE_HOLD} and pitch > 0:
        return _clip_midi_pitch(pitch)
    return None


def _previous_note_pitch(rows: list[list[int]], voice: int) -> int | None:
    state_i = V5_FIELD_NAMES.index(f"v{voice}_state")
    pitch_i = V5_FIELD_NAMES.index(f"v{voice}_pitch")
    for previous in reversed(rows):
        if previous[state_i] == STATE_NOTE and previous[pitch_i] > 0:
            return _clip_midi_pitch(previous[pitch_i])
    return None


def _same_active_pitch_run(rows: list[list[int]], voice: int, pitch: int) -> int:
    run = 0
    for previous in reversed(rows):
        active = _active_pitch_from_row(previous, voice)
        if active != pitch:
            break
        run += 1
    return run


def _plan_pitch(row: list[int], voice: int) -> int | None:
    pc_name = "plan_bass_pc" if voice == 0 else "plan_top_pc"
    oct_name = "plan_bass_oct" if voice == 0 else "plan_top_oct"
    pc = row[V5_FIELD_NAMES.index(pc_name)]
    octave = row[V5_FIELD_NAMES.index(oct_name)]
    if pc >= 12 or octave >= 10:
        return None
    pitch = octave * 12 + pc
    if pitch <= 0:
        return None
    return _clip_midi_pitch(pitch)


def _voice_range(voice: int) -> tuple[int, int]:
    # Keyboard inventions frequently cross registers, but hard bounds prevent sampled outliers.
    return (36, 72) if voice == 0 else (48, 96)


def _scale_degree_id(pitch: int, key_pc: int, mode: int) -> int:
    if key_pc >= 12 or pitch <= 0:
        return 0
    rel = (pitch - key_pc) % 12
    major = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    minor = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}
    return (minor if mode == 1 else major).get(rel, 8 + rel % 5)


def _clip_midi_pitch(pitch: int) -> int:
    return max(1, min(127, int(pitch)))


if __name__ == "__main__":
    main()
