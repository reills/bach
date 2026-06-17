#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
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
from src.instrumental_v4.data import load_v4_dataset
from src.instrumental_v4.model import CompoundConfig, build_generator, build_planner
from src.instrumental_v4.representation import PLAN_FIELD_NAMES, PLAN_FEATURE_SPECS, V4_FIELD_NAMES, V4Piece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instrumental_v4 planned continuation and export MusicXML/MIDI.")
    parser.add_argument("--checkpoint", default="out/instrumental_v4_broad_planner/instrumental_v4.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--prompt-bars", type=int, default=8)
    parser.add_argument("--generate-bars", type=int, default=16)
    parser.add_argument("--plan-temperature", type=float, default=0.85)
    parser.add_argument("--plan-top-k", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.82)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--out-dir", default="out/instrumental_v4_broad_planner/listen")
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--device", default="cuda")
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
    dataset_path = args.dataset or ckpt.get("dataset")
    if not dataset_path:
        raise SystemExit("dataset path must be supplied when checkpoint has no dataset metadata")
    pieces, _ = load_v4_dataset(dataset_path)
    template = pieces[args.piece_index]
    config = CompoundConfig(**ckpt["config"])
    planner = build_planner(config).to(device)
    generator = build_generator(config).to(device)
    planner.load_state_dict(ckpt["planner_state"])
    generator.load_state_dict(ckpt["generator_state"])
    planner.eval()
    generator.eval()

    prompt_bars = min(args.prompt_bars, len(template.plans))
    prompt_slices = prompt_bars * template.steps_per_bar
    generate_slices = args.generate_bars * template.steps_per_bar
    prompt_rows = [row[:] for row in template.rows[:prompt_slices]]
    plans = _generate_plans(
        planner,
        template,
        prompt_bars=prompt_bars,
        generate_bars=args.generate_bars,
        device=device,
        temperature=args.plan_temperature,
        top_k=args.plan_top_k,
        argmax=args.argmax,
        max_seq_len=config.max_seq_len,
    )
    rows = _generate_rows(
        generator,
        template,
        prompt_rows=prompt_rows,
        plans=plans,
        generate_slices=generate_slices,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        argmax=args.argmax,
        max_seq_len=config.max_seq_len,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_v3 = _reset_v3_positions(_to_v3_rows(prompt_rows), template, start_bar=0)
    continuation_v3 = _reset_v3_positions(_to_v3_rows(rows[prompt_slices:]), template, start_bar=0)
    combined_v3 = _reset_v3_positions(_to_v3_rows(rows), template, start_bar=0)

    source_v3_pieces = [_as_v3_piece(piece) for piece in pieces]
    primary_report = _write_piece_exports(
        out_dir,
        template=_as_v3_piece(template),
        rows=continuation_v3,
        piece_id=f"instrumental_v4_{template.piece_id}_continuation",
        source_path=str(args.checkpoint),
        role="continuation_generated_only",
        prompt_slices=prompt_slices,
        generated_slices=generate_slices,
        source_pieces=source_v3_pieces,
        extra={"plans": plans[prompt_bars:], "prompt_bars": prompt_bars, "generated_bars": args.generate_bars},
    )
    _write_piece_exports(
        out_dir,
        template=_as_v3_piece(template),
        rows=prompt_v3,
        piece_id=f"instrumental_v4_{template.piece_id}_prompt",
        source_path=str(template.source_path),
        role="prompt_model_input",
        prompt_slices=prompt_slices,
        generated_slices=0,
        source_pieces=source_v3_pieces,
        extra={"plans": plans[:prompt_bars], "prompt_bars": prompt_bars, "generated_bars": 0},
    )
    _write_piece_exports(
        out_dir,
        template=_as_v3_piece(template),
        rows=combined_v3,
        piece_id=f"instrumental_v4_{template.piece_id}_prompt_plus_continuation",
        source_path=str(args.checkpoint),
        role="prompt_plus_continuation",
        prompt_slices=prompt_slices,
        generated_slices=generate_slices,
        source_pieces=source_v3_pieces,
        extra={"plans": plans, "prompt_bars": prompt_bars, "generated_bars": args.generate_bars},
    )
    print(json.dumps(primary_report.to_dict(), indent=2))


def _generate_plans(
    planner: torch.nn.Module,
    template: V4Piece,
    *,
    prompt_bars: int,
    generate_bars: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    argmax: bool,
    max_seq_len: int,
) -> list[list[int]]:
    plans = [plan.values[:] for plan in template.plans[:prompt_bars]]
    total = prompt_bars + generate_bars
    while len(plans) < total:
        context = plans[-max_seq_len:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = planner(x)
        next_plan = []
        for name in PLAN_FIELD_NAMES:
            value = _sample(logits[name][0, -1], temperature=temperature, top_k=top_k, argmax=argmax)
            next_plan.append(max(0, min(PLAN_FEATURE_SPECS[name] - 1, value)))
        bar = len(plans)
        next_plan[PLAN_FIELD_NAMES.index("plan_phrase_pos")] = bar % PLAN_FEATURE_SPECS["plan_phrase_pos"]
        next_plan[PLAN_FIELD_NAMES.index("plan_cadence_zone")] = 1 if next_plan[0] in {6, 7} else 0
        plans.append(next_plan)
    return plans


def _generate_rows(
    generator: torch.nn.Module,
    template: V4Piece,
    *,
    prompt_rows: list[list[int]],
    plans: list[list[int]],
    generate_slices: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    argmax: bool,
    max_seq_len: int,
) -> list[list[int]]:
    rows = [row[:] for row in prompt_rows]
    total = len(prompt_rows) + generate_slices
    while len(rows) < total:
        context = rows[-max_seq_len:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = generator(x)
        next_row = _sample_v4_row(
            logits,
            rows,
            template,
            plans=plans,
            temperature=temperature,
            top_k=top_k,
            argmax=argmax,
        )
        rows.append(next_row)
    return rows


def _sample_v4_row(
    logits: dict[str, torch.Tensor],
    rows: list[list[int]],
    template: V4Piece,
    *,
    plans: list[list[int]],
    temperature: float,
    top_k: int,
    argmax: bool,
) -> list[int]:
    prev = rows[-1]
    idx = len(rows)
    row = prev[:]
    bar = min(V3_FEATURE_SPECS["bar"] - 1, idx // template.steps_per_bar)
    pos = min(V3_FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
    row[V4_FIELD_NAMES.index("bar")] = bar
    row[V4_FIELD_NAMES.index("pos")] = pos
    row[V4_FIELD_NAMES.index("phrase_pos")] = bar % V3_FEATURE_SPECS["phrase_pos"]
    row[V4_FIELD_NAMES.index("cadence_zone")] = 1 if row[V4_FIELD_NAMES.index("phrase_pos")] in {6, 7} else 0
    row[V4_FIELD_NAMES.index("key_pc")] = template.key_pc
    row[V4_FIELD_NAMES.index("mode")] = template.mode
    row[V4_FIELD_NAMES.index("voice_count")] = 2

    active: list[int | None] = []
    for voice in range(2):
        state_i = V4_FIELD_NAMES.index(f"v{voice}_state")
        pitch_i = V4_FIELD_NAMES.index(f"v{voice}_pitch")
        mel_i = V4_FIELD_NAMES.index(f"v{voice}_mel")
        dur_i = V4_FIELD_NAMES.index(f"v{voice}_dur")
        tie_i = V4_FIELD_NAMES.index(f"v{voice}_tie")
        degree_i = V4_FIELD_NAMES.index(f"v{voice}_degree")

        state_logits = logits[f"v{voice}_state"][0, -1].clone()
        prev_state = prev[state_i]
        prev_pitch = prev[pitch_i]
        if not (prev_state in {STATE_NOTE, STATE_HOLD} and prev_pitch > 0):
            state_logits[STATE_HOLD] -= 100.0
        state = _sample(state_logits, temperature=temperature, top_k=3, argmax=argmax)

        if state == STATE_REST:
            row[state_i] = STATE_REST
            row[pitch_i] = 0
            row[mel_i] = 0
            row[dur_i] = 0
            row[tie_i] = 0
            row[degree_i] = 0
            active.append(None)
            continue

        if state == STATE_HOLD and prev_state in {STATE_NOTE, STATE_HOLD} and prev_pitch > 0:
            pitch = prev_pitch
            row[state_i] = STATE_HOLD
            row[pitch_i] = pitch
            row[mel_i] = 0
            row[dur_i] = max(1, _sample(logits[f"v{voice}_dur"][0, -1], temperature=temperature, top_k=top_k, argmax=argmax))
            row[tie_i] = 1
            row[degree_i] = _scale_degree_id(pitch, template.key_pc, template.mode)
            active.append(pitch)
            continue

        pitch_logits = logits[f"v{voice}_pitch"][0, -1].clone()
        pitch = _sample(pitch_logits, temperature=temperature, top_k=top_k, argmax=argmax)
        if pitch <= 0:
            pitch = prev_pitch if prev_pitch > 0 else (48 if voice == 0 else 60)
        row[state_i] = STATE_NOTE
        row[pitch_i] = pitch
        row[mel_i] = _encode_melody(pitch - _last_note_pitch(rows, voice, fallback=pitch))
        row[dur_i] = max(1, _sample(logits[f"v{voice}_dur"][0, -1], temperature=temperature, top_k=top_k, argmax=argmax))
        row[tie_i] = 0
        row[degree_i] = _scale_degree_id(pitch, template.key_pc, template.mode)
        active.append(pitch)

    _derive_vertical(row, active)
    plan = plans[min(len(plans) - 1, idx // template.steps_per_bar)]
    offset = len(V3_FIELD_NAMES)
    row[offset : offset + len(PLAN_FIELD_NAMES)] = plan[:]
    return row


def _sample(logits: torch.Tensor, *, temperature: float, top_k: int, argmax: bool) -> int:
    if argmax or temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(0.05, temperature)
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, k=top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _write_piece_exports(
    out_dir: Path,
    *,
    template: InstrumentalV3Piece,
    rows: list[list[int]],
    piece_id: str,
    source_path: str,
    role: str,
    prompt_slices: int,
    generated_slices: int,
    source_pieces: list[InstrumentalV3Piece],
    extra: dict[str, object],
):
    piece = slice_rows_to_piece(rows, template=template, piece_id=piece_id, source_path=source_path)
    score = piece_to_canonical_score(piece)
    report = evaluate_slices(piece.slices)
    novelty = source_overlap_report(piece.slices, [source.slices for source in source_pieces], ngram=16)

    xml_path = out_dir / f"{piece_id}.musicxml"
    midi_path = out_dir / f"{piece_id}.mid"
    metrics_path = out_dir / f"{piece_id}.metrics.json"
    slices_path = out_dir / f"{piece_id}.slices.json"

    xml_path.write_text(canonical_score_to_musicxml(score), encoding="utf-8")
    midi_path.write_bytes(canonical_score_to_midi(score))
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({**report.to_dict(), "source_overlap": novelty}, f, indent=2)
    with slices_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "role": role,
                "piece_id": piece_id,
                "source_path": source_path,
                "slice_count": len(rows),
                "prompt_slices": prompt_slices,
                "generated_slices": generated_slices,
                "rows": rows,
                **extra,
            },
            f,
            indent=2,
        )

    print(f"wrote {xml_path}")
    print(f"wrote {midi_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {slices_path}")
    return report


def _to_v3_rows(rows: list[list[int]]) -> list[list[int]]:
    return [row[: len(V3_FIELD_NAMES)] for row in rows]


def _as_v3_piece(piece: V4Piece) -> InstrumentalV3Piece:
    return InstrumentalV3Piece(
        piece_id=piece.piece_id,
        source_path=piece.source_path,
        tpq=piece.tpq,
        grid_ticks=piece.grid_ticks,
        time_signature=piece.time_signature,
        key=piece.key,
        key_pc=piece.key_pc,
        mode=piece.mode,
        bar_len_ticks=piece.bar_len_ticks,
        steps_per_bar=piece.steps_per_bar,
        slices=[SliceEvent(row[: len(V3_FIELD_NAMES)]) for row in piece.rows],
    )


def _reset_v3_positions(rows: list[list[int]], template: V4Piece, *, start_bar: int) -> list[list[int]]:
    out = []
    for idx, row in enumerate(rows):
        new_row = row[:]
        bar = start_bar + idx // template.steps_per_bar
        new_row[V3_FIELD_NAMES.index("bar")] = min(V3_FEATURE_SPECS["bar"] - 1, bar)
        new_row[V3_FIELD_NAMES.index("pos")] = min(V3_FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
        phrase_pos = bar % V3_FEATURE_SPECS["phrase_pos"]
        new_row[V3_FIELD_NAMES.index("phrase_pos")] = phrase_pos
        new_row[V3_FIELD_NAMES.index("cadence_zone")] = 1 if phrase_pos in {6, 7} else 0
        out.append(new_row)
    return out


def _last_note_pitch(rows: list[list[int]], voice: int, *, fallback: int) -> int:
    state_i = V4_FIELD_NAMES.index(f"v{voice}_state")
    pitch_i = V4_FIELD_NAMES.index(f"v{voice}_pitch")
    for row in reversed(rows):
        if row[state_i] == STATE_NOTE and row[pitch_i] > 0:
            return row[pitch_i]
    return fallback


def _encode_melody(delta: int) -> int:
    return max(1, min(49, int(delta) + 25))


def _scale_degree_id(pitch: int, key_pc: int, mode: int) -> int:
    if key_pc >= 12 or pitch <= 0:
        return 0
    rel = (pitch - key_pc) % 12
    major = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    minor = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}
    return (minor if mode == 1 else major).get(rel, 8 + rel % 5)


def _derive_vertical(row: list[int], active: list[int | None]) -> None:
    vi = V4_FIELD_NAMES.index("vertical_interval")
    ci = V4_FIELD_NAMES.index("consonance")
    si = V4_FIELD_NAMES.index("spacing")
    if active[0] is None or active[1] is None:
        row[vi] = row[ci] = row[si] = 0
        return
    spacing = min(48, abs(active[1] - active[0])) + 1
    row[vi] = row[si] = spacing
    pc = (spacing - 1) % 12
    row[ci] = 1 if pc in {0, 7} else 2 if pc in {3, 4, 8, 9} else 3


if __name__ == "__main__":
    main()
