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
from src.instrumental_v3.data import load_dataset
from src.instrumental_v3.metrics import evaluate_slices, source_overlap_report
from src.instrumental_v3.model import InstrumentalV3Config, InstrumentalV3Transformer
from src.instrumental_v3.representation import (
    FIELD_NAMES,
    FEATURE_SPECS,
    InstrumentalV3Piece,
    STATE_HOLD,
    STATE_NOTE,
    STATE_REST,
    piece_to_canonical_score,
    slice_rows_to_piece,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instrumental_v3 continuation and export MusicXML/MIDI.")
    parser.add_argument("--checkpoint", default="out/instrumental_v3_tiny/instrumental_v3_tiny.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--prompt-slices", type=int, default=64)
    parser.add_argument("--generate-slices", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument(
        "--export-generated-only",
        action="store_true",
        help="Make the primary *_continuation export contain generated slices only.",
    )
    parser.add_argument(
        "--write-separated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write explicit *_prompt and *_prompt_plus_continuation files.",
    )
    parser.add_argument("--key-bias", type=float, default=1.0)
    parser.add_argument(
        "--max-rest-run-slices",
        type=int,
        default=8,
        help="Force a voice back in after this many consecutive rest slices.",
    )
    parser.add_argument(
        "--max-same-pitch-slices",
        type=int,
        default=8,
        help="Strongly discourage a voice from staying on one pitch longer than this many slices.",
    )
    parser.add_argument(
        "--same-pitch-penalty",
        type=float,
        default=10.0,
        help="Logit penalty for repeating a pitch after --max-same-pitch-slices.",
    )
    parser.add_argument(
        "--tail-force-active-ratio",
        type=float,
        default=0.15,
        help="In the final fraction of generation, strongly discourage voice dropout.",
    )
    parser.add_argument(
        "--motif-bias",
        type=float,
        default=0.8,
        help="Bias new notes toward melodic intervals already used in the prompt/recent context.",
    )
    parser.add_argument(
        "--cadence-bias",
        type=float,
        default=1.2,
        help="Bias the tail toward tonic/dominant/triad tones.",
    )
    parser.add_argument("--out-dir", default="out/instrumental_v3_tiny/listen")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
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
    pieces, _ = load_dataset(dataset_path)
    template = pieces[args.piece_index]
    config = InstrumentalV3Config(**ckpt["config"])
    model = InstrumentalV3Transformer(config, feature_specs=ckpt.get("feature_specs", FEATURE_SPECS)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = [s.values[:] for s in template.slices[: args.prompt_slices]]
    rows = [row[:] for row in prompt]
    total = args.prompt_slices + args.generate_slices
    while len(rows) < total:
        context = rows[-config.max_seq_len :]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        next_row = _sample_musical_row(
            logits,
            rows,
            template,
            temperature=args.temperature,
            top_k=args.top_k,
            argmax=args.argmax,
            key_bias=args.key_bias,
            generated_index=len(rows) - args.prompt_slices,
            generated_total=args.generate_slices,
            max_rest_run_slices=args.max_rest_run_slices,
            max_same_pitch_slices=args.max_same_pitch_slices,
            same_pitch_penalty=args.same_pitch_penalty,
            tail_force_active_ratio=args.tail_force_active_ratio,
            motif_bias=args.motif_bias,
            cadence_bias=args.cadence_bias,
        )
        rows.append(next_row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = _reset_positions(prompt, template)
    continuation_rows = _reset_positions(rows[args.prompt_slices :], template)
    combined_rows = _reset_positions(rows, template)

    primary_rows = continuation_rows if args.export_generated_only else combined_rows
    primary_report = _write_piece_exports(
        out_dir,
        template=template,
        rows=primary_rows,
        piece_id=f"instrumental_v3_{template.piece_id}_continuation",
        source_path=str(args.checkpoint),
        role="continuation_generated_only" if args.export_generated_only else "prompt_plus_continuation",
        prompt_slices=args.prompt_slices,
        generated_slices=args.generate_slices,
        source_pieces=pieces,
    )

    if args.write_separated:
        _write_piece_exports(
            out_dir,
            template=template,
            rows=prompt_rows,
            piece_id=f"instrumental_v3_{template.piece_id}_prompt",
            source_path=str(template.source_path),
            role="prompt_model_input",
            prompt_slices=args.prompt_slices,
            generated_slices=0,
            source_pieces=pieces,
        )
        _write_piece_exports(
            out_dir,
            template=template,
            rows=combined_rows,
            piece_id=f"instrumental_v3_{template.piece_id}_prompt_plus_continuation",
            source_path=str(args.checkpoint),
            role="prompt_plus_continuation",
            prompt_slices=args.prompt_slices,
            generated_slices=args.generate_slices,
            source_pieces=pieces,
        )

    print(json.dumps(primary_report.to_dict(), indent=2))


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
):
    piece = slice_rows_to_piece(
        rows,
        template=template,
        piece_id=piece_id,
        source_path=source_path,
    )
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
            },
            f,
            indent=2,
        )

    print(f"wrote {xml_path}")
    print(f"wrote {midi_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {slices_path}")
    return report


def _sample_musical_row(
    logits: dict[str, torch.Tensor],
    rows: list[list[int]],
    template: InstrumentalV3Piece,
    *,
    temperature: float,
    top_k: int,
    argmax: bool,
    key_bias: float,
    generated_index: int,
    generated_total: int,
    max_rest_run_slices: int,
    max_same_pitch_slices: int,
    same_pitch_penalty: float,
    tail_force_active_ratio: float,
    motif_bias: float,
    cadence_bias: float,
) -> list[int]:
    prev = rows[-1]
    idx = len(rows)
    row = prev[:]
    row[FIELD_NAMES.index("bar")] = min(FEATURE_SPECS["bar"] - 1, idx // template.steps_per_bar)
    row[FIELD_NAMES.index("pos")] = min(FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
    _set_phrase_fields(row)
    row[FIELD_NAMES.index("key_pc")] = template.key_pc
    row[FIELD_NAMES.index("mode")] = template.mode
    row[FIELD_NAMES.index("voice_count")] = 2

    active: list[int | None] = []
    tail_start = max(0, int(generated_total * (1.0 - tail_force_active_ratio)))
    in_tail = generated_index >= tail_start
    tail_phase = 0.0
    if generated_total > tail_start:
        tail_phase = (generated_index - tail_start) / max(1, generated_total - tail_start - 1)

    for voice in range(2):
        state_i = FIELD_NAMES.index(f"v{voice}_state")
        pitch_i = FIELD_NAMES.index(f"v{voice}_pitch")
        mel_i = FIELD_NAMES.index(f"v{voice}_mel")
        dur_i = FIELD_NAMES.index(f"v{voice}_dur")
        tie_i = FIELD_NAMES.index(f"v{voice}_tie")
        degree_i = FIELD_NAMES.index(f"v{voice}_degree")

        prev_state = prev[state_i]
        prev_pitch = prev[pitch_i]
        state_logits = logits[f"v{voice}_state"][0, -1].clone()
        rest_run = _consecutive_rest_slices(rows, voice)
        same_pitch_run = _consecutive_same_pitch_slices(rows, voice)
        force_active = in_tail or rest_run >= max_rest_run_slices
        if force_active:
            state_logits[STATE_REST] -= 100.0
        if not (prev_state in {STATE_NOTE, STATE_HOLD} and prev_pitch > 0):
            state_logits[STATE_HOLD] -= 100.0
        if same_pitch_run >= max_same_pitch_slices:
            state_logits[STATE_HOLD] -= 100.0
        state = _sample(state_logits, temperature=temperature, top_k=3, argmax=argmax)
        if state == STATE_HOLD and not (prev_state in {STATE_NOTE, STATE_HOLD} and prev_pitch > 0):
            state = STATE_NOTE
        if force_active and state == STATE_REST:
            state = STATE_NOTE

        if state == STATE_REST:
            row[state_i] = STATE_REST
            row[pitch_i] = 0
            row[mel_i] = 0
            row[dur_i] = 0
            row[tie_i] = 0
            row[degree_i] = 0
            active.append(None)
            continue

        if state == STATE_HOLD:
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
        _apply_pitch_bias(pitch_logits, template, key_bias=key_bias)
        _apply_motif_pitch_bias(pitch_logits, rows, voice, motif_bias=motif_bias)
        _apply_same_pitch_penalty(
            pitch_logits,
            rows,
            voice,
            max_same_pitch_slices=max_same_pitch_slices,
            same_pitch_penalty=same_pitch_penalty,
        )
        if in_tail:
            _apply_cadence_pitch_bias(
                pitch_logits,
                template,
                voice,
                cadence_bias=cadence_bias,
                tail_phase=tail_phase,
            )
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
    return row


def _normalize_next_row(row: list[int], rows: list[list[int]], template: InstrumentalV3Piece) -> list[int]:
    out = [max(0, min(FEATURE_SPECS[name] - 1, int(value))) for name, value in zip(FIELD_NAMES, row)]
    idx = len(rows)
    out[FIELD_NAMES.index("bar")] = min(FEATURE_SPECS["bar"] - 1, idx // template.steps_per_bar)
    out[FIELD_NAMES.index("pos")] = min(FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
    _set_phrase_fields(out)
    out[FIELD_NAMES.index("key_pc")] = template.key_pc
    out[FIELD_NAMES.index("mode")] = template.mode
    out[FIELD_NAMES.index("voice_count")] = 2

    prev = rows[-1]
    active = []
    for voice in range(2):
        state_i = FIELD_NAMES.index(f"v{voice}_state")
        pitch_i = FIELD_NAMES.index(f"v{voice}_pitch")
        dur_i = FIELD_NAMES.index(f"v{voice}_dur")
        tie_i = FIELD_NAMES.index(f"v{voice}_tie")
        state = out[state_i]
        prev_state = prev[state_i]
        prev_pitch = prev[pitch_i]
        if state == STATE_HOLD and prev_state in {STATE_NOTE, STATE_HOLD} and prev_pitch > 0:
            out[pitch_i] = prev_pitch
            out[tie_i] = 1
            out[dur_i] = max(1, out[dur_i])
        elif state == STATE_REST:
            out[pitch_i] = 0
            out[dur_i] = 0
            out[tie_i] = 0
        else:
            out[state_i] = STATE_NOTE
            if out[pitch_i] <= 0:
                out[pitch_i] = prev_pitch if prev_pitch > 0 else (48 if voice == 0 else 60)
            out[dur_i] = max(1, out[dur_i])
            out[tie_i] = 0
        active.append(out[pitch_i] if out[pitch_i] > 0 else None)

    vi = FIELD_NAMES.index("vertical_interval")
    ci = FIELD_NAMES.index("consonance")
    si = FIELD_NAMES.index("spacing")
    if active[0] is None or active[1] is None:
        out[vi] = out[ci] = out[si] = 0
    else:
        spacing = min(48, abs(active[1] - active[0])) + 1
        out[vi] = out[si] = spacing
        pc = (spacing - 1) % 12
        out[ci] = 1 if pc in {0, 7} else 2 if pc in {3, 4, 8, 9} else 3
    return out


def _reset_positions(rows: list[list[int]], template: InstrumentalV3Piece) -> list[list[int]]:
    out = []
    for idx, row in enumerate(rows):
        new_row = row[:]
        new_row[FIELD_NAMES.index("bar")] = min(FEATURE_SPECS["bar"] - 1, idx // template.steps_per_bar)
        new_row[FIELD_NAMES.index("pos")] = min(FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
        _set_phrase_fields(new_row)
        out.append(new_row)
    return out


def _set_phrase_fields(row: list[int]) -> None:
    if "phrase_pos" not in FIELD_NAMES:
        return
    bar = row[FIELD_NAMES.index("bar")]
    phrase_pos = bar % FEATURE_SPECS["phrase_pos"]
    row[FIELD_NAMES.index("phrase_pos")] = phrase_pos
    row[FIELD_NAMES.index("cadence_zone")] = 1 if phrase_pos in {6, 7} else 0


def _last_note_pitch(rows: list[list[int]], voice: int, *, fallback: int) -> int:
    state_i = FIELD_NAMES.index(f"v{voice}_state")
    pitch_i = FIELD_NAMES.index(f"v{voice}_pitch")
    for row in reversed(rows):
        if row[state_i] == STATE_NOTE and row[pitch_i] > 0:
            return row[pitch_i]
    return fallback


def _consecutive_rest_slices(rows: list[list[int]], voice: int) -> int:
    state_i = FIELD_NAMES.index(f"v{voice}_state")
    count = 0
    for row in reversed(rows):
        if row[state_i] != STATE_REST:
            break
        count += 1
    return count


def _consecutive_same_pitch_slices(rows: list[list[int]], voice: int) -> int:
    state_i = FIELD_NAMES.index(f"v{voice}_state")
    pitch_i = FIELD_NAMES.index(f"v{voice}_pitch")
    current_pitch = None
    count = 0
    for row in reversed(rows):
        state = row[state_i]
        pitch = row[pitch_i]
        if state not in {STATE_NOTE, STATE_HOLD} or pitch <= 0:
            break
        if current_pitch is None:
            current_pitch = pitch
        if pitch != current_pitch:
            break
        count += 1
    return count


def _encode_melody(delta: int) -> int:
    return max(1, min(49, int(delta) + 25))


def _scale_degree_id(pitch: int, key_pc: int, mode: int) -> int:
    if key_pc >= 12 or pitch <= 0:
        return 0
    rel = (pitch - key_pc) % 12
    major = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    minor = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}
    return (minor if mode == 1 else major).get(rel, 8 + rel % 5)


def _apply_pitch_bias(logits: torch.Tensor, template: InstrumentalV3Piece, *, key_bias: float) -> None:
    if key_bias <= 0 or template.key_pc >= 12:
        return
    scale = {0, 2, 4, 5, 7, 9, 11} if template.mode == 0 else {0, 2, 3, 5, 7, 8, 10, 11}
    for pitch in range(1, min(129, logits.numel())):
        if (pitch - template.key_pc) % 12 not in scale:
            logits[pitch] -= key_bias


def _apply_motif_pitch_bias(
    logits: torch.Tensor,
    rows: list[list[int]],
    voice: int,
    *,
    motif_bias: float,
) -> None:
    if motif_bias <= 0:
        return
    previous_pitch = _last_note_pitch(rows, voice, fallback=0)
    if previous_pitch <= 0:
        return

    mel_i = FIELD_NAMES.index(f"v{voice}_mel")
    state_i = FIELD_NAMES.index(f"v{voice}_state")
    motif_intervals: set[int] = set()
    for row in rows[-96:]:
        if row[state_i] != STATE_NOTE:
            continue
        encoded = row[mel_i]
        if encoded <= 0:
            continue
        interval = encoded - 25
        if -12 <= interval <= 12:
            motif_intervals.add(interval)

    # Keep motivic recurrence, but always allow idiomatic stepwise motion.
    motif_intervals.update({-3, -2, -1, 1, 2, 3})
    for pitch in range(1, min(129, logits.numel())):
        interval = pitch - previous_pitch
        if interval in motif_intervals:
            logits[pitch] += motif_bias
        elif abs(interval) > 12:
            logits[pitch] -= motif_bias


def _apply_same_pitch_penalty(
    logits: torch.Tensor,
    rows: list[list[int]],
    voice: int,
    *,
    max_same_pitch_slices: int,
    same_pitch_penalty: float,
) -> None:
    if same_pitch_penalty <= 0:
        return
    run = _consecutive_same_pitch_slices(rows, voice)
    if run < max_same_pitch_slices:
        return
    previous_pitch = _last_note_pitch(rows, voice, fallback=0)
    if 0 < previous_pitch < logits.numel():
        logits[previous_pitch] -= same_pitch_penalty


def _apply_cadence_pitch_bias(
    logits: torch.Tensor,
    template: InstrumentalV3Piece,
    voice: int,
    *,
    cadence_bias: float,
    tail_phase: float,
) -> None:
    if cadence_bias <= 0 or template.key_pc >= 12:
        return
    tonic = template.key_pc
    dominant = (tonic + 7) % 12
    third = (tonic + (3 if template.mode == 1 else 4)) % 12
    fifth = dominant
    tonic_triad = {tonic, third, fifth}
    dominant_triad = {dominant, (dominant + 4) % 12, (dominant + 7) % 12}

    # Early tail can lean dominant; final tail should resolve to tonic-triad tones.
    target_pcs = tonic_triad if tail_phase >= 0.55 else tonic_triad | dominant_triad
    for pitch in range(1, min(129, logits.numel())):
        pc = pitch % 12
        if pc in target_pcs:
            logits[pitch] += cadence_bias
        if tail_phase >= 0.8:
            if voice == 0 and pc == tonic and 36 <= pitch <= 60:
                logits[pitch] += cadence_bias
            elif voice == 1 and pc in tonic_triad and 55 <= pitch <= 84:
                logits[pitch] += cadence_bias * 0.75


def _derive_vertical(row: list[int], active: list[int | None]) -> None:
    vi = FIELD_NAMES.index("vertical_interval")
    ci = FIELD_NAMES.index("consonance")
    si = FIELD_NAMES.index("spacing")
    if active[0] is None or active[1] is None:
        row[vi] = row[ci] = row[si] = 0
        return
    spacing = min(48, abs(active[1] - active[0])) + 1
    row[vi] = row[si] = spacing
    pc = (spacing - 1) % 12
    row[ci] = 1 if pc in {0, 7} else 2 if pc in {3, 4, 8, 9} else 3


if __name__ == "__main__":
    main()
