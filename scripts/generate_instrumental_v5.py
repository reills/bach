#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
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
from src.instrumental_v5.representation import (
    V5_COUNTERPOINT_FIELD_NAMES,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
    counterpoint_features_for_transition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/export from a v5 checkpoint.")
    parser.add_argument("--checkpoint", default="out/instrumental_v5_overfit_sample/checkpoint_latest.pt")
    parser.add_argument("--data-dir", default="data/instrumental_v5/keyboard_overture_cnorm_outer2_v5_sample")
    parser.add_argument("--out-dir", default="out/instrumental_v5_overfit_sample/generated")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--prompt-rows", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
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
        )
        reports.append(report)
    print(json.dumps({"samples": reports}, indent=2, sort_keys=True))


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
        _repair_generated_row(next_row, rows, template)
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
        hybrid_context=hybrid_context,
        candidate_count=candidate_count,
    )
    for candidate_idx, rows in enumerate(batched_rows):
        score, diagnostics = _score_candidate_rows(
            rows,
            template=template,
            prompt_row_count=len(prompt_rows),
            source_pieces=source_pieces,
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
            _repair_generated_row(next_row, rows, template)
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


def _repair_generated_row(row: list[int], rows: list[list[int]], template: InstrumentalV3Piece) -> None:
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
        if row[pitch_i] <= 0:
            row[pitch_i] = prev_pitch if prev_pitch > 0 else (48 if voice == 0 else 60)
        row[pitch_i] = _clip_midi_pitch(row[pitch_i])
        row[state_i] = STATE_NOTE
        row[dur_i] = max(1, row[dur_i])
        row[tie_i] = 0
        row[degree_i] = _scale_degree_id(row[pitch_i], template.key_pc, template.mode)
        active.append(row[pitch_i])

    _derive_vertical(row, active)
    _derive_counterpoint_transition(row, prev)
    # Plan fields are conditioning summaries. Keeping sampled values is fine, but clamp defensively.
    for name in PLAN_FIELD_NAMES:
        idx = V5_FIELD_NAMES.index(name)
        row[idx] = max(0, min(V5_FEATURE_SPECS[name] - 1, row[idx]))


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
) -> dict[str, object]:
    v3_rows = [row[: len(V3_FIELD_NAMES)] for row in rows]
    piece = slice_rows_to_piece(v3_rows, template=template, piece_id=piece_id, source_path=checkpoint)
    score = piece_to_canonical_score(piece)
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
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "piece_id": piece_id,
        "musicxml": str(xml_path),
        "midi": str(midi_path),
        "metrics": str(metrics_path),
        "rows": str(rows_path),
        "counterpoint": report.to_dict(),
        "source_overlap": novelty,
        "hybrid": hybrid_diagnostics,
        "candidate_rerank": rerank_diagnostics,
    }


def _hybrid_context_from_args(
    args: argparse.Namespace,
    *,
    template: InstrumentalV3Piece,
    data_dir: Path,
    total_rows: int,
) -> HybridContext | None:
    if not getattr(args, "hybrid_conditioning", False):
        return None
    fragment_path = _resolve_fragment_path(getattr(args, "fragment_path", None), data_dir=data_dir)
    measures = int(getattr(args, "measures", 0) or _rows_to_measures(total_rows, template.steps_per_bar))
    key = getattr(args, "key", None) or template.key or "C"
    return build_hybrid_context(
        ComposeControls(
            key=key,
            measures=measures,
            texture=int(getattr(args, "texture", 2) or 2),
        ),
        fragment_path=fragment_path,
        retrieval_limit=int(getattr(args, "retrieval_limit", 1) or 1),
    )


def _resolve_fragment_path(fragment_path: str | None, *, data_dir: Path) -> Path:
    if fragment_path:
        return Path(fragment_path)
    return data_dir / "train_emi_fragments.jsonl"


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
