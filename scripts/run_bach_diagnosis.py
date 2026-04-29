"""Run the Bach overfit diagnostics as one command and build a listening bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_overfit_continuation import run_eval as run_continuation_eval  # noqa: E402
from scripts.eval_teacher_forcing import run_eval as run_teacher_forcing_eval  # noqa: E402
from src.api.canonical import CanonicalScore, PartInfo, tokens_to_canonical_score  # noqa: E402
from src.api.canonical.from_tokens import ParseDiagnostics  # noqa: E402
from src.api.render import canonical_score_to_midi, canonical_score_to_standard_musicxml  # noqa: E402
from src.tokens.repair import repair_harm_tokens  # noqa: E402


def _load_eval_basic():
    spec = importlib.util.spec_from_file_location(
        "eval_basic", ROOT / "scripts" / "eval_basic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summary_metric(summary: dict[str, Any], metric: str, key: str = "avg") -> float | None:
    value = summary.get("metrics", {}).get(metric, {}).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _summarize(values: Sequence[Any]) -> dict[str, float | None]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric:
        return {"avg": None, "min": None, "max": None}
    return {"avg": round(mean(numeric), 6), "min": min(numeric), "max": max(numeric)}


def _continuation_args(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    temperature: float,
    use_grammar_mask: bool,
) -> Namespace:
    return Namespace(
        checkpoint=args.checkpoint,
        vocab=args.vocab,
        events=args.events,
        out_dir=out_dir,
        samples=args.samples,
        prompt_bars=args.prompt_bars,
        continuation_bars=args.continuation_bars,
        temperature=temperature,
        top_p=1.0 if temperature <= 0 else args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_grammar_mask=use_grammar_mask,
        use_scg=args.use_scg,
        texture=args.texture,
        alpha=args.alpha,
        gamma=args.gamma,
        eos_token=args.eos_token,
        max_length=args.max_length,
        min_new_tokens=args.min_new_tokens,
        max_new_multiplier=args.max_new_multiplier,
        key=args.key,
        style=args.style,
        difficulty=args.difficulty,
        measures_token_prefix=args.measures_token_prefix,
        key_from_plan=args.key_from_plan,
        device=args.device,
        seed=args.seed,
        quiet=args.quiet,
    )


def _teacher_args(args: argparse.Namespace, *, out_dir: Path) -> Namespace:
    return Namespace(
        checkpoint=args.checkpoint,
        vocab=args.vocab,
        events=args.events,
        out_dir=out_dir,
        batch_size=args.batch_size,
        bars_per_seq=args.bars_per_seq,
        max_seq_len=args.max_length,
        allow_truncate=args.allow_truncate,
        mask_prefix_loss=args.mask_prefix_loss,
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        prepend_bos=args.prepend_bos,
        append_eos=args.append_eos,
        key=args.key,
        style=args.style,
        difficulty=args.difficulty,
        measures=args.measures,
        measures_token_prefix=args.measures_token_prefix,
        key_from_plan=args.key_from_plan,
        device=args.device,
        seed=args.seed,
        max_batches=0,
        quiet=args.quiet,
    )


def _harm_repair_report(summary: dict[str, Any], eval_basic) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sample in summary.get("samples", []):
        generated_path = sample.get("paths", {}).get("generated_tokens")
        if not generated_path:
            continue
        tokens = Path(generated_path).read_text(encoding="utf-8").split()
        raw_metrics = sample.get("metrics", {})
        repair = repair_harm_tokens(tokens)
        repaired_metrics = eval_basic.evaluate(repair.tokens)
        rows.append(
            {
                "index": sample.get("index"),
                "piece_id": sample.get("piece_id"),
                "raw_harm_mismatch_count": raw_metrics.get("harm_mismatch_count"),
                "repaired_harm_mismatch_count": repaired_metrics.get("harm_mismatch_count"),
                "repair_rewrote_events": repair.repaired_event_count,
                "repair_skipped_events": repair.skipped_event_count,
            }
        )

    return {
        "raw_harm_mismatch_count": _summarize(
            [row["raw_harm_mismatch_count"] for row in rows]
        ),
        "repaired_harm_mismatch_count": _summarize(
            [row["repaired_harm_mismatch_count"] for row in rows]
        ),
        "repair_rewrote_events": _summarize([row["repair_rewrote_events"] for row in rows]),
        "samples": rows,
    }


def _selected_samples(summary: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    samples = [
        sample
        for sample in summary.get("samples", [])
        if isinstance(sample.get("token_match_rate"), (int, float))
    ]
    if not samples:
        return []
    sorted_samples = sorted(samples, key=lambda sample: float(sample["token_match_rate"]))
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    for sample in sorted_samples[:limit]:
        selected.append({**sample, "selection": "worst"})
        if isinstance(sample.get("index"), int):
            selected_indices.add(sample["index"])
    for sample in reversed(sorted_samples[-limit:]):
        index = sample.get("index")
        if not isinstance(index, int) or index not in selected_indices:
            selected.append({**sample, "selection": "best"})
            if isinstance(index, int):
                selected_indices.add(index)
    return selected


def _render_tokens_file(tokens_path: Path, out_dir: Path, label: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = tokens_path.read_text(encoding="utf-8").split()
    diagnostics = ParseDiagnostics()
    try:
        score, rendered_tokens, render_warning = _tokens_to_listenable_score(
            tokens,
            diagnostics=diagnostics,
        )
        musicxml_path = out_dir / f"{label}.musicxml"
        midi_path = out_dir / f"{label}.mid"
        tokens_copy_path = out_dir / f"{label}.tokens.txt"
        midi_path.write_bytes(canonical_score_to_midi(score))
        tokens_copy_path.write_text(" ".join(rendered_tokens), encoding="utf-8")
        musicxml_error = None
        try:
            musicxml_path.write_text(canonical_score_to_standard_musicxml(score), encoding="utf-8")
            musicxml_value: str | None = str(musicxml_path)
        except Exception as exc:
            musicxml_error = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            musicxml_value = None
        return {
            "ok": True,
            "musicxml": musicxml_value,
            "musicxml_error": musicxml_error,
            "midi": str(midi_path),
            "tokens": str(tokens_copy_path),
            "render_warning": render_warning,
            "parse_diagnostics": diagnostics.to_dict(),
        }
    except Exception as exc:
        error_path = out_dir / f"{label}.render_error.json"
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "source_tokens": str(tokens_path),
            "parse_diagnostics": diagnostics.to_dict(),
        }
        error_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {**payload, "error": str(error_path)}


def _tokens_to_listenable_score(
    tokens: list[str],
    *,
    diagnostics: ParseDiagnostics,
) -> tuple[CanonicalScore, list[str], str | None]:
    part_info = PartInfo(id="part-0", instrument="piano", midi_program=0)
    try:
        return (
            tokens_to_canonical_score(
                tokens,
                part_info=part_info,
                ignore_invalid_events=True,
                diagnostics=diagnostics,
            ),
            tokens,
            None,
        )
    except ValueError as exc:
        if "event dur_tick must stay inside the score" not in str(exc):
            raise
        padded_tokens = [*tokens, *_closing_bar_tokens(tokens)]
        padded_diagnostics = ParseDiagnostics(max_issues=diagnostics.max_issues)
        score = tokens_to_canonical_score(
            padded_tokens,
            part_info=part_info,
            ignore_invalid_events=True,
            diagnostics=padded_diagnostics,
        )
        diagnostics.skipped_invalid_voice_events = padded_diagnostics.skipped_invalid_voice_events
        diagnostics.skipped_voice_before_pos = padded_diagnostics.skipped_voice_before_pos
        diagnostics.skipped_missing_anchor = padded_diagnostics.skipped_missing_anchor
        diagnostics.parsed_pitched_events = padded_diagnostics.parsed_pitched_events
        diagnostics.parsed_rest_events = padded_diagnostics.parsed_rest_events
        diagnostics.issues = padded_diagnostics.issues
        return score, padded_tokens, "appended one empty closing bar so sustained notes fit inside the score"


def _closing_bar_tokens(tokens: Sequence[str]) -> list[str]:
    time_sig = next((token for token in reversed(tokens) if token.startswith("TIME_SIG_")), "TIME_SIG_4_4")
    key = next((token for token in reversed(tokens) if token.startswith("KEY_")), "KEY_C")
    return ["BAR", time_sig, key]


def _tokens_to_bars(tokens: Sequence[str]) -> list[list[str]]:
    bars: list[list[str]] = []
    current: list[str] | None = None
    for token in tokens:
        if token == "BAR":
            if current is not None:
                bars.append(current)
            current = ["BAR"]
            continue
        if current is not None:
            current.append(token)
    if current is not None:
        bars.append(current)
    return bars


def _write_generated_continuation_tokens(
    generated_tokens_path: Path,
    sample_dir: Path,
    *,
    prompt_bars: int,
    continuation_bars: int,
) -> Path:
    tokens = generated_tokens_path.read_text(encoding="utf-8").split()
    bars = _tokens_to_bars(tokens)
    continuation = bars[prompt_bars : prompt_bars + continuation_bars]
    out_path = sample_dir / "generated_continuation.source_tokens.txt"
    out_path.write_text(
        " ".join(token for bar in continuation for token in bar),
        encoding="utf-8",
    )
    return out_path


def _build_listening_bundle(summary: dict[str, Any], out_dir: Path, *, limit: int) -> list[dict[str, Any]]:
    bundle_dir = out_dir / "listening"
    rendered: list[dict[str, Any]] = []
    for sample in _selected_samples(summary, limit):
        sample_label = f"{sample['selection']}_sample_{int(sample['index']):03d}"
        sample_dir = bundle_dir / sample_label
        paths = sample.get("paths", {})
        generated = paths.get("generated_tokens")
        prompt = paths.get("prompt_tokens")
        reference = paths.get("reference_tokens")
        reference_continuation = paths.get("reference_continuation_tokens")
        item = {
            "selection": sample["selection"],
            "index": sample["index"],
            "piece_id": sample.get("piece_id"),
            "token_match_rate": sample.get("token_match_rate"),
            "prompt": None,
            "generated_full": None,
            "generated_continuation": None,
            "reference_full": None,
            "reference_continuation": None,
        }
        if prompt:
            item["prompt"] = _render_tokens_file(Path(prompt), sample_dir, "prompt")
        if generated:
            item["generated_full"] = _render_tokens_file(Path(generated), sample_dir, "generated_full")
            generated_continuation_path = _write_generated_continuation_tokens(
                Path(generated),
                sample_dir,
                prompt_bars=int(sample.get("prompt_bars", 2)),
                continuation_bars=int(sample.get("continuation_bars", 6)),
            )
            item["generated_continuation"] = _render_tokens_file(
                generated_continuation_path,
                sample_dir,
                "generated_continuation",
            )
        if reference:
            item["reference_full"] = _render_tokens_file(Path(reference), sample_dir, "reference_full")
        if reference_continuation:
            item["reference_continuation"] = _render_tokens_file(
                Path(reference_continuation),
                sample_dir,
                "reference_continuation",
            )
        rendered.append(item)
    return rendered


def _plain_diagnosis(
    *,
    teacher: dict[str, Any],
    sampled: dict[str, Any],
    greedy_grammar: dict[str, Any],
    greedy_no_grammar: dict[str, Any],
    harm_repair: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    top1 = teacher.get("overall", {}).get("top1_accuracy")
    sampled_match = sampled.get("token_match_rate", {}).get("avg")
    greedy_match = greedy_grammar.get("token_match_rate", {}).get("avg")
    raw_harm = harm_repair.get("raw_harm_mismatch_count", {}).get("avg")
    repaired_harm = harm_repair.get("repaired_harm_mismatch_count", {}).get("avg")

    if isinstance(top1, (int, float)) and top1 >= 0.98:
        lines.append("Teacher forcing is excellent: the model can predict the training chorales when kept on the true path.")
    elif isinstance(top1, (int, float)):
        lines.append("Teacher forcing is not yet overfit-clean: fix training/data/representation before scaling the model.")
    else:
        lines.append("Teacher forcing did not produce a usable accuracy number.")

    if isinstance(greedy_match, (int, float)) and isinstance(sampled_match, (int, float)):
        if greedy_match < sampled_match:
            lines.append("Greedy generation is worse than sampled generation, so decoding path constraints are likely hurting continuation.")
        else:
            lines.append("Greedy generation is at least as good as sampled generation, so sampling randomness is not the main culprit.")

    grammar_violations = _summary_metric(greedy_grammar, "token_grammar_violations")
    no_grammar_violations = _summary_metric(greedy_no_grammar, "token_grammar_violations")
    if isinstance(grammar_violations, (int, float)) and isinstance(no_grammar_violations, (int, float)):
        lines.append(
            "Grammar-mask check: "
            f"masked avg violations={grammar_violations}, unmasked avg violations={no_grammar_violations}."
        )

    if isinstance(raw_harm, (int, float)) and isinstance(repaired_harm, (int, float)):
        if raw_harm > 0 and repaired_harm < raw_harm:
            lines.append("HARM_* drift is real; treat HARM_* as derived metadata after generation, not as musical truth.")
        else:
            lines.append("HARM_* repair did not materially change mismatch counts in this run.")

    lines.append("Use the MIDI files in listening/ first; the metrics only explain what you are hearing.")
    return lines


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_basic = _load_eval_basic()

    if not args.quiet:
        print("[1/4] sampled continuation", flush=True)
    sampled = run_continuation_eval(
        _continuation_args(
            args,
            out_dir=args.out_dir / "sampled",
            temperature=args.temperature,
            use_grammar_mask=True,
        )
    )

    if not args.quiet:
        print("[2/4] greedy continuation with grammar mask", flush=True)
    greedy_grammar = run_continuation_eval(
        _continuation_args(
            args,
            out_dir=args.out_dir / "greedy_grammar",
            temperature=0.0,
            use_grammar_mask=True,
        )
    )

    if not args.quiet:
        print("[3/4] greedy continuation without grammar mask", flush=True)
    greedy_no_grammar = run_continuation_eval(
        _continuation_args(
            args,
            out_dir=args.out_dir / "greedy_no_grammar",
            temperature=0.0,
            use_grammar_mask=False,
        )
    )

    if not args.quiet:
        print("[4/4] teacher forcing, harmonic repair comparison, listening bundle", flush=True)
    teacher = run_teacher_forcing_eval(_teacher_args(args, out_dir=args.out_dir / "teacher_forcing"))
    harm_repair = _harm_repair_report(sampled, eval_basic)
    listening = _build_listening_bundle(sampled, args.out_dir, limit=args.listen_limit)
    diagnosis = _plain_diagnosis(
        teacher=teacher,
        sampled=sampled,
        greedy_grammar=greedy_grammar,
        greedy_no_grammar=greedy_no_grammar,
        harm_repair=harm_repair,
    )

    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "vocab": str(args.vocab),
            "events": str(args.events),
            "samples": args.samples,
            "prompt_bars": args.prompt_bars,
            "continuation_bars": args.continuation_bars,
            "texture": args.texture,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "device": args.device,
            "seed": args.seed,
        },
        "plain_diagnosis": diagnosis,
        "teacher_forcing": {
            "path": str(args.out_dir / "teacher_forcing" / "teacher_forcing_summary.json"),
            "overall": teacher.get("overall"),
            "loss": teacher.get("loss"),
            "by_category": teacher.get("by_category"),
        },
        "continuation": {
            "sampled": {
                "path": str(args.out_dir / "sampled" / "summary.json"),
                "token_match_rate": sampled.get("token_match_rate"),
                "pass_rates": sampled.get("pass_rates"),
                "metrics": sampled.get("metrics"),
            },
            "greedy_grammar": {
                "path": str(args.out_dir / "greedy_grammar" / "summary.json"),
                "token_match_rate": greedy_grammar.get("token_match_rate"),
                "pass_rates": greedy_grammar.get("pass_rates"),
                "metrics": greedy_grammar.get("metrics"),
            },
            "greedy_no_grammar": {
                "path": str(args.out_dir / "greedy_no_grammar" / "summary.json"),
                "token_match_rate": greedy_no_grammar.get("token_match_rate"),
                "pass_rates": greedy_no_grammar.get("pass_rates"),
                "metrics": greedy_no_grammar.get("metrics"),
            },
        },
        "harmonic_repair": harm_repair,
        "listening": listening,
    }

    summary_path = args.out_dir / "diagnosis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme_path = args.out_dir / "READ_ME_FIRST.txt"
    readme_path.write_text(
        "\n".join(
            [
                "Bach overfit diagnosis",
                "",
                "Listen first:",
                *[
                    f"- {item['selection']} sample {item['index']}: "
                    f"generated continuation = "
                    f"{item.get('generated_continuation', {}).get('midi') if item.get('generated_continuation') else 'render failed'}; "
                    f"reference continuation = "
                    f"{item.get('reference_continuation', {}).get('midi') if item.get('reference_continuation') else 'render failed'}"
                    for item in listening
                ],
                "",
                "Boundary note: prompt.mid is the real chorale opening; generated_continuation.mid is only the model-composed continuation.",
                "",
                "What the automation thinks:",
                *[f"- {line}" for line in diagnosis],
                "",
                f"Full JSON summary: {summary_path}",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sampled, greedy, teacher-forced, repair, and listening diagnostics together."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("out/overfit_20_chorales/notelm_step6000.pt"))
    parser.add_argument("--vocab", type=Path, default=Path("out/overfit_20_chorales/vocab.json"))
    parser.add_argument("--events", type=Path, default=Path("data/overfit_20_chorales/events.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/diagnostics/overfit_20_chorales"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--prompt-bars", type=int, default=2)
    parser.add_argument("--continuation-bars", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--texture", type=int, default=4)
    parser.add_argument("--use-scg", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--eos-token", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=512)
    parser.add_argument("--max-new-multiplier", type=float, default=3.0)
    parser.add_argument("--key", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--measures", type=int, default=None)
    parser.add_argument("--measures-token-prefix", default="MEAS")
    parser.add_argument("--no-key-from-plan", dest="key_from_plan", action="store_false")
    parser.set_defaults(key_from_plan=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bars-per-seq", type=int, default=8)
    parser.add_argument("--allow-truncate", action="store_true")
    parser.add_argument("--mask-prefix-loss", action="store_true")
    parser.add_argument("--pad-token", default="<pad>")
    parser.add_argument("--bos-token", default=None)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--listen-limit", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.listen_limit <= 0:
        raise SystemExit("--listen-limit must be positive")
    summary = run_diagnosis(args)
    if not args.quiet:
        print(f"summary: {args.out_dir / 'diagnosis_summary.json'}")
        print(f"read first: {args.out_dir / 'READ_ME_FIRST.txt'}")
        for line in summary["plain_diagnosis"]:
            print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
