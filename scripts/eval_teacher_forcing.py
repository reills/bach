"""Evaluate teacher-forced next-token accuracy for a trained NoteLM checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataio.collate_miditok import (  # noqa: E402
    MidiTokCollator,
    PackedBarDataset,
    PrefixControlConfig,
)
from src.dataio.dataset import BarDataset  # noqa: E402
from src.models.notelm import load_notelm_checkpoint  # noqa: E402
from src.utils.decoding.rules import token_category  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize_losses(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": round(mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _empty_counter() -> dict[str, int]:
    return {"count": 0, "top1": 0, "top5": 0}


def _rate(hits: int, count: int) -> float | None:
    if count <= 0:
        return None
    return round(hits / count, 6)


def _format_category_stats(raw: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float | None]]:
    return {
        category: {
            "count": values["count"],
            "top1_accuracy": _rate(values["top1"], values["count"]),
            "top5_accuracy": _rate(values["top5"], values["count"]),
        }
        for category, values in sorted(raw.items())
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_notelm_checkpoint(
        args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
    )
    pad_id = loaded.vocab[args.pad_token]

    dataset = BarDataset(
        str(args.events),
        str(loaded.vocab_path),
        return_tokens=False,
    )
    packed = PackedBarDataset(
        dataset,
        max_seq_len=max(1, min(args.max_seq_len, loaded.config.max_seq_len)),
        bars_per_seq=args.bars_per_seq,
        allow_truncate=args.allow_truncate,
    )
    if len(packed) == 0:
        raise SystemExit("no packed sequences; check --events, --bars-per-seq, and --max-seq-len")

    collator = MidiTokCollator(
        loaded.vocab,
        pad_token=args.pad_token,
        prefix_config=PrefixControlConfig(
            style=args.style,
            difficulty=args.difficulty,
            measures=args.measures,
            measures_token_prefix=args.measures_token_prefix,
            key_from_plan=args.key_from_plan,
            key_override=args.key,
        ),
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        prepend_bos=args.prepend_bos,
        append_eos=args.append_eos,
    )
    loader = DataLoader(
        packed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collator,
    )

    inv_vocab = {idx: token for token, idx in loaded.vocab.items()}
    model = loaded.model
    model.eval()
    device = next(model.parameters()).device

    total = _empty_counter()
    by_category: dict[str, dict[str, int]] = defaultdict(_empty_counter)
    losses: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    processed_batches = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch.ids.to(device)
            attn_mask = batch.attn_mask.to(device)
            prefix_len = batch.prefix_len.to(device)
            desc_embed = None
            if loaded.config.desc_embed_dim > 0 and batch.desc_embed is not None:
                desc_embed = batch.desc_embed.to(device)

            inputs = ids[:, :-1].contiguous()
            labels = ids[:, 1:].contiguous()
            input_attn_mask = attn_mask[:, :-1]
            label_mask = labels != pad_id

            if args.mask_prefix_loss:
                max_len = labels.size(1)
                mask_len = torch.clamp(prefix_len - 1, min=0, max=max_len)
                range_idx = torch.arange(max_len, device=device)[None, :]
                label_mask &= range_idx >= mask_len[:, None]

            logits = model(inputs, attn_mask=input_attn_mask, desc_embed=desc_embed)
            if label_mask.any():
                loss = torch.nn.functional.cross_entropy(
                    logits[label_mask],
                    labels[label_mask],
                    reduction="mean",
                )
                losses.append(float(loss.item()))

            top1 = torch.argmax(logits, dim=-1)
            k = min(5, logits.size(-1))
            top5 = torch.topk(logits, k=k, dim=-1).indices

            label_ids = labels.detach().cpu()
            label_mask_cpu = label_mask.detach().cpu()
            top1_cpu = top1.detach().cpu()
            top5_cpu = top5.detach().cpu()

            for row_idx in range(label_ids.size(0)):
                row_count = 0
                row_top1 = 0
                row_top5 = 0
                for pos_idx in range(label_ids.size(1)):
                    if not bool(label_mask_cpu[row_idx, pos_idx]):
                        continue
                    label_id = int(label_ids[row_idx, pos_idx])
                    pred_top1 = int(top1_cpu[row_idx, pos_idx])
                    pred_top5 = [int(value) for value in top5_cpu[row_idx, pos_idx].tolist()]
                    category = token_category(inv_vocab.get(label_id, str(label_id)))

                    total["count"] += 1
                    by_category[category]["count"] += 1
                    row_count += 1
                    if pred_top1 == label_id:
                        total["top1"] += 1
                        by_category[category]["top1"] += 1
                        row_top1 += 1
                    if label_id in pred_top5:
                        total["top5"] += 1
                        by_category[category]["top5"] += 1
                        row_top5 += 1

                sample_rows.append(
                    {
                        "piece_id": batch.piece_id[row_idx],
                        "bar_index": int(batch.bar_index[row_idx]),
                        "bar_count": int(batch.bar_count[row_idx]),
                        "token_count": row_count,
                        "top1_accuracy": _rate(row_top1, row_count),
                        "top5_accuracy": _rate(row_top5, row_count),
                    }
                )

            processed_batches += 1
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break

    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "vocab": str(loaded.vocab_path),
            "events": str(args.events),
            "batch_size": args.batch_size,
            "bars_per_seq": args.bars_per_seq,
            "max_seq_len": min(args.max_seq_len, loaded.config.max_seq_len),
            "mask_prefix_loss": args.mask_prefix_loss,
            "device": args.device,
            "seed": args.seed,
        },
        "sequence_count": len(sample_rows),
        "token_count": total["count"],
        "loss": _summarize_losses(losses),
        "overall": {
            "top1_accuracy": _rate(total["top1"], total["count"]),
            "top5_accuracy": _rate(total["top5"], total["count"]),
        },
        "by_category": _format_category_stats(by_category),
        "sequences": sample_rows,
    }
    (args.out_dir / "teacher_forcing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure teacher-forced next-token accuracy by token family."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("data/overfit_20_chorales/events.parquet"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bars-per-seq", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--allow-truncate", action="store_true")
    parser.add_argument("--mask-prefix-loss", action="store_true")
    parser.add_argument("--pad-token", default="<pad>")
    parser.add_argument("--bos-token", default=None)
    parser.add_argument("--eos-token", default=None)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--key", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--measures", type=int, default=None)
    parser.add_argument("--measures-token-prefix", default="MEAS")
    parser.add_argument("--no-key-from-plan", dest="key_from_plan", action="store_false")
    parser.set_defaults(key_from_plan=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.bars_per_seq <= 0:
        raise SystemExit("--bars-per-seq must be positive")
    summary = run_eval(args)
    if not args.quiet:
        print(f"summary: {args.out_dir / 'teacher_forcing_summary.json'}")
        print(f"loss: {summary['loss']}")
        print(f"overall: {summary['overall']}")
        for category, stats in summary["by_category"].items():
            print(
                f"{category}: count={stats['count']} "
                f"top1={stats['top1_accuracy']} top5={stats['top5_accuracy']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
