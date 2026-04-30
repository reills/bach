#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v3.data import InstrumentalV3Dataset, load_dataset
from src.instrumental_v3.metrics import evaluate_slices
from src.instrumental_v3.model import InstrumentalV3Config, InstrumentalV3Transformer, multihead_next_slice_loss, per_head_accuracy
from src.instrumental_v3.representation import FIELD_NAMES, FEATURE_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train instrumental_v3 next-slice Transformer.")
    parser.add_argument("--dataset", default="data/instrumental_v3/inventions_tiny.json")
    parser.add_argument("--output-dir", default="out/instrumental_v3_tiny")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--log-every", type=int, default=50)
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
        torch.backends.cuda.matmul.allow_tf32 = True

    pieces, meta = load_dataset(args.dataset)
    dataset = InstrumentalV3Dataset(pieces, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    config = InstrumentalV3Config(
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    )
    model = InstrumentalV3Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    field_weights = {
        "v0_state": 2.0,
        "v1_state": 2.0,
        "v0_pitch": 2.0,
        "v1_pitch": 2.0,
        "v0_mel": 1.5,
        "v1_mel": 1.5,
        "vertical_interval": 1.5,
        "consonance": 1.25,
    }

    step = 0
    last_metrics: dict[str, float] = {}
    while step < args.steps:
        for batch in loader:
            batch = batch.to(device)
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :]
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(inputs)
                loss, metrics = multihead_next_slice_loss(logits, targets, field_weights=field_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            last_metrics = metrics
            if step == 1 or step % args.log_every == 0:
                pitch_acc = (metrics["v0_pitch_acc"] + metrics["v1_pitch_acc"]) / 2
                state_acc = (metrics["v0_state_acc"] + metrics["v1_state_acc"]) / 2
                print(f"step={step} loss={loss.item():.4f} state_acc={state_acc:.3f} pitch_acc={pitch_acc:.3f} vertical={metrics['vertical_interval_acc']:.3f}")
            if step >= args.steps:
                break

    model.eval()
    full_rows = torch.tensor([s.values for s in pieces[0].slices[: args.seq_len]], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_rows[:, :-1, :])
        final_acc = per_head_accuracy(logits, full_rows[:, 1:, :])
    report = evaluate_slices(pieces[0].slices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "instrumental_v3_tiny.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "feature_specs": FEATURE_SPECS,
            "field_names": FIELD_NAMES,
            "dataset": str(args.dataset),
            "dataset_meta": meta,
            "args": vars(args),
            "step": step,
            "final_accuracy": final_acc,
        },
        ckpt_path,
    )
    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"last_batch": last_metrics, "first_piece_accuracy": final_acc, "first_piece_report": report.to_dict()}, f, indent=2)
    print(f"wrote {ckpt_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
