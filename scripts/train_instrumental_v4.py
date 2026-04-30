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

from src.instrumental_v4.data import V4PlanDataset, V4SliceDataset, load_v4_dataset
from src.instrumental_v4.model import (
    CompoundConfig,
    build_generator,
    build_planner,
    multihead_loss,
    per_head_accuracy,
)
from src.instrumental_v4.representation import (
    PLAN_FEATURE_SPECS,
    PLAN_FIELD_NAMES,
    V4_FEATURE_SPECS,
    V4_FIELD_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train instrumental_v4 planner and slice generator.")
    parser.add_argument("--dataset", default="data/instrumental_v4/keyboard_overture_cnorm_outer2_v4.json")
    parser.add_argument("--output-dir", default="out/instrumental_v4_broad_planner")
    parser.add_argument("--planner-steps", type=int, default=800)
    parser.add_argument("--generator-steps", type=int, default=1600)
    parser.add_argument("--plan-seq-len", type=int, default=32)
    parser.add_argument("--slice-seq-len", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--log-every", type=int, default=100)
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
        torch.backends.cudnn.allow_tf32 = True

    pieces, meta = load_v4_dataset(args.dataset)
    plan_data = V4PlanDataset(pieces, seq_len=args.plan_seq_len)
    slice_data = V4SliceDataset(pieces, seq_len=args.slice_seq_len)
    plan_loader = DataLoader(plan_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    slice_loader = DataLoader(slice_data, batch_size=args.batch_size, shuffle=True, drop_last=False)

    config = CompoundConfig(
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        max_seq_len=max(args.plan_seq_len, args.slice_seq_len),
    )
    planner = build_planner(config).to(device)
    generator = build_generator(config).to(device)

    planner_weights = {
        "plan_bass_pc": 1.5,
        "plan_top_pc": 1.5,
        "plan_bass_oct": 1.25,
        "plan_top_oct": 1.25,
        "plan_v0_density": 1.25,
        "plan_v1_density": 1.25,
        "plan_final_interval_class": 1.5,
    }
    generator_weights = {
        "v0_state": 2.0,
        "v1_state": 2.0,
        "v0_pitch": 2.0,
        "v1_pitch": 2.0,
        "v0_mel": 1.5,
        "v1_mel": 1.5,
        "vertical_interval": 1.5,
        "consonance": 1.25,
        "plan_bass_pc": 1.2,
        "plan_top_pc": 1.2,
        "plan_final_interval_class": 1.2,
    }

    planner_metrics = _train_stage(
        name="planner",
        model=planner,
        loader=plan_loader,
        steps=args.planner_steps,
        lr=args.lr,
        device=device,
        amp=args.amp,
        field_names=PLAN_FIELD_NAMES,
        field_weights=planner_weights,
        log_every=args.log_every,
    )
    generator_metrics = _train_stage(
        name="generator",
        model=generator,
        loader=slice_loader,
        steps=args.generator_steps,
        lr=args.lr,
        device=device,
        amp=args.amp,
        field_names=V4_FIELD_NAMES,
        field_weights=generator_weights,
        log_every=args.log_every,
    )

    planner.eval()
    generator.eval()
    first_plan = torch.tensor([p.values for p in pieces[0].plans[: args.plan_seq_len]], dtype=torch.long, device=device).unsqueeze(0)
    first_slice = torch.tensor(pieces[0].rows[: args.slice_seq_len], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        plan_acc = per_head_accuracy(planner(first_plan[:, :-1, :]), first_plan[:, 1:, :], PLAN_FIELD_NAMES)
        slice_acc = per_head_accuracy(generator(first_slice[:, :-1, :]), first_slice[:, 1:, :], V4_FIELD_NAMES)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "instrumental_v4.pt"
    torch.save(
        {
            "planner_state": planner.state_dict(),
            "generator_state": generator.state_dict(),
            "config": asdict(config),
            "plan_feature_specs": PLAN_FEATURE_SPECS,
            "v4_feature_specs": V4_FEATURE_SPECS,
            "plan_field_names": PLAN_FIELD_NAMES,
            "v4_field_names": V4_FIELD_NAMES,
            "dataset": str(args.dataset),
            "dataset_meta": meta,
            "args": vars(args),
            "planner_steps": args.planner_steps,
            "generator_steps": args.generator_steps,
            "first_piece_plan_accuracy": plan_acc,
            "first_piece_slice_accuracy": slice_acc,
        },
        ckpt_path,
    )
    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "planner_last_batch": planner_metrics,
                "generator_last_batch": generator_metrics,
                "first_piece_plan_accuracy": plan_acc,
                "first_piece_slice_accuracy": slice_acc,
            },
            f,
            indent=2,
        )
    print(f"wrote {ckpt_path}")
    print(f"wrote {metrics_path}")


def _train_stage(
    *,
    name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
    amp: bool,
    field_names: list[str],
    field_weights: dict[str, float],
    log_every: int,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    step = 0
    last: dict[str, float] = {}
    while step < steps:
        for batch in loader:
            batch = batch.to(device)
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :]
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(inputs)
                loss, metrics = multihead_loss(logits, targets, field_names, field_weights=field_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            last = metrics
            if step == 1 or step % log_every == 0:
                focus = _focus_line(name, metrics)
                print(f"{name} step={step} loss={loss.item():.4f} {focus}")
            if step >= steps:
                break
    return last


def _focus_line(name: str, metrics: dict[str, float]) -> str:
    if name == "planner":
        keys = ["plan_bass_pc_acc", "plan_top_pc_acc", "plan_v0_density_acc", "plan_v1_density_acc", "plan_final_interval_class_acc"]
    else:
        keys = ["v0_state_acc", "v1_state_acc", "v0_pitch_acc", "v1_pitch_acc", "vertical_interval_acc"]
    return " ".join(f"{key}={metrics.get(key, 0.0):.3f}" for key in keys)


if __name__ == "__main__":
    main()
