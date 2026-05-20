#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v4.model import CompoundConfig
from src.instrumental_v5.model import build_generator, masked_multihead_loss
from src.instrumental_v5.representation import V5_FEATURE_SPECS, V5_FIELD_NAMES
from src.instrumental_v5.tokenize import load_tokenized_split, load_v5_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny/experimental instrumental_v5 generator from tokenized tensors.")
    parser.add_argument("--data-dir", default="data/instrumental_v5/keyboard_overture_cnorm_outer2_v5_sample")
    parser.add_argument("--tokenized-dir", default=None)
    parser.add_argument("--output-dir", default="out/instrumental_v5_overfit_sample")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", "--heads", dest="n_heads", type=int, default=4)
    parser.add_argument("--n-layers", "--layers", dest="n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
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

    data_dir = Path(args.data_dir)
    tokenized_dir = Path(args.tokenized_dir) if args.tokenized_dir else data_dir / "tokenized"
    vocab = load_v5_vocab(data_dir / "vocab.json")
    train_data = load_tokenized_split(tokenized_dir / "train.pt")
    val_data = load_tokenized_split(tokenized_dir / "val.pt") if (tokenized_dir / "val.pt").exists() else None

    train_windows, train_mask = _crop(train_data["windows"], train_data["mask"], max_seq_len=args.max_seq_len)
    train_loader = DataLoader(TensorDataset(train_windows, train_mask), batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_data is not None:
        val_windows, val_mask = _crop(val_data["windows"], val_data["mask"], max_seq_len=args.max_seq_len)
        val_loader = DataLoader(TensorDataset(val_windows, val_mask), batch_size=args.batch_size, shuffle=False)

    config = CompoundConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    )
    model = build_generator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log: list[dict[str, object]] = []
    step = 0
    last_metrics: dict[str, float] = {}
    model.train()
    while step < args.max_steps:
        for windows, mask in train_loader:
            windows = windows.to(device)
            mask = mask.to(device)
            inputs = windows[:, :-1, :]
            targets = windows[:, 1:, :]
            target_mask = mask[:, 1:]
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(inputs)
                loss, metrics = masked_multihead_loss(logits, targets, target_mask, field_weights=_field_weights())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            last_metrics = metrics

            if step == 1 or step % args.log_every == 0:
                line = {
                    "step": step,
                    "loss": float(loss.item()),
                    "train": _focus_metrics(metrics),
                }
                metrics_log.append(line)
                print(json.dumps(line, sort_keys=True))
            if val_loader is not None and args.val_every > 0 and step % args.val_every == 0:
                val_loss, val_metrics = _evaluate(model, val_loader, device=device)
                line = {
                    "step": step,
                    "val_loss": val_loss,
                    "val": _focus_metrics(val_metrics),
                }
                metrics_log.append(line)
                print(json.dumps(line, sort_keys=True))
            if args.save_every > 0 and step % args.save_every == 0:
                _save_checkpoint(output_dir / f"checkpoint_step{step}.pt", model, config, args, vocab, step, last_metrics)
            if step >= args.max_steps:
                break

    ckpt_path = output_dir / f"checkpoint_step{step}.pt"
    _save_checkpoint(ckpt_path, model, config, args, vocab, step, last_metrics)
    latest_path = output_dir / "checkpoint_latest.pt"
    _save_checkpoint(latest_path, model, config, args, vocab, step, last_metrics)
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"log": metrics_log, "last_train_metrics": last_metrics}, f, indent=2, sort_keys=True)
    print(f"wrote {ckpt_path}")
    print(f"wrote {latest_path}")


def _crop(windows: torch.Tensor, mask: torch.Tensor, *, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")
    return windows[:, :max_seq_len, :].contiguous(), mask[:, :max_seq_len].contiguous()


def _evaluate(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    last_metrics: dict[str, float] = {}
    with torch.no_grad():
        for windows, mask in loader:
            windows = windows.to(device)
            mask = mask.to(device)
            logits = model(windows[:, :-1, :])
            loss, metrics = masked_multihead_loss(logits, windows[:, 1:, :], mask[:, 1:], field_weights=_field_weights())
            total_loss += float(loss.item())
            total_batches += 1
            last_metrics = metrics
    model.train()
    return total_loss / max(1, total_batches), last_metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: CompoundConfig,
    args: argparse.Namespace,
    vocab: dict[str, object],
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "field_names": V5_FIELD_NAMES,
            "feature_specs": V5_FEATURE_SPECS,
            "vocab": vocab,
            "args": vars(args),
            "step": step,
            "last_train_metrics": metrics,
        },
        path,
    )


def _field_weights() -> dict[str, float]:
    return {
        "v0_state": 2.0,
        "v1_state": 2.0,
        "v0_pitch": 2.0,
        "v1_pitch": 2.0,
        "v0_mel": 1.5,
        "v1_mel": 1.5,
        "vertical_interval": 1.5,
        "cp_v0_motion": 1.5,
        "cp_v1_motion": 1.5,
        "cp_motion_type": 1.6,
        "cp_parallel_perfect": 2.0,
        "cp_direct_perfect": 1.8,
        "cp_voice_crossing": 1.8,
        "cp_spacing_violation": 1.5,
        "phrase_role": 1.25,
        "speac_label": 1.15,
        "cmmc_function": 1.2,
        "cadence_target": 1.15,
        "harmonic_function": 1.15,
        "retrieved_contour_bucket": 1.25,
        "retrieved_rhythm_bucket": 1.25,
    }


def _focus_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "v0_state_acc",
        "v1_state_acc",
        "v0_pitch_acc",
        "v1_pitch_acc",
        "cp_motion_type_acc",
        "cp_parallel_perfect_acc",
        "cp_direct_perfect_acc",
        "phrase_role_acc",
        "speac_label_acc",
        "cmmc_function_acc",
        "cadence_target_acc",
        "retrieved_contour_bucket_acc",
        "retrieved_rhythm_bucket_acc",
    ]
    return {key: round(float(metrics.get(key, 0.0)), 4) for key in keys}


if __name__ == "__main__":
    main()
