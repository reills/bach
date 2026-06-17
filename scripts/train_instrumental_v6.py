#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v6.model import (
    FactorizedConfig,
    build_generator,
    config_from_checkpoint,
    multihead_loss,
    objective_metadata,
)
from src.instrumental_v6.representation import GLOBAL_FIELD_NAMES
from src.instrumental_v6.tokenize import load_tokenized_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train factorized 2-6 voice instrumental_v6 model.")
    parser.add_argument("--data-dir", default="data/instrumental_v6/mixed_bach_v1")
    parser.add_argument("--output-dir", default="out/instrumental_v6_mixed_bach_v1")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-cross-layers", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=["pooled_v1", "voice_aware_v2"],
        default="voice_aware_v2",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-min-ratio", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--balanced-voice-counts",
        action="store_true",
        help="Sample training windows inversely to their active voice-count frequency.",
    )
    parser.add_argument("--resume", default=None, help="Resume model and optimizer state from a v6 checkpoint.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights from a v6 checkpoint but start a fresh optimizer schedule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    train_data = load_tokenized_split(data_dir / "tokenized/train.pt")
    val_data = load_tokenized_split(data_dir / "tokenized/val.pt")
    max_voices = int(train_data["max_voices"])
    config = FactorizedConfig(
        max_voices=max_voices,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_cross_layers=args.n_cross_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        architecture=args.architecture,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": asdict(config),
                "objective": objective_metadata(config),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model = build_generator(config).to(device)
    if args.resume and args.init_checkpoint:
        raise SystemExit("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        initial_config = config_from_checkpoint(initial["config"])
        if not _weight_compatible(initial_config, config):
            raise SystemExit("initial checkpoint config does not match requested model config")
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.max_steps),
        eta_min=args.lr * args.lr_min_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if config_from_checkpoint(checkpoint["config"]) != config:
            raise SystemExit("resume checkpoint config does not match requested model config")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        step = int(checkpoint.get("step", 0))
    train_loader = _loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        seq_len=args.max_seq_len,
        balanced_voice_counts=args.balanced_voice_counts,
    )
    val_loader = _loader(val_data, batch_size=args.batch_size, shuffle=False, seq_len=args.max_seq_len)

    log: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_step = 0
    model.train()
    while step < args.max_steps:
        for global_values, voice_values, pair_values, mask in train_loader:
            global_values = global_values.to(device)
            voice_values = voice_values.to(device)
            pair_values = pair_values.to(device)
            mask = mask.to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(
                    global_values[:, :-1],
                    voice_values[:, :-1],
                    pair_values[:, :-1],
                )
                loss, metrics = multihead_loss(
                    logits,
                    global_values[:, 1:],
                    voice_values[:, 1:],
                    pair_values[:, 1:],
                    mask[:, 1:],
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                line = {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": scheduler.get_last_lr()[0],
                    "train": _focus(metrics),
                }
                _emit(line, log=log, log_path=log_path)
            if step % args.val_every == 0:
                val_loss, val_metrics = _evaluate(model, val_loader, device=device)
                line = {"step": step, "val_loss": val_loss, "val": _focus(val_metrics)}
                _emit(line, log=log, log_path=log_path)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    _save(
                        output_dir / "checkpoint_best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        args,
                        step,
                    )
            if step % args.save_every == 0:
                _save(
                    output_dir / f"checkpoint_step{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    args,
                    step,
                )
                _save(
                    output_dir / "checkpoint_latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    args,
                    step,
                )
            if step >= args.max_steps:
                break

    train_loss, train_metrics = _evaluate(model, train_loader, device=device)
    val_loss, val_metrics = _evaluate(model, val_loader, device=device)
    final = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "train": _focus(train_metrics),
        "val": _focus(val_metrics),
    }
    _emit(final, log=log, log_path=log_path)
    _save(output_dir / "checkpoint_latest.pt", model, optimizer, scheduler, scaler, config, args, step)
    _save(
        output_dir / f"checkpoint_step{step}.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        args,
        step,
    )
    (output_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "objective": objective_metadata(config),
                "log": log,
                "final_train_metrics": train_metrics,
                "final_val_metrics": val_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _emit(
    line: dict[str, object],
    *,
    log: list[dict[str, object]],
    log_path: Path,
) -> None:
    log.append(line)
    encoded = json.dumps(line, sort_keys=True)
    print(encoded, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def _loader(
    data: dict[str, object],
    *,
    batch_size: int,
    shuffle: bool,
    seq_len: int,
    balanced_voice_counts: bool = False,
) -> DataLoader:
    global_values = data["global_values"][:, :seq_len]  # type: ignore[index]
    dataset = TensorDataset(
        global_values,
        data["voice_values"][:, :seq_len],  # type: ignore[index]
        data["pair_values"][:, :seq_len],  # type: ignore[index]
        data["mask"][:, :seq_len],  # type: ignore[index]
    )
    sampler = None
    if shuffle and balanced_voice_counts:
        voice_counts = global_values[:, 0, GLOBAL_FIELD_NAMES.index("voice_count")]
        frequencies = torch.bincount(voice_counts)
        weights = torch.tensor(
            [1.0 / max(1, int(frequencies[value])) for value in voice_counts],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    was_training = model.training
    model.eval()
    total_loss = total_tokens = 0.0
    correct: dict[str, float] = {}
    counts: dict[str, float] = {}
    with torch.no_grad():
        for global_values, voice_values, pair_values, mask in loader:
            global_values = global_values.to(device)
            voice_values = voice_values.to(device)
            pair_values = pair_values.to(device)
            mask = mask.to(device)
            logits = model(global_values[:, :-1], voice_values[:, :-1], pair_values[:, :-1])
            loss, metrics = multihead_loss(
                logits,
                global_values[:, 1:],
                voice_values[:, 1:],
                pair_values[:, 1:],
                mask[:, 1:],
            )
            tokens = float(mask[:, 1:].sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            for key, value in metrics.items():
                if not key.endswith("_acc"):
                    continue
                base = key.removesuffix("_acc")
                count = metrics.get(f"{base}_count", 0.0)
                correct[key] = correct.get(key, 0.0) + value * count
                counts[base] = counts.get(base, 0.0) + count
    if was_training:
        model.train()
    aggregate = {
        key: value / max(1.0, counts[key.removesuffix("_acc")])
        for key, value in correct.items()
    }
    return total_loss / max(1.0, total_tokens), aggregate


def _focus(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "voice.state_acc",
        "voice.pitch_acc",
        "voice.mel_acc",
        "voice.dur_acc",
        "pair.motion_acc",
        "pair.parallel_perfect_acc",
        "global.development_acc",
        "global.section_role_acc",
    ]
    return {key: round(float(metrics.get(key, 0.0)), 4) for key in keys}


def _save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: FactorizedConfig,
    args: argparse.Namespace,
    step: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": asdict(config),
            "args": vars(args),
            "step": step,
            "objective": objective_metadata(config),
        },
        path,
    )


def _weight_compatible(left: FactorizedConfig, right: FactorizedConfig) -> bool:
    return (
        left.max_voices == right.max_voices
        and left.d_model == right.d_model
        and left.n_heads == right.n_heads
        and left.n_layers == right.n_layers
        and left.n_cross_layers == right.n_cross_layers
        and left.max_seq_len == right.max_seq_len
        and left.meter_vocab_size == right.meter_vocab_size
        and left.architecture == right.architecture
    )


if __name__ == "__main__":
    main()
