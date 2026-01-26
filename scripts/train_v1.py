import argparse
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataio.dataset import BarDataset
from src.dataio.collate_miditok import (
    MidiTokCollator,
    PackedBarDataset,
    PrefixControlConfig,
)
from src.models.notelm.model import NoteLM, NoteLMConfig


def _load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"vocab file must be a dict: {path}")
    return {str(k): int(v) for k, v in data.items()}


def _save_vocab(path: Path, vocab: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def _extend_vocab(vocab: Dict[str, int], tokens: Iterable[str]) -> List[str]:
    added = []
    next_id = max(vocab.values(), default=-1) + 1
    for token in tokens:
        if token in vocab:
            continue
        vocab[token] = next_id
        next_id += 1
        added.append(token)
    return added


def _normalize_device(value: Optional[str]) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_harm_class_ids(vocab: Dict[str, int]) -> List[int]:
    ids = []
    for token, idx in vocab.items():
        if token.startswith("HARM_CLASS_"):
            ids.append(idx)
    return ids


def _build_measure_tokens(prefix: str, max_measures: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(1, max_measures + 1)]


def _normalize_control(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return cleaned.strip("_").upper()


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: NoteLM,
    optimizer: torch.optim.Optimizer,
    config: NoteLMConfig,
    vocab_path: Path,
) -> Path:
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "vocab_path": str(vocab_path),
    }
    path = output_dir / f"notelm_step{step}.pt"
    torch.save(ckpt, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NoteLM v1 (AR).")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--vocab", default="data/processed/vocab.json")
    parser.add_argument("--output-dir", default="out/notelm_v1")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--bars-per-seq", type=int, default=1)
    parser.add_argument("--allow-truncate", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rotary-pct", type=float, default=1.0)
    parser.add_argument("--rotary-base", type=int, default=10000)
    parser.add_argument("--tie-weights", action="store_true", default=True)
    parser.add_argument("--no-tie-weights", dest="tie_weights", action="store_false")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--harm-class-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--key", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--measures", type=int, default=None)
    parser.add_argument("--measures-token-prefix", default="MEAS")
    parser.add_argument("--measures-max", type=int, default=None)
    parser.add_argument("--no-key-from-plan", dest="key_from_plan", action="store_false")
    parser.set_defaults(key_from_plan=True)
    parser.add_argument("--mask-prefix-loss", action="store_true")

    parser.add_argument("--pad-token", default="<pad>")
    parser.add_argument("--bos-token", default=None)
    parser.add_argument("--eos-token", default=None)
    parser.add_argument("--unk-token", default="<unk>")
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--extend-vocab", action="store_true", default=True)
    parser.add_argument("--no-extend-vocab", dest="extend_vocab", action="store_false")

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.vocab)
    vocab = _load_vocab(vocab_path)

    required_tokens: List[str] = [args.pad_token]
    if args.prepend_bos and args.bos_token:
        required_tokens.append(args.bos_token)
    if args.append_eos and args.eos_token:
        required_tokens.append(args.eos_token)
    if args.harm_class_dropout > 0:
        required_tokens.append(args.unk_token)

    if args.key:
        required_tokens.append(f"KEY_{args.key}")
    if args.style:
        required_tokens.append(f"STYLE_{_normalize_control(args.style)}")
    if args.difficulty:
        required_tokens.append(f"DIFFICULTY_{_normalize_control(args.difficulty)}")

    if args.measures is None:
        measures_max = args.measures_max
        if measures_max is None:
            if args.bars_per_seq > 0:
                measures_max = args.bars_per_seq
            else:
                raise SystemExit(
                    "Provide --measures-max when --bars-per-seq <= 0 and --measures is not set."
                )
        required_tokens.extend(
            _build_measure_tokens(args.measures_token_prefix, measures_max)
        )
    else:
        required_tokens.append(f"{args.measures_token_prefix}_{args.measures}")

    if args.extend_vocab:
        added = _extend_vocab(vocab, required_tokens)
        vocab_path = output_dir / "vocab.json"
        _save_vocab(vocab_path, vocab)
        if added:
            print(f"Extended vocab with {len(added)} tokens.")
    else:
        missing = [tok for tok in required_tokens if tok not in vocab]
        if missing:
            raise SystemExit(
                "Missing required tokens in vocab: "
                + ", ".join(missing)
                + ". Rebuild vocab or enable --extend-vocab."
            )

    bar_token = "BAR"
    if bar_token not in vocab:
        raise SystemExit("Missing BAR token in vocab.")

    dataset = BarDataset(
        str(args.events),
        str(vocab_path),
        unk_token=args.unk_token,
    )

    packed = PackedBarDataset(
        dataset,
        max_seq_len=args.max_seq_len,
        bars_per_seq=args.bars_per_seq,
        allow_truncate=args.allow_truncate,
    )
    if len(packed) == 0:
        raise SystemExit("No sequences were built; check max_seq_len or dataset.")

    prefix_config = PrefixControlConfig(
        style=args.style,
        difficulty=args.difficulty,
        measures=args.measures,
        measures_token_prefix=args.measures_token_prefix,
        key_from_plan=args.key_from_plan,
        key_override=args.key,
    )
    collator = MidiTokCollator(
        vocab,
        pad_token=args.pad_token,
        prefix_config=prefix_config,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        prepend_bos=args.prepend_bos,
        append_eos=args.append_eos,
    )

    loader = DataLoader(
        packed,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        collate_fn=collator,
    )

    config = NoteLMConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        rotary_pct=args.rotary_pct,
        rotary_base=args.rotary_base,
        bar_token_id=vocab[bar_token],
        tie_weights=args.tie_weights,
    )

    device = _normalize_device(args.device)
    _set_seed(args.seed)

    model = NoteLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    harm_class_ids = _collect_harm_class_ids(vocab)
    harm_class_ids_tensor = (
        torch.tensor(harm_class_ids, device=device) if harm_class_ids else None
    )
    unk_id = vocab.get(args.unk_token)
    if args.harm_class_dropout > 0 and unk_id is None:
        raise SystemExit("harm_class_dropout requires unk_token in vocab.")

    pad_id = vocab[args.pad_token]

    (output_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    step = 0
    start_time = time.time()
    model.train()

    for epoch in range(args.epochs):
        for batch in loader:
            ids = batch.ids.to(device)
            attn_mask = batch.attn_mask.to(device)
            prefix_len = batch.prefix_len.to(device)

            inputs = ids[:, :-1].contiguous()
            labels = ids[:, 1:].contiguous()
            attn_mask = attn_mask[:, :-1]

            if args.harm_class_dropout > 0 and harm_class_ids_tensor is not None:
                harm_mask = (inputs[..., None] == harm_class_ids_tensor).any(-1)
                drop = (
                    torch.rand_like(inputs.float()) < args.harm_class_dropout
                ) & harm_mask
                inputs = inputs.masked_fill(drop, unk_id)

            if args.mask_prefix_loss:
                max_len = labels.size(1)
                mask_len = torch.clamp(prefix_len - 1, min=0, max=max_len)
                range_idx = torch.arange(max_len, device=device)[None, :]
                mask = range_idx < mask_len[:, None]
                labels = labels.masked_fill(mask, pad_id)

            logits = model(inputs, attn_mask=attn_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id,
                label_smoothing=args.label_smoothing,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"elapsed {elapsed:.1f}s"
                )

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                path = _save_checkpoint(
                    output_dir, step, model, optimizer, config, vocab_path
                )
                print(f"Saved checkpoint to {path}")

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    path = _save_checkpoint(output_dir, step, model, optimizer, config, vocab_path)
    print(f"Saved final checkpoint to {path}")


if __name__ == "__main__":
    main()
