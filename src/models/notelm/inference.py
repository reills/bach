import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from src.models.notelm.model import NoteLM, NoteLMConfig


PathLike = Union[str, Path]


@dataclass(frozen=True)
class LoadedNoteLM:
    model: NoteLM
    vocab: Dict[str, int]
    config: NoteLMConfig
    checkpoint_path: Path
    vocab_path: Path
    step: Optional[int]


def _load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"vocab file must be a dict: {path}")
    return {str(k): int(v) for k, v in data.items()}


def _resolve_vocab_path(
    checkpoint_path: Path,
    vocab_path: Optional[PathLike],
) -> Path:
    candidates = []
    if vocab_path is None:
        candidates.append(checkpoint_path.parent / "vocab.json")
    else:
        candidate = Path(vocab_path)
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.append(checkpoint_path.parent / candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"vocab file not found; tried: {tried}")


def load_notelm_checkpoint(
    checkpoint_path: PathLike,
    *,
    vocab_path: Optional[PathLike] = None,
    device: Union[str, torch.device] = "cpu",
) -> LoadedNoteLM:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a dict: {checkpoint_file}")

    config_data = checkpoint.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(f"checkpoint missing config dict: {checkpoint_file}")

    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint missing model_state: {checkpoint_file}")

    config = NoteLMConfig(**config_data)
    model = NoteLM(config)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    resolved_vocab_path = _resolve_vocab_path(
        checkpoint_file,
        vocab_path or checkpoint.get("vocab_path"),
    )
    vocab = _load_vocab(resolved_vocab_path)

    step = checkpoint.get("step")
    if step is not None:
        step = int(step)

    return LoadedNoteLM(
        model=model,
        vocab=vocab,
        config=config,
        checkpoint_path=checkpoint_file.resolve(),
        vocab_path=resolved_vocab_path,
        step=step,
    )
