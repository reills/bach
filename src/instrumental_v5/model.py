from __future__ import annotations

import torch
import torch.nn.functional as F

from src.instrumental_v4.model import CompoundConfig, CompoundTransformer
from src.instrumental_v5.representation import V5_FEATURE_SPECS, V5_FIELD_NAMES


def build_generator(config: CompoundConfig) -> CompoundTransformer:
    return CompoundTransformer(config, V5_FIELD_NAMES, V5_FEATURE_SPECS)


def masked_multihead_loss(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    field_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if targets.shape[:2] != mask.shape:
        raise ValueError("targets and mask must agree on batch/sequence shape")
    active = mask.reshape(-1).bool()
    if not bool(active.any()):
        raise ValueError("loss mask has no active targets")

    weights = field_weights or {}
    losses = []
    metrics: dict[str, float] = {}
    for field_idx, name in enumerate(V5_FIELD_NAMES):
        field_logits = logits[name].reshape(-1, logits[name].size(-1))
        field_targets = targets[..., field_idx].reshape(-1).clamp(min=0, max=logits[name].size(-1) - 1)
        selected_logits = field_logits[active]
        selected_targets = field_targets[active]
        loss = F.cross_entropy(selected_logits, selected_targets)
        losses.append(loss * float(weights.get(name, 1.0)))
        with torch.no_grad():
            pred = selected_logits.argmax(dim=-1)
            metrics[f"{name}_acc"] = (pred == selected_targets).float().mean().item()
    return torch.stack(losses).sum(), metrics
