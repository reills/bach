from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.instrumental_v3.representation import FEATURE_SPECS, FIELD_NAMES


@dataclass
class InstrumentalV3Config:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048


class InstrumentalV3Transformer(nn.Module):
    def __init__(self, config: InstrumentalV3Config, feature_specs: dict[str, int] | None = None) -> None:
        super().__init__()
        self.config = config
        self.feature_specs = dict(feature_specs or FEATURE_SPECS)
        self.field_names = list(FIELD_NAMES)
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(self.feature_specs[name], config.d_model) for name in self.field_names}
        )
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.ln = nn.LayerNorm(config.d_model)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model, self.feature_specs[name]) for name in self.field_names}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.size(-1) != len(self.field_names):
            raise ValueError(f"expected (batch, seq, {len(self.field_names)}) input")
        batch, seq_len, _ = x.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        hidden = torch.zeros(batch, seq_len, self.config.d_model, device=x.device)
        for field_idx, name in enumerate(self.field_names):
            values = x[..., field_idx].clamp(min=0, max=self.feature_specs[name] - 1)
            hidden = hidden + self.embeddings[name](values)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        hidden = hidden + self.pos_embedding(positions)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.blocks(hidden, mask=causal_mask)
        hidden = self.ln(hidden)
        return {name: self.heads[name](hidden) for name in self.field_names}


def multihead_next_slice_loss(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    field_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    metrics: dict[str, float] = {}
    weights = field_weights or {}
    for field_idx, name in enumerate(FIELD_NAMES):
        field_logits = logits[name].reshape(-1, logits[name].size(-1))
        field_targets = targets[..., field_idx].reshape(-1).clamp(min=0, max=logits[name].size(-1) - 1)
        loss = F.cross_entropy(field_logits, field_targets)
        weight = float(weights.get(name, 1.0))
        losses.append(loss * weight)
        with torch.no_grad():
            pred = field_logits.argmax(dim=-1)
            metrics[f"{name}_acc"] = (pred == field_targets).float().mean().item()
    return torch.stack(losses).sum(), metrics


def per_head_accuracy(logits: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    with torch.no_grad():
        for field_idx, name in enumerate(FIELD_NAMES):
            field_logits = logits[name]
            field_targets = targets[..., field_idx].clamp(min=0, max=field_logits.size(-1) - 1)
            pred = field_logits.argmax(dim=-1)
            out[name] = (pred == field_targets).float().mean().item()
    return out
