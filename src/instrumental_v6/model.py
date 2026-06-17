from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.instrumental_v6.representation import (
    GLOBAL_FEATURE_SPECS,
    GLOBAL_FIELD_NAMES,
    METER_NAMES,
    PAIR_FEATURE_SPECS,
    PAIR_FIELD_NAMES,
    STATE_NOTE,
    STATE_REST,
    VOICE_FEATURE_SPECS,
    VOICE_FIELD_NAMES,
)

V6_OBJECTIVE_NAME = "factorized_voice_axis_v1"
VOICE_AWARE_OBJECTIVE_NAME = "factorized_voice_axis_v2"
LEGACY_METER_VOCAB_SIZE = 10


@dataclass
class FactorizedConfig:
    max_voices: int = 6
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    n_cross_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 512
    meter_vocab_size: int = len(METER_NAMES)
    architecture: str = "voice_aware_v2"


class FactorizedCounterpointTransformer(nn.Module):
    def __init__(self, config: FactorizedConfig) -> None:
        super().__init__()
        self.config = config
        self.global_feature_specs = dict(GLOBAL_FEATURE_SPECS)
        self.global_feature_specs["meter"] = config.meter_vocab_size
        self.global_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, config.d_model)
                for name, size in self.global_feature_specs.items()
            }
        )
        self.voice_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, config.d_model) for name, size in VOICE_FEATURE_SPECS.items()}
        )
        self.pair_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, config.d_model) for name, size in PAIR_FEATURE_SPECS.items()}
        )
        self.voice_index_embedding = nn.Embedding(config.max_voices, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
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
        self.global_norm = nn.LayerNorm(config.d_model)
        self.voice_norm = nn.LayerNorm(config.d_model)
        self.pair_norm = nn.LayerNorm(config.d_model)
        self.global_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.d_model, size)
                for name, size in self.global_feature_specs.items()
            }
        )
        self.voice_heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model, size) for name, size in VOICE_FEATURE_SPECS.items()}
        )
        self.pair_heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model, size) for name, size in PAIR_FEATURE_SPECS.items()}
        )

    def forward(
        self,
        global_values: torch.Tensor,
        voice_values: torch.Tensor,
        pair_values: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if global_values.dim() != 3:
            raise ValueError("global_values must have shape (batch, seq, global_fields)")
        if voice_values.dim() != 4:
            raise ValueError("voice_values must have shape (batch, seq, voices, voice_fields)")
        if pair_values.dim() != 5:
            raise ValueError("pair_values must have shape (batch, seq, voices, voices, pair_fields)")
        batch, seq_len, voice_count, _ = voice_values.shape
        if voice_count != self.config.max_voices:
            raise ValueError("voice axis does not match model max_voices")
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds model max_seq_len")

        global_hidden = torch.zeros(batch, seq_len, self.config.d_model, device=global_values.device)
        for index, name in enumerate(GLOBAL_FIELD_NAMES):
            global_hidden += self.global_embeddings[name](
                global_values[..., index].clamp(0, self.global_feature_specs[name] - 1)
            )

        voice_hidden = torch.zeros(
            batch,
            seq_len,
            voice_count,
            self.config.d_model,
            device=voice_values.device,
        )
        for index, name in enumerate(VOICE_FIELD_NAMES):
            voice_hidden += self.voice_embeddings[name](
                voice_values[..., index].clamp(0, VOICE_FEATURE_SPECS[name] - 1)
            )
        voice_indices = torch.arange(voice_count, device=voice_values.device)
        voice_hidden += self.voice_index_embedding(voice_indices).view(1, 1, voice_count, -1)

        pair_hidden = torch.zeros(
            batch,
            seq_len,
            voice_count,
            voice_count,
            self.config.d_model,
            device=pair_values.device,
        )
        for index, name in enumerate(PAIR_FIELD_NAMES):
            pair_hidden += self.pair_embeddings[name](
                pair_values[..., index].clamp(0, PAIR_FEATURE_SPECS[name] - 1)
            )

        voice_count_values = global_values[..., GLOBAL_FIELD_NAMES.index("voice_count")]
        voice_mask = (
            torch.arange(voice_count, device=global_values.device).view(1, 1, voice_count)
            < voice_count_values.unsqueeze(-1)
        )
        voice_denominator = voice_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(global_hidden.dtype)
        pooled_voice = (voice_hidden * voice_mask.unsqueeze(-1)).sum(dim=2) / voice_denominator

        pair_mask = voice_mask.unsqueeze(3) & voice_mask.unsqueeze(2)
        pair_mask &= torch.triu(
            torch.ones(voice_count, voice_count, dtype=torch.bool, device=global_values.device),
            diagonal=1,
        ).view(1, 1, voice_count, voice_count)
        pair_denominator = pair_mask.sum(dim=(2, 3), keepdim=False).clamp_min(1).unsqueeze(-1).to(global_hidden.dtype)
        pooled_pair = (pair_hidden * pair_mask.unsqueeze(-1)).sum(dim=(2, 3)) / pair_denominator

        positions = torch.arange(seq_len, device=global_values.device).view(1, seq_len)
        temporal = global_hidden + pooled_voice + pooled_pair + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=global_values.device),
            diagonal=1,
        )
        temporal = self.global_norm(self.blocks(temporal, mask=causal_mask))
        per_voice = self.voice_norm(temporal.unsqueeze(2) + voice_hidden)
        per_pair = self.pair_norm(
            temporal.unsqueeze(2).unsqueeze(3)
            + voice_hidden.unsqueeze(3)
            + voice_hidden.unsqueeze(2)
            + pair_hidden
        )
        return {
            "global": {name: head(temporal) for name, head in self.global_heads.items()},
            "voice": {name: head(per_voice) for name, head in self.voice_heads.items()},
            "pair": {name: head(per_pair) for name, head in self.pair_heads.items()},
        }


class CrossVoiceBlock(nn.Module):
    def __init__(self, config: FactorizedConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        inactive_voices: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=inactive_voices,
            need_weights=False,
        )
        values = values + self.attention_dropout(attended)
        return values + self.feedforward(self.feedforward_norm(values))


class VoiceAwareFactorizedCounterpointTransformer(nn.Module):
    """Preserve each voice's temporal history before combining the ensemble."""

    def __init__(self, config: FactorizedConfig) -> None:
        super().__init__()
        self.config = config
        self.global_feature_specs = dict(GLOBAL_FEATURE_SPECS)
        self.global_feature_specs["meter"] = config.meter_vocab_size
        self.global_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, config.d_model)
                for name, size in self.global_feature_specs.items()
            }
        )
        self.voice_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, config.d_model) for name, size in VOICE_FEATURE_SPECS.items()}
        )
        self.pair_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, config.d_model) for name, size in PAIR_FEATURE_SPECS.items()}
        )
        self.voice_index_embedding = nn.Embedding(config.max_voices, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_blocks = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.n_layers,
        )
        self.cross_voice_blocks = nn.ModuleList(
            CrossVoiceBlock(config) for _ in range(config.n_cross_layers)
        )
        self.global_norm = nn.LayerNorm(config.d_model)
        self.voice_norm = nn.LayerNorm(config.d_model)
        self.pair_norm = nn.LayerNorm(config.d_model)
        self.global_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.d_model, size)
                for name, size in self.global_feature_specs.items()
            }
        )
        self.voice_heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model, size) for name, size in VOICE_FEATURE_SPECS.items()}
        )
        self.pair_heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model, size) for name, size in PAIR_FEATURE_SPECS.items()}
        )

    def forward(
        self,
        global_values: torch.Tensor,
        voice_values: torch.Tensor,
        pair_values: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if global_values.dim() != 3:
            raise ValueError("global_values must have shape (batch, seq, global_fields)")
        if voice_values.dim() != 4:
            raise ValueError("voice_values must have shape (batch, seq, voices, voice_fields)")
        if pair_values.dim() != 5:
            raise ValueError("pair_values must have shape (batch, seq, voices, voices, pair_fields)")
        batch, seq_len, voice_count, _ = voice_values.shape
        if voice_count != self.config.max_voices:
            raise ValueError("voice axis does not match model max_voices")
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds model max_seq_len")

        global_hidden = torch.zeros(batch, seq_len, self.config.d_model, device=global_values.device)
        for index, name in enumerate(GLOBAL_FIELD_NAMES):
            global_hidden += self.global_embeddings[name](
                global_values[..., index].clamp(0, self.global_feature_specs[name] - 1)
            )

        voice_hidden = torch.zeros(
            batch,
            seq_len,
            voice_count,
            self.config.d_model,
            device=voice_values.device,
        )
        for index, name in enumerate(VOICE_FIELD_NAMES):
            voice_hidden += self.voice_embeddings[name](
                voice_values[..., index].clamp(0, VOICE_FEATURE_SPECS[name] - 1)
            )
        voice_indices = torch.arange(voice_count, device=voice_values.device)
        voice_hidden += self.voice_index_embedding(voice_indices).view(1, 1, voice_count, -1)

        positions = self.position_embedding(
            torch.arange(seq_len, device=global_values.device)
        ).view(1, seq_len, 1, -1)
        per_voice = global_hidden.unsqueeze(2) + voice_hidden + positions
        per_voice = per_voice.transpose(1, 2).reshape(
            batch * voice_count,
            seq_len,
            self.config.d_model,
        )
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=global_values.device),
            diagonal=1,
        )
        per_voice = self.temporal_blocks(per_voice, mask=causal_mask)
        per_voice = per_voice.reshape(
            batch,
            voice_count,
            seq_len,
            self.config.d_model,
        ).transpose(1, 2)

        active_voice_count = global_values[..., GLOBAL_FIELD_NAMES.index("voice_count")]
        voice_axis = torch.arange(voice_count, device=global_values.device).view(1, 1, voice_count)
        voice_mask = voice_axis < active_voice_count.unsqueeze(-1)
        cross_values = per_voice.reshape(batch * seq_len, voice_count, self.config.d_model)
        inactive_voices = (~voice_mask).reshape(batch * seq_len, voice_count)
        for block in self.cross_voice_blocks:
            cross_values = block(cross_values, inactive_voices=inactive_voices)
        per_voice = self.voice_norm(
            cross_values.reshape(batch, seq_len, voice_count, self.config.d_model)
        )

        denominator = voice_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(per_voice.dtype)
        ensemble = self.global_norm(
            (per_voice * voice_mask.unsqueeze(-1)).sum(dim=2) / denominator
        )

        pair_hidden = torch.zeros(
            batch,
            seq_len,
            voice_count,
            voice_count,
            self.config.d_model,
            device=pair_values.device,
        )
        for index, name in enumerate(PAIR_FIELD_NAMES):
            pair_hidden += self.pair_embeddings[name](
                pair_values[..., index].clamp(0, PAIR_FEATURE_SPECS[name] - 1)
            )
        per_pair = self.pair_norm(
            ensemble.unsqueeze(2).unsqueeze(3)
            + per_voice.unsqueeze(3)
            + per_voice.unsqueeze(2)
            + pair_hidden
        )
        return {
            "global": {name: head(ensemble) for name, head in self.global_heads.items()},
            "voice": {name: head(per_voice) for name, head in self.voice_heads.items()},
            "pair": {name: head(per_pair) for name, head in self.pair_heads.items()},
        }


def build_generator(config: FactorizedConfig) -> nn.Module:
    if config.architecture == "pooled_v1":
        return FactorizedCounterpointTransformer(config)
    if config.architecture == "voice_aware_v2":
        return VoiceAwareFactorizedCounterpointTransformer(config)
    raise ValueError(f"unknown instrumental_v6 architecture: {config.architecture}")


def config_from_checkpoint(values: dict[str, object]) -> FactorizedConfig:
    normalized = dict(values)
    normalized.setdefault("meter_vocab_size", LEGACY_METER_VOCAB_SIZE)
    normalized.setdefault("architecture", "pooled_v1")
    normalized.setdefault("n_cross_layers", 2)
    return FactorizedConfig(**normalized)


def multihead_loss(
    logits: dict[str, dict[str, torch.Tensor]],
    global_targets: torch.Tensor,
    voice_targets: torch.Tensor,
    pair_targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = objective_weights()
    losses: list[torch.Tensor] = []
    active_weights: list[float] = []
    metrics: dict[str, float] = {}
    base = mask.bool()
    voice_count = global_targets[..., GLOBAL_FIELD_NAMES.index("voice_count")]
    voice_axis = torch.arange(voice_targets.size(2), device=voice_targets.device).view(1, 1, -1)
    valid_voice = base.unsqueeze(-1) & voice_axis.lt(voice_count.unsqueeze(-1))
    state = voice_targets[..., VOICE_FIELD_NAMES.index("state")]
    note = valid_voice & state.eq(STATE_NOTE)
    active_voice = valid_voice & state.ne(STATE_REST)

    for index, name in enumerate(GLOBAL_FIELD_NAMES):
        _append_loss(
            losses,
            active_weights,
            metrics,
            logits["global"][name],
            global_targets[..., index],
            base,
            weights[f"global.{name}"],
            f"global.{name}",
        )
    for index, name in enumerate(VOICE_FIELD_NAMES):
        field_mask = valid_voice
        if name in {"pitch", "dur", "degree"}:
            field_mask = note
        elif name == "mel":
            field_mask = note & voice_targets[..., index].ne(0)
        elif name == "tie":
            field_mask = active_voice
        _append_loss(
            losses,
            active_weights,
            metrics,
            logits["voice"][name],
            voice_targets[..., index],
            field_mask,
            weights[f"voice.{name}"],
            f"voice.{name}",
        )
        for voice in range(voice_targets.size(2)):
            _record_accuracy(
                metrics,
                logits["voice"][name][:, :, voice],
                voice_targets[:, :, voice, index],
                field_mask[:, :, voice],
                f"voice.{name}.v{voice}",
            )

    pair_axis = torch.arange(voice_targets.size(2), device=voice_targets.device)
    upper_triangle = pair_axis.view(-1, 1) < pair_axis.view(1, -1)
    valid_pair = (
        active_voice.unsqueeze(3)
        & active_voice.unsqueeze(2)
        & upper_triangle.view(1, 1, voice_targets.size(2), voice_targets.size(2))
    )
    for index, name in enumerate(PAIR_FIELD_NAMES):
        _append_loss(
            losses,
            active_weights,
            metrics,
            logits["pair"][name],
            pair_targets[..., index],
            valid_pair,
            weights[f"pair.{name}"],
            f"pair.{name}",
        )
    if not losses:
        raise ValueError("no active v6 loss targets")
    return torch.stack(losses).sum() / sum(active_weights), metrics


def objective_weights() -> dict[str, float]:
    weights = {f"global.{name}": 0.08 for name in GLOBAL_FIELD_NAMES}
    weights.update(
        {
            "voice.state": 2.5,
            "voice.pitch": 2.0,
            "voice.mel": 3.0,
            "voice.dur": 1.5,
            "voice.tie": 0.1,
            "voice.degree": 0.5,
            "pair.interval": 0.75,
            "pair.interval_class": 0.5,
            "pair.consonance": 0.35,
            "pair.motion": 0.75,
            "pair.parallel_perfect": 0.25,
            "pair.direct_perfect": 0.2,
            "pair.crossing": 0.25,
            "pair.spacing_violation": 0.25,
        }
    )
    return weights


def objective_metadata(config: FactorizedConfig) -> dict[str, object]:
    return {
        "name": (
            VOICE_AWARE_OBJECTIVE_NAME
            if config.architecture == "voice_aware_v2"
            else V6_OBJECTIVE_NAME
        ),
        "config": asdict(config),
        "weights": objective_weights(),
        "shared_voice_heads": True,
        "all_pair_heads": True,
    }


def _append_loss(
    losses: list[torch.Tensor],
    active_weights: list[float],
    metrics: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
    weight: float,
    name: str,
) -> None:
    flat_active = active.reshape(-1)
    count = int(flat_active.sum().item())
    metrics[f"{name}_count"] = float(count)
    if count == 0:
        return
    classes = logits.size(-1)
    selected_logits = logits.reshape(-1, classes)[flat_active]
    selected_targets = targets.reshape(-1)[flat_active].clamp(0, classes - 1)
    losses.append(F.cross_entropy(selected_logits, selected_targets) * weight)
    active_weights.append(weight)
    with torch.no_grad():
        metrics[f"{name}_acc"] = (
            selected_logits.argmax(dim=-1).eq(selected_targets).float().mean().item()
        )


def _record_accuracy(
    metrics: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
    name: str,
) -> None:
    flat_active = active.reshape(-1)
    count = int(flat_active.sum().item())
    metrics[f"{name}_count"] = float(count)
    if count == 0:
        return
    classes = logits.size(-1)
    selected_logits = logits.reshape(-1, classes)[flat_active]
    selected_targets = targets.reshape(-1)[flat_active].clamp(0, classes - 1)
    with torch.no_grad():
        metrics[f"{name}_acc"] = (
            selected_logits.argmax(dim=-1).eq(selected_targets).float().mean().item()
        )
