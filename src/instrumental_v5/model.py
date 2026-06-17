from __future__ import annotations

import torch
import torch.nn.functional as F

from src.instrumental_v3.representation import STATE_NOTE, STATE_REST
from src.instrumental_v4.model import CompoundConfig, CompoundTransformer
from src.instrumental_v5.representation import (
    V5_EMI_FIELD_NAMES,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
)

V5_OBJECTIVE_NAME = "onset_aware_v1"


def generation_field_weights() -> dict[str, float]:
    """Prioritize fields that directly control generated note onsets and motion."""

    weights = {name: 0.2 for name in V5_FIELD_NAMES}
    weights.update(
        {
            "bar": 0.05,
            "pos": 0.05,
            "phrase_pos": 0.05,
            "cadence_zone": 0.1,
            "key_pc": 0.05,
            "mode": 0.05,
            "voice_count": 0.05,
            "vertical_interval": 0.75,
            "consonance": 0.35,
            "spacing": 0.25,
            "cp_v0_motion": 1.0,
            "cp_v1_motion": 1.0,
            "cp_motion_type": 0.75,
            "cp_prev_interval_class": 0.35,
            "cp_curr_interval_class": 0.5,
            "cp_parallel_perfect": 0.2,
            "cp_direct_perfect": 0.2,
            "cp_voice_crossing": 0.2,
            "cp_spacing_violation": 0.2,
        }
    )
    for voice in (0, 1):
        weights.update(
            {
                f"v{voice}_state": 2.5,
                f"v{voice}_pitch": 1.0,
                f"v{voice}_mel": 3.0,
                f"v{voice}_dur": 1.25,
                f"v{voice}_tie": 0.1,
                f"v{voice}_degree": 0.5,
            }
        )
    for name in V5_EMI_FIELD_NAMES:
        weights[name] = 0.2
    for name in V5_FIELD_NAMES:
        if name.startswith("plan_"):
            weights[name] = 0.1
    return weights


def generation_target_masks(targets: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build per-head masks that match how fields are consumed during generation."""

    if targets.dim() != 3 or targets.size(-1) != len(V5_FIELD_NAMES):
        raise ValueError(f"expected targets with shape (batch, seq, {len(V5_FIELD_NAMES)})")
    if targets.shape[:2] != mask.shape:
        raise ValueError("targets and mask must agree on batch/sequence shape")

    base = mask.bool()
    masks = {name: base for name in V5_FIELD_NAMES}
    note_masks: dict[int, torch.Tensor] = {}
    active_masks: dict[int, torch.Tensor] = {}
    for voice in (0, 1):
        state = targets[..., V5_FIELD_NAMES.index(f"v{voice}_state")]
        note = base & state.eq(STATE_NOTE)
        active = base & state.ne(STATE_REST)
        note_masks[voice] = note
        active_masks[voice] = active
        masks[f"v{voice}_pitch"] = note
        masks[f"v{voice}_mel"] = note & targets[..., V5_FIELD_NAMES.index(f"v{voice}_mel")].ne(0)
        masks[f"v{voice}_dur"] = note
        masks[f"v{voice}_tie"] = active
        masks[f"v{voice}_degree"] = note

    both_active = active_masks[0] & active_masks[1]
    for name in ("vertical_interval", "consonance", "spacing", "cp_voice_crossing", "cp_spacing_violation"):
        masks[name] = both_active

    transition_valid = base & targets[
        ..., V5_FIELD_NAMES.index("cp_motion_type")
    ].ne(0)
    for name in (
        "cp_v0_motion",
        "cp_v1_motion",
        "cp_motion_type",
        "cp_prev_interval_class",
        "cp_curr_interval_class",
        "cp_parallel_perfect",
        "cp_direct_perfect",
    ):
        masks[name] = transition_valid
    return masks


def objective_metadata() -> dict[str, object]:
    return {
        "name": V5_OBJECTIVE_NAME,
        "field_weights": generation_field_weights(),
        "target_masking": {
            "pitch_duration_degree": "NOTE rows",
            "melodic_interval": "NOTE rows with a defined previous-note interval",
            "vertical": "rows with both voices active",
            "counterpoint_transition": "rows with a defined two-voice transition",
        },
    }


def build_generator(config: CompoundConfig) -> CompoundTransformer:
    return CompoundTransformer(config, V5_FIELD_NAMES, V5_FEATURE_SPECS)


def masked_multihead_loss(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    field_weights: dict[str, float] | None = None,
    field_masks: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if targets.shape[:2] != mask.shape:
        raise ValueError("targets and mask must agree on batch/sequence shape")
    base_active = mask.reshape(-1).bool()
    if not bool(base_active.any()):
        raise ValueError("loss mask has no active targets")

    weights = field_weights or {}
    losses = []
    active_weights = []
    metrics: dict[str, float] = {}
    for field_idx, name in enumerate(V5_FIELD_NAMES):
        weight = float(weights.get(name, 1.0))
        if weight <= 0:
            continue
        active = base_active
        if field_masks is not None and name in field_masks:
            field_mask = field_masks[name]
            if field_mask.shape != mask.shape:
                raise ValueError(f"field mask for {name!r} must match the base mask shape")
            active = active & field_mask.reshape(-1).bool()
        target_count = int(active.sum().item())
        if target_count == 0:
            metrics[f"{name}_count"] = 0.0
            continue
        field_logits = logits[name].reshape(-1, logits[name].size(-1))
        field_targets = targets[..., field_idx].reshape(-1).clamp(min=0, max=logits[name].size(-1) - 1)
        selected_logits = field_logits[active]
        selected_targets = field_targets[active]
        loss = F.cross_entropy(selected_logits, selected_targets)
        losses.append(loss * weight)
        active_weights.append(weight)
        with torch.no_grad():
            pred = selected_logits.argmax(dim=-1)
            metrics[f"{name}_acc"] = (pred == selected_targets).float().mean().item()
            metrics[f"{name}_count"] = float(target_count)
    if not losses:
        raise ValueError("no active field targets after applying field masks and weights")
    return torch.stack(losses).sum() / sum(active_weights), metrics
