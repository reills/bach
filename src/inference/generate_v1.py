import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from src.models.notelm import LoadedNoteLM, load_notelm_checkpoint
from src.utils.decoding.rules import MusicRules, grammar_constraints
from src.utils.decoding.sampler import Sampler
from src.utils.decoding.scg import SCGSampler, build_grammar_mask
from src.utils.decoding.voice_state import build_voice_leading_mask

_log = logging.getLogger(__name__)

TokenLike = Union[int, str]


@dataclass(frozen=True)
class GenerationConfig:
    max_length: int
    temperature: float = 0.75
    top_p: float = 0.85
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    use_scg: bool = False
    use_grammar_mask: bool = False
    use_voice_leading_mask: bool = True
    target_texture: int = 1
    bar_voice_survival_penalty: float = 8.0
    alpha: float = 0.6
    gamma: float = 0.4
    eos_token: Optional[TokenLike] = None
    use_chorale_v2_mask: bool = False
    v2_max_sonority_repeats: int = 0


@dataclass(frozen=True)
class GenerationResult:
    ids: List[int]
    tokens: List[str]
    stopped_on_eos: bool


def _resolve_token_id(token: TokenLike, vocab: dict[str, int]) -> int:
    if isinstance(token, int):
        if token < 0:
            raise ValueError(f"token ids must be non-negative: {token}")
        return token
    if token not in vocab:
        raise KeyError(f"token not found in vocab: {token}")
    return vocab[token]


def _resolve_seed_ids(seed_tokens: Sequence[TokenLike], vocab: dict[str, int]) -> List[int]:
    if not seed_tokens:
        raise ValueError("seed_tokens must not be empty")
    return [_resolve_token_id(token, vocab) for token in seed_tokens]


def _resolve_eos_id(config: GenerationConfig, vocab: dict[str, int]) -> Optional[int]:
    if config.eos_token is not None:
        return _resolve_token_id(config.eos_token, vocab)
    for token in ("<eos>", "EOS"):
        if token in vocab:
            return vocab[token]
    return None


def _device_for_model(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_sampler(vocab: dict[str, int], config: GenerationConfig) -> Union[Sampler, SCGSampler]:
    sampler = Sampler(
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
    )
    if not config.use_scg:
        return sampler
    return SCGSampler(
        sampler=sampler,
        rules=MusicRules(vocab),
        alpha=config.alpha,
        gamma=config.gamma,
    )


def _decode_ids(token_ids: Sequence[int], vocab: dict[str, int]) -> List[str]:
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return [inv_vocab.get(token_id, str(token_id)) for token_id in token_ids]


def _generate_from_loaded(
    loaded: LoadedNoteLM,
    *,
    seed_tokens: Sequence[TokenLike],
    generation_config: GenerationConfig,
) -> GenerationResult:
    seed_ids = _resolve_seed_ids(seed_tokens, loaded.vocab)
    if generation_config.max_length < len(seed_ids):
        raise ValueError(
            "max_length must be at least as large as the seed length"
        )

    model = loaded.model
    model.eval()
    device = _device_for_model(model)
    generated = torch.tensor([seed_ids], dtype=torch.long, device=device)
    sampler = _build_sampler(loaded.vocab, generation_config)
    eos_id = _resolve_eos_id(generation_config, loaded.vocab)
    stopped_on_eos = False
    inv_vocab = {idx: token for token, idx in loaded.vocab.items()}
    grammar_fallbacks = 0
    voice_leading_fallbacks = 0

    with torch.no_grad():
        while generated.size(1) < generation_config.max_length:
            window = generated[:, -loaded.config.max_seq_len :]
            logits = model(window)

            decoded_so_far = None
            constraints = None
            if (
                generation_config.use_grammar_mask
                or generation_config.use_voice_leading_mask
                or generation_config.use_chorale_v2_mask
            ):
                decoded_so_far = [inv_vocab.get(int(t), str(int(t))) for t in generated[0].tolist()]
                constraints = grammar_constraints(decoded_so_far)

            if generation_config.use_grammar_mask and decoded_so_far is not None and constraints is not None:
                mask = build_grammar_mask(
                    decoded_so_far,
                    loaded.vocab,
                    allowed_categories=set(constraints.allowed_categories),
                    allow_eos=constraints.allow_eos,
                ).to(device)
                # Apply mask by setting disallowed tokens to -inf
                if mask.any():
                    logits[:, -1, :] = logits[:, -1, :].masked_fill(~mask, float("-inf"))
                else:
                    grammar_fallbacks += 1
                    _log.debug("Grammar mask produced empty allowed set; skipping mask at step %d", generated.size(1))

            if (
                generation_config.use_voice_leading_mask
                and decoded_so_far is not None
                and constraints is not None
                and "MEL_INT12" in constraints.allowed_categories
            ):
                mask = build_voice_leading_mask(
                    decoded_so_far,
                    loaded.vocab,
                    allowed_categories=set(constraints.allowed_categories),
                ).to(device)
                masked_logits = logits[:, -1, :].masked_fill(~mask, float("-inf"))
                if torch.isfinite(masked_logits).any():
                    logits[:, -1, :] = masked_logits
                else:
                    voice_leading_fallbacks += 1
                    _log.debug(
                        "Voice-leading mask produced empty allowed set; skipping mask at step %d",
                        generated.size(1),
                    )

            if generation_config.use_chorale_v2_mask and decoded_so_far is not None:
                mask = _build_chorale_v2_mask(decoded_so_far, loaded.vocab).to(device)
                if mask.any():
                    logits[:, -1, :] = logits[:, -1, :].masked_fill(~mask, float("-inf"))

                if generation_config.v2_max_sonority_repeats > 0:
                    masked_logits = _apply_chorale_v2_repeat_mask(
                        logits[:, -1, :],
                        decoded_so_far,
                        loaded.vocab,
                        max_repeats=generation_config.v2_max_sonority_repeats,
                    )
                    logits[:, -1, :] = masked_logits

            if (
                generation_config.bar_voice_survival_penalty > 0
                and decoded_so_far is not None
                and constraints is not None
                and "BAR" in constraints.allowed_categories
            ):
                bar_token_id = loaded.vocab.get("BAR")
                missing_voice_count = _bar_survival_missing_voice_count(
                    decoded_so_far,
                    target_texture=generation_config.target_texture,
                )
                if bar_token_id is not None and missing_voice_count > 0:
                    logits[:, -1, bar_token_id] -= (
                        generation_config.bar_voice_survival_penalty * missing_voice_count
                    )

            next_token = sampler.sample(logits[:, -1, :], input_ids=window)
            generated = torch.cat([generated, next_token.to(device=device)], dim=1)
            if eos_id is not None and int(next_token.item()) == eos_id:
                stopped_on_eos = True
                break

    if grammar_fallbacks > 0:
        _log.warning("Grammar mask fell back (empty allowed set) %d times during generation.", grammar_fallbacks)
    if voice_leading_fallbacks > 0:
        _log.warning(
            "Voice-leading mask fell back (empty allowed set) %d times during generation.",
            voice_leading_fallbacks,
        )

    token_ids = generated[0].detach().cpu().tolist()
    return GenerationResult(
        ids=token_ids,
        tokens=_decode_ids(token_ids, loaded.vocab),
        stopped_on_eos=stopped_on_eos,
    )


def _bar_survival_missing_voice_count(tokens: Sequence[str], *, target_texture: int) -> int:
    required_voices = _required_bar_voice_count(target_texture)
    if required_voices <= 1:
        return 0
    voices_seen: set[int] = set()
    for token in reversed(tokens):
        if token == "BAR":
            break
        if not token.startswith("VOICE_"):
            continue
        try:
            voices_seen.add(int(token.split("_", 1)[1]))
        except ValueError:
            continue
    return max(0, required_voices - len(voices_seen))


def _required_bar_voice_count(target_texture: int) -> int:
    if isinstance(target_texture, bool):
        return 1
    return max(1, min(int(target_texture), 4) - 1)


def _build_chorale_v2_mask(tokens: Sequence[str], vocab: dict[str, int]) -> torch.Tensor:
    allowed = _chorale_v2_allowed_tokens(tokens, vocab)
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    if not allowed:
        return mask
    for token_id in allowed:
        mask[token_id] = True
    return mask


def _chorale_v2_allowed_tokens(tokens: Sequence[str], vocab: dict[str, int]) -> set[int]:
    prefixes = _chorale_v2_allowed_prefixes(tokens)
    if not prefixes:
        return set()
    partial = _chorale_v2_current_partial_sonority(tokens)
    allowed: set[int] = set()
    for token, token_id in vocab.items():
        for prefix in prefixes:
            if prefix.endswith("_"):
                if token.startswith(prefix):
                    if not _chorale_v2_pitch_fits_partial(token, prefix, partial):
                        continue
                    allowed.add(token_id)
            elif token == prefix:
                allowed.add(token_id)
    return allowed


def _chorale_v2_allowed_prefixes(tokens: Sequence[str]) -> tuple[str, ...]:
    if not tokens:
        return ()
    if tokens[-1] == "BAR":
        return ("STYLE_CHORALE",)

    try:
        bar_start = len(tokens) - 1 - list(reversed(tokens)).index("BAR")
    except ValueError:
        return ("BAR",)
    current_bar = list(tokens[bar_start + 1:])

    if not current_bar:
        return ("STYLE_CHORALE",)
    if len(current_bar) == 1 and current_bar[0] == "STYLE_CHORALE":
        return ("KEY_",)
    if len(current_bar) == 2 and current_bar[1].startswith("KEY_"):
        return ("TIME_",)
    if len(current_bar) == 3 and current_bar[2].startswith("TIME_"):
        return ("TEXTURE_",)
    if len(current_bar) == 4 and current_bar[3].startswith("TEXTURE_"):
        return ("POS_",)

    last = tokens[-1]
    if last.startswith("DUR_"):
        return ("POS_", "BAR")
    if last.startswith("POS_"):
        return ("BASS_",)
    if last.startswith("BASS_"):
        return ("TENOR_",)
    if last.startswith("TENOR_"):
        return ("ALTO_",)
    if last.startswith("ALTO_"):
        return ("SOP_",)
    if last.startswith("SOP_"):
        return ("DUR_",)
    return ()


def _apply_chorale_v2_repeat_mask(
    logits: torch.Tensor,
    tokens: Sequence[str],
    vocab: dict[str, int],
    *,
    max_repeats: int,
) -> torch.Tensor:
    if max_repeats <= 0:
        return logits

    previous_sonorities = _chorale_v2_sonorities(tokens)
    if not previous_sonorities:
        return logits
    run_len = _trailing_equal_run(previous_sonorities)
    if run_len < max_repeats:
        return logits

    partial = _chorale_v2_current_partial_sonority(tokens)
    if partial is None:
        return logits
    previous = previous_sonorities[-1]
    if tuple(partial) != previous[: len(partial)]:
        return logits
    if len(partial) >= len(previous):
        return logits

    voice_prefix = ("BASS_", "TENOR_", "ALTO_", "SOP_")[len(partial)]
    banned_token = f"{voice_prefix}{previous[len(partial)]}"
    banned_id = vocab.get(banned_token)
    if banned_id is None:
        return logits

    candidate = logits.clone()
    candidate[:, banned_id] = float("-inf")
    if torch.isfinite(candidate).any(dim=1).all():
        return candidate
    return logits


def _chorale_v2_current_partial_sonority(tokens: Sequence[str]) -> tuple[int, ...] | None:
    try:
        pos_idx = max(idx for idx, token in enumerate(tokens) if token.startswith("POS_"))
    except ValueError:
        return None
    if any(token == "BAR" for token in tokens[pos_idx + 1:]):
        return None

    expected = ("BASS_", "TENOR_", "ALTO_", "SOP_")
    partial: list[int] = []
    for token in tokens[pos_idx + 1:]:
        if len(partial) >= len(expected):
            return None
        pitch = _chorale_v2_pitch(token, expected[len(partial)])
        if pitch is None:
            return None
        partial.append(pitch)
    return tuple(partial)


def _chorale_v2_sonorities(tokens: Sequence[str]) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    idx = 0
    while idx + 5 < len(tokens):
        if not tokens[idx].startswith("POS_"):
            idx += 1
            continue
        pitches = (
            _chorale_v2_pitch(tokens[idx + 1], "BASS_"),
            _chorale_v2_pitch(tokens[idx + 2], "TENOR_"),
            _chorale_v2_pitch(tokens[idx + 3], "ALTO_"),
            _chorale_v2_pitch(tokens[idx + 4], "SOP_"),
        )
        if all(pitch is not None for pitch in pitches) and tokens[idx + 5].startswith("DUR_"):
            result.append((int(pitches[0]), int(pitches[1]), int(pitches[2]), int(pitches[3])))
            idx += 6
            continue
        idx += 1
    return result


def _chorale_v2_pitch(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    try:
        return int(token[len(prefix):])
    except ValueError:
        return None


def _chorale_v2_pitch_fits_partial(token: str, prefix: str, partial: tuple[int, ...] | None) -> bool:
    voice_prefixes = ("BASS_", "TENOR_", "ALTO_", "SOP_")
    if prefix not in voice_prefixes:
        return True
    voice_idx = voice_prefixes.index(prefix)
    if partial is None or len(partial) != voice_idx or not partial:
        return True
    pitch = _chorale_v2_pitch(token, prefix)
    if pitch is None:
        return False
    return pitch >= partial[-1]


def _trailing_equal_run(values: Sequence[object]) -> int:
    if not values:
        return 0
    last = values[-1]
    count = 0
    for value in reversed(values):
        if value != last:
            break
        count += 1
    return count


def generate_v1(
    checkpoint_path: Union[str, Path],
    *,
    seed_tokens: Sequence[TokenLike],
    generation_config: GenerationConfig,
    vocab_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
) -> GenerationResult:
    loaded = load_notelm_checkpoint(
        checkpoint_path,
        vocab_path=vocab_path,
        device=device,
    )
    return _generate_from_loaded(
        loaded,
        seed_tokens=seed_tokens,
        generation_config=generation_config,
    )
