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
            if generation_config.use_grammar_mask or generation_config.use_voice_leading_mask:
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
