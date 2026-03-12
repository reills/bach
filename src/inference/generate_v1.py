from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from src.models.notelm import LoadedNoteLM, load_notelm_checkpoint
from src.utils.decoding.rules import MusicRules
from src.utils.decoding.sampler import Sampler
from src.utils.decoding.scg import SCGSampler


TokenLike = Union[int, str]


@dataclass(frozen=True)
class GenerationConfig:
    max_length: int
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    use_scg: bool = False
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

    with torch.no_grad():
        while generated.size(1) < generation_config.max_length:
            window = generated[:, -loaded.config.max_seq_len :]
            logits = model(window)
            next_token = sampler.sample(logits[:, -1, :], input_ids=window)
            generated = torch.cat([generated, next_token.to(device=device)], dim=1)
            if eos_id is not None and int(next_token.item()) == eos_id:
                stopped_on_eos = True
                break

    token_ids = generated[0].detach().cpu().tolist()
    return GenerationResult(
        ids=token_ids,
        tokens=_decode_ids(token_ids, loaded.vocab),
        stopped_on_eos=stopped_on_eos,
    )


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
