import torch
from typing import Dict, List, Optional, Set
from src.utils.decoding.sampler import Sampler
from src.utils.decoding.rules import MusicRules, token_category


def build_grammar_mask(
    prefix_tokens: List[str],
    vocab: Dict[str, int],
    allowed_categories: Set[str],
    allow_eos: bool = True,
) -> torch.Tensor:
    """
    Build a boolean mask tensor of shape (vocab_size,) where True means the
    token is allowed at this position given the grammar constraints.
    """
    vocab_size = max(vocab.values()) + 1
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for token, idx in vocab.items():
        cat = token_category(token)
        if cat in allowed_categories:
            mask[idx] = True
        elif allow_eos and cat in ("EOS",):
            mask[idx] = True
    return mask


class SCGSampler:
    def __init__(
        self,
        sampler: Sampler,
        rules: MusicRules,
        alpha: float = 0.6,
        gamma: float = 0.4,
    ) -> None:
        self.sampler = sampler
        self.rules = rules
        self.alpha = alpha
        self.gamma = gamma

    def sample(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply alpha (rules)
        logits = self.rules.apply_rules(logits, self.alpha)

        # Apply gamma (density nudge)
        # If gamma > 0, we might favor POS_ tokens or VOICE_ tokens to increase density
        if self.gamma != 0:
            logits = self._apply_density_nudge(logits, self.gamma)

        return self.sampler.sample(logits, input_ids=input_ids)

    def _apply_density_nudge(self, logits: torch.Tensor, gamma: float) -> torch.Tensor:
        # Simplistic implementation: nudge POS_ tokens and VOICE_ tokens
        for token, idx in self.rules.vocab.items():
            if token.startswith("POS_") or token.startswith("VOICE_"):
                logits[:, idx] += gamma * 1.0
        return logits
