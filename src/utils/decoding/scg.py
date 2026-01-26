import torch
from typing import Dict, List, Optional
from src.utils.decoding.sampler import Sampler
from src.utils.decoding.rules import MusicRules

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
