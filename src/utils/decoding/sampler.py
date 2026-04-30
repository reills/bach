import torch
import torch.nn.functional as F
from typing import Optional

class Sampler:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def sample(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits: (batch, vocab_size)
        input_ids: (batch, seq_len) used for repetition penalty
        """
        logits = logits.clone()

        if self.repetition_penalty != 1.0 and input_ids is not None:
            # Simple repetition penalty: scale down logits of tokens already present
            score = torch.gather(logits, 1, input_ids)
            # if score < 0 then score = score * penalty else score = score / penalty
            score = torch.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
            logits.scatter_(1, input_ids, score)

        if self.no_repeat_ngram_size > 0 and input_ids is not None:
            logits = self._apply_no_repeat_ngram(logits, input_ids)

        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / self.temperature

        if self.top_p < 1.0:
            logits = self._top_p_filter(logits, self.top_p)

        probs = F.softmax(logits, dim=-1)
        prev_token = torch.multinomial(probs, num_samples=1)
        return prev_token

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep at least the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    def _apply_no_repeat_ngram(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        ngram_size = self.no_repeat_ngram_size
        if ngram_size <= 0 or input_ids.size(1) < ngram_size - 1:
            return logits

        result = logits.clone()
        for batch_idx in range(input_ids.size(0)):
            banned = self._banned_next_tokens(input_ids[batch_idx].tolist(), ngram_size)
            if not banned:
                continue
            candidate = result[batch_idx].clone()
            candidate[banned] = float("-inf")
            if torch.isfinite(candidate).any():
                result[batch_idx] = candidate
        return result

    @staticmethod
    def _banned_next_tokens(sequence: list[int], ngram_size: int) -> list[int]:
        if ngram_size == 1:
            return sorted(set(sequence))
        if len(sequence) < ngram_size - 1:
            return []

        current_prefix = tuple(sequence[-(ngram_size - 1):])
        banned: set[int] = set()
        stop = len(sequence) - ngram_size + 1
        for idx in range(max(0, stop)):
            ngram = sequence[idx: idx + ngram_size]
            if tuple(ngram[:-1]) == current_prefix:
                banned.add(ngram[-1])
        return sorted(banned)
