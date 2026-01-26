from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError("torch is required to use MidiTok collators.") from exc

from src.dataio.dataset import BarDataset, BarSample
from src.tokens.schema import BarPlan


@dataclass
class SequenceSample:
    ids: List[int]
    piece_id: str
    bar_index: int
    plan: Optional[BarPlan]
    bar_count: int


class PackedBarDataset(Dataset):
    def __init__(
        self,
        dataset: BarDataset,
        *,
        max_seq_len: int,
        bars_per_seq: int = 1,
        allow_truncate: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.bars_per_seq = bars_per_seq
        self.allow_truncate = allow_truncate
        self._sequences = self._build_sequences()

    def _build_sequences(self) -> List[List[int]]:
        piece_ids = getattr(self.dataset, "_piece_id", None)
        bar_indices = getattr(self.dataset, "_bar_index", None)
        ids_list = getattr(self.dataset, "_ids", None)

        indices = list(range(len(self.dataset)))
        if piece_ids is not None and bar_indices is not None:
            indices.sort(key=lambda i: (piece_ids[i], bar_indices[i]))
        else:
            samples = [self.dataset[i] for i in indices]
            indices = sorted(
                indices,
                key=lambda i: (samples[i].piece_id, samples[i].bar_index),
            )

        sequences: List[List[int]] = []
        current: List[int] = []
        current_len = 0
        current_bars = 0
        last_piece: Optional[str] = None

        for idx in indices:
            if piece_ids is None or bar_indices is None:
                sample = self.dataset[idx]
                piece_id = sample.piece_id
                bar_len = len(sample.ids)
            else:
                piece_id = piece_ids[idx]
                bar_len = len(ids_list[idx]) if ids_list is not None else len(self.dataset[idx].ids)

            if last_piece is not None and piece_id != last_piece:
                if current:
                    sequences.append(current)
                current = []
                current_len = 0
                current_bars = 0
            last_piece = piece_id

            if bar_len > self.max_seq_len and not self.allow_truncate:
                continue

            should_flush = False
            if current and self.bars_per_seq > 0 and current_bars >= self.bars_per_seq:
                should_flush = True
            if current and current_len + bar_len > self.max_seq_len:
                should_flush = True

            if should_flush:
                sequences.append(current)
                current = []
                current_len = 0
                current_bars = 0

            current.append(idx)
            current_len += min(bar_len, self.max_seq_len)
            current_bars += 1

        if current:
            sequences.append(current)
        return sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> SequenceSample:
        indices = self._sequences[idx]
        if not indices:
            raise IndexError("empty sequence at index")

        ids: List[int] = []
        first_sample: Optional[BarSample] = None
        for bar_idx in indices:
            sample = self.dataset[bar_idx]
            if first_sample is None:
                first_sample = sample
            ids.extend(sample.ids)
        if first_sample is None:
            raise IndexError("empty sequence at index")

        if len(ids) > self.max_seq_len and self.allow_truncate:
            ids = ids[: self.max_seq_len]

        return SequenceSample(
            ids=ids,
            piece_id=first_sample.piece_id,
            bar_index=first_sample.bar_index,
            plan=first_sample.plan,
            bar_count=len(indices),
        )


@dataclass
class PrefixControlConfig:
    style: Optional[str] = None
    difficulty: Optional[str] = None
    measures: Optional[int] = None
    measures_token_prefix: str = "MEAS"
    key_from_plan: bool = True
    key_override: Optional[str] = None


def _normalize_token_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return cleaned.strip("_").upper()


def build_prefix_tokens(
    plan: Optional[BarPlan],
    bar_count: int,
    config: PrefixControlConfig,
) -> List[str]:
    tokens: List[str] = []

    key_value = config.key_override
    if key_value is None and config.key_from_plan and plan is not None:
        key_value = plan.key
    if key_value:
        tokens.append(f"KEY_{key_value}")

    if config.style:
        tokens.append(f"STYLE_{_normalize_token_label(config.style)}")
    if config.difficulty:
        tokens.append(f"DIFFICULTY_{_normalize_token_label(config.difficulty)}")

    measures_value = config.measures if config.measures is not None else bar_count
    if measures_value:
        tokens.append(f"{config.measures_token_prefix}_{measures_value}")

    return tokens


@dataclass
class MidiTokBatch:
    ids: torch.Tensor
    attn_mask: torch.Tensor
    prefix_len: torch.Tensor
    bar_count: torch.Tensor
    piece_id: List[str]
    bar_index: torch.Tensor


class MidiTokCollator:
    def __init__(
        self,
        vocab: Dict[str, int],
        *,
        pad_token: str,
        prefix_config: PrefixControlConfig,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        prepend_bos: bool = False,
        append_eos: bool = False,
    ) -> None:
        self.vocab = vocab
        self.pad_id = self._require_token(pad_token)
        self.bos_id = self._optional_token(bos_token)
        self.eos_id = self._optional_token(eos_token)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.prefix_config = prefix_config

        if self.prepend_bos and self.bos_id is None:
            raise ValueError("prepend_bos is set but bos_token is missing in vocab")
        if self.append_eos and self.eos_id is None:
            raise ValueError("append_eos is set but eos_token is missing in vocab")

    def _require_token(self, token: str) -> int:
        if token not in self.vocab:
            raise KeyError(f"token not found in vocab: {token}")
        return self.vocab[token]

    def _optional_token(self, token: Optional[str]) -> Optional[int]:
        if token is None:
            return None
        return self._require_token(token)

    def __call__(self, samples: Sequence[SequenceSample]) -> MidiTokBatch:
        batch_ids: List[List[int]] = []
        prefix_lens: List[int] = []
        bar_counts: List[int] = []
        piece_ids: List[str] = []
        bar_indices: List[int] = []

        for sample in samples:
            prefix_tokens = build_prefix_tokens(sample.plan, sample.bar_count, self.prefix_config)
            prefix_ids = [self.vocab[token] for token in prefix_tokens]

            seq: List[int] = []
            if self.prepend_bos:
                seq.append(self.bos_id)
            seq.extend(prefix_ids)
            prefix_len = len(seq)
            seq.extend(sample.ids)
            if self.append_eos:
                seq.append(self.eos_id)

            batch_ids.append(seq)
            prefix_lens.append(prefix_len)
            bar_counts.append(sample.bar_count)
            piece_ids.append(sample.piece_id)
            bar_indices.append(sample.bar_index)

        max_len = max(len(seq) for seq in batch_ids)
        padded = [seq + [self.pad_id] * (max_len - len(seq)) for seq in batch_ids]

        ids = torch.tensor(padded, dtype=torch.long)
        attn_mask = ids != self.pad_id
        return MidiTokBatch(
            ids=ids,
            attn_mask=attn_mask,
            prefix_len=torch.tensor(prefix_lens, dtype=torch.long),
            bar_count=torch.tensor(bar_counts, dtype=torch.long),
            piece_id=piece_ids,
            bar_index=torch.tensor(bar_indices, dtype=torch.long),
        )
