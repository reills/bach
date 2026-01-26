import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "pandas is required to read datasets; install pandas and a parquet engine."
    ) from exc

try:
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError("torch is required to use BarDataset.") from exc

from src.tokens.schema import BarPlan


@dataclass
class BarSample:
    ids: List[int]
    bar_index: int
    piece_id: str
    plan: Optional[BarPlan]
    tokens: Optional[List[str]] = None


def _load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"vocab file must be a dict: {path}")
    return {str(k): int(v) for k, v in data.items()}


def _split_tokens(value: object) -> List[str]:
    if isinstance(value, str):
        return [tok for tok in value.split() if tok]
    if isinstance(value, (list, tuple)):
        return [str(tok) for tok in value if tok]
    raise ValueError(f"Unsupported tokens value type: {type(value)}")


def _parse_plan(plan_value: object, bar_index: int) -> Optional[BarPlan]:
    if plan_value is None:
        return None
    if pd.isna(plan_value):
        return None
    if isinstance(plan_value, str):
        data = json.loads(plan_value)
    elif isinstance(plan_value, dict):
        data = dict(plan_value)
    else:
        raise ValueError(f"Unsupported plan value type: {type(plan_value)}")
    if "bar_index" not in data:
        data["bar_index"] = bar_index
    return BarPlan(**data)


class BarDataset(Dataset):
    def __init__(
        self,
        events_path: str,
        vocab_path: str,
        *,
        tokens_col: str = "tokens",
        plan_col: str = "plan_json",
        bar_index_col: str = "bar_index",
        piece_id_col: str = "piece_id",
        unk_token: Optional[str] = "<unk>",
        return_tokens: bool = False,
    ) -> None:
        self.events_path = Path(events_path)
        self.vocab_path = Path(vocab_path)
        self.tokens_col = tokens_col
        self.plan_col = plan_col
        self.bar_index_col = bar_index_col
        self.piece_id_col = piece_id_col
        self.return_tokens = return_tokens

        if not self.events_path.exists():
            raise FileNotFoundError(f"events file not found: {self.events_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"vocab file not found: {self.vocab_path}")

        self.vocab = _load_vocab(self.vocab_path)
        self.unk_id = self.vocab.get(unk_token) if unk_token else None

        df = self._read_events(self.events_path, self._columns_to_read())

        self._ids: List[List[int]] = []
        self._tokens: Optional[List[List[str]]] = [] if return_tokens else None
        self._plans: List[Optional[BarPlan]] = []
        self._bar_index: List[int] = []
        self._piece_id: List[str] = []

        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            tokens = _split_tokens(row_dict[tokens_col])
            ids = self._encode_tokens(tokens)
            bar_index = int(row_dict[bar_index_col])
            piece_id = str(row_dict[piece_id_col])

            plan_value = row_dict.get(plan_col)
            plan = _parse_plan(plan_value, bar_index) if plan_col in row_dict else None

            self._ids.append(ids)
            if self._tokens is not None:
                self._tokens.append(tokens)
            self._plans.append(plan)
            self._bar_index.append(bar_index)
            self._piece_id.append(piece_id)

    def _columns_to_read(self) -> List[str]:
        cols = [self.tokens_col, self.bar_index_col, self.piece_id_col]
        if self.plan_col:
            cols.append(self.plan_col)
        return cols

    @staticmethod
    def _read_events(events_path: Path, columns: Sequence[str]) -> "pd.DataFrame":
        if events_path.suffix.lower() == ".parquet":
            return pd.read_parquet(events_path, columns=list(columns))
        return pd.read_csv(events_path, usecols=list(columns))

    def _encode_tokens(self, tokens: List[str]) -> List[int]:
        ids: List[int] = []
        for tok in tokens:
            tok_id = self.vocab.get(tok)
            if tok_id is None:
                if self.unk_id is None:
                    raise KeyError(f"token not in vocab: {tok}")
                tok_id = self.unk_id
            ids.append(tok_id)
        return ids

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> BarSample:
        tokens = self._tokens[idx] if self._tokens is not None else None
        return BarSample(
            ids=self._ids[idx],
            bar_index=self._bar_index[idx],
            piece_id=self._piece_id[idx],
            plan=self._plans[idx],
            tokens=tokens,
        )
