from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.instrumental_v6.representation import InstrumentalV6Piece


def save_dataset(
    path: str | Path,
    pieces: list[InstrumentalV6Piece],
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "pieces": [piece.to_dict() for piece in pieces],
            }
        ),
        encoding="utf-8",
    )


def load_dataset(path: str | Path) -> tuple[list[InstrumentalV6Piece], dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pieces = [InstrumentalV6Piece.from_dict(item) for item in data["pieces"]]
    return pieces, dict(data.get("metadata", {}))
