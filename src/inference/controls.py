from dataclasses import dataclass
import re
from typing import List, Optional

from src.dataio.collate_miditok import PrefixControlConfig, build_prefix_tokens


@dataclass(frozen=True)
class ComposeControls:
    key: Optional[str] = None
    style: Optional[str] = None
    difficulty: Optional[str] = None
    measures: Optional[int] = None
    texture: Optional[int] = None


def normalize_compose_key(value: str) -> str:
    cleaned = value.strip().replace("♭", "b").replace("♯", "#")
    cleaned = re.sub(r"\s+", "", cleaned)
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(.*)", cleaned)
    if match is None:
        raise ValueError(f"unsupported key value: {value!r}")

    tonic = match.group(1).upper()
    accidental = match.group(2)
    remainder = match.group(3).lower()

    if remainder in ("", "maj", "major"):
        mode = ""
    elif remainder in ("m", "min", "minor"):
        mode = "m"
    else:
        raise ValueError(f"unsupported key value: {value!r}")

    return f"{tonic}{accidental}{mode}"


def build_control_prefix_tokens(
    controls: ComposeControls,
    *,
    measures_token_prefix: str = "MEAS",
) -> List[str]:
    measures = _normalize_measures(controls.measures)
    config = PrefixControlConfig(
        style=_normalize_optional_text(controls.style),
        difficulty=_normalize_optional_text(controls.difficulty),
        measures=measures,
        measures_token_prefix=measures_token_prefix,
        key_from_plan=False,
        key_override=_normalize_optional_key(controls.key),
    )
    return build_prefix_tokens(None, 0, config)


def build_compose_seed_tokens(
    controls: ComposeControls,
    *,
    measures_token_prefix: str = "MEAS",
    default_key: str = "C",
    default_time_sig: str = "4_4",
) -> List[str]:
    tokens = build_control_prefix_tokens(
        controls,
        measures_token_prefix=measures_token_prefix,
    )
    texture = normalize_texture(controls.texture)
    if texture <= 1:
        return tokens

    key = _normalize_optional_key(controls.key) or normalize_compose_key(default_key)
    tokens.extend(
        [
            "BAR",
            f"TIME_SIG_{default_time_sig}",
            f"KEY_{key}",
            *_voice_anchor_tokens(texture),
            "POS_0",
        ]
    )
    return tokens


def normalize_texture(value: Optional[int]) -> int:
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("texture must be an integer")
    return min(max(value, 1), 4)


def _normalize_optional_key(value: Optional[str]) -> Optional[str]:
    stripped = _normalize_optional_text(value)
    if stripped is None:
        return None
    return normalize_compose_key(stripped)


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _normalize_measures(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or value <= 0:
        raise ValueError("measures must be a positive integer")
    return value


def _voice_anchor_tokens(texture: int) -> List[str]:
    anchor_pitches = {
        2: [48, 67],
        3: [48, 60, 67],
        4: [48, 55, 64, 72],
    }
    return [
        f"ABS_VOICE_{voice}_{pitch}"
        for voice, pitch in enumerate(anchor_pitches[texture])
    ]
