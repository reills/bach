import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI

from src.api.app import _default_repository, create_app
from src.api.compose_service import ComposeServiceResult, compose_baseline
from src.api.routes.scores import ComposeHandler, ComposeRequest
from src.inference.controls import ComposeControls, build_control_prefix_tokens
from src.inference.generate_v1 import GenerationConfig, GenerationResult, _generate_from_loaded
from src.models.notelm import load_notelm_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "out" / "notelm_v1" / "notelm_step5000.pt"
DEFAULT_VOCAB_PATH = REPO_ROOT / "out" / "notelm_v1" / "vocab.json"


@dataclass(frozen=True)
class ComposeRuntimeConfig:
    checkpoint_path: Path
    vocab_path: Optional[Path] = None
    device: str = "cpu"
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    use_scg: bool = False
    alpha: float = 0.6
    gamma: float = 0.4
    eos_token: Optional[str] = None


def build_compose_service(config: ComposeRuntimeConfig) -> ComposeHandler:
    loaded = load_notelm_checkpoint(
        config.checkpoint_path,
        vocab_path=config.vocab_path,
        device=config.device,
    )

    def generator(
        checkpoint_path: str | Path,
        *,
        seed_tokens: list[str | int],
        generation_config: GenerationConfig,
        vocab_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> GenerationResult:
        del checkpoint_path, vocab_path, device
        return _generate_from_loaded(
            loaded,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
        )

    def compose_service(request: ComposeRequest) -> ComposeServiceResult:
        constraints = _constraints_dict(request.constraints)
        controls = ComposeControls(
            key=_constraint_text(constraints, "key", default="C"),
            style=_constraint_text(constraints, "style"),
            difficulty=_constraint_text(constraints, "difficulty"),
            measures=_constraint_int(constraints, "measures", default=4),
        )
        seed_tokens = build_control_prefix_tokens(controls)

        generation_config = GenerationConfig(
            max_length=_constraint_int(constraints, "max_length", "maxLength", default=config.max_length),
            temperature=_constraint_float(constraints, "temperature", default=config.temperature),
            top_p=_constraint_float(constraints, "top_p", "topP", default=config.top_p),
            repetition_penalty=_constraint_float(
                constraints,
                "repetition_penalty",
                "repetitionPenalty",
                default=config.repetition_penalty,
            ),
            no_repeat_ngram_size=_constraint_int(
                constraints,
                "no_repeat_ngram_size",
                "noRepeatNgramSize",
                default=config.no_repeat_ngram_size,
            ),
            use_scg=_constraint_bool(constraints, "use_scg", "useScg", default=config.use_scg),
            alpha=_constraint_float(constraints, "alpha", default=config.alpha),
            gamma=_constraint_float(constraints, "gamma", default=config.gamma),
            eos_token=_constraint_text(constraints, "eos_token", "eosToken", default=config.eos_token),
        )
        return compose_baseline(
            config.checkpoint_path,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
            vocab_path=loaded.vocab_path,
            device=config.device,
            render_mode=request.render_mode,
            generator=generator,
        )

    return compose_service


def create_configured_app(config: ComposeRuntimeConfig) -> FastAPI:
    return create_app(
        compose_service=build_compose_service(config),
        repository=_default_repository(),
    )


def create_configured_app_from_env() -> FastAPI:
    return create_configured_app(_runtime_config_from_env())


def _runtime_config_from_env() -> ComposeRuntimeConfig:
    checkpoint_path = Path(os.environ.get("BACH_GEN_CHECKPOINT", str(DEFAULT_CHECKPOINT_PATH))).expanduser()
    vocab_value = os.environ.get("BACH_GEN_VOCAB")
    vocab_path = Path(vocab_value).expanduser() if vocab_value else DEFAULT_VOCAB_PATH
    device = os.environ.get("BACH_GEN_DEVICE") or _default_device()
    return ComposeRuntimeConfig(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device,
        max_length=_env_int("BACH_GEN_MAX_LENGTH", 512),
        temperature=_env_float("BACH_GEN_TEMPERATURE", 1.0),
        top_p=_env_float("BACH_GEN_TOP_P", 0.9),
        repetition_penalty=_env_float("BACH_GEN_REPETITION_PENALTY", 1.0),
        no_repeat_ngram_size=_env_int("BACH_GEN_NO_REPEAT_NGRAM_SIZE", 0),
        use_scg=_env_bool("BACH_GEN_USE_SCG", False),
        alpha=_env_float("BACH_GEN_ALPHA", 0.6),
        gamma=_env_float("BACH_GEN_GAMMA", 0.4),
        eos_token=os.environ.get("BACH_GEN_EOS_TOKEN"),
    )


def _constraints_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("constraints must be an object")
    return value


def _constraint_text(constraints: dict[str, Any], *names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = constraints.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        stripped = value.strip()
        return stripped or None
    return default


def _constraint_int(constraints: dict[str, Any], *names: str, default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = constraints.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value
    return default


def _constraint_float(constraints: dict[str, Any], *names: str, default: float) -> float:
    for name in names:
        value = constraints.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)
    return default


def _constraint_bool(constraints: dict[str, Any], *names: str, default: bool) -> bool:
    for name in names:
        value = constraints.get(name)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value
    return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean string")


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
