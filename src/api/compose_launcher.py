import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from fastapi import FastAPI

from src.api.app import _default_repository, create_app
from src.api.compose_service import ComposeServiceResult, compose_baseline, compose_canonical_score
from src.api.routes.scores import ComposeHandler, ComposeRequest
from src.emi.composer import EMI_ENGINE_VERSION, EmiComposerConfig, compose_emi
from src.inference.controls import ComposeControls, build_compose_seed_tokens, normalize_texture
from src.inference.generate_v1 import GenerationConfig, GenerationResult, _generate_from_loaded
from src.inference.hybrid import build_hybrid_context
from src.inference.rerank import (
    QUALITY_PASSES_DEFAULT,
    evaluate_novelty_metrics,
    evaluate_quality_metrics,
    is_high_copy_risk,
    normalize_quality_passes,
)
from src.models.notelm import load_notelm_checkpoint
from src.utils.decoding.voice_state import (
    VOICE_LEADING_DEFAULT,
    VoiceLeadingMode,
    normalize_voice_leading,
    voice_leading_enabled,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "out" / "notelm_v1" / "notelm_step5000.pt"
DEFAULT_VOCAB_PATH = REPO_ROOT / "out" / "notelm_v1" / "vocab.json"
EngineMode = Literal["transformer", "emi", "hybrid"]


@dataclass(frozen=True)
class ComposeRuntimeConfig:
    checkpoint_path: Path
    vocab_path: Optional[Path] = None
    engine: EngineMode = "hybrid"
    emi_fragment_path: Optional[Path] = None
    hybrid_allow_emi_debug_fallback: bool = False
    device: str = "cpu"
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    use_scg: bool = False
    use_grammar_mask: bool = True
    alpha: float = 0.6
    gamma: float = 0.4
    eos_token: Optional[str] = None
    quality_passes: int = QUALITY_PASSES_DEFAULT
    voice_leading: VoiceLeadingMode = VOICE_LEADING_DEFAULT


def build_compose_service(config: ComposeRuntimeConfig) -> ComposeHandler:
    loaded_cache = None

    def loaded_notelm():
        nonlocal loaded_cache
        if loaded_cache is None:
            loaded_cache = load_notelm_checkpoint(
                config.checkpoint_path,
                vocab_path=config.vocab_path,
                device=config.device,
            )
        return loaded_cache

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
            loaded_notelm(),
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
            texture=_constraint_texture(constraints),
        )
        engine = _constraint_engine(constraints, default=config.engine)
        seed_tokens = build_compose_seed_tokens(controls)
        voice_leading = _constraint_voice_leading(constraints, default=config.voice_leading)
        quality_passes = _constraint_quality_passes(constraints, default=config.quality_passes)
        fragment_path = _constraint_fragment_path(constraints, default=config.emi_fragment_path)
        hybrid_context = (
            build_hybrid_context(
                controls,
                fragment_path=fragment_path,
            )
            if engine == "hybrid"
            else None
        )

        if engine == "emi":
            return _compose_emi_result(
                request,
                controls=controls,
                config=config,
                constraints=constraints,
                requested_engine=engine,
            )

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
            use_grammar_mask=_constraint_bool(
                constraints,
                "use_grammar_mask",
                "useGrammarMask",
                default=config.use_grammar_mask,
            ),
            use_voice_leading_mask=voice_leading_enabled(voice_leading),
            target_texture=controls.texture,
            bar_voice_survival_penalty=(8.0 if voice_leading_enabled(voice_leading) else 0.0),
            alpha=_constraint_float(constraints, "alpha", default=config.alpha),
            gamma=_constraint_float(constraints, "gamma", default=config.gamma),
            eos_token=_constraint_text(constraints, "eos_token", "eosToken", default=config.eos_token),
        )
        if hybrid_context is not None:
            generation_config = replace(
                generation_config,
                conditioning=hybrid_context.model_conditioning(),
            )
        try:
            loaded = loaded_notelm()
            result = compose_baseline(
                config.checkpoint_path,
                seed_tokens=seed_tokens,
                generation_config=generation_config,
                vocab_path=loaded.vocab_path,
                device=config.device,
                render_mode=request.render_mode,
                generator=generator,
                quality_passes=quality_passes,
            )
        except Exception as exc:
            if engine == "hybrid" and _hybrid_debug_fallback_enabled(constraints, config=config):
                return _compose_emi_result(
                    request,
                    controls=controls,
                    config=config,
                    constraints=constraints,
                    requested_engine=engine,
                    extra_diagnostics={
                        "hybridFallbackReason": "transformer_exception_debug_only",
                        "transformerError": str(exc),
                        "hybrid": hybrid_context.diagnostics() if hybrid_context is not None else None,
                    },
                )
            raise

        generated_metrics = evaluate_quality_metrics(result.generation.tokens)
        novelty_metrics = (
            evaluate_novelty_metrics(
                result.generation.tokens,
                _constraint_token_sequences(constraints, "source_token_sequences", "sourceTokenSequences"),
                ngram=_constraint_int(constraints, "novelty_ngram", "noveltyNgram", default=16) or 16,
            )
            if engine == "hybrid"
            else None
        )
        if engine == "hybrid" and novelty_metrics is not None and is_high_copy_risk(novelty_metrics):
            raise ValueError("hybrid novelty gate rejected candidate for excessive source overlap")

        diagnostics: dict[str, object] = {
            "engine": "hybrid" if engine == "hybrid" else "transformer",
            "requestedEngine": engine,
            "proposalEngine": "transformer",
            "texture": controls.texture,
            "qualityPasses": quality_passes,
            "useGrammarMask": generation_config.use_grammar_mask,
            "useVoiceLeadingMask": generation_config.use_voice_leading_mask,
            "barVoiceSurvivalPenalty": generation_config.bar_voice_survival_penalty,
            "voiceLeading": voice_leading,
            "seed_tokens": [str(token) for token in seed_tokens],
            "generated_metrics": generated_metrics,
        }
        if hybrid_context is not None:
            diagnostics.update(
                {
                    "hybrid": hybrid_context.diagnostics(),
                    "hybridQualityGateFailed": _quality_gate_failed(
                        generated_metrics,
                        target_texture=controls.texture,
                    ),
                    "novelty": novelty_metrics,
                }
            )
        object.__setattr__(
            result,
            "diagnostics",
            diagnostics,
        )
        return result

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
    fragment_value = os.environ.get("BACH_GEN_EMI_FRAGMENTS")
    fragment_path = Path(fragment_value).expanduser() if fragment_value else None
    device = os.environ.get("BACH_GEN_DEVICE") or _default_device()
    return ComposeRuntimeConfig(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        engine=normalize_engine(os.environ.get("BACH_GEN_ENGINE"), default="hybrid"),
        emi_fragment_path=fragment_path,
        hybrid_allow_emi_debug_fallback=_env_bool("BACH_GEN_HYBRID_EMI_DEBUG_FALLBACK", False),
        device=device,
        max_length=_env_int("BACH_GEN_MAX_LENGTH", 512),
        temperature=_env_float("BACH_GEN_TEMPERATURE", 1.0),
        top_p=_env_float("BACH_GEN_TOP_P", 0.9),
        repetition_penalty=_env_float("BACH_GEN_REPETITION_PENALTY", 1.0),
        no_repeat_ngram_size=_env_int("BACH_GEN_NO_REPEAT_NGRAM_SIZE", 0),
        use_scg=_env_bool("BACH_GEN_USE_SCG", False),
        use_grammar_mask=_env_bool("BACH_GEN_USE_GRAMMAR_MASK", True),
        alpha=_env_float("BACH_GEN_ALPHA", 0.6),
        gamma=_env_float("BACH_GEN_GAMMA", 0.4),
        eos_token=os.environ.get("BACH_GEN_EOS_TOKEN"),
        quality_passes=normalize_quality_passes(
            _env_int("BACH_GEN_QUALITY_PASSES", QUALITY_PASSES_DEFAULT)
        ),
        voice_leading=normalize_voice_leading(os.environ.get("BACH_GEN_VOICE_LEADING")),
    )


def _compose_emi_result(
    request: ComposeRequest,
    *,
    controls: ComposeControls,
    config: ComposeRuntimeConfig,
    constraints: dict[str, Any],
    requested_engine: EngineMode,
    extra_diagnostics: dict[str, object] | None = None,
) -> ComposeServiceResult:
    seed = _constraint_int(constraints, "seed", "randomSeed", default=0) or 0
    fragment_path = _constraint_fragment_path(constraints, default=config.emi_fragment_path)
    composition = compose_emi(
        EmiComposerConfig(
            key=controls.key or "C",
            measures=controls.measures or 4,
            texture=controls.texture or 2,
            seed=seed,
            fragment_path=fragment_path,
        )
    )
    diagnostics: dict[str, object] = {
        "engine": "emi",
        "requestedEngine": requested_engine,
        "emiVersion": EMI_ENGINE_VERSION,
        "texture": controls.texture,
        "qualityPasses": 0,
        "seed": seed,
        "seed_tokens": [],
        "generated_metrics": None,
        "emi": composition.diagnostics,
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return compose_canonical_score(
        composition.score,
        generation=GenerationResult(
            ids=[],
            tokens=[
                "EMI_SYMBOLIC",
                f"ENGINE_{EMI_ENGINE_VERSION}",
                f"KEY_{composition.diagnostics['key']}",
                f"MEAS_{composition.diagnostics['measures']}",
                f"TEXTURE_{composition.diagnostics['texture']}",
            ],
            stopped_on_eos=True,
        ),
        render_mode=request.render_mode,
        diagnostics=diagnostics,
    )


def _quality_gate_failed(metrics: dict[str, int | float | None], *, target_texture: int) -> bool:
    if _metric(metrics.get("harm_mismatch_count")) > 0:
        return True
    if _metric(metrics.get("token_grammar_violations")) > 0:
        return True
    if _metric(metrics.get("duplicate_bar_rate")) >= 0.35:
        return True
    if _metric(metrics.get("repeated_pitch_rate")) >= 0.6:
        return True
    if _metric(metrics.get("repeated_interval_rate")) >= 0.7:
        return True
    if _metric(metrics.get("counterpoint_voice_crossings")) > 0:
        return True
    if _metric(metrics.get("counterpoint_parallel_octaves")) > 2:
        return True
    if _metric(metrics.get("counterpoint_parallel_fifths")) > 2:
        return True
    avg_active = metrics.get("counterpoint_avg_active_voices")
    if isinstance(avg_active, (int, float)) and target_texture >= 2:
        return float(avg_active) < max(1.5, target_texture - 1.0)
    return False


def _metric(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def normalize_engine(value: str | None, *, default: EngineMode = "transformer") -> EngineMode:
    if value is None:
        return default
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "transformer": "transformer",
        "neural": "transformer",
        "notelm": "transformer",
        "emi": "emi",
        "cope": "emi",
        "symbolic": "emi",
        "hybrid": "hybrid",
        "both": "hybrid",
    }
    engine = aliases.get(normalized)
    if engine is None:
        raise ValueError("engine must be one of: transformer, emi, hybrid")
    return engine  # type: ignore[return-value]


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


def _constraint_path(
    constraints: dict[str, Any],
    *names: str,
    default: Optional[Path] = None,
) -> Optional[Path]:
    value = _constraint_text(constraints, *names)
    if value is None:
        return default
    return Path(value).expanduser()


def _constraint_fragment_path(
    constraints: dict[str, Any],
    *,
    default: Optional[Path] = None,
) -> Optional[Path]:
    return _constraint_path(
        constraints,
        "emi_fragment_path",
        "emiFragmentPath",
        "fragmentPath",
        default=default,
    )


def _constraint_int(constraints: dict[str, Any], *names: str, default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = constraints.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value
    return default


def _constraint_texture(constraints: dict[str, Any]) -> int:
    value = _constraint_int(constraints, "texture", "voices", "voiceCount", default=1)
    return normalize_texture(value)


def _constraint_quality_passes(constraints: dict[str, Any], *, default: int) -> int:
    value = _constraint_int(constraints, "quality_passes", "qualityPasses", default=default)
    return normalize_quality_passes(value)


def _constraint_engine(constraints: dict[str, Any], *, default: EngineMode) -> EngineMode:
    value = _constraint_text(constraints, "engine", "compositionEngine")
    return normalize_engine(value, default=default)


def _constraint_voice_leading(
    constraints: dict[str, Any],
    *,
    default: VoiceLeadingMode,
) -> VoiceLeadingMode:
    value = _constraint_text(constraints, "voice_leading", "voiceLeading", default=default)
    return normalize_voice_leading(value, default=default)


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


def _hybrid_debug_fallback_enabled(
    constraints: dict[str, Any],
    *,
    config: ComposeRuntimeConfig,
) -> bool:
    return _constraint_bool(
        constraints,
        "hybrid_allow_emi_fallback",
        "hybridAllowEmiFallback",
        default=config.hybrid_allow_emi_debug_fallback,
    )


def _constraint_token_sequences(constraints: dict[str, Any], *names: str) -> list[list[str]]:
    for name in names:
        value = constraints.get(name)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list of token lists")
        sequences: list[list[str]] = []
        for sequence in value:
            if not isinstance(sequence, list) or not all(isinstance(token, str) for token in sequence):
                raise ValueError(f"{name} must be a list of token lists")
            sequences.append(list(sequence))
        return sequences
    return []


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
