import os
import random
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
EngineMode = Literal["transformer", "emi", "hybrid", "instrumental_v6"]


@dataclass(frozen=True)
class ComposeRuntimeConfig:
    checkpoint_path: Path
    vocab_path: Optional[Path] = None
    engine: EngineMode = "hybrid"
    emi_fragment_path: Optional[Path] = None
    v5_checkpoint_path: Optional[Path] = None
    v5_data_dir: Optional[Path] = None
    v5_prompt_rows: int = 64
    v5_piece_index: int = 0
    v5_candidates: int = 4
    v6_checkpoint_path: Optional[Path] = None
    v6_data_dir: Optional[Path] = None
    v6_prompt_rows: int = 64
    v6_piece_index: int = 0
    v6_candidates: int = 4
    v6_temperature: float = 0.4
    v6_duration_temperature: float = 0.8
    v6_duration_prior_strength: float = 0.1
    v6_top_k: int = 12
    v6_beam_size: int = 96
    v6_tempo: int = 88
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


@dataclass(frozen=True)
class _LoadedV6Runtime:
    model: Any
    model_config: Any
    pieces: list[Any]
    device: torch.device
    checkpoint_step: int | None


def build_compose_service(config: ComposeRuntimeConfig) -> ComposeHandler:
    loaded_cache = None
    loaded_v6_cache = None

    def loaded_notelm():
        nonlocal loaded_cache
        if loaded_cache is None:
            loaded_cache = load_notelm_checkpoint(
                config.checkpoint_path,
                vocab_path=config.vocab_path,
                device=config.device,
            )
        return loaded_cache

    def loaded_v6() -> _LoadedV6Runtime:
        nonlocal loaded_v6_cache
        if loaded_v6_cache is None:
            loaded_v6_cache = _load_v6_runtime(config)
        return loaded_v6_cache

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

        if engine == "instrumental_v6":
            return _compose_v6_result(
                request,
                controls=controls,
                config=config,
                constraints=constraints,
                runtime=loaded_v6(),
            )

        if engine == "hybrid" and config.v5_checkpoint_path is not None and config.v5_data_dir is not None:
            return _compose_v5_hybrid_result(
                request,
                controls=controls,
                config=config,
                constraints=constraints,
                hybrid_context=hybrid_context,
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
        v5_checkpoint_path=_env_optional_path("BACH_GEN_V5_CHECKPOINT"),
        v5_data_dir=_env_optional_path("BACH_GEN_V5_DATA_DIR"),
        v5_prompt_rows=_env_int("BACH_GEN_V5_PROMPT_ROWS", 64),
        v5_piece_index=_env_int("BACH_GEN_V5_PIECE_INDEX", 0),
        v5_candidates=_env_int("BACH_GEN_V5_CANDIDATES", 4),
        v6_checkpoint_path=_env_optional_path("BACH_GEN_V6_CHECKPOINT"),
        v6_data_dir=_env_optional_path("BACH_GEN_V6_DATA_DIR"),
        v6_prompt_rows=_env_int("BACH_GEN_V6_PROMPT_ROWS", 64),
        v6_piece_index=_env_int("BACH_GEN_V6_PIECE_INDEX", 0),
        v6_candidates=_env_int("BACH_GEN_V6_CANDIDATES", 4),
        v6_temperature=_env_float("BACH_GEN_V6_TEMPERATURE", 0.4),
        v6_duration_temperature=_env_float("BACH_GEN_V6_DURATION_TEMPERATURE", 0.8),
        v6_duration_prior_strength=_env_float("BACH_GEN_V6_DURATION_PRIOR_STRENGTH", 0.1),
        v6_top_k=_env_int("BACH_GEN_V6_TOP_K", 12),
        v6_beam_size=_env_int("BACH_GEN_V6_BEAM_SIZE", 96),
        v6_tempo=_env_int("BACH_GEN_V6_TEMPO", 88),
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


def _load_v6_runtime(config: ComposeRuntimeConfig) -> _LoadedV6Runtime:
    if config.v6_checkpoint_path is None or config.v6_data_dir is None:
        raise ValueError("instrumental_v6 requires BACH_GEN_V6_CHECKPOINT and BACH_GEN_V6_DATA_DIR")

    from src.instrumental_v6.data import load_dataset
    from src.instrumental_v6.model import build_generator, config_from_checkpoint

    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.init()

    checkpoint = torch.load(config.v6_checkpoint_path, map_location=device, weights_only=False)
    model_config = config_from_checkpoint(checkpoint["config"])
    model = build_generator(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    pieces, _ = load_dataset(config.v6_data_dir / "pieces.json")
    if not pieces:
        raise ValueError("instrumental_v6 dataset contains no pieces")
    return _LoadedV6Runtime(
        model=model,
        model_config=model_config,
        pieces=pieces,
        device=device,
        checkpoint_step=checkpoint.get("step"),
    )


def _compose_v6_result(
    request: ComposeRequest,
    *,
    controls: ComposeControls,
    config: ComposeRuntimeConfig,
    constraints: dict[str, Any],
    runtime: _LoadedV6Runtime,
) -> ComposeServiceResult:
    from scripts.generate_instrumental_v6 import (
        _candidate_score,
        _continuation_piece,
        _duration_log_prior,
        _motif_report,
        _select_template,
        _subject_contour,
        _with_tempo,
        generate_rows,
    )
    from src.instrumental_v6.metrics import evaluate_piece_rows, source_overlap_report
    from src.instrumental_v6.representation import piece_to_canonical_score, rows_to_piece

    voice_count = controls.texture or 2
    if not 2 <= voice_count <= runtime.model_config.max_voices:
        raise ValueError(
            f"instrumental_v6 voices must be between 2 and {runtime.model_config.max_voices}"
        )

    form = _constraint_v6_form(constraints, voice_count=voice_count)
    piece_id = _constraint_text(constraints, "piece_id", "pieceId")
    piece_index = _constraint_int(
        constraints,
        "piece_index",
        "pieceIndex",
        default=config.v6_piece_index,
    ) or 0
    template = _select_template(
        runtime.pieces,
        voices=voice_count,
        requested_form=form,
        piece_id=piece_id,
        piece_index=piece_index,
    )
    prompt_count = min(max(2, config.v6_prompt_rows), len(template.global_rows))
    prompt = (
        [row[:] for row in template.global_rows[:prompt_count]],
        [[voice[:] for voice in row] for row in template.voice_rows[:prompt_count]],
        [
            [[pair[:] for pair in left] for left in row]
            for row in template.pair_rows[:prompt_count]
        ],
    )
    measure_count = controls.measures or 4
    max_new_rows = _constraint_int(
        constraints,
        "max_new_rows",
        "maxNewRows",
        default=measure_count * template.steps_per_bar,
    ) or 1
    max_new_rows = max(1, max_new_rows)
    candidate_count = max(
        1,
        _constraint_int(
            constraints,
            "candidates",
            "candidateCount",
            default=config.v6_candidates,
        )
        or 1,
    )
    temperature = _constraint_float(
        constraints,
        "temperature",
        default=config.v6_temperature,
    )
    duration_temperature = _constraint_float(
        constraints,
        "duration_temperature",
        "durationTemperature",
        default=config.v6_duration_temperature,
    )
    duration_prior_strength = max(
        0.0,
        _constraint_float(
            constraints,
            "duration_prior_strength",
            "durationPriorStrength",
            default=config.v6_duration_prior_strength,
        ),
    )
    top_k = max(
        1,
        _constraint_int(constraints, "top_k", "topK", default=config.v6_top_k) or 1,
    )
    beam_size = max(
        1,
        _constraint_int(
            constraints,
            "beam_size",
            "beamSize",
            default=config.v6_beam_size,
        )
        or 1,
    )
    tempo = max(
        1,
        _constraint_int(constraints, "tempo", default=config.v6_tempo) or config.v6_tempo,
    )
    seed = _constraint_int(constraints, "seed", "randomSeed", default=2604) or 2604
    random.seed(seed)
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    source_rows = [
        piece.voice_rows
        for piece in runtime.pieces
        if piece.voice_count == voice_count
    ]
    duration_log_prior = _duration_log_prior(
        runtime.pieces,
        voices=voice_count,
        form=form,
        device=runtime.device,
    )
    source_start = prompt_count
    source_end = min(len(template.global_rows), source_start + max_new_rows)
    source_baseline = evaluate_piece_rows(
        template.global_rows[source_start:source_end],
        template.voice_rows[source_start:source_end],
        template.pair_rows[source_start:source_end],
        voice_count=voice_count,
    )
    subject = _subject_contour(prompt[1], voice_count)

    candidates: list[tuple[float, Any, dict[str, object]]] = []
    for candidate_index in range(candidate_count):
        generated = generate_rows(
            runtime.model,
            prompt=prompt,
            template=template,
            form=form,
            voice_count=voice_count,
            max_new_rows=max_new_rows,
            device=runtime.device,
            max_context=runtime.model_config.max_seq_len,
            temperature=temperature,
            duration_temperature=duration_temperature,
            top_k=top_k,
            beam_size=beam_size,
            duration_log_prior=duration_log_prior,
            duration_prior_strength=duration_prior_strength,
        )
        generated_piece = rows_to_piece(
            global_rows=generated[0],
            voice_rows=generated[1],
            pair_rows=generated[2],
            template=replace(template, form=form, voice_count=voice_count),
            piece_id=f"instrumental_v6_{voice_count}v_candidate{candidate_index:02d}",
        )
        continuation = _continuation_piece(generated_piece, prompt_count)
        report = evaluate_piece_rows(
            continuation.global_rows,
            continuation.voice_rows,
            continuation.pair_rows,
            voice_count=voice_count,
        )
        overlap = source_overlap_report(
            continuation.voice_rows,
            source_rows,
            voice_count=voice_count,
            ngram=16,
        )
        motif = _motif_report(
            generated_piece.voice_rows,
            subject=subject,
            voice_count=voice_count,
        )
        score_value = _candidate_score(
            report,
            overlap,
            source_baseline=source_baseline,
            motif_report=motif,
        )
        candidates.append(
            (
                score_value,
                continuation,
                {
                    "candidate_index": candidate_index,
                    "score": score_value,
                    "counterpoint": report,
                    "motif": motif,
                    "source_overlap": overlap,
                },
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_piece, selected = candidates[0]
    best_piece = replace(
        best_piece,
        piece_id=f"instrumental_v6_{voice_count}v_{form.lower()}",
    )
    score = _with_tempo(piece_to_canonical_score(best_piece), tempo)
    diagnostics: dict[str, object] = {
        "engine": "instrumental_v6",
        "requestedEngine": "instrumental_v6",
        "proposalEngine": "voice_aware_v2",
        "checkpoint": str(config.v6_checkpoint_path),
        "checkpointStep": runtime.checkpoint_step,
        "dataDir": str(config.v6_data_dir),
        "architecture": runtime.model_config.architecture,
        "voices": voice_count,
        "form": form,
        "templatePieceId": template.piece_id,
        "promptRows": prompt_count,
        "generatedRows": max_new_rows,
        "candidates": candidate_count,
        "temperature": temperature,
        "durationTemperature": duration_temperature,
        "durationPriorStrength": duration_prior_strength,
        "tempo": tempo,
        "seed": seed,
        "sourceBaseline": source_baseline,
        "selected": selected,
        "candidateScores": [candidate[2] for candidate in candidates],
    }
    return compose_canonical_score(
        score,
        generation=GenerationResult(
            ids=[],
            tokens=[
                "INSTRUMENTAL_V6",
                "VOICE_AWARE_V2",
                f"VOICES_{voice_count}",
                f"FORM_{form}",
                f"ROWS_{max_new_rows}",
            ],
            stopped_on_eos=True,
        ),
        render_mode=request.render_mode,
        diagnostics=diagnostics,
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


def _compose_v5_hybrid_result(
    request: ComposeRequest,
    *,
    controls: ComposeControls,
    config: ComposeRuntimeConfig,
    constraints: dict[str, Any],
    hybrid_context: Any,
) -> ComposeServiceResult:
    if config.v5_checkpoint_path is None or config.v5_data_dir is None:
        raise ValueError("v5 hybrid requires BACH_GEN_V5_CHECKPOINT and BACH_GEN_V5_DATA_DIR")

    import pandas as pd

    from scripts.generate_instrumental_v5 import _generate_best_rows, _template_piece
    from scripts.make_v5_listening_batch import _reset_positions
    from src.instrumental_v3.metrics import evaluate_slices, source_overlap_report
    from src.instrumental_v3.representation import (
        FIELD_NAMES as V3_FIELD_NAMES,
        piece_to_canonical_score,
        slice_rows_to_piece,
    )
    from src.instrumental_v4.model import CompoundConfig
    from src.instrumental_v5.model import build_generator
    from src.instrumental_v5.representation import V5_FIELD_NAMES

    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.init()

    ckpt = torch.load(config.v5_checkpoint_path, map_location=device)
    if list(ckpt.get("field_names", [])) != V5_FIELD_NAMES:
        raise ValueError("v5 checkpoint field_names do not match current v5 representation")
    model_config = CompoundConfig(**ckpt["config"])
    model = build_generator(model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    events = pd.read_parquet(config.v5_data_dir / "events.parquet")
    grouped = [group.sort_values("row_index").copy() for _, group in events.groupby("piece_id", sort=False)]
    if not grouped:
        raise ValueError("v5 events.parquet has no pieces")
    template_df = grouped[min(max(0, config.v5_piece_index), len(grouped) - 1)]
    template = _template_piece(template_df)
    prompt_rows = template_df[V5_FIELD_NAMES].to_numpy(dtype="int64").tolist()[: config.v5_prompt_rows]
    if len(prompt_rows) < 2:
        raise ValueError("v5 prompt must contain at least two rows")

    if hybrid_context is not None:
        from src.inference.hybrid import apply_conditioning_to_v5_rows

        prompt_rows = apply_conditioning_to_v5_rows(
            prompt_rows,
            hybrid_context,
            steps_per_bar=template.steps_per_bar,
        )

    measure_count = controls.measures or 4
    max_new_rows = max(1, measure_count * template.steps_per_bar)
    source_pieces = [_template_piece(group) for group in grouped]
    rows, rerank_diagnostics = _generate_best_rows(
        model,
        prompt_rows=[row[:] for row in prompt_rows],
        template=template,
        max_new_rows=max_new_rows,
        device=device,
        max_context=model_config.max_seq_len,
        temperature=_constraint_float(constraints, "temperature", default=config.temperature),
        top_p=_constraint_float(constraints, "top_p", "topP", default=config.top_p),
        top_k=_constraint_int(constraints, "top_k", "topK", default=0) or 0,
        hybrid_context=hybrid_context,
        candidate_count=_constraint_int(constraints, "candidates", "candidateCount", default=config.v5_candidates) or 1,
        source_pieces=source_pieces,
    )
    generated_only = _reset_positions(rows[len(prompt_rows) :], template=template)
    v3_rows = [row[: len(V3_FIELD_NAMES)] for row in generated_only]
    piece = slice_rows_to_piece(
        v3_rows,
        template=template,
        piece_id="instrumental_v5_hybrid_compose",
        source_path=str(config.v5_checkpoint_path),
    )
    score = piece_to_canonical_score(piece)
    quality = evaluate_slices(piece.slices).to_dict()
    novelty = source_overlap_report(piece.slices, [source.slices for source in source_pieces], ngram=16)
    diagnostics: dict[str, object] = {
        "engine": "hybrid",
        "requestedEngine": "hybrid",
        "proposalEngine": "instrumental_v5_transformer",
        "v5": {
            "checkpoint": str(config.v5_checkpoint_path),
            "dataDir": str(config.v5_data_dir),
            "promptRows": len(prompt_rows),
            "generatedRows": len(generated_only),
            "pieceIndex": config.v5_piece_index,
            "fieldCount": len(V5_FIELD_NAMES),
            "candidates": rerank_diagnostics["candidate_count"],
        },
        "candidate_rerank": rerank_diagnostics,
        "hybrid": hybrid_context.diagnostics() if hybrid_context is not None else None,
        "generated_metrics": quality,
        "novelty": novelty,
    }
    return compose_canonical_score(
        score,
        generation=GenerationResult(
            ids=[],
            tokens=[
                "V5_INSTRUMENTAL",
                "HYBRID_RETRIEVAL_CONDITIONED",
                f"ROWS_{len(generated_only)}",
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
        "instrumental-v6": "instrumental_v6",
        "v6": "instrumental_v6",
        "voice-aware-v2": "instrumental_v6",
    }
    engine = aliases.get(normalized)
    if engine is None:
        raise ValueError("engine must be one of: transformer, emi, hybrid, instrumental_v6")
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


def _constraint_v6_form(constraints: dict[str, Any], *, voice_count: int) -> str:
    value = _constraint_text(constraints, "form")
    if value is None:
        return {2: "INVENTION", 3: "SINFONIA"}.get(voice_count, "FUGUE")
    normalized = value.strip().upper()
    valid = {"INVENTION", "SINFONIA", "FUGUE", "SUITE", "PARTITA", "PRELUDE"}
    if normalized not in valid:
        raise ValueError(f"instrumental_v6 form must be one of: {', '.join(sorted(valid))}")
    return normalized


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


def _env_optional_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
