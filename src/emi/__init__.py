"""EMI-inspired symbolic fragment analysis, retrieval, and composition helpers."""

from src.emi.composer import (
    EMI_ENGINE_VERSION,
    EmiComposerConfig,
    EmiComposition,
    compose_emi,
)
from src.emi.cmmc import (
    CmmcEvent,
    CmmcPieceAnalysis,
    analyze_rows,
    get_function,
    gradus_evaluate,
    interval_translator,
    pattern_match,
    run_the_speac_weightings,
    simple_matcher,
)
from src.emi.fragments import (
    Fragment,
    FragmentMatch,
    FragmentQuery,
    extract_fragments,
    fragment_from_jsonl,
    fragment_to_jsonl,
    rank_fragments,
    score_fragment,
    summarize_fragments,
)
from src.emi.planner import PhrasePlanStep, build_phrase_plan

__all__ = [
    "EMI_ENGINE_VERSION",
    "EmiComposerConfig",
    "EmiComposition",
    "Fragment",
    "FragmentMatch",
    "FragmentQuery",
    "PhrasePlanStep",
    "CmmcEvent",
    "CmmcPieceAnalysis",
    "analyze_rows",
    "build_phrase_plan",
    "compose_emi",
    "extract_fragments",
    "fragment_from_jsonl",
    "fragment_to_jsonl",
    "get_function",
    "gradus_evaluate",
    "interval_translator",
    "pattern_match",
    "rank_fragments",
    "run_the_speac_weightings",
    "score_fragment",
    "simple_matcher",
    "summarize_fragments",
]
