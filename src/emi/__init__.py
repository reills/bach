"""EMI-inspired symbolic fragment analysis, retrieval, and composition helpers."""

from src.emi.composer import (
    EMI_ENGINE_VERSION,
    EmiComposerConfig,
    EmiComposition,
    compose_emi,
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
    "build_phrase_plan",
    "compose_emi",
    "extract_fragments",
    "fragment_from_jsonl",
    "fragment_to_jsonl",
    "rank_fragments",
    "score_fragment",
    "summarize_fragments",
]
