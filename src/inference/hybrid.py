from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.emi.cmmc import cmmc_function_id
from src.emi.fragments import Fragment, FragmentMatch, FragmentQuery, fragment_from_jsonl, rank_fragments
from src.emi.planner import (
    PhrasePlanStep,
    build_phrase_plan,
    cadence_type_id,
    harmonic_function_id,
    phrase_role_id,
    speac_label_id,
)
from src.inference.controls import ComposeControls, normalize_texture
from src.instrumental_v5.representation import (
    CONTOUR_BUCKET_TO_ID,
    RHYTHM_BUCKET_TO_ID,
    V5_EMI_FIELD_NAMES,
    V5_FIELD_NAMES,
    contour_bucket_id,
    rhythm_bucket_id,
)

HYBRID_ENGINE_VERSION = "hybrid_retrieval_conditioned_v1"


@dataclass(frozen=True)
class HybridContext:
    plan: list[PhrasePlanStep]
    conditioning_rows: list[dict[str, int]]
    fragment_count: int
    retrieved_matches: list[FragmentMatch]
    copy_hashes: tuple[str, ...]

    def model_conditioning(self) -> dict[str, object]:
        return {
            "version": HYBRID_ENGINE_VERSION,
            "field_names": V5_EMI_FIELD_NAMES,
            "rows": self.conditioning_rows,
        }

    def diagnostics(self) -> dict[str, object]:
        role_counts = Counter(step.phrase_role for step in self.plan)
        match_reason_counts = Counter()
        for match in self.retrieved_matches:
            match_reason_counts.update(name for name, value in match.reasons.items() if value > 0)
        return {
            "hybridVersion": HYBRID_ENGINE_VERSION,
            "fragmentMemoryCount": self.fragment_count,
            "retrievedFragmentCount": len(self.retrieved_matches),
            "rolePlan": [step.phrase_role for step in self.plan],
            "speacLabels": [step.speac_label for step in self.plan],
            "cmmcFunctions": [step.cmmc_function for step in self.plan],
            "cadenceTargets": [step.cadence_target for step in self.plan],
            "harmonicFunctions": [step.harmonic_function for step in self.plan],
            "conditioningFields": list(V5_EMI_FIELD_NAMES),
            "conditioningRows": self.conditioning_rows,
            "roleCounts": dict(sorted(role_counts.items())),
            "retrievalReasonCounts": dict(sorted(match_reason_counts.items())),
            "copyRiskHashCount": len(self.copy_hashes),
        }


def build_hybrid_context(
    controls: ComposeControls,
    *,
    fragment_path: Path | None = None,
    retrieval_limit: int = 1,
) -> HybridContext:
    measures = controls.measures or 4
    texture = normalize_texture(controls.texture)
    plan = build_phrase_plan(
        measures=measures,
        key=controls.key or "C",
        texture=texture,
    )
    fragments = load_fragment_memory(fragment_path)
    matches = retrieve_plan_fragments(plan, fragments, limit=retrieval_limit)
    conditioning_rows = _conditioning_rows(plan, matches)
    copy_hashes = tuple(sorted({fragment.copy_hash for fragment in fragments if fragment.copy_hash}))
    return HybridContext(
        plan=plan,
        conditioning_rows=conditioning_rows,
        fragment_count=len(fragments),
        retrieved_matches=matches,
        copy_hashes=copy_hashes,
    )


def load_fragment_memory(path: Path | None) -> list[Fragment]:
    if path is None or not path.exists():
        return []
    fragments: list[Fragment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                fragments.append(fragment_from_jsonl(stripped))
    return fragments


def retrieve_plan_fragments(
    plan: Sequence[PhrasePlanStep],
    fragments: Sequence[Fragment],
    *,
    limit: int = 1,
) -> list[FragmentMatch]:
    if limit <= 0 or not fragments:
        return []
    selected: list[FragmentMatch] = []
    used_copy_hashes: set[str] = set()
    for step in plan:
        matches = rank_fragments(
            FragmentQuery(
                phrase_role=step.phrase_role,
                speac_label=step.speac_label,
                cmmc_function=step.cmmc_function,
                cadence_type=step.cadence_target,
                local_key_pc=step.local_key_pc,
                mode=step.mode,
                harmonic_function=step.harmonic_function,
                target_beats=step.target_beats,
                avoid_copy_hashes=tuple(sorted(used_copy_hashes)),
            ),
            fragments,
            limit=limit,
        )
        if not matches:
            continue
        match = matches[0]
        selected.append(match)
        if match.fragment.copy_hash:
            used_copy_hashes.add(match.fragment.copy_hash)
    return selected


def conditioning_has_raw_fragment_ids(conditioning: dict[str, object]) -> bool:
    serialized = repr(conditioning)
    return "fragment_id" in serialized or "fragmentId" in serialized or "_v0_s" in serialized or "_v1_s" in serialized


def apply_conditioning_to_v5_rows(
    rows: Sequence[Sequence[int]],
    context: HybridContext,
    *,
    steps_per_bar: int,
) -> list[list[int]]:
    return [
        apply_conditioning_to_v5_row(row, context, row_index=row_index, steps_per_bar=steps_per_bar)
        for row_index, row in enumerate(rows)
    ]


def apply_conditioning_to_v5_row(
    row: Sequence[int],
    context: HybridContext,
    *,
    row_index: int,
    steps_per_bar: int,
) -> list[int]:
    if steps_per_bar <= 0:
        raise ValueError("steps_per_bar must be positive")
    conditioned = list(row)
    if len(conditioned) != len(V5_FIELD_NAMES):
        raise ValueError(f"expected v5 row width {len(V5_FIELD_NAMES)}, got {len(conditioned)}")
    if not context.conditioning_rows:
        return conditioned
    field_indices = {field: V5_FIELD_NAMES.index(field) for field in V5_EMI_FIELD_NAMES}
    plan_index = min(max(0, row_index) // steps_per_bar, len(context.conditioning_rows) - 1)
    conditioning_row = context.conditioning_rows[plan_index]
    for field, value in conditioning_row.items():
        if field in field_indices:
            conditioned[field_indices[field]] = int(value)
    return conditioned


def _conditioning_rows(
    plan: Sequence[PhrasePlanStep],
    matches: Sequence[FragmentMatch],
) -> list[dict[str, int]]:
    match_by_index = {idx: match.fragment for idx, match in enumerate(matches)}
    rows: list[dict[str, int]] = []
    for idx, step in enumerate(plan):
        fragment = match_by_index.get(idx)
        rows.append(
            {
                "phrase_role": phrase_role_id(step.phrase_role),
                "speac_label": speac_label_id(step.speac_label),
                "cmmc_function": cmmc_function_id(step.cmmc_function),
                "cadence_target": cadence_type_id(step.cadence_target),
                "harmonic_function": harmonic_function_id(step.harmonic_function),
                "local_key_pc": max(0, min(12, int(step.local_key_pc))),
                "retrieved_contour_bucket": (
                    contour_bucket_id(fragment.contour_bucket)
                    if fragment is not None
                    else CONTOUR_BUCKET_TO_ID["UNKNOWN"]
                ),
                "retrieved_rhythm_bucket": (
                    rhythm_bucket_id(fragment.rhythm_bucket)
                    if fragment is not None
                    else RHYTHM_BUCKET_TO_ID["UNKNOWN"]
                ),
            }
        )
    return rows


def plan_to_jsonable(plan: Sequence[PhrasePlanStep]) -> list[dict[str, object]]:
    return [asdict(step) for step in plan]
