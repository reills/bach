"""Instrumental v5 EMI-conditioned compound counterpoint representation."""

from src.instrumental_v5.representation import (
    CONTOUR_BUCKET_NAMES,
    PHRASE_ROLE_NAMES,
    RHYTHM_BUCKET_NAMES,
    V5_EMI_FIELD_NAMES,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
    V5Piece,
    build_v5_piece,
    classify_contour_bucket,
    classify_rhythm_bucket,
)

__all__ = [
    "CONTOUR_BUCKET_NAMES",
    "PHRASE_ROLE_NAMES",
    "RHYTHM_BUCKET_NAMES",
    "V5_EMI_FIELD_NAMES",
    "V5_FEATURE_SPECS",
    "V5_FIELD_NAMES",
    "V5Piece",
    "build_v5_piece",
    "classify_contour_bucket",
    "classify_rhythm_bucket",
]
