from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tokens.schema import DescriptorSpec, EventSpec, SCHEMA_VERSION, STAGE1_TPQ


def test_event_spec_defaults():
    spec = EventSpec()
    assert spec.version == SCHEMA_VERSION
    assert spec.tpq == STAGE1_TPQ

    assert set(spec.tokens) == {
        "structural",
        "voice_events",
        "anchors",
        "intervals",
        "doubling",
    }
    assert "BAR" in spec.tokens["structural"]
    assert "MEL_INT12_{-24..+24}" in spec.tokens["intervals"]


def test_event_spec_enforces_stage1_constraints():
    with pytest.raises(ValueError, match="unsupported event schema version"):
        EventSpec(version="remi_tab_v2")

    with pytest.raises(ValueError, match="requires tpq=24"):
        EventSpec(tpq=12)


def test_event_spec_tokens_not_shared_between_instances():
    first = EventSpec()
    second = EventSpec()
    first.tokens["structural"].append("EXTRA_TOKEN")
    assert "EXTRA_TOKEN" not in second.tokens["structural"]


def test_descriptor_spec_defaults():
    spec = DescriptorSpec()
    assert spec.fields == [
        "TIME_SIG",
        "KEY",
        "CHORD_FN",
        "DENSITY",
        "CADENCE",
        "DIFFICULTY",
    ]
