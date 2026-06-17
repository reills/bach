from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.dataio.musicxml_cleaner import MeterOverride, clean_musicxml_file
from src.instrumental_v6.model import LEGACY_METER_VOCAB_SIZE, config_from_checkpoint
from src.instrumental_v6.representation import METER_TO_ID, meter_id


def test_cleaner_repairs_isolated_meter_that_conflicts_with_bar_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.xml"
    output = tmp_path / "clean.xml"
    raw = _score_xml(["4/4", "4/4", "4/4", "3/4", "4/4", "4/4", "4/4"])
    source.write_text(raw, encoding="utf-8")

    report = clean_musicxml_file(source, output, relative_path="raw.xml")

    assert source.read_text(encoding="utf-8") == raw
    assert report.status == "repaired"
    assert report.training_approved
    assert report.changes[0]["before"] == "3/4"
    assert _meters(output) == ["4/4"] * 7


def test_reviewed_override_handles_same_duration_compound_meter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.xml"
    output = tmp_path / "clean.xml"
    source.write_text(_score_xml(["3/4", "3/4", "3/4"], notes_per_bar=3), encoding="utf-8")

    report = clean_musicxml_file(
        source,
        output,
        relative_path="works/BWV.xml",
        overrides=[
            MeterOverride(
                path="works/BWV.xml",
                movement_index=0,
                time_signature="6/8",
                source="reviewed score",
            )
        ],
    )

    assert report.status == "repaired"
    assert report.changes[0]["reason"] == "reviewed_override"
    assert _meters(output) == ["6/8", "6/8", "6/8"]


def test_reviewed_override_does_not_hide_corrupt_bar_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.xml"
    output = tmp_path / "clean.xml"
    source.write_text(_score_xml(["4/4"] * 5, notes_per_bar=3), encoding="utf-8")

    report = clean_musicxml_file(
        source,
        output,
        relative_path="works/BWV.xml",
        overrides=[
            MeterOverride(
                path="works/BWV.xml",
                movement_index=0,
                time_signature="4/4",
                source="reviewed score",
            )
        ],
    )

    assert report.status == "review_required"
    assert not report.training_approved
    assert any(issue["kind"] == "override_content_mismatch" for issue in report.issues)


def test_cleaner_does_not_guess_between_equal_duration_meters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.xml"
    output = tmp_path / "clean.xml"
    raw = _score_xml(["8/8", "8/8", "8/8", "4/4", "8/8", "8/8", "8/8"])
    source.write_text(raw, encoding="utf-8")

    report = clean_musicxml_file(source, output, relative_path="raw.xml")

    assert report.status == "review_required"
    assert not report.training_approved
    assert report.changes == []
    assert _meters(output) == ["8/8", "8/8", "8/8", "4/4", "8/8", "8/8", "8/8"]
    assert any(issue["kind"] == "equivalent_meter_change" for issue in report.issues)


def test_common_equivalent_meters_do_not_become_unknown() -> None:
    assert meter_id("8/8") == METER_TO_ID["8/8"]
    assert meter_id("6/4") == METER_TO_ID["6/4"]


def test_legacy_checkpoint_keeps_original_meter_embedding_size() -> None:
    config = config_from_checkpoint({"max_voices": 6, "d_model": 48, "n_heads": 6})

    assert config.meter_vocab_size == LEGACY_METER_VOCAB_SIZE


def test_suspicious_meter_is_quarantined_for_review(tmp_path: Path) -> None:
    source = tmp_path / "raw.xml"
    output = tmp_path / "clean.xml"
    source.write_text(_score_xml(["33/32"] * 4), encoding="utf-8")

    report = clean_musicxml_file(source, output, relative_path="raw.xml")

    assert report.status == "review_required"
    assert not report.training_approved
    assert any(issue["kind"] == "suspicious_meter" for issue in report.issues)


def _score_xml(meters: list[str], *, notes_per_bar: int = 4) -> str:
    measures: list[str] = []
    for index, meter in enumerate(meters, start=1):
        beats, beat_type = meter.split("/")
        notes = "".join(
            """
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>1</duration><type>quarter</type>
      </note>"""
            for _ in range(notes_per_bar)
        )
        barline = (
            '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
            if index == len(meters)
            else ""
        )
        measures.append(
            f"""
    <measure number="{index}">
      <attributes>
        <divisions>1</divisions>
        <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>{notes}
      {barline}
    </measure>"""
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="4.0">\n'
        '  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n'
        '  <part id="P1">'
        + "".join(measures)
        + "\n  </part>\n</score-partwise>\n"
    )


def _meters(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        f"{time.findtext('beats')}/{time.findtext('beat-type')}"
        for time in root.findall("./part/measure/attributes/time")
    ]
