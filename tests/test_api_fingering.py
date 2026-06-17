import asyncio
import xml.etree.ElementTree as ET

import httpx
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.canonical import (
    CanonicalScore,
    Event,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
)
from src.api.render import canonical_score_to_musicxml
from src.api.store import InMemoryScoreRepository


class CompatTestClient(TestClient):
    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def run_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=str(self.base_url)) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run_request())


def _build_score() -> CanonicalScore:
    measures = [
        Measure(id="m0", index=0, start_tick=0, length_ticks=24),
        Measure(id="m1", index=1, start_tick=24, length_ticks=24),
    ]
    part = Part(
        info=PartInfo(
            id="part-0",
            instrument="classical_guitar",
            tuning=[40, 45, 50, 55, 59, 64],
            midi_program=24,
        ),
        events=[
            Event(id="lead", start_tick=0, dur_tick=6, voice_id=0, pitch_midi=64),
            Event(
                id="carry",
                start_tick=0,
                dur_tick=30,
                voice_id=1,
                pitch_midi=60,
                fingering=GuitarFingering(string_index=4, fret=1),
            ),
            Event(id="answer", start_tick=12, dur_tick=6, voice_id=0, pitch_midi=62),
            Event(id="next", start_tick=24, dur_tick=12, voice_id=0, pitch_midi=65),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


def _pitch_summary(score_xml: str) -> list[tuple[str, str, str, str]]:
    root = ET.fromstring(score_xml)
    summary: list[tuple[str, str, str, str]] = []
    for note_el in root.findall("./part/measure/note"):
        pitch_el = note_el.find("./pitch")
        if pitch_el is None:
            continue
        summary.append(
            (
                note_el.attrib.get("{http://www.w3.org/XML/1998/namespace}id", ""),
                pitch_el.findtext("./step", ""),
                pitch_el.findtext("./alter", ""),
                pitch_el.findtext("./octave", ""),
            )
        )
    return summary


def _technical_summary(score_xml: str, *, event_id: str) -> list[tuple[str, str]]:
    root = ET.fromstring(score_xml)
    summary: list[tuple[str, str]] = []
    for note_el in root.findall("./part/measure/note"):
        if note_el.attrib.get("{http://www.w3.org/XML/1998/namespace}id") != event_id:
            continue
        summary_el = note_el.find("./notations/technical")
        if summary_el is None:
            continue
        summary.append(
            (
                summary_el.findtext("./string", ""),
                summary_el.findtext("./fret", ""),
            )
        )
    return summary


def test_alt_positions_resolves_hit_key_to_event_and_returns_compact_picker_options():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/alt_positions",
            json={
                "scoreId": stored_score.score_id,
                "measureId": "m1",
                "eventHitKey": {
                    "barIndex": 1,
                    "voiceIndex": 1,
                    "beatIndex": 0,
                    "noteIndex": 0,
                },
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {
        "eventId": "carry",
        "options": [
            {"stringIndex": 4, "fret": 1, "selected": True},
            {"stringIndex": 3, "fret": 5, "selected": False},
            {"stringIndex": 2, "fret": 10, "selected": False},
            {"stringIndex": 1, "fret": 15, "selected": False},
            {"stringIndex": 0, "fret": 20, "selected": False},
        ],
    }


def test_alt_positions_returns_404_when_event_hit_key_is_missing():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/alt_positions",
            json={
                "scoreId": stored_score.score_id,
                "measureId": "m1",
                "eventHitKey": {
                    "barIndex": 1,
                    "voiceIndex": 1,
                    "beatIndex": 9,
                    "noteIndex": 0,
                },
            },
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown event hit key: 1|1|9|0"}


def test_apply_fingering_updates_musicxml_technical_tags_without_changing_pitches():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())
    original_xml = canonical_score_to_musicxml(stored_score.score)

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/apply_fingering",
            json={
                "scoreId": stored_score.score_id,
                "revision": stored_score.revision,
                "fingeringSelections": [
                    {"eventId": "answer", "stringIndex": 1, "fret": 7},
                    {"eventId": "next", "stringIndex": 0, "fret": 20},
                ],
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 2
    assert _pitch_summary(payload["scoreXML"]) == _pitch_summary(original_xml)
    assert _technical_summary(original_xml, event_id="answer") == []
    assert _technical_summary(original_xml, event_id="next") == []
    assert _technical_summary(payload["scoreXML"], event_id="answer") == [("5", "7")]
    assert _technical_summary(payload["scoreXML"], event_id="next") == [("6", "20")]

    committed = repository.get_score(stored_score.score_id)
    assert committed.revision == 2
    assert committed.score.parts[0].events[2].fingering == GuitarFingering(string_index=1, fret=7)
    assert committed.score.parts[0].events[3].fingering == GuitarFingering(string_index=0, fret=20)


def test_apply_fingering_returns_409_for_stale_revision():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())
    fresh_draft = repository.create_draft(
        stored_score.score_id,
        base_revision=stored_score.revision,
    )
    repository.commit_draft(fresh_draft.draft_id)

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/apply_fingering",
            json={
                "scoreId": stored_score.score_id,
                "revision": stored_score.revision,
                "fingeringSelections": [
                    {"eventId": "answer", "stringIndex": 1, "fret": 7},
                ],
            },
        )
    finally:
        client.close()

    assert response.status_code == 409
    assert "is at revision 2, not 1" in response.text
