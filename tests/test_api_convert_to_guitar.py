import asyncio
import base64

import httpx
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.store import InMemoryScoreRepository


class CompatTestClient(TestClient):
    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def run_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=str(self.base_url)) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run_request())


def _piano_score() -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(
            tpq=24,
            key_sig_map={0: "C"},
            time_sig_map={0: "4/4"},
            tempo_map={0: 88},
        ),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(id="part-0", instrument="piano", midi_program=0),
                events=[
                    Event(id="bass", start_tick=0, dur_tick=96, voice_id=0, pitch_midi=48),
                    Event(id="third", start_tick=0, dur_tick=96, voice_id=1, pitch_midi=52),
                    Event(id="fifth", start_tick=0, dur_tick=96, voice_id=2, pitch_midi=55),
                    Event(id="root", start_tick=0, dur_tick=96, voice_id=3, pitch_midi=60),
                    Event(id="melody", start_tick=0, dur_tick=96, voice_id=4, pitch_midi=64),
                ],
            )
        ],
    )


def test_convert_to_guitar_creates_independent_guitar_branch_without_mutating_piano():
    repository = InMemoryScoreRepository[CanonicalScore]()
    piano_score = _piano_score()
    stored_piano = repository.create_score(piano_score, name="Piano draft")
    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/api/convert-to-guitar",
            json={
                "scoreId": stored_piano.score_id,
                "revision": stored_piano.revision,
                "settings": {"difficulty": "medium", "maxFret": 12},
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["scoreId"] == "score-2"
    assert payload["revision"] == 1
    assert payload["branch"] == "guitar"
    assert payload["instrumentMode"] == "guitar"
    assert payload["sourcePianoScoreId"] == stored_piano.score_id
    assert payload["sourcePianoRevision"] == stored_piano.revision
    assert payload["sourcePianoRevisionId"] == f"{stored_piano.score_id}@{stored_piano.revision}"
    assert payload["scoreXML"] == payload["guitarMusicXml"]
    assert payload["document"]["instrumentMode"] == "guitar"
    assert payload["document"]["views"]["score"]["xml"] == payload["guitarMusicXml"]
    assert payload["document"]["views"]["tab"]["xml"] == payload["guitarTabXml"]
    assert "<staff-tuning" in payload["guitarTabXml"]
    assert base64.b64decode(payload["midi"])
    assert payload["conversionSettings"]["difficulty"] == "medium"
    assert payload["conversionSettings"]["maxFret"] == 12
    assert payload["diagnostics"]["droppedNotes"]
    assert len(payload["sourceMap"]) == 5
    assert any(note["dropped"] for note in payload["sourceMap"])

    unchanged_piano = repository.get_score(stored_piano.score_id)
    assert unchanged_piano.revision == stored_piano.revision
    assert unchanged_piano.score == piano_score

    stored_guitar = repository.get_score(payload["scoreId"])
    assert stored_guitar.revision == 1
    assert stored_guitar.score.parts[0].info.instrument == "classical_guitar"
    assert all(event.fingering is not None for event in stored_guitar.score.parts[0].events)


def test_convert_to_guitar_rejects_stale_piano_revision():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_piano = repository.create_score(_piano_score())
    draft = repository.create_draft(stored_piano.score_id, base_revision=stored_piano.revision)
    repository.save_draft(draft.draft_id, draft.score)
    repository.commit_draft(draft.draft_id)
    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/api/convert-to-guitar",
            json={"scoreId": stored_piano.score_id, "revision": stored_piano.revision},
        )
    finally:
        client.close()

    assert response.status_code == 409
    assert "is at revision 2, not 1" in response.json()["detail"]
