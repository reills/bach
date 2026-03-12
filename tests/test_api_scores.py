import asyncio
import base64

import httpx
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.compose_service import ComposeServiceResult, build_event_hit_map, build_measure_map
from src.api.render import canonical_score_to_midi, canonical_score_to_musicxml
from src.api.services import preview_window_inpaint
from src.api.store import InMemoryScoreRepository
from src.inference.generate_v1 import GenerationResult


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
        Measure(id="m2", index=2, start_tick=48, length_ticks=24),
    ]
    part = Part(
        info=PartInfo(
            id="part-0",
            instrument="classical_guitar",
            tuning=[40, 45, 50, 55, 59, 64],
            midi_program=24,
        ),
        events=[
            Event(id="carry", start_tick=0, dur_tick=30, voice_id=0, pitch_midi=60),
            Event(id="m0-note", start_tick=12, dur_tick=6, voice_id=1, pitch_midi=64),
            Event(id="m1-note-a", start_tick=24, dur_tick=12, voice_id=0, pitch_midi=62),
            Event(id="m1-note-b", start_tick=36, dur_tick=12, voice_id=1, pitch_midi=65),
            Event(id="m2-note", start_tick=48, dur_tick=12, voice_id=0, pitch_midi=67),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


def test_compose_route_stores_generated_score_and_returns_frontend_payload():
    repository = InMemoryScoreRepository[CanonicalScore]()
    score = _build_score()
    captured: dict[str, object] = {}

    def fake_compose_service(request) -> ComposeServiceResult:
        captured["request"] = request.model_dump()
        return ComposeServiceResult(
            generation=GenerationResult(ids=[1, 2], tokens=["BAR", "EOS"], stopped_on_eos=True),
            score=score,
            score_xml=canonical_score_to_musicxml(score),
            midi=canonical_score_to_midi(score),
            measure_map=build_measure_map(score),
            event_hit_map=build_event_hit_map(score),
        )

    client = CompatTestClient(
        create_app(
            compose_service=fake_compose_service,
            repository=repository,
        )
    )
    try:
        response = client.post(
            "/compose",
            json={
                "prompt": "Prelude",
                "constraints": {"temperature": 0.5},
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert captured == {
        "request": {
            "prompt": "Prelude",
            "constraints": {"temperature": 0.5},
        }
    }

    payload = response.json()
    assert payload["scoreId"] == "score-1"
    assert payload["revision"] == 1
    assert payload["scoreXML"] == canonical_score_to_musicxml(score)
    assert payload["measureMap"] == build_measure_map(score)
    assert payload["eventHitMap"] == build_event_hit_map(score)
    assert base64.b64decode(payload["midi"]) == canonical_score_to_midi(score)

    stored = repository.get_score(payload["scoreId"])
    assert stored.revision == 1
    assert stored.score == score


def test_preview_then_commit_routes_update_score_and_revision():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        preview_response = client.post(
            "/inpaint_preview",
            json={
                "scoreId": stored_score.score_id,
                "measureId": "m1",
                "revision": stored_score.revision,
                "mode": "window",
            },
        )
        assert preview_response.status_code == 200

        preview_payload = preview_response.json()
        assert preview_payload["draftId"] == "draft-1"
        assert preview_payload["baseRevision"] == 1
        assert preview_payload["highlightMeasureId"] == "m1"
        assert preview_payload["changedMeasureIds"] == ["m1"]
        assert preview_payload["lockedEventIds"] == ["carry"]
        assert preview_payload["measureMap"] == {"0": "m0", "1": "m1", "2": "m2"}
        assert "part-0-m1-regen-0" in preview_payload["scoreXML"]

        commit_response = client.post(
            "/commit_draft",
            json={
                "scoreId": stored_score.score_id,
                "draftId": preview_payload["draftId"],
            },
        )
    finally:
        client.close()

    assert commit_response.status_code == 200
    commit_payload = commit_response.json()
    assert commit_payload["revision"] == 2
    assert "part-0-m1-regen-0" in commit_payload["scoreXML"]
    assert commit_payload["measureMap"] == {"0": "m0", "1": "m1", "2": "m2"}
    assert "part-0-m1-regen-0" in commit_payload["eventHitMap"].values()

    committed = repository.get_score(stored_score.score_id)
    assert committed.revision == 2
    assert committed.score.parts[0].events[2].id == "part-0-m1-regen-0"


def test_commit_draft_returns_409_for_stale_revision_conflicts():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())

    stale_preview = preview_window_inpaint(
        repository,
        stored_score.score_id,
        revision=stored_score.revision,
        measure_id="m1",
    )
    fresh_draft = repository.create_draft(
        stored_score.score_id,
        base_revision=stored_score.revision,
    )
    repository.commit_draft(fresh_draft.draft_id)

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/commit_draft",
            json={
                "scoreId": stored_score.score_id,
                "draftId": stale_preview.draft_id,
            },
        )
    finally:
        client.close()

    assert response.status_code == 409
    assert "based on revision 1" in response.text
