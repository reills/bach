import asyncio

import httpx
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.compose_service import ComposeServiceResult, export_score
from src.api.render import canonical_score_to_midi
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
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[
            Measure(id="m0", index=0, start_tick=0, length_ticks=96),
            Measure(id="m1", index=1, start_tick=96, length_ticks=96),
        ],
        parts=[
            Part(
                info=PartInfo(id="part-0", instrument="piano"),
                events=[
                    Event(id="soprano-0", start_tick=0, dur_tick=96, voice_id=0, pitch_midi=72),
                    Event(id="bass-0", start_tick=0, dur_tick=96, voice_id=1, pitch_midi=48),
                    Event(id="soprano-1", start_tick=96, dur_tick=96, voice_id=0, pitch_midi=74),
                    Event(id="bass-1", start_tick=96, dur_tick=96, voice_id=1, pitch_midi=50),
                ],
            )
        ],
    )


def _generated_score(measure_count: int, *, soprano: int = 80, bass: int = 56) -> CanonicalScore:
    measures = [
        Measure(id=f"g{i}", index=i, start_tick=i * 96, length_ticks=96)
        for i in range(measure_count)
    ]
    events: list[Event] = []
    for measure in measures:
        events.extend(
            [
                Event(
                    id=f"gen-s-{measure.index}",
                    start_tick=measure.start_tick,
                    dur_tick=96,
                    voice_id=0,
                    pitch_midi=soprano + measure.index,
                ),
                Event(
                    id=f"gen-b-{measure.index}",
                    start_tick=measure.start_tick,
                    dur_tick=96,
                    voice_id=1,
                    pitch_midi=bass + measure.index,
                ),
            ]
        )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[Part(info=PartInfo(id="part-0", instrument="piano"), events=events)],
    )


def _fake_compose_service(captured: dict[str, object]):
    def fake_compose_service(request) -> ComposeServiceResult:
        captured["request"] = request.model_dump()
        measure_count = int(request.constraints["measures"])
        generated = _generated_score(measure_count)
        exported = export_score(generated)
        return ComposeServiceResult(
            generation=GenerationResult(ids=[], tokens=["GENERATED"], stopped_on_eos=True),
            score=generated,
            score_xml=exported.score_xml,
            midi=canonical_score_to_midi(generated),
            measure_map=exported.measure_map,
            event_hit_map=exported.event_hit_map,
            render_mode=request.render_mode,
        )

    return fake_compose_service


def test_generate_measures_appends_generated_context_fitted_material():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())
    captured: dict[str, object] = {}
    client = CompatTestClient(
        create_app(compose_service=_fake_compose_service(captured), repository=repository)
    )
    try:
        response = client.post(
            "/generate_measures",
            json={
                "scoreId": stored.score_id,
                "revision": stored.revision,
                "operation": "append",
                "count": 2,
                "constraints": {"engine": "instrumental_v6", "seed": 1234},
                "render_mode": "piano",
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 2
    assert payload["insertedMeasureIds"] == ["m2", "m3"]
    assert payload["changedMeasureIds"] == ["m2", "m3"]
    assert payload["diagnostics"]["contextFitTransposition"] == -6
    assert payload["document"]["views"]["score"]["measureMap"] == {
        "0": "m0",
        "1": "m1",
        "2": "m2",
        "3": "m3",
    }

    request = captured["request"]
    assert request["constraints"]["measures"] == 2
    assert request["constraints"]["texture"] == 2
    assert request["constraints"]["seed"] == 1234
    assert request["constraints"]["measureContext"]["insertIndex"] == 2
    assert [item["id"] for item in request["constraints"]["measureContext"]["before"]] == ["m0", "m1"]

    committed = repository.get_score(stored.score_id)
    appended_events = [
        event
        for event in committed.score.parts[0].events
        if event.start_tick >= 192
    ]
    assert [(event.start_tick, event.voice_id, event.pitch_midi) for event in appended_events] == [
        (192, 0, 74),
        (192, 1, 50),
        (288, 0, 75),
        (288, 1, 51),
    ]


def test_generate_measures_inserts_after_selected_measure_and_exports_updated_score():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())
    captured: dict[str, object] = {}
    client = CompatTestClient(
        create_app(compose_service=_fake_compose_service(captured), repository=repository)
    )
    try:
        response = client.post(
            "/generate_measures",
            json={
                "scoreId": stored.score_id,
                "revision": stored.revision,
                "operation": "insert_after",
                "measureId": "m0",
                "count": 1,
                "constraints": {"engine": "instrumental_v6", "seed": 5678},
                "render_mode": "piano",
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 2
    assert payload["insertedMeasureIds"] == ["m1-2"]
    assert payload["replacedMeasureIds"] == []
    assert payload["changedMeasureIds"] == ["m1-2"]
    assert payload["document"]["views"]["score"]["measureMap"] == {
        "0": "m0",
        "1": "m1-2",
        "2": "m1",
    }
    assert payload["scoreXML"] is not None

    request = captured["request"]
    assert request["constraints"]["measures"] == 1
    assert request["constraints"]["texture"] == 2
    assert request["constraints"]["seed"] == 5678
    assert request["constraints"]["measureContext"]["insertIndex"] == 1
    assert [item["id"] for item in request["constraints"]["measureContext"]["before"]] == ["m0"]
    assert [item["id"] for item in request["constraints"]["measureContext"]["after"]] == ["m1"]

    committed = repository.get_score(stored.score_id)
    assert [measure.id for measure in committed.score.measures] == ["m0", "m1-2", "m1"]
    assert [(measure.index, measure.start_tick) for measure in committed.score.measures] == [
        (0, 0),
        (1, 96),
        (2, 192),
    ]
    inserted_events = [
        event
        for event in committed.score.parts[0].events
        if event.start_tick == 96
    ]
    shifted_events = [
        event
        for event in committed.score.parts[0].events
        if event.start_tick == 192
    ]
    assert [(event.voice_id, event.pitch_midi) for event in inserted_events] == [
        (0, 73),
        (1, 49),
    ]
    assert [(event.voice_id, event.pitch_midi) for event in shifted_events] == [
        (0, 74),
        (1, 50),
    ]


def test_generate_measures_rewrites_selected_measure_with_new_id():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())
    captured: dict[str, object] = {}
    client = CompatTestClient(
        create_app(compose_service=_fake_compose_service(captured), repository=repository)
    )
    try:
        response = client.post(
            "/generate_measures",
            json={
                "scoreId": stored.score_id,
                "revision": stored.revision,
                "operation": "replace",
                "measureId": "m0",
                "count": 1,
                "render_mode": "piano",
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["insertedMeasureIds"] == ["m0-2"]
    assert payload["replacedMeasureIds"] == ["m0"]
    assert payload["changedMeasureIds"] == ["m0", "m0-2"]
    assert payload["document"]["views"]["score"]["measureMap"] == {
        "0": "m0-2",
        "1": "m1",
    }

    committed = repository.get_score(stored.score_id)
    assert [measure.id for measure in committed.score.measures] == ["m0-2", "m1"]
    rewritten_events = [
        event
        for event in committed.score.parts[0].events
        if event.start_tick == 0
    ]
    assert [(event.voice_id, event.pitch_midi) for event in rewritten_events] == [
        (0, 74),
        (1, 50),
    ]


def test_generate_measures_rejects_missing_selected_measure_for_replace():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())
    captured: dict[str, object] = {}
    client = CompatTestClient(
        create_app(compose_service=_fake_compose_service(captured), repository=repository)
    )
    try:
        response = client.post(
            "/generate_measures",
            json={
                "scoreId": stored.score_id,
                "revision": stored.revision,
                "operation": "replace",
                "count": 1,
            },
        )
    finally:
        client.close()

    assert response.status_code == 400
    assert response.json() == {"detail": "measureId is required for replace"}
