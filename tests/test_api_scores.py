import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.compose_service import ComposeServiceResult, build_event_hit_map, build_measure_map, export_score
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


def _assert_measure_map(payload: dict, *, expected: dict[str, str]) -> None:
    assert payload["measureMap"] == expected
    assert list(payload["measureMap"].keys()) == ["0", "1", "2"]
    assert all(isinstance(measure_id, str) and measure_id for measure_id in payload["measureMap"].values())


def _assert_event_hit_map(payload: dict, *, expected: dict[str, str]) -> None:
    assert payload["eventHitMap"] == expected
    assert payload["eventHitMap"]
    for hit_key, event_id in payload["eventHitMap"].items():
        parts = hit_key.split("|")
        assert len(parts) == 4
        assert all(part.isdigit() for part in parts)
        assert isinstance(event_id, str) and event_id


def test_compose_route_stores_generated_score_and_returns_frontend_payload():
    repository = InMemoryScoreRepository[CanonicalScore]()
    score = _build_score()
    captured: dict[str, object] = {}
    exported = export_score(score)

    def fake_compose_service(request) -> ComposeServiceResult:
        captured["request"] = request.model_dump()
        return ComposeServiceResult(
            generation=GenerationResult(ids=[1, 2], tokens=["BAR", "EOS"], stopped_on_eos=True),
            score=score,
            score_xml=exported.score_xml,
            midi=canonical_score_to_midi(score),
            measure_map=exported.measure_map,
            event_hit_map=exported.event_hit_map,
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
            "name": None,
        }
    }

    payload = response.json()
    assert payload["scoreId"] == "score-1"
    assert payload["revision"] == 1
    assert payload["scoreXML"] == canonical_score_to_musicxml(score)
    _assert_measure_map(payload, expected=build_measure_map(score))
    _assert_event_hit_map(payload, expected=build_event_hit_map(score))
    assert base64.b64decode(payload["midi"]) == canonical_score_to_midi(score)

    stored = repository.get_score(payload["scoreId"])
    assert stored.revision == 1
    assert stored.score == score


def test_compose_route_allows_localhost_dev_origins_on_nondefault_ports():
    repository = InMemoryScoreRepository[CanonicalScore]()
    score = _build_score()
    exported = export_score(score)

    def fake_compose_service(request) -> ComposeServiceResult:
        return ComposeServiceResult(
            generation=GenerationResult(ids=[1, 2], tokens=["BAR", "EOS"], stopped_on_eos=True),
            score=score,
            score_xml=exported.score_xml,
            midi=canonical_score_to_midi(score),
            measure_map=exported.measure_map,
            event_hit_map=exported.event_hit_map,
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
            headers={"Origin": "http://localhost:5174"},
            json={},
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"


def test_compose_route_returns_json_error_for_invalid_generation_with_cors_header():
    def fake_compose_service(request) -> ComposeServiceResult:
        raise ValueError("generated token stream does not contain a complete event prefix")

    client = CompatTestClient(create_app(compose_service=fake_compose_service))
    try:
        response = client.post(
            "/compose",
            headers={"Origin": "http://localhost:5173"},
            json={},
        )
    finally:
        client.close()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "generated token stream does not contain a complete event prefix"
    }
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_compose_launcher_binds_real_compose_service(monkeypatch):
    from src.api.compose_launcher import ComposeRuntimeConfig, create_configured_app

    score = _build_score()
    exported = export_score(score)
    load_calls: list[dict[str, object]] = []
    captured: dict[str, object] = {}

    def fake_load_notelm_checkpoint(checkpoint_path, *, vocab_path=None, device="cpu"):
        load_calls.append(
            {
                "checkpoint_path": Path(checkpoint_path),
                "vocab_path": None if vocab_path is None else Path(vocab_path),
                "device": device,
            }
        )
        return SimpleNamespace(vocab_path=Path("/tmp/runtime-vocab.json"))

    def fake_compose_baseline(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
        device="cpu",
        generator=None,
    ) -> ComposeServiceResult:
        captured["checkpoint_path"] = Path(checkpoint_path)
        captured["seed_tokens"] = list(seed_tokens)
        captured["generation_config"] = generation_config
        captured["vocab_path"] = None if vocab_path is None else Path(vocab_path)
        captured["device"] = device
        captured["generator"] = generator
        return ComposeServiceResult(
            generation=GenerationResult(ids=[1, 2], tokens=["BAR", "EOS"], stopped_on_eos=True),
            score=score,
            score_xml=exported.score_xml,
            midi=canonical_score_to_midi(score),
            measure_map=exported.measure_map,
            event_hit_map=exported.event_hit_map,
        )

    monkeypatch.setattr("src.api.compose_launcher.load_notelm_checkpoint", fake_load_notelm_checkpoint)
    monkeypatch.setattr("src.api.compose_launcher.compose_baseline", fake_compose_baseline)

    app = create_configured_app(
        ComposeRuntimeConfig(
            checkpoint_path=Path("/tmp/notelm.pt"),
            vocab_path=Path("/tmp/vocab.json"),
            device="cpu",
            max_length=96,
            top_p=0.95,
        )
    )
    client = CompatTestClient(app)
    try:
        response = client.post(
            "/compose",
            json={
                "name": "Launcher test",
                "constraints": {
                    "key": "g minor",
                    "style": "chorale",
                    "difficulty": "easy",
                    "measures": 8,
                    "temperature": 0.5,
                    "topP": 0.8,
                    "maxLength": 64,
                },
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert load_calls == [
        {
            "checkpoint_path": Path("/tmp/notelm.pt"),
            "vocab_path": Path("/tmp/vocab.json"),
            "device": "cpu",
        }
    ]
    assert captured["checkpoint_path"] == Path("/tmp/notelm.pt")
    assert captured["seed_tokens"] == [
        "KEY_Gm",
        "STYLE_CHORALE",
        "DIFFICULTY_EASY",
        "MEAS_8",
    ]
    assert captured["generation_config"].max_length == 64
    assert captured["generation_config"].temperature == 0.5
    assert captured["generation_config"].top_p == 0.8
    assert captured["vocab_path"] == Path("/tmp/runtime-vocab.json")
    assert captured["device"] == "cpu"
    assert callable(captured["generator"])


def test_compose_launcher_uses_local_defaults_when_env_is_unset(monkeypatch):
    from src.api.compose_launcher import (
        DEFAULT_CHECKPOINT_PATH,
        DEFAULT_VOCAB_PATH,
        _runtime_config_from_env,
    )

    monkeypatch.delenv("BACH_GEN_CHECKPOINT", raising=False)
    monkeypatch.delenv("BACH_GEN_VOCAB", raising=False)
    monkeypatch.delenv("BACH_GEN_DEVICE", raising=False)

    config = _runtime_config_from_env()

    assert config.checkpoint_path == DEFAULT_CHECKPOINT_PATH
    assert config.vocab_path == DEFAULT_VOCAB_PATH


def test_compose_launcher_defaults_seed_controls_when_request_has_no_constraints(monkeypatch):
    from src.api.compose_launcher import ComposeRuntimeConfig, create_configured_app

    score = _build_score()
    exported = export_score(score)
    captured: dict[str, object] = {}

    def fake_load_notelm_checkpoint(checkpoint_path, *, vocab_path=None, device="cpu"):
        return SimpleNamespace(vocab_path=Path("/tmp/runtime-vocab.json"))

    def fake_compose_baseline(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
        device="cpu",
        generator=None,
    ) -> ComposeServiceResult:
        captured["seed_tokens"] = list(seed_tokens)
        captured["generation_config"] = generation_config
        return ComposeServiceResult(
            generation=GenerationResult(ids=[1, 2], tokens=["BAR", "EOS"], stopped_on_eos=True),
            score=score,
            score_xml=exported.score_xml,
            midi=canonical_score_to_midi(score),
            measure_map=exported.measure_map,
            event_hit_map=exported.event_hit_map,
        )

    monkeypatch.setattr("src.api.compose_launcher.load_notelm_checkpoint", fake_load_notelm_checkpoint)
    monkeypatch.setattr("src.api.compose_launcher.compose_baseline", fake_compose_baseline)

    app = create_configured_app(
        ComposeRuntimeConfig(
            checkpoint_path=Path("/tmp/notelm.pt"),
            vocab_path=Path("/tmp/vocab.json"),
            device="cpu",
        )
    )
    client = CompatTestClient(app)
    try:
        response = client.post("/compose", json={})
    finally:
        client.close()

    assert response.status_code == 200
    assert captured["seed_tokens"] == ["KEY_C", "MEAS_4"]
    assert captured["generation_config"].max_length == 512


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
        _assert_measure_map(
            preview_payload,
            expected={"0": "m0", "1": "m1", "2": "m2"},
        )
        _assert_event_hit_map(
            preview_payload,
            expected=build_event_hit_map(preview_payload["scoreXML"]),
        )
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
    _assert_measure_map(
        commit_payload,
        expected={"0": "m0", "1": "m1", "2": "m2"},
    )
    _assert_event_hit_map(
        commit_payload,
        expected=build_event_hit_map(commit_payload["scoreXML"]),
    )
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
