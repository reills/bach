"""Focused tests for the API error contract.

Covers: missing score IDs (404), stale revisions (409), bad measure IDs (404),
bad event IDs (404), and missing draft IDs (404) across all routes.
"""
from __future__ import annotations

import asyncio

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
            Event(id="e0", start_tick=0, dur_tick=12, voice_id=0, pitch_midi=60),
            Event(id="e1", start_tick=24, dur_tick=12, voice_id=0, pitch_midi=62),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


# ---------------------------------------------------------------------------
# /inpaint_preview — score not found
# ---------------------------------------------------------------------------

def test_inpaint_preview_returns_404_for_unknown_score_id():
    client = CompatTestClient(create_app())
    try:
        response = client.post(
            "/inpaint_preview",
            json={"scoreId": "score-99", "measureId": "m0", "revision": 1},
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "score-99" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /inpaint_preview — stale revision
# ---------------------------------------------------------------------------

def test_inpaint_preview_returns_409_for_stale_revision():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())
    # Advance revision by committing a draft
    draft = repository.create_draft(stored.score_id, base_revision=stored.revision)
    repository.commit_draft(draft.draft_id)

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/inpaint_preview",
            json={"scoreId": stored.score_id, "measureId": "m0", "revision": 1},
        )
    finally:
        client.close()

    assert response.status_code == 409
    assert "revision" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /inpaint_preview — bad measure ID (must be 404, not 400)
# ---------------------------------------------------------------------------

def test_inpaint_preview_returns_404_for_unknown_measure_id():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/inpaint_preview",
            json={
                "scoreId": stored.score_id,
                "measureId": "no-such-measure",
                "revision": stored.revision,
            },
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "no-such-measure" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /commit_draft — draft not found
# ---------------------------------------------------------------------------

def test_commit_draft_returns_404_for_unknown_draft_id():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/commit_draft",
            json={"scoreId": stored.score_id, "draftId": "draft-99"},
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "draft-99" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /commit_draft — draft belongs to a different score
# ---------------------------------------------------------------------------

def test_commit_draft_returns_404_when_draft_belongs_to_different_score():
    repository = InMemoryScoreRepository[CanonicalScore]()
    score_a = repository.create_score(_build_score())
    score_b = repository.create_score(_build_score())
    draft_b = repository.create_draft(score_b.score_id, base_revision=score_b.revision)

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/commit_draft",
            json={"scoreId": score_a.score_id, "draftId": draft_b.draft_id},
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert score_a.score_id in response.json()["detail"] or draft_b.draft_id in response.json()["detail"]


# ---------------------------------------------------------------------------
# /discard_draft — draft not found
# ---------------------------------------------------------------------------

def test_discard_draft_returns_404_for_unknown_draft_id():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/discard_draft",
            json={"scoreId": stored.score_id, "draftId": "draft-99"},
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "draft-99" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /alt_positions — score not found
# ---------------------------------------------------------------------------

def test_alt_positions_returns_404_for_unknown_score_id():
    client = CompatTestClient(create_app())
    try:
        response = client.post(
            "/alt_positions",
            json={
                "scoreId": "score-99",
                "measureId": "m0",
                "eventHitKey": {"barIndex": 0, "voiceIndex": 0, "beatIndex": 0, "noteIndex": 0},
            },
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "score-99" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /alt_positions — bad measure ID
# ---------------------------------------------------------------------------

def test_alt_positions_returns_404_for_unknown_measure_id():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/alt_positions",
            json={
                "scoreId": stored.score_id,
                "measureId": "no-such-measure",
                "eventHitKey": {"barIndex": 0, "voiceIndex": 0, "beatIndex": 0, "noteIndex": 0},
            },
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "no-such-measure" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /apply_fingering — score not found
# ---------------------------------------------------------------------------

def test_apply_fingering_returns_404_for_unknown_score_id():
    client = CompatTestClient(create_app())
    try:
        response = client.post(
            "/apply_fingering",
            json={
                "scoreId": "score-99",
                "revision": 1,
                "fingeringSelections": [{"eventId": "e0", "stringIndex": 0, "fret": 5}],
            },
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "score-99" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /apply_fingering — unknown event ID (must be 404, not 400)
# ---------------------------------------------------------------------------

def test_apply_fingering_returns_404_for_unknown_event_id():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    client = CompatTestClient(create_app(repository=repository))
    try:
        response = client.post(
            "/apply_fingering",
            json={
                "scoreId": stored.score_id,
                "revision": stored.revision,
                "fingeringSelections": [{"eventId": "no-such-event", "stringIndex": 0, "fret": 5}],
            },
        )
    finally:
        client.close()

    assert response.status_code == 404
    assert "no-such-event" in response.json()["detail"]
