from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.compose_service import ComposeServiceResult, export_score
from src.api.render import canonical_score_to_midi
from src.emi.fragments import Fragment, fragment_to_jsonl
from src.inference.generate_v1 import GenerationResult


class CompatTestClient(TestClient):
    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def run_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=str(self.base_url)) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run_request())


def _score() -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=24)],
        parts=[
            Part(
                info=PartInfo(id="part-0", instrument="piano", midi_program=0),
                events=[Event(id="e0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=60)],
            )
        ],
    )


def _fragment() -> Fragment:
    return Fragment(
        id="toy_v0_s0_l8",
        piece_id="toy",
        source_path="toy.musicxml",
        voice=0,
        start_slice=0,
        length_slices=8,
        start_bar=0,
        start_pos=0,
        beats=2.0,
        phrase_role="SUBJECT_ENTRY",
        key="C",
        key_pc=0,
        mode=0,
        start_pitch=60,
        end_pitch=67,
        start_degree=1,
        end_degree=5,
        melodic_intervals=[2, 2, 1],
        rhythm_steps=[2, 2, 2],
        vertical_intervals=[12, 10],
        state_pattern=[2, 1, 2, 1],
        contour_hash="contour",
        fingerprint="fingerprint",
        speac_label="S",
        cadence_type="NONE",
        contour_bucket="ASCENDING_STEPWISE",
        rhythm_bucket="EVEN_8THS",
        local_key_pc=0,
        harmonic_function="TONIC",
        entry_degree=1,
        exit_degree=5,
        copy_hash="copyhash",
        transposition_hash="transhash",
    )


def test_emi_engine_bypasses_notelm_checkpoint(monkeypatch) -> None:
    from src.api.compose_launcher import ComposeRuntimeConfig, create_configured_app

    def fail_load_notelm_checkpoint(*args, **kwargs):
        raise AssertionError("EMI engine should not load a transformer checkpoint")

    monkeypatch.setattr("src.api.compose_launcher.load_notelm_checkpoint", fail_load_notelm_checkpoint)

    app = create_configured_app(
        ComposeRuntimeConfig(
            checkpoint_path=Path("/tmp/missing-notelm.pt"),
            vocab_path=Path("/tmp/missing-vocab.json"),
        )
    )
    client = CompatTestClient(app)
    try:
        response = client.post(
            "/compose",
            json={
                "render_mode": "piano",
                "constraints": {
                    "engine": "emi",
                    "key": "C",
                    "measures": 2,
                    "texture": 2,
                },
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["instrumentMode"] == "piano"
    assert payload["diagnostics"]["engine"] == "emi"
    assert payload["diagnostics"]["requestedEngine"] == "emi"
    assert len(payload["measureMap"]) == 2


def test_hybrid_engine_calls_transformer_with_retrieval_conditioning(monkeypatch, tmp_path: Path) -> None:
    from src.api.compose_launcher import ComposeRuntimeConfig, create_configured_app

    score = _score()
    exported = export_score(score)
    captured = {}
    fragment_path = tmp_path / "fragments.jsonl"
    fragment_path.write_text(fragment_to_jsonl(_fragment()) + "\n", encoding="utf-8")

    def fake_load_notelm_checkpoint(*args, **kwargs):
        return SimpleNamespace(vocab_path=Path("/tmp/runtime-vocab.json"))

    def fake_compose_baseline(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
        device="cpu",
        render_mode="guitar",
        generator=None,
        quality_passes=1,
    ):
        captured["checkpoint_path"] = Path(checkpoint_path)
        captured["seed_tokens"] = list(seed_tokens)
        captured["generation_config"] = generation_config
        captured["quality_passes"] = quality_passes
        return ComposeServiceResult(
            generation=GenerationResult(ids=[1], tokens=["BAR", "EOS"], stopped_on_eos=True),
            score=score,
            score_xml=exported.score_xml,
            midi=canonical_score_to_midi(score),
            measure_map=exported.measure_map,
            event_hit_map=exported.event_hit_map,
            render_mode=render_mode,
        )

    monkeypatch.setattr("src.api.compose_launcher.load_notelm_checkpoint", fake_load_notelm_checkpoint)
    monkeypatch.setattr("src.api.compose_launcher.compose_baseline", fake_compose_baseline)

    app = create_configured_app(
        ComposeRuntimeConfig(
            checkpoint_path=Path("/tmp/notelm.pt"),
            vocab_path=Path("/tmp/vocab.json"),
            engine="hybrid",
            emi_fragment_path=fragment_path,
        )
    )
    client = CompatTestClient(app)
    try:
        response = client.post(
            "/compose",
            json={
                "render_mode": "piano",
                "constraints": {
                    "key": "C",
                    "measures": 2,
                    "texture": 2,
                    "qualityPasses": 2,
                },
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    assert captured["checkpoint_path"] == Path("/tmp/notelm.pt")
    assert captured["generation_config"].conditioning["version"] == "hybrid_retrieval_conditioned_v1"
    assert "retrieved_contour_bucket" in captured["generation_config"].conditioning["field_names"]
    assert "toy_v0_s0_l8" not in repr(captured["generation_config"].conditioning)
    assert captured["quality_passes"] == 2
    assert diagnostics["engine"] == "hybrid"
    assert diagnostics["requestedEngine"] == "hybrid"
    assert diagnostics["proposalEngine"] == "transformer"
    assert diagnostics["hybrid"]["retrievedFragmentCount"] > 0
    assert "hybridFallbackReason" not in diagnostics
