from __future__ import annotations

import json
from pathlib import Path

from src.instrumental_v5.ace_step import (
    ACE_STEP_DEFAULT_BRANCH,
    ACE_STEP_DEFAULT_TAG,
    ACE_STEP_DEFAULT_REPO_DIR,
    ACE_STEP_REPO_URL,
    build_ace_step_prompt,
    build_ace_step_setup_plan,
    write_ace_step_handoff,
    write_ace_step_manifest,
)


def test_setup_plan_uses_pinned_shallow_filtered_git_clone() -> None:
    plan = build_ace_step_setup_plan(recommended_tag="v0.1.8", install=True)

    assert plan.repo_url == ACE_STEP_REPO_URL
    assert plan.branch == ACE_STEP_DEFAULT_BRANCH
    assert plan.recommended_tag == ACE_STEP_DEFAULT_TAG
    assert plan.repo_dir == ACE_STEP_DEFAULT_REPO_DIR
    assert Path(plan.repo_dir).name == "ACE"
    assert plan.clone_command() == [
        "git",
        "clone",
        "--filter=blob:none",
        "--depth",
        "1",
        "--branch",
        "main",
        ACE_STEP_REPO_URL,
        ACE_STEP_DEFAULT_REPO_DIR,
    ]
    assert plan.checkout_command() == [
        "git",
        "-C",
        ACE_STEP_DEFAULT_REPO_DIR,
        "switch",
        "-C",
        "main",
        "origin/main",
    ]
    assert plan.install_command() == ["uv", "sync"]


def test_ace_step_prompt_keeps_symbolic_generator_authoritative() -> None:
    prompt = build_ace_step_prompt(
        form="invention",
        key="D minor",
        instrument="classical_guitar",
        voices=2,
        bars=16,
        subject="D4 E4 F4 A4",
    )

    assert "Bach-style counterpoint" in prompt
    assert "classical guitar" in prompt
    assert "D minor" in prompt
    assert "no vocals" in prompt
    assert "MusicXML" not in prompt


def test_write_ace_step_handoff_uses_official_lora_sidecar_names(tmp_path: Path) -> None:
    handoff = write_ace_step_handoff(
        tmp_path,
        sample_id="sample 01",
        musicxml_path=tmp_path / "sample.musicxml",
        midi_path=tmp_path / "sample.mid",
        key="D minor",
        time_signature="4/4",
        bpm=88,
        duration_seconds=12.5,
        form="invention",
        instrument="classical_guitar",
        voices=2,
        bars=8,
        subject="D4 E4 F4 A4",
    )

    caption_path = Path(handoff.caption_path)
    lyrics_path = Path(handoff.lyrics_path)
    metadata_path = Path(handoff.metadata_path)
    request_path = Path(handoff.request_path)
    assert caption_path.name == "sample_01.caption.txt"
    assert lyrics_path.name == "sample_01.lyrics.txt"
    assert metadata_path.name == "sample_01.json"
    assert caption_path.exists()
    assert lyrics_path.read_text(encoding="utf-8") == "[Instrumental]\n"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["keyscale"] == "D minor"
    assert metadata["timesignature"] == "4"
    assert metadata["expected_audio"].endswith("sample_01.wav")
    assert handoff.ready_for_lora_training is False

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["prompt"] == caption_path.read_text(encoding="utf-8").strip()
    assert request["lyrics"] == "[Instrumental]\n"
    assert request["task_type"] == "text2music"
    assert request["use_cot_caption"] is False

    manifest = write_ace_step_manifest(tmp_path, [handoff])
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["official_repo"] == ACE_STEP_REPO_URL
    assert manifest_data["recommended_tag"] == ACE_STEP_DEFAULT_TAG
    assert manifest_data["canonical_generator"] == "instrumental_v5_symbolic"
    assert manifest_data["ready_entry_count"] == 0
