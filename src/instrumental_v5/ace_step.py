from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

ACE_STEP_REPO_URL = "https://github.com/ace-step/ACE-Step-1.5.git"
ACE_STEP_DEFAULT_BRANCH = "main"
ACE_STEP_RECOMMENDED_TAG = "v0.1.8"
ACE_STEP_DEFAULT_TAG = ACE_STEP_RECOMMENDED_TAG
ACE_STEP_LICENSE = "MIT"
ACE_STEP_DEFAULT_MODEL = "acestep-v15-turbo"
ACE_STEP_DEFAULT_REPO_DIR = str(Path(__file__).resolve().parents[3] / "ACE")

_INSTRUMENT_TAGS = {
    "piano": "piano, clean contrapuntal articulation",
    "harpsichord": "harpsichord, crisp Baroque articulation",
    "classical_guitar": "classical guitar, nylon strings, clean fingerstyle articulation",
    "nylon_guitar": "classical guitar, nylon strings, clean fingerstyle articulation",
    "lute": "lute, intimate plucked Baroque articulation",
}


@dataclass(frozen=True)
class AceStepSetupPlan:
    repo_url: str = ACE_STEP_REPO_URL
    branch: str = ACE_STEP_DEFAULT_BRANCH
    repo_dir: str = ACE_STEP_DEFAULT_REPO_DIR
    install: bool = False
    recommended_tag: str = ACE_STEP_RECOMMENDED_TAG

    def clone_command(self) -> list[str]:
        return [
            "git",
            "clone",
            "--filter=blob:none",
            "--depth",
            "1",
            "--branch",
            self.branch,
            self.repo_url,
            self.repo_dir,
        ]

    def fetch_command(self) -> list[str]:
        return [
            "git",
            "-C",
            self.repo_dir,
            "fetch",
            "--depth",
            "1",
            "origin",
            f"{self.branch}:refs/remotes/origin/{self.branch}",
        ]

    def checkout_command(self) -> list[str]:
        return ["git", "-C", self.repo_dir, "switch", "-C", self.branch, f"origin/{self.branch}"]

    def upstream_command(self) -> list[str]:
        return ["git", "-C", self.repo_dir, "branch", f"--set-upstream-to=origin/{self.branch}", self.branch]

    def install_command(self) -> list[str]:
        return ["uv", "sync"]

    def to_dict(self) -> dict[str, object]:
        commands = [self.clone_command()]
        if self.install:
            commands.append(self.install_command())
        return {
            "repo_url": self.repo_url,
            "branch": self.branch,
            "recommended_tag": self.recommended_tag,
            "repo_dir": self.repo_dir,
            "license": ACE_STEP_LICENSE,
            "install": self.install,
            "commands": commands,
        }


@dataclass(frozen=True)
class AceStepRenderRequest:
    sample_id: str
    prompt: str
    lyrics: str
    bpm: int
    key_scale: str
    time_signature: str
    duration_seconds: float
    model: str = ACE_STEP_DEFAULT_MODEL
    thinking: bool = False
    audio_format: str = "wav"
    task_type: str = "text2music"

    def api_payload(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "lyrics": self.lyrics,
            "thinking": self.thinking,
            "model": self.model,
            "task_type": self.task_type,
            "audio_format": self.audio_format,
            "bpm": self.bpm,
            "key_scale": self.key_scale,
            "time_signature": self.time_signature,
            "audio_duration": round(self.duration_seconds, 3),
            "use_cot_caption": False,
            "use_cot_language": False,
            "vocal_language": "instrumental",
        }


@dataclass(frozen=True)
class AceStepHandoff:
    sample_id: str
    dataset_dir: str
    caption_path: str
    lyrics_path: str
    metadata_path: str
    request_path: str
    expected_audio_path: str
    musicxml_path: str
    midi_path: str
    ready_for_lora_training: bool
    request: AceStepRenderRequest

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["request"] = self.request.api_payload()
        return data


def build_ace_step_setup_plan(
    *,
    repo_dir: str | Path = ACE_STEP_DEFAULT_REPO_DIR,
    branch: str = ACE_STEP_DEFAULT_BRANCH,
    recommended_tag: str = ACE_STEP_RECOMMENDED_TAG,
    repo_url: str = ACE_STEP_REPO_URL,
    install: bool = False,
) -> AceStepSetupPlan:
    return AceStepSetupPlan(
        repo_url=repo_url,
        branch=branch,
        repo_dir=str(repo_dir),
        install=install,
        recommended_tag=recommended_tag,
    )


def setup_ace_step_repo(plan: AceStepSetupPlan, *, dry_run: bool = False) -> dict[str, object]:
    repo_dir = Path(plan.repo_dir)
    if dry_run:
        return {"dry_run": True, **plan.to_dict()}

    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            raise ValueError(f"{repo_dir} exists but is not a git repository")
        remote = _run(["git", "-C", str(repo_dir), "remote", "get-url", "origin"], capture=True).strip()
        if _normalize_repo_url(remote) != _normalize_repo_url(plan.repo_url):
            raise ValueError(f"ACE-Step repo origin mismatch: {remote!r}")
        _run(plan.fetch_command())
        _run(plan.checkout_command())
        _run(plan.upstream_command())
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(plan.clone_command())

    commit = _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True).strip()
    result: dict[str, object] = {**plan.to_dict(), "dry_run": False, "commit": commit}
    if plan.install:
        _run(plan.install_command(), cwd=repo_dir)
        result["installed"] = True
    return result


def build_ace_step_prompt(
    *,
    form: str | None,
    key: str | None,
    instrument: str,
    voices: int,
    bars: int,
    subject: str | None = None,
    extra_tags: Sequence[str] | None = None,
) -> str:
    tags = [
        "baroque instrumental",
        "Bach-style counterpoint",
        f"{max(1, voices)} independent voices",
        _INSTRUMENT_TAGS.get(instrument, instrument.replace("_", " ")),
        "notation-derived performance",
        "clear subject entries",
        "clean articulation",
        "no vocals",
    ]
    if form:
        tags.append(f"{form.replace('_', ' ')} form")
    if key:
        tags.append(_ace_key_scale(key))
    if bars > 0:
        tags.append(f"{bars} bars")
    if subject:
        tags.append("recognizable recurring subject")
    tags.extend(tag.strip() for tag in (extra_tags or []) if tag.strip())
    return ", ".join(dict.fromkeys(tags))


def write_ace_step_handoff(
    out_dir: str | Path,
    *,
    sample_id: str,
    musicxml_path: str | Path,
    midi_path: str | Path,
    key: str | None,
    time_signature: str,
    bpm: int,
    duration_seconds: float,
    form: str | None,
    instrument: str,
    voices: int,
    bars: int,
    subject: str | None = None,
    model: str = ACE_STEP_DEFAULT_MODEL,
    thinking: bool = False,
    audio_format: str = "wav",
) -> AceStepHandoff:
    root = Path(out_dir)
    dataset_dir = root / "ace_step_handoff"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(sample_id)
    caption_path = dataset_dir / f"{stem}.caption.txt"
    lyrics_path = dataset_dir / f"{stem}.lyrics.txt"
    metadata_path = dataset_dir / f"{stem}.json"
    request_path = root / f"{stem}.ace_step_request.json"
    expected_audio_path = dataset_dir / f"{stem}.{audio_format}"

    prompt = build_ace_step_prompt(
        form=form,
        key=key,
        instrument=instrument,
        voices=voices,
        bars=bars,
        subject=subject,
    )
    lyrics = "[Instrumental]\n"
    request = AceStepRenderRequest(
        sample_id=sample_id,
        prompt=prompt,
        lyrics=lyrics,
        bpm=int(bpm),
        key_scale=_ace_key_scale(key),
        time_signature=_ace_time_signature(time_signature),
        duration_seconds=max(10.0, float(duration_seconds)),
        model=model,
        thinking=thinking,
        audio_format=audio_format,
    )

    caption_path.write_text(prompt + "\n", encoding="utf-8")
    lyrics_path.write_text(lyrics, encoding="utf-8")
    metadata = {
        "caption": prompt,
        "bpm": int(bpm),
        "keyscale": request.key_scale,
        "timesignature": request.time_signature,
        "language": "instrumental",
        "source_musicxml": str(musicxml_path),
        "source_midi": str(midi_path),
        "expected_audio": str(expected_audio_path),
        "note": "Render the MIDI/MusicXML to this audio file before ACE-Step LoRA preprocessing.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    request_path.write_text(json.dumps(request.api_payload(), indent=2, sort_keys=True), encoding="utf-8")

    return AceStepHandoff(
        sample_id=sample_id,
        dataset_dir=str(dataset_dir),
        caption_path=str(caption_path),
        lyrics_path=str(lyrics_path),
        metadata_path=str(metadata_path),
        request_path=str(request_path),
        expected_audio_path=str(expected_audio_path),
        musicxml_path=str(musicxml_path),
        midi_path=str(midi_path),
        ready_for_lora_training=expected_audio_path.exists(),
        request=request,
    )


def write_ace_step_manifest(
    out_dir: str | Path,
    handoffs: Sequence[AceStepHandoff | dict[str, Any]],
    *,
    setup_plan: AceStepSetupPlan | None = None,
) -> Path:
    path = Path(out_dir) / "ace_step_manifest.json"
    plan = setup_plan or build_ace_step_setup_plan()
    entries = [handoff.to_dict() if isinstance(handoff, AceStepHandoff) else dict(handoff) for handoff in handoffs]
    payload = {
        "role": "downstream_audio_renderer_lora_dataset_handoff",
        "canonical_generator": "instrumental_v5_symbolic",
        "official_repo": ACE_STEP_REPO_URL,
        "recommended_tag": plan.recommended_tag,
        "license": ACE_STEP_LICENSE,
        "setup": plan.to_dict(),
        "entries": entries,
        "ready_entry_count": sum(1 for entry in entries if bool(entry.get("ready_for_lora_training"))),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _ace_key_scale(key: str | None) -> str:
    if not key:
        return "C Major"
    normalized = key.strip().replace("_", " ")
    compact = normalized.replace(" ", "")
    try:
        compact = normalize_key_for_ace(compact)
    except ValueError:
        return normalized
    if compact.endswith("m"):
        return f"{compact[:-1]} minor"
    return f"{compact} Major"


def normalize_key_for_ace(value: str) -> str:
    from src.inference.controls import normalize_compose_key

    return normalize_compose_key(value)


def _ace_time_signature(value: str) -> str:
    try:
        numerator, denominator = value.split("/", 1)
        num = int(numerator)
        den = int(denominator)
    except Exception:
        return "4"
    if (num, den) == (6, 8):
        return "6"
    return str(num)


def _safe_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "sample"


def _normalize_repo_url(value: str) -> str:
    lowered = value.strip().lower().removesuffix(".git")
    if lowered.startswith("git@github.com:"):
        lowered = "https://github.com/" + lowered[len("git@github.com:") :]
    return lowered


def _run(cmd: Sequence[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return completed.stdout or ""
