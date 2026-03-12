"""Smoke tests for scripts/train_v1.py using a tiny synthetic dataset."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_events(tmp_path: Path, n_bars: int = 6) -> tuple[Path, Path]:
    """Create minimal vocab.json and events.parquet for smoke testing."""
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "BAR": 2,
        "MEAS_1": 3,
        "POS_0": 4,
        "VOICE_0": 5,
        "DUR_24": 6,
        "MEL_INT12_0": 7,
        "HARM_OCT_0": 8,
        "HARM_CLASS_0": 9,
    }
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    bar_tokens = "BAR POS_0 VOICE_0 DUR_24 MEL_INT12_0 HARM_OCT_0 HARM_CLASS_0"
    rows = [
        {
            "piece_id": f"piece_{i // 3}",
            "bar_index": i % 3,
            "tokens": bar_tokens,
            "plan_json": json.dumps(
                {
                    "bar_index": i % 3,
                    "time_sig": "4/4",
                    "key": "C",
                    "density_bucket": "LOW",
                    "pitch_range": 0,
                    "polyphony_max": 1,
                }
            ),
        }
        for i in range(n_bars)
    ]
    df = pd.DataFrame(rows)
    events_path = tmp_path / "events.parquet"
    df.to_parquet(events_path, index=False)
    return vocab_path, events_path


def _run_trainer(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_v1.py")] + args,
        capture_output=True,
        text=True,
    )


def _base_args(vocab_path: Path, events_path: Path, out_dir: Path) -> list[str]:
    """Minimal args shared across all smoke tests.

    --no-key-from-plan disables plan-key prefix tokens so the minimal vocab
    (which lacks KEY_* entries) does not cause a KeyError in the collator.
    """
    return [
        "--events", str(events_path),
        "--vocab", str(vocab_path),
        "--output-dir", str(out_dir),
        "--d-model", "32",
        "--n-heads", "2",
        "--n-layers", "1",
        "--batch-size", "2",
        "--max-seq-len", "64",
        "--bars-per-seq", "1",
        "--allow-truncate",
        "--no-key-from-plan",
        "--seed", "42",
        "--device", "cpu",
    ]


@pytest.fixture()
def synthetic_data(tmp_path):
    vocab_path, events_path = _make_events(tmp_path)
    return tmp_path, vocab_path, events_path


def test_dry_run_completes(synthetic_data, tmp_path):
    """--dry-run-batches 1 should complete without error and produce a log line."""
    base_dir, vocab_path, events_path = synthetic_data
    out_dir = tmp_path / "out"
    result = _run_trainer(
        _base_args(vocab_path, events_path, out_dir)
        + ["--dry-run-batches", "1", "--log-every", "1"]
    )
    assert result.returncode == 0, result.stderr
    assert "Dry-run complete" in result.stderr or "Dry-run complete" in result.stdout


def test_dry_run_saves_no_final_checkpoint(synthetic_data, tmp_path):
    """Dry-run exits early; the final checkpoint should NOT be written."""
    base_dir, vocab_path, events_path = synthetic_data
    out_dir = tmp_path / "out"
    _run_trainer(
        _base_args(vocab_path, events_path, out_dir)
        + ["--dry-run-batches", "1"]
    )
    checkpoints = list(out_dir.glob("notelm_step*.pt")) if out_dir.exists() else []
    assert checkpoints == [], f"Unexpected checkpoint saved during dry-run: {checkpoints}"


def test_full_run_saves_checkpoint(synthetic_data, tmp_path):
    """A short full run should produce a final checkpoint with required keys."""
    base_dir, vocab_path, events_path = synthetic_data
    out_dir = tmp_path / "out"
    result = _run_trainer(
        _base_args(vocab_path, events_path, out_dir)
        + ["--epochs", "1", "--log-every", "1", "--save-every", "0"]
    )
    assert result.returncode == 0, result.stderr
    checkpoints = list(out_dir.glob("notelm_step*.pt"))
    assert len(checkpoints) == 1, f"Expected 1 checkpoint, got {checkpoints}"
    ckpt = __import__("torch").load(checkpoints[0], map_location="cpu")
    for key in ("step", "model_state", "optimizer_state", "config", "vocab_path", "timestamp", "args"):
        assert key in ckpt, f"Missing key '{key}' in checkpoint"


def test_resume_continues_from_step(synthetic_data, tmp_path):
    """Resuming from a checkpoint should restore the step counter."""
    import torch
    from dataclasses import asdict
    from src.models.notelm.model import NoteLM, NoteLMConfig

    base_dir, vocab_path, events_path = synthetic_data
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a tiny checkpoint at step=10 manually
    vocab = json.loads(vocab_path.read_text())
    config = NoteLMConfig(
        vocab_size=len(vocab),
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
        dropout=0.0,
        bar_token_id=vocab["BAR"],
    )
    model = NoteLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Copy vocab to out_dir (train script writes it there after extend)
    import shutil
    shutil.copy(vocab_path, out_dir / "vocab.json")

    ckpt_path = out_dir / "notelm_step10.pt"
    from datetime import datetime, timezone
    torch.save(
        {
            "step": 10,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "vocab_path": str(out_dir / "vocab.json"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": {},
        },
        ckpt_path,
    )

    result = _run_trainer(
        _base_args(vocab_path, events_path, out_dir)
        + ["--epochs", "1", "--log-every", "1", "--save-every", "0",
           "--resume", str(ckpt_path)]
    )
    assert result.returncode == 0, result.stderr
    # Final checkpoint step should be > 10
    checkpoints = sorted(out_dir.glob("notelm_step*.pt"))
    steps = [int(p.stem.replace("notelm_step", "")) for p in checkpoints]
    final_step = max(s for s in steps if s != 10)
    assert final_step > 10, f"Expected step > 10, got {final_step}"


def test_val_split_runs(synthetic_data, tmp_path):
    """--val-split should split the dataset and log val_loss."""
    base_dir, vocab_path, events_path = synthetic_data
    out_dir = tmp_path / "out"
    result = _run_trainer(
        _base_args(vocab_path, events_path, out_dir)
        + ["--epochs", "1", "--val-split", "0.34", "--val-every", "1",
           "--log-every", "1", "--save-every", "0"]
    )
    assert result.returncode == 0, result.stderr
    combined = result.stderr + result.stdout
    assert "val_loss" in combined, "Expected val_loss in training output"


def test_checkpoint_metadata_fields(synthetic_data, tmp_path):
    """Checkpoint saved by the trainer should contain timestamp and args fields."""
    import torch
    base_dir, vocab_path, events_path = synthetic_data
    out_dir = tmp_path / "out"
    _run_trainer(
        _base_args(vocab_path, events_path, out_dir)
        + ["--epochs", "1", "--save-every", "0"]
    )
    ckpt_files = list(out_dir.glob("notelm_step*.pt"))
    assert ckpt_files, "No checkpoint written"
    ckpt = torch.load(ckpt_files[0], map_location="cpu")
    assert "timestamp" in ckpt, "checkpoint missing 'timestamp'"
    assert "args" in ckpt, "checkpoint missing 'args'"
    # timestamp should be a valid ISO string
    from datetime import datetime
    datetime.fromisoformat(ckpt["timestamp"])
