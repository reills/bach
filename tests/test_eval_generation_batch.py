import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "eval_generation_batch", ROOT / "scripts" / "eval_generation_batch.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


VALID_TOKENS = [
    "KEY_C",
    "MEAS_1",
    "BAR",
    "TIME_SIG_4_4",
    "KEY_C",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0",
    "DUR_24",
    "MEL_INT12_0",
    "HARM_OCT_0",
    "HARM_CLASS_0",
]


def test_summarize_samples_reports_avg_min_max_for_quality_metrics():
    samples = [
        {
            "ok": True,
            "metrics": {
                "counterpoint_avg_active_voices": 2.0,
                "counterpoint_voice_crossings": 3,
            },
        },
        {
            "ok": True,
            "metrics": {
                "counterpoint_avg_active_voices": 4.0,
                "counterpoint_voice_crossings": 1,
            },
        },
        {
            "ok": False,
            "metrics": {
                "counterpoint_avg_active_voices": 100.0,
                "counterpoint_voice_crossings": 100,
            },
        },
    ]

    summary = _mod.summarize_samples(samples)

    assert summary["counterpoint_avg_active_voices"] == {
        "avg": 3.0,
        "min": 2.0,
        "max": 4.0,
    }
    assert summary["counterpoint_voice_crossings"] == {
        "avg": 2.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_run_batch_writes_summary_and_sample_outputs(monkeypatch, tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps({token: index for index, token in enumerate(sorted(set(VALID_TOKENS)))}),
        encoding="utf-8",
    )

    def fake_load_notelm_checkpoint(checkpoint_path, *, vocab_path=None, device="cpu"):
        return SimpleNamespace(vocab_path=vocab_path, vocab={})

    def fake_generate_from_loaded(loaded, *, seed_tokens, generation_config):
        return _mod.GenerationResult(
            ids=list(range(len(VALID_TOKENS))),
            tokens=list(VALID_TOKENS),
            stopped_on_eos=False,
        )

    monkeypatch.setattr(_mod, "load_notelm_checkpoint", fake_load_notelm_checkpoint)
    monkeypatch.setattr(_mod, "_generate_from_loaded", fake_generate_from_loaded)

    out_dir = tmp_path / "batch"
    rc = _mod.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--vocab",
            str(vocab_path),
            "--samples",
            "2",
            "--out-dir",
            str(out_dir),
            "--texture",
            "1",
            "--render-mode",
            "piano",
            "--quiet",
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 2
    assert summary["successful_count"] == 2
    assert summary["failed_count"] == 0
    assert summary["config"]["seed_tokens"] == ["KEY_C", "MEAS_8"]
    assert summary["metrics"]["harm_mismatch_count"] == {
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
    assert (out_dir / "sample_001" / "tokens.txt").exists()
    assert (out_dir / "sample_001" / "metrics.json").exists()
    assert (out_dir / "sample_001" / "example.musicxml").exists()
    assert (out_dir / "sample_001" / "example.mid").exists()
