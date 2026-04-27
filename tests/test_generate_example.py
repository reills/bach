import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "generate_example", ROOT / "scripts" / "generate_example.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def test_generate_with_checkpoint_passes_grammar_mask_to_generation_config(monkeypatch):
    captured = {}

    def fake_compose_baseline(
        checkpoint_path,
        *,
        seed_tokens,
        generation_config,
        vocab_path=None,
    ):
        captured["checkpoint_path"] = checkpoint_path
        captured["seed_tokens"] = seed_tokens
        captured["generation_config"] = generation_config
        captured["vocab_path"] = vocab_path
        return SimpleNamespace(generation=SimpleNamespace(tokens=["BAR"]))

    monkeypatch.setattr(
        "src.api.compose_service.compose_baseline",
        fake_compose_baseline,
    )

    tokens = _mod._generate_with_checkpoint(
        SimpleNamespace(
            checkpoint="/tmp/checkpoint.pt",
            vocab="/tmp/vocab.json",
            key="C",
            style="baroque",
            difficulty=None,
            measures=8,
            max_length=128,
            temperature=0.7,
            top_p=0.85,
            use_grammar_mask=True,
            use_scg=True,
        )
    )

    assert tokens == ["BAR"]
    assert captured["generation_config"].use_grammar_mask is True
    assert captured["generation_config"].use_scg is True
    assert captured["generation_config"].temperature == 0.7
    assert captured["generation_config"].top_p == 0.85
