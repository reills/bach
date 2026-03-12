import json
from dataclasses import asdict

import torch

from src.models.notelm import NoteLM, NoteLMConfig, load_notelm_checkpoint


def test_load_notelm_checkpoint_restores_model_and_vocab(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    vocab = {
        "<pad>": 0,
        "BAR": 1,
        "POS_0": 2,
        "VOICE_0": 3,
    }
    vocab_path = checkpoint_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    config = NoteLMConfig(
        vocab_size=len(vocab),
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_seq_len=8,
        dropout=0.0,
        bar_token_id=vocab["BAR"],
    )
    model = NoteLM(config)
    model.eval()

    ids = torch.tensor([[vocab["BAR"], vocab["POS_0"], vocab["VOICE_0"]]])
    expected_logits = model(ids)

    checkpoint_path = checkpoint_dir / "notelm_step7.pt"
    torch.save(
        {
            "step": 7,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "vocab_path": "vocab.json",
        },
        checkpoint_path,
    )

    loaded = load_notelm_checkpoint(checkpoint_path)

    assert loaded.step == 7
    assert loaded.config == config
    assert loaded.vocab == vocab
    assert loaded.vocab_path == vocab_path.resolve()
    assert loaded.model.training is False
    assert next(loaded.model.parameters()).device.type == "cpu"

    actual_logits = loaded.model(ids)
    torch.testing.assert_close(actual_logits, expected_logits)
