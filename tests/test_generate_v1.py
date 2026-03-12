import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.notelm.inference import LoadedNoteLM


generate_module = importlib.import_module("src.inference.generate_v1")


class FakeAutoregressiveModel(torch.nn.Module):
    def __init__(self, logits_by_context, *, vocab_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self._logits_by_context = logits_by_context
        self._vocab_size = vocab_size
        self.calls = []

    def forward(self, ids, attn_mask=None, desc_embed=None):
        context = tuple(int(token) for token in ids[0].tolist())
        self.calls.append(context)
        logits = torch.full(
            (ids.size(0), ids.size(1), self._vocab_size),
            -1e9,
            dtype=torch.float32,
            device=ids.device,
        )
        logits[:, -1, :] = self._logits_by_context[context].to(ids.device)
        return logits


def make_loaded_model(model, vocab):
    return LoadedNoteLM(
        model=model,
        vocab=vocab,
        config=SimpleNamespace(max_seq_len=model.config.max_seq_len),
        checkpoint_path=Path("/tmp/notelm.pt"),
        vocab_path=Path("/tmp/vocab.json"),
        step=None,
    )


def test_generate_v1_stops_on_eos(tmp_path, monkeypatch):
    vocab = {"BAR": 0, "POS_0": 1, "<eos>": 2}
    model = FakeAutoregressiveModel(
        {
            (vocab["BAR"],): torch.tensor([-1e9, 0.0, -1e9]),
            (vocab["BAR"], vocab["POS_0"]): torch.tensor([-1e9, -1e9, 0.0]),
        },
        vocab_size=len(vocab),
        max_seq_len=8,
    )
    loaded = make_loaded_model(model, vocab)
    captured = {}

    def fake_loader(checkpoint_path, *, vocab_path=None, device="cpu"):
        captured["checkpoint_path"] = checkpoint_path
        captured["vocab_path"] = vocab_path
        captured["device"] = device
        return loaded

    monkeypatch.setattr(generate_module, "load_notelm_checkpoint", fake_loader)

    result = generate_module.generate_v1(
        tmp_path / "checkpoint.pt",
        seed_tokens=["BAR"],
        generation_config=generate_module.GenerationConfig(max_length=4, top_p=1.0),
    )

    assert captured == {
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "vocab_path": None,
        "device": "cpu",
    }
    assert result.ids == [vocab["BAR"], vocab["POS_0"], vocab["<eos>"]]
    assert result.tokens == ["BAR", "POS_0", "<eos>"]
    assert result.stopped_on_eos is True
    assert model.calls == [
        (vocab["BAR"],),
        (vocab["BAR"], vocab["POS_0"]),
    ]


def test_generate_v1_uses_scg_and_max_length(monkeypatch):
    vocab = {"BAR": 0, "POS_0": 1, "OTHER": 2}
    base_logits = torch.tensor([0.0, -1000.0, -1e9], dtype=torch.float32)
    model = FakeAutoregressiveModel(
        {
            (vocab["BAR"],): base_logits,
            (vocab["BAR"], vocab["POS_0"]): base_logits,
            (vocab["POS_0"], vocab["POS_0"]): base_logits,
        },
        vocab_size=len(vocab),
        max_seq_len=2,
    )
    loaded = make_loaded_model(model, vocab)

    monkeypatch.setattr(generate_module, "load_notelm_checkpoint", lambda *args, **kwargs: loaded)

    result = generate_module.generate_v1(
        "unused.pt",
        seed_tokens=["BAR"],
        generation_config=generate_module.GenerationConfig(
            max_length=4,
            top_p=1.0,
            use_scg=True,
            gamma=2000.0,
        ),
    )

    assert result.ids == [vocab["BAR"], vocab["POS_0"], vocab["POS_0"], vocab["POS_0"]]
    assert result.tokens == ["BAR", "POS_0", "POS_0", "POS_0"]
    assert result.stopped_on_eos is False
    assert model.calls == [
        (vocab["BAR"],),
        (vocab["BAR"], vocab["POS_0"]),
        (vocab["POS_0"], vocab["POS_0"]),
    ]
