import torch
import pytest
from pathlib import Path
import json
import pandas as pd
from src.dataio.dataset import BarDataset
from src.models.notelm.model import NoteLM, NoteLMConfig

@pytest.fixture
def mock_vocab(tmp_path):
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "BAR": 3,
        "TIME_SIG_4_4": 4,
        "KEY_C": 5,
        "POS_0": 6,
        "VOICE_0": 7,
        "DUR_24": 8,
        "MEL_INT12_0": 9,
        "HARM_OCT_0": 10,
        "HARM_CLASS_0": 11,
        "DENSITY_LOW": 12,
    }
    vocab_path = tmp_path / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    return vocab_path

@pytest.fixture
def mock_events(tmp_path):
    data = [
        {
            "piece_id": "test_piece",
            "bar_index": 0,
            "tokens": "BAR TIME_SIG_4_4 KEY_C POS_0 VOICE_0 DUR_24 MEL_INT12_0 HARM_OCT_0 HARM_CLASS_0",
            "plan_json": json.dumps({
                "bar_index": 0,
                "time_sig": "4/4",
                "key": "C",
                "density_bucket": "DENSITY_LOW",
                "pitch_range": 0,
                "polyphony_max": 1
            }),
            "bar_len_ticks": 96
        }
    ]
    df = pd.DataFrame(data)
    events_path = tmp_path / "events.parquet"
    df.to_parquet(events_path)
    return events_path

def test_dataset_and_model_forward(mock_vocab, mock_events):
    dataset = BarDataset(str(mock_events), str(mock_vocab))
    sample = dataset[0]
    
    # Prepare batch
    ids = torch.tensor([sample.ids]) # (1, seq_len)
    
    # Model config
    config = NoteLMConfig(
        vocab_size=len(dataset.vocab),
        d_model=128,
        n_heads=4,
        n_layers=2,
        desc_embed_dim=16,
        bar_token_id=dataset.vocab["BAR"]
    )
    model = NoteLM(config)
    
    # Prepare dummy desc_embed
    # We have 1 bar in the sample
    desc_embed = torch.randn(1, 1, 16)
    
    # Forward pass
    logits = model(ids, desc_embed=desc_embed)
    
    assert logits.shape == (1, len(sample.ids), config.vocab_size)
    print("\nForward pass successful!")
