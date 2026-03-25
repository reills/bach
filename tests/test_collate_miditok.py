"""Tests for Phase 4 bar-level descriptor embeddings in collate_miditok."""

import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")

from src.dataio.collate_miditok import (
    DESC_EMBED_DIM,
    MidiTokBatch,
    MidiTokCollator,
    PackedBarDataset,
    PrefixControlConfig,
    SequenceSample,
    bar_plan_to_desc_vector,
    build_prefix_tokens,
)
from src.tokens.schema import BarPlan


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_plan(
    key: Optional[str] = None,
    time_sig=None,
    tempo: Optional[float] = None,
) -> BarPlan:
    plan = MagicMock(spec=BarPlan)
    plan.key = key
    plan.time_sig = time_sig
    plan.tempo = tempo
    return plan


def _make_vocab() -> dict:
    tokens = [
        "<pad>", "<bos>", "<eos>",
        "BAR", "POS_0", "POS_24", "POS_48",
        "VOICE_0", "VOICE_1",
        "DUR_24", "DUR_12",
        "MEL_INT12_+0", "MEL_INT12_+2", "MEL_INT12_-2",
        "HARM_OCT_0", "HARM_OCT_NA",
        "HARM_CLASS_0", "HARM_CLASS_4", "HARM_CLASS_NA",
        "REST_24",
        "KEY_C", "KEY_Am", "KEY_G",
        "MEAS_1", "MEAS_2", "MEAS_4",
        "STYLE_BAROQUE", "DIFFICULTY_EASY",
        "ABS_VOICE_0_60", "ABS_BASS_48", "ABS_SOP_72",
    ]
    return {tok: i for i, tok in enumerate(tokens)}


def _make_bar_sample(ids: List[int], piece_id: str = "piece_1", bar_index: int = 0, plan=None):
    sample = MagicMock()
    sample.ids = ids
    sample.piece_id = piece_id
    sample.bar_index = bar_index
    sample.plan = plan
    return sample


def _make_sequence_sample(
    ids: List[int],
    plans: List[Optional[BarPlan]] = None,
    piece_id: str = "piece_1",
    bar_index: int = 0,
    bar_count: int = 1,
) -> SequenceSample:
    return SequenceSample(
        ids=ids,
        piece_id=piece_id,
        bar_index=bar_index,
        plans=plans if plans is not None else [None],
        bar_count=bar_count,
    )


# ---------------------------------------------------------------------------
# DESC_EMBED_DIM constant
# ---------------------------------------------------------------------------

def test_desc_embed_dim_is_48():
    assert DESC_EMBED_DIM == 48


# ---------------------------------------------------------------------------
# bar_plan_to_desc_vector
# ---------------------------------------------------------------------------

def test_desc_vector_none_plan():
    vec = bar_plan_to_desc_vector(None)
    assert len(vec) == DESC_EMBED_DIM
    assert all(v == 0.0 for v in vec)


def test_desc_vector_length():
    plan = _make_plan(key="C", time_sig=(4, 4), tempo=120.0)
    vec = bar_plan_to_desc_vector(plan)
    assert len(vec) == DESC_EMBED_DIM


def test_desc_vector_key_c_major():
    plan = _make_plan(key="C")
    vec = bar_plan_to_desc_vector(plan)
    # C = pitch class 0, dim 0 should be 1.0
    assert vec[0] == 1.0
    # Major mode: dim 12 = 1.0, dim 13 = 0.0
    assert vec[12] == 1.0
    assert vec[13] == 0.0


def test_desc_vector_key_a_minor():
    plan = _make_plan(key="Am")
    vec = bar_plan_to_desc_vector(plan)
    # A = pitch class 9, dim 9 should be 1.0
    assert vec[9] == 1.0
    # Minor mode: dim 12 = 0.0, dim 13 = 1.0
    assert vec[12] == 0.0
    assert vec[13] == 1.0


def test_desc_vector_key_f_sharp():
    plan = _make_plan(key="F#")
    vec = bar_plan_to_desc_vector(plan)
    # F# = pitch class 6
    assert vec[6] == 1.0
    assert vec[12] == 1.0  # major


def test_desc_vector_key_bb_major():
    plan = _make_plan(key="Bb")
    vec = bar_plan_to_desc_vector(plan)
    # Bb = pitch class 10
    assert vec[10] == 1.0


def test_desc_vector_key_unknown():
    plan = _make_plan(key="X#")
    vec = bar_plan_to_desc_vector(plan)
    # Unknown key: no pitch class dim set, mode dim 12 also 0
    assert vec[12] == 0.0


def test_desc_vector_time_sig_4_4():
    plan = _make_plan(time_sig=(4, 4))
    vec = bar_plan_to_desc_vector(plan)
    # 4/4 = index 0, dim 14
    assert vec[14] == 1.0
    assert vec[15] == 0.0
    assert vec[16] == 0.0
    assert vec[17] == 0.0


def test_desc_vector_time_sig_3_4():
    plan = _make_plan(time_sig=(3, 4))
    vec = bar_plan_to_desc_vector(plan)
    assert vec[14] == 0.0
    assert vec[15] == 1.0


def test_desc_vector_time_sig_6_8():
    plan = _make_plan(time_sig=(6, 8))
    vec = bar_plan_to_desc_vector(plan)
    assert vec[16] == 1.0


def test_desc_vector_time_sig_unknown():
    plan = _make_plan(time_sig=(5, 4))
    vec = bar_plan_to_desc_vector(plan)
    # Unknown time sig: all time sig dims 0
    assert vec[14] == 0.0
    assert vec[15] == 0.0
    assert vec[16] == 0.0
    assert vec[17] == 0.0


def test_desc_vector_tempo_120():
    plan = _make_plan(tempo=120.0)
    vec = bar_plan_to_desc_vector(plan)
    # 120 bpm falls in bin for threshold 120 (index 5 in _DESC_TEMPO_BINS)
    tempo_sum = sum(vec[18:26])
    assert tempo_sum == 1.0


def test_desc_vector_tempo_none():
    plan = _make_plan(tempo=None)
    vec = bar_plan_to_desc_vector(plan)
    assert all(v == 0.0 for v in vec[18:26])


def test_desc_vector_full_plan():
    plan = _make_plan(key="G", time_sig=(4, 4), tempo=140.0)
    vec = bar_plan_to_desc_vector(plan)
    assert len(vec) == DESC_EMBED_DIM
    # G = pitch class 7
    assert vec[7] == 1.0
    # Major mode
    assert vec[12] == 1.0
    # 4/4
    assert vec[14] == 1.0
    # Tempo bin set
    assert sum(vec[18:26]) == 1.0


def test_desc_vector_real_bar_plan_schema():
    plan = BarPlan(
        bar_index=0,
        time_sig="4/4",
        key="C",
        density_bucket="LOW",
    )
    vec = bar_plan_to_desc_vector(plan)
    assert len(vec) == DESC_EMBED_DIM
    assert vec[0] == 1.0
    assert vec[12] == 1.0
    assert vec[14] == 1.0
    assert all(v == 0.0 for v in vec[18:26])


# ---------------------------------------------------------------------------
# SequenceSample uses plans (list) not plan (single)
# ---------------------------------------------------------------------------

def test_sequence_sample_has_plans_field():
    sample = _make_sequence_sample(ids=[1, 2, 3], plans=[None])
    assert hasattr(sample, "plans")
    assert isinstance(sample.plans, list)


def test_sequence_sample_plans_single():
    plan = _make_plan(key="C")
    sample = _make_sequence_sample(ids=[1, 2, 3], plans=[plan])
    assert len(sample.plans) == 1
    assert sample.plans[0] is plan


def test_sequence_sample_plans_multiple():
    plan1 = _make_plan(key="C")
    plan2 = _make_plan(key="G")
    sample = _make_sequence_sample(ids=[1, 2, 3, 4, 5, 6], plans=[plan1, plan2], bar_count=2)
    assert len(sample.plans) == 2


def test_sequence_sample_no_plan_attribute():
    """SequenceSample should not have a 'plan' attribute (replaced by 'plans')."""
    sample = _make_sequence_sample(ids=[1, 2], plans=[None])
    assert not hasattr(sample, "plan"), "SequenceSample should not have singular 'plan' field"


# ---------------------------------------------------------------------------
# MidiTokBatch has desc_embed field
# ---------------------------------------------------------------------------

def test_midi_tok_batch_has_desc_embed():
    ids = torch.tensor([[1, 2, 3]])
    attn_mask = torch.ones(1, 3, dtype=torch.bool)
    batch = MidiTokBatch(
        ids=ids,
        attn_mask=attn_mask,
        prefix_len=torch.tensor([0]),
        bar_count=torch.tensor([1]),
        piece_id=["p1"],
        bar_index=torch.tensor([0]),
        desc_embed=None,
    )
    assert hasattr(batch, "desc_embed")
    assert batch.desc_embed is None


def test_midi_tok_batch_desc_embed_tensor():
    ids = torch.tensor([[1, 2, 3]])
    attn_mask = torch.ones(1, 3, dtype=torch.bool)
    desc = torch.zeros(1, DESC_EMBED_DIM)
    batch = MidiTokBatch(
        ids=ids,
        attn_mask=attn_mask,
        prefix_len=torch.tensor([0]),
        bar_count=torch.tensor([1]),
        piece_id=["p1"],
        bar_index=torch.tensor([0]),
        desc_embed=desc,
    )
    assert batch.desc_embed is not None
    assert batch.desc_embed.shape == (1, DESC_EMBED_DIM)


# ---------------------------------------------------------------------------
# MidiTokCollator produces desc_embed in batch
# ---------------------------------------------------------------------------

def _make_collator(vocab=None, key_from_plan=True):
    if vocab is None:
        vocab = _make_vocab()
    prefix_config = PrefixControlConfig(
        key_from_plan=key_from_plan,
        measures_token_prefix="MEAS",
    )
    return MidiTokCollator(
        vocab,
        pad_token="<pad>",
        prefix_config=prefix_config,
    )


def test_collator_produces_desc_embed():
    collator = _make_collator()
    vocab = _make_vocab()
    ids = [vocab["BAR"], vocab["POS_0"], vocab["VOICE_0"], vocab["DUR_24"],
           vocab["MEL_INT12_+0"], vocab["HARM_OCT_0"], vocab["HARM_CLASS_0"]]
    sample = _make_sequence_sample(ids=ids, plans=[None], bar_count=1)
    batch = collator([sample])
    assert batch.desc_embed is not None
    assert batch.desc_embed.shape[0] == 1
    assert batch.desc_embed.shape[1] == DESC_EMBED_DIM


def test_collator_desc_embed_shape_multi_sample():
    collator = _make_collator()
    vocab = _make_vocab()
    ids = [vocab["BAR"], vocab["POS_0"]]
    samples = [
        _make_sequence_sample(ids=ids, plans=[None], bar_count=1),
        _make_sequence_sample(ids=ids, plans=[None], bar_count=1),
        _make_sequence_sample(ids=ids, plans=[None], bar_count=1),
    ]
    batch = collator(samples)
    assert batch.desc_embed.shape == (3, DESC_EMBED_DIM)


def test_collator_desc_embed_with_plan():
    collator = _make_collator()
    vocab = _make_vocab()
    plan = _make_plan(key="C", time_sig=(4, 4), tempo=120.0)
    ids = [vocab["BAR"], vocab["POS_0"]]
    sample = _make_sequence_sample(ids=ids, plans=[plan], bar_count=1)
    batch = collator([sample])
    assert batch.desc_embed is not None
    # C major pitch class 0 should be set
    assert batch.desc_embed[0, 0].item() == pytest.approx(1.0)
    # major mode dim 12
    assert batch.desc_embed[0, 12].item() == pytest.approx(1.0)


def test_collator_desc_embed_dtype():
    collator = _make_collator()
    vocab = _make_vocab()
    ids = [vocab["BAR"]]
    sample = _make_sequence_sample(ids=ids, plans=[None])
    batch = collator([sample])
    assert batch.desc_embed.dtype == torch.float


def test_collator_desc_embed_averaged_over_bars():
    """When a sequence has multiple bars with different plans, desc_embed is averaged."""
    collator = _make_collator()
    vocab = _make_vocab()
    plan_c = _make_plan(key="C")   # pitch class 0 -> dim 0
    plan_g = _make_plan(key="G")   # pitch class 7 -> dim 7
    ids = [vocab["BAR"], vocab["POS_0"], vocab["BAR"], vocab["POS_0"]]
    sample = _make_sequence_sample(ids=ids, plans=[plan_c, plan_g], bar_count=2)
    batch = collator([sample])
    # Average of C (dim0=1, dim7=0) and G (dim0=0, dim7=1) = 0.5 each
    assert batch.desc_embed[0, 0].item() == pytest.approx(0.5)
    assert batch.desc_embed[0, 7].item() == pytest.approx(0.5)


def test_collator_uses_plans_0_for_prefix():
    """Collator uses plans[0] for prefix token generation."""
    vocab = _make_vocab()
    prefix_config = PrefixControlConfig(
        key_from_plan=True,
        measures_token_prefix="MEAS",
    )
    collator = MidiTokCollator(
        vocab,
        pad_token="<pad>",
        prefix_config=prefix_config,
    )
    plan = _make_plan(key="C")
    ids = [vocab["BAR"], vocab["POS_0"]]
    sample = _make_sequence_sample(ids=ids, plans=[plan, None], bar_count=2)
    batch = collator([sample])
    # Sequence should start with KEY_C token
    id_list = batch.ids[0].tolist()
    key_c_id = vocab["KEY_C"]
    assert key_c_id in id_list


def test_collator_empty_plans_list():
    """Collator handles samples with empty plans list gracefully."""
    collator = _make_collator()
    vocab = _make_vocab()
    ids = [vocab["BAR"]]
    sample = _make_sequence_sample(ids=ids, plans=[], bar_count=1)
    batch = collator([sample])
    assert batch.desc_embed is not None
    assert all(v == pytest.approx(0.0) for v in batch.desc_embed[0].tolist())


# ---------------------------------------------------------------------------
# PackedBarDataset __getitem__ returns SequenceSample with plans list
# ---------------------------------------------------------------------------

def _make_mock_bar_dataset(bars_per_piece=3, num_pieces=2):
    """Create a mock BarDataset with predictable bars."""
    dataset = MagicMock()
    samples = []
    idx = 0
    piece_ids = []
    bar_indices_list = []
    ids_list = []
    plans_list = []

    for piece_num in range(num_pieces):
        for bar_num in range(bars_per_piece):
            plan = _make_plan(key="C" if piece_num == 0 else "G")
            sample = _make_bar_sample(
                ids=[idx * 10 + i for i in range(5)],
                piece_id=f"piece_{piece_num}",
                bar_index=bar_num,
                plan=plan,
            )
            samples.append(sample)
            piece_ids.append(f"piece_{piece_num}")
            bar_indices_list.append(bar_num)
            ids_list.append(sample.ids)
            plans_list.append(plan)
            idx += 1

    dataset.__len__ = MagicMock(return_value=len(samples))
    dataset.__getitem__ = MagicMock(side_effect=lambda i: samples[i])
    dataset._piece_id = piece_ids
    dataset._bar_index = bar_indices_list
    dataset._ids = ids_list
    return dataset, samples


def test_packed_dataset_getitem_returns_sequence_sample():
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=2, num_pieces=1)
    packed = PackedBarDataset(dataset, max_seq_len=100, bars_per_seq=2)
    assert len(packed) > 0
    item = packed[0]
    assert isinstance(item, SequenceSample)


def test_packed_dataset_getitem_has_plans():
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=2, num_pieces=1)
    packed = PackedBarDataset(dataset, max_seq_len=100, bars_per_seq=2)
    item = packed[0]
    assert hasattr(item, "plans")
    assert isinstance(item.plans, list)


def test_packed_dataset_plans_length_matches_bars():
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=2, num_pieces=1)
    packed = PackedBarDataset(dataset, max_seq_len=100, bars_per_seq=2)
    item = packed[0]
    assert len(item.plans) == item.bar_count


def test_packed_dataset_plans_single_bar():
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=3, num_pieces=1)
    packed = PackedBarDataset(dataset, max_seq_len=100, bars_per_seq=1)
    item = packed[0]
    assert len(item.plans) == 1


def test_packed_dataset_plans_are_bar_plans():
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=2, num_pieces=1)
    packed = PackedBarDataset(dataset, max_seq_len=100, bars_per_seq=2)
    item = packed[0]
    for plan in item.plans:
        # Plans can be None or BarPlan
        assert plan is None or hasattr(plan, "key")


def test_packed_dataset_truncation_truncates_plans():
    """When allow_truncate=True and ids overflow, plans list is also truncated."""
    # Each bar has 20 tokens; max_seq_len=25 should fit only 1 bar
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=3, num_pieces=1)

    # Override ids to be 20 tokens each
    for i in range(len(dataset._ids)):
        dataset._ids[i] = list(range(20))
        dataset.__getitem__.side_effect = lambda idx, ds=dataset: type(
            "S", (), {
                "ids": list(range(20)),
                "piece_id": ds._piece_id[idx],
                "bar_index": ds._bar_index[idx],
                "plan": MagicMock(spec=BarPlan),
            }
        )()

    packed = PackedBarDataset(dataset, max_seq_len=25, bars_per_seq=3, allow_truncate=True)
    if len(packed) > 0:
        item = packed[0]
        # Tokens should not exceed max_seq_len
        assert len(item.ids) <= 25


# ---------------------------------------------------------------------------
# Integration: collator + packed dataset
# ---------------------------------------------------------------------------

def test_collator_with_packed_dataset():
    dataset, _ = _make_mock_bar_dataset(bars_per_piece=2, num_pieces=2)
    packed = PackedBarDataset(dataset, max_seq_len=200, bars_per_seq=2)
    vocab = _make_vocab()
    prefix_config = PrefixControlConfig(key_from_plan=True, measures_token_prefix="MEAS")
    collator = MidiTokCollator(vocab, pad_token="<pad>", prefix_config=prefix_config)

    samples = [packed[i] for i in range(min(2, len(packed)))]
    if not samples:
        pytest.skip("No packed sequences available")

    batch = collator(samples)
    assert batch.desc_embed is not None
    assert batch.desc_embed.shape[0] == len(samples)
    assert batch.desc_embed.shape[1] == DESC_EMBED_DIM
