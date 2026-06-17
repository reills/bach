import torch

from src.utils.decoding.sampler import Sampler


def test_sampler_temperature_zero_uses_argmax():
    sampler = Sampler(temperature=0.0, top_p=0.01)

    token = sampler.sample(torch.tensor([[0.0, 4.0, 2.0]]))

    assert token.tolist() == [[1]]


def test_sampler_blocks_repeated_ngram_completion():
    sampler = Sampler(temperature=0.0, top_p=1.0, no_repeat_ngram_size=3)

    token = sampler.sample(
        torch.tensor([[0.0, 10.0, 0.0, 1.0]]),
        input_ids=torch.tensor([[1, 2, 1, 2]]),
    )

    assert token.tolist() == [[3]]
