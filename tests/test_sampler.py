import torch

from src.utils.decoding.sampler import Sampler


def test_sampler_temperature_zero_uses_argmax():
    sampler = Sampler(temperature=0.0, top_p=0.01)

    token = sampler.sample(torch.tensor([[0.0, 4.0, 2.0]]))

    assert token.tolist() == [[1]]
