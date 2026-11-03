from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    n: int = 1  # outputs per prompt: the other n - 1 are forked once the prompt is computed

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.n >= 1
