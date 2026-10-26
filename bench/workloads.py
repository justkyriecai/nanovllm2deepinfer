from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Request:
    prompt_len: int
    max_tokens: int
    arrival_step: int = 0
    temperature: float = 0.6


def steady_decode(seed: int = 0) -> list[Request]:
    """nano-vllm's bench.py: 256 requests at once, decode-heavy."""
    rng = Random(seed)
    return [Request(rng.randint(100, 1024), rng.randint(100, 1024)) for _ in range(256)]


def mixed_long_short(seed: int = 0) -> list[Request]:
    """128 short requests in steady decode, a 3072-token prompt every 20 steps."""
    rng = Random(seed)
    short = [Request(rng.randint(100, 300), rng.randint(600, 1000)) for _ in range(128)]
    long = [Request(3072, 64, arrival_step=100 + 20 * i) for i in range(20)]
    return short + long


def kv_pressure(seed: int = 0) -> list[Request]:
    """More KV than the GPU can hold: forces preemption."""
    rng = Random(seed)
    return [Request(rng.randint(256, 1024), rng.randint(768, 1024)) for _ in range(512)]


def many_small(seed: int = 0) -> list[Request]:
    """Tiny requests: CPU overhead per step dominates."""
    rng = Random(seed)
    return [Request(rng.randint(16, 32), rng.randint(32, 64)) for _ in range(512)]


SCENARIOS = {f.__name__: f for f in (steady_decode, mixed_long_short, kv_pressure, many_small)}
