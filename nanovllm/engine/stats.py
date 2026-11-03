from dataclasses import dataclass, field


@dataclass
class StepRecord:
    """One engine step, all durations in seconds. Pure Python: no torch here."""
    is_prefill: bool
    num_seqs: int
    num_tokens: int
    wall: float
    cpu_schedule: float      # scheduler.schedule()
    cpu_prepare: float       # prepare_prefill / prepare_decode / prepare_sample
    cpu_launch: float        # run_model: launching forward, not executing it
    cpu_sample_wait: float   # sampler(...).tolist(): CPU waiting for GPU
    cpu_postprocess: float   # scheduler.postprocess()
    gpu_forward: float       # CUDA events around run_model
    gpu_sample: float        # CUDA events around sampler
    seq_ids: list[int] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)
    recomputed_tokens: int = 0  # sum of num_tokens of the seqs preempted this step: KV thrown away
    num_used_blocks: int = 0    # KV blocks held after scheduling this step

    @property
    def gpu_busy(self) -> float:
        return (self.gpu_forward + self.gpu_sample) / self.wall


@dataclass
class RequestRecord:
    seq_id: int
    num_prompt_tokens: int
    num_output_tokens: int
    arrival_time: float
    token_times: list[float]
    num_preemptions: int
    finish_reason: str | None = None

    @property
    def ttft(self) -> float:
        return self.token_times[0] - self.arrival_time

    @property
    def tbt(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

