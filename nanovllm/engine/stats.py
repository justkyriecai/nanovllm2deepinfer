from dataclasses import dataclass


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

    @property
    def gpu_busy(self) -> float:
        return (self.gpu_forward + self.gpu_sample) / self.wall

