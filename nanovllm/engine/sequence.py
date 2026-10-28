from copy import copy
from enum import IntEnum, auto
from itertools import count
from time import perf_counter

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(IntEnum):
    # DeepInfer csrc/engine/sequence.hpp:30. Order matters: every FINISHED_* comes after RUNNING.
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()
    FINISHED_IGNORED = auto()
    FINISHED_PREEMPTED = auto()


FINISH_REASONS = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH_CAPPED: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
    SequenceStatus.FINISHED_IGNORED: "prompt_length_exceeded",
    SequenceStatus.FINISHED_PREEMPTED: "insufficient_system_resource",
}


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.arrival_time = perf_counter()
        self.token_times: list[float] = []
        self.num_preemptions = 0
        self.total_inflight_tokens = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status >= SequenceStatus.FINISHED_STOPPED

    @property
    def finish_reason(self) -> str | None:
        return FINISH_REASONS.get(self.status)

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.total_inflight_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def reset_for_recompute(self):
        """Back to an unprefilled prompt after preemption: its KV is gone, prompt + output so far are redone (sequence.hpp:223, :258)."""
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.total_inflight_tokens = 0

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.token_times.append(perf_counter())

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
