"""2.1 acceptance: every way a request can end has its own state and a clean exit, and no block leaks.

Pure CPU: LLMEngine.step() runs against a fake model runner, the scheduler and block manager are real.
"""
from itertools import count
from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import FINISH_REASONS, Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK = 256
EOS = 7
TOKENS = count(100)  # a different token per request, so prefix caching never shares blocks between them


class FakeRunner:
    last_timing = dict(cpu_prepare=0., cpu_launch=0., cpu_sample_wait=0., gpu_forward=0., gpu_sample=0.)

    def __init__(self, token=1, on_run=None):
        self.token = token
        self.on_run = on_run

    def call(self, method, seqs, is_prefill):
        assert method == "run"
        if self.on_run:
            self.on_run(seqs)
        return [self.token] * len(seqs)


def make_engine(num_blocks=8, max_model_len=4096, max_num_seqs=8, token=1, on_run=None):
    Sequence.block_size = BLOCK
    config = SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=16384, max_model_len=max_model_len,
                             eos=EOS, kvcache_block_size=BLOCK, num_kvcache_blocks=num_blocks,
                             prefill_attempt_interval=1, prefill_watermark=1.0)  # 2.2's knobs off: prefill whenever waiting
    engine = LLMEngine.__new__(LLMEngine)  # skip loading a model
    engine.scheduler = Scheduler(config)
    engine.model_runner = FakeRunner(token, on_run)
    engine.step_records, engine.request_records, engine.trace = [], [], None
    return engine


def add(engine, prompt_len, max_tokens=4, ignore_eos=True) -> Sequence:
    seq = Sequence([next(TOKENS)] * prompt_len, SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))
    engine.scheduler.add(seq)
    return seq


def check_blocks(engine):
    bm = engine.scheduler.block_manager
    free = set(bm.free_block_ids)
    assert len(free) == len(bm.free_block_ids), "a block is in free_block_ids twice"
    assert not free & bm.used_block_ids
    assert free | bm.used_block_ids == set(range(len(bm.blocks)))


def run(engine, max_steps=200):
    """Step until idle. Returns {seq_id: finish_reason} and the seq_ids scheduled in each step."""
    reasons, scheduled = {}, []
    for _ in range(max_steps):
        if engine.is_finished():
            break
        n = len(engine.step_records)
        outputs, _ = engine.step()
        scheduled.append(engine.step_records[-1].seq_ids if len(engine.step_records) > n else [])
        reasons.update({seq_id: reason for seq_id, _, reason in outputs})
        check_blocks(engine)
    assert engine.is_finished(), "engine did not drain"
    bm = engine.scheduler.block_manager
    assert len(bm.free_block_ids) == len(bm.blocks), "blocks leaked"
    return reasons, scheduled


# task A

def test_status_and_finish_reason():
    seq = Sequence([1, 2, 3])
    for status in (SequenceStatus.WAITING, SequenceStatus.RUNNING):
        seq.status = status
        assert not seq.is_finished and seq.finish_reason is None
    expected = {"FINISHED_STOPPED": "stop", "FINISHED_LENGTH_CAPPED": "length", "FINISHED_ABORTED": "abort",
                "FINISHED_IGNORED": "prompt_length_exceeded", "FINISHED_PREEMPTED": "insufficient_system_resource"}
    assert {s.name: r for s, r in FINISH_REASONS.items()} == expected
    for status, reason in FINISH_REASONS.items():
        seq.status = status
        assert seq.is_finished and seq.finish_reason == reason


def test_inflight_seed():
    Sequence.block_size = BLOCK
    seq = Sequence([1] * BLOCK)
    assert seq.total_inflight_tokens == 0 and seq.num_blocks == 1
    seq.total_inflight_tokens = 1
    assert seq.num_blocks == 2


# task B

def test_never_is_ignored_and_does_not_block_the_queue():
    """Doc test 1: a prompt that fills every block can never run (decode needs one more). The ones behind it still run."""
    engine = make_engine(num_blocks=8)
    a, huge, c = add(engine, 100), add(engine, 8 * BLOCK), add(engine, 100)
    reasons, _ = run(engine)
    assert huge.status == SequenceStatus.FINISHED_IGNORED
    assert reasons[huge.seq_id] == "prompt_length_exceeded"
    assert reasons[a.seq_id] == reasons[c.seq_id] == "length"
    assert a.num_completion_tokens == c.num_completion_tokens == 4


def test_later_waits_for_space():
    """Doc test 2: does not fit now, fits once the first one is done."""
    engine = make_engine(num_blocks=8)
    first, second = add(engine, 6 * BLOCK - 10, max_tokens=4), add(engine, 4 * BLOCK - 10, max_tokens=4)
    reasons, scheduled = run(engine)
    first_done = max(i for i, ids in enumerate(scheduled) if first.seq_id in ids)
    second_start = min(i for i, ids in enumerate(scheduled) if second.seq_id in ids)
    assert second_start > first_done
    assert reasons[first.seq_id] == reasons[second.seq_id] == "length"


def test_prompt_longer_than_max_model_len_is_ignored_on_add():
    engine = make_engine(max_model_len=1024)
    seq = add(engine, 1024)
    assert seq.status == SequenceStatus.FINISHED_IGNORED
    reasons, _ = run(engine)
    assert reasons[seq.seq_id] == "prompt_length_exceeded"


def test_prompt_at_max_model_len_minus_one():
    """Doc test 6: the longest prompt that is accepted still produces its token."""
    engine = make_engine(max_model_len=1024)
    seq = add(engine, 1023, max_tokens=1)
    reasons, _ = run(engine)
    assert reasons[seq.seq_id] == "length" and seq.num_completion_tokens == 1


def test_length_capped_at_max_model_len():
    """Experiment (c): never ask for a block past max_model_len (the CUDA graph's block_tables is that wide)."""
    engine = make_engine(max_model_len=1024)
    seq = add(engine, 1000, max_tokens=100)
    widest = 0

    def on_run(seqs):
        nonlocal widest
        widest = max([widest] + [len(s.block_table) for s in seqs])
    engine.model_runner.on_run = on_run
    reasons, _ = run(engine)
    assert reasons[seq.seq_id] == "length"
    assert seq.num_tokens == 1024
    assert widest <= 1024 // BLOCK


# task C

def test_stop_on_eos():
    engine = make_engine(token=EOS)
    seq = add(engine, 100, max_tokens=10, ignore_eos=False)
    reasons, _ = run(engine)
    assert reasons[seq.seq_id] == "stop" and seq.num_completion_tokens == 1


def test_abort_running():
    """Doc test 3: gone from the next step, its blocks come back."""
    engine = make_engine(num_blocks=16)
    seqs = [add(engine, 300, max_tokens=32) for _ in range(3)]
    for _ in range(5):
        engine.step()
    victim = seqs[1]
    tokens_before = victim.num_tokens
    engine.abort_request([victim.seq_id])
    assert victim.status == SequenceStatus.FINISHED_ABORTED
    assert victim in engine.scheduler.running, "abort only marks; the queue is cleaned by schedule()"
    assert victim.block_table, "abort only marks; blocks are freed by step()"
    reasons, scheduled = run(engine)
    assert all(victim.seq_id not in ids for ids in scheduled)
    assert not victim.block_table and victim.num_tokens == tokens_before
    assert reasons[victim.seq_id] == "abort"
    assert all(reasons[s.seq_id] == "length" for s in (seqs[0], seqs[2]))


def test_abort_waiting():
    """Doc test 4: aborted before it ever ran."""
    engine = make_engine(num_blocks=4)
    running, queued = add(engine, 3 * BLOCK - 10, max_tokens=8), add(engine, 2 * BLOCK, max_tokens=8)
    engine.step()
    assert queued in engine.scheduler.waiting
    engine.abort_request([queued.seq_id])
    reasons, scheduled = run(engine)
    assert all(queued.seq_id not in ids for ids in scheduled)
    assert reasons[queued.seq_id] == "abort" and queued.num_completion_tokens == 0
    assert reasons[running.seq_id] == "length"


def test_abort_while_step_is_running():
    """abort lands after schedule() picked the seq: postprocess must not append to it."""
    engine = make_engine()
    seqs = [add(engine, 100, max_tokens=8) for _ in range(2)]
    engine.step()
    victim = seqs[0]
    tokens_before = victim.num_tokens
    engine.model_runner.on_run = lambda scheduled: engine.abort_request([victim.seq_id])
    engine.step()
    engine.model_runner.on_run = None
    assert victim.num_tokens == tokens_before
    reasons, _ = run(engine)
    assert reasons[victim.seq_id] == "abort" and reasons[seqs[1].seq_id] == "length"


def test_request_record_for_every_end():
    """1.1's RequestRecord is written for ignored and aborted requests too."""
    engine = make_engine(num_blocks=8, max_model_len=4096)
    ok, huge, gone = add(engine, 100), add(engine, 8 * BLOCK), add(engine, 100, max_tokens=32)
    engine.step()
    engine.abort_request([gone.seq_id])
    run(engine)
    by_id = {r.seq_id: r.finish_reason for r in engine.request_records}
    assert by_id == {ok.seq_id: "length", huge.seq_id: "prompt_length_exceeded", gone.seq_id: "abort"}
