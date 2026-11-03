"""2.3 acceptance: when KV runs out in decode, the victim is the request with the longest output; a victim is
either reset and put back at the head of waiting, or (stop_preempted_request) ended as FINISHED_PREEMPTED.

Pure CPU: LLMEngine.step() runs against a fake model runner, the scheduler and block manager are real.

The scene, 3 KV blocks of 256 tokens: A (prompt 200) runs alone for a while, then B (prompt 230), then C (prompt 240).
Each holds one block. C crosses 256 tokens first and needs a fourth block that does not exist. At that step
A has the longest output (38), C the shortest (17); C is also the newest, at the tail of running.
"""
from itertools import count
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK = 256
EOS = 7
TOKENS = count(3000)  # a different token per request, so prefix caching never shares blocks between them


class FakeRunner:
    last_timing = dict(cpu_prepare=0., cpu_launch=0., cpu_sample_wait=0., gpu_forward=0., gpu_sample=0.)

    def call(self, method, seqs, is_prefill):
        return [1] * len(seqs)


def make_engine(stop_preempted_request=False, num_blocks=3):
    Sequence.block_size = BLOCK
    config = SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=16384, max_model_len=4096,
                             eos=EOS, kvcache_block_size=BLOCK, num_kvcache_blocks=num_blocks,
                             prefill_attempt_interval=1, prefill_watermark=1.0,  # 2.2's knobs off
                             stop_preempted_request=stop_preempted_request)
    engine = LLMEngine.__new__(LLMEngine)  # skip loading a model
    engine.scheduler = Scheduler(config)
    engine.model_runner = FakeRunner()
    engine.step_records, engine.request_records, engine.trace = [], [], None
    return engine


def add(engine, prompt_len, max_tokens=100) -> Sequence:
    seq = Sequence([next(TOKENS)] * prompt_len, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
    engine.scheduler.add(seq)
    return seq


def steps(engine, n):
    for _ in range(n):
        engine.step()


def until_preemption(engine, limit=50):
    """Step until some step preempts. Returns that step's record."""
    for _ in range(limit):
        engine.step()
        record = engine.step_records[-1]
        if record.preempted:
            return record
    raise AssertionError(f"no preemption in {limit} steps")


def scene(engine):
    """A, B, C as in the module docstring, stepped right up to the preemption."""
    a = add(engine, 200)
    steps(engine, 1 + 20)         # A: prefill, 20 decodes
    b = add(engine, 230)
    steps(engine, 1 + 1)          # B: prefill; A and B: 1 decode
    c = add(engine, 240)
    steps(engine, 1)              # C: prefill. 3 of 3 blocks used
    return a, b, c


# task A

def test_victim_is_the_longest_output():
    """(a) Not the tail of running (C, the newest), but the one with the most output tokens (A)."""
    engine = make_engine()
    a, b, c = scene(engine)
    assert a.num_completion_tokens > b.num_completion_tokens > c.num_completion_tokens
    record = until_preemption(engine)
    assert record.preempted == [a.seq_id], f"preempted {record.preempted}, want A={a.seq_id} (B={b.seq_id}, C={c.seq_id})"
    assert sorted(record.seq_ids) == sorted([b.seq_id, c.seq_id]), "the block A freed should go to C in the same step"


# task B

def test_reset_for_recompute():
    """(b) The rolling counters go back to an unprefilled prompt; the tokens stay."""
    Sequence.block_size = BLOCK
    seq = Sequence([1] * 300, SamplingParams(max_tokens=100, ignore_eos=True))
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill, seq.total_inflight_tokens = 300, 1, False, 2
    seq.append_token(5)
    seq.reset_for_recompute()
    assert (seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill, seq.total_inflight_tokens) == (0, 0, True, 0)
    assert seq.num_tokens == 301 and seq.token_ids[-1] == 5, "output so far is kept and recomputed as part of the prompt"


def test_recompute_goes_back_to_the_head_of_waiting():
    """(b) The victim holds no blocks, is WAITING at waiting[0], and eventually finishes with its full output."""
    engine = make_engine()
    a, b, c = scene(engine)
    d = add(engine, 100)  # a newcomer that must not jump ahead of A
    until_preemption(engine)
    sched = engine.scheduler
    assert a.status == SequenceStatus.WAITING and sched.waiting[0] is a
    assert a.block_table == [] and a.num_cached_tokens == 0 and a.is_prefill
    assert a.num_preemptions == 1
    for _ in range(3000):
        if engine.is_finished():
            break
        engine.step()
    assert engine.is_finished()
    records = {r.seq_id: r for r in engine.request_records}
    for seq in (a, b, c, d):
        assert records[seq.seq_id].finish_reason == "length" and records[seq.seq_id].num_output_tokens == 100
    assert len(sched.block_manager.free_block_ids) == 3, "every block back in the free list"


def test_stop_preempted_request():
    """(c) The victim ends as FINISHED_PREEMPTED in the same step, and its blocks are already C's."""
    engine = make_engine(stop_preempted_request=True)
    a, b, c = scene(engine)
    record = until_preemption(engine)
    sched = engine.scheduler
    assert a.status == SequenceStatus.FINISHED_PREEMPTED
    assert a not in sched.waiting and a not in sched.running
    assert c.seq_id in record.seq_ids, "A's block must be freed at preemption, not after the step"
    records = {r.seq_id: r for r in engine.request_records}
    assert records[a.seq_id].finish_reason == "insufficient_system_resource"
