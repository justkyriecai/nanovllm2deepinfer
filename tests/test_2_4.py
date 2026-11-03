"""2.4 acceptance: SamplingParams(n) computes the prompt once. The other n - 1 outputs are forked children that share
the parent's blocks (ref_count), wait in prompted, and copy-on-write their last block when they enter running.

Pure CPU: LLMEngine.step() runs against a fake model runner, the scheduler and block manager are real.
"""
from collections import Counter
from itertools import count
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK = 256
EOS = 7
TOKENS = count(5000)  # a different token per request, so prefix caching never shares blocks between them


class FakeRunner:
    last_timing = dict(cpu_prepare=0., cpu_launch=0., cpu_sample_wait=0., gpu_forward=0., gpu_sample=0.)

    def __init__(self):
        self.clones = []      # every clone_blocks call, in order
        self.calls = []       # method names, in order

    def call(self, method, *args):
        self.calls.append(method)
        if method == "clone_blocks":
            self.clones.append(list(args[0]))
            return None
        seqs, is_prefill = args
        return [1] * len(seqs)


def make_engine(num_blocks=64):
    Sequence.block_size = BLOCK
    config = SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=16384, max_model_len=4096,
                             eos=EOS, kvcache_block_size=BLOCK, num_kvcache_blocks=num_blocks,
                             prefill_attempt_interval=1, prefill_watermark=1.0,  # 2.2's knobs off
                             stop_preempted_request=False)
    engine = LLMEngine.__new__(LLMEngine)  # skip loading a model
    engine.scheduler = Scheduler(config)
    engine.model_runner = FakeRunner()
    engine.step_records, engine.request_records, engine.trace = [], [], None
    return engine


def add(engine, prompt_len, n=4, max_tokens=10) -> Sequence:
    seq = Sequence([next(TOKENS)] * prompt_len, SamplingParams(max_tokens=max_tokens, ignore_eos=True, n=n))
    engine.scheduler.add(seq)
    return seq


def ref_counts(engine, block_ids):
    return [engine.scheduler.block_manager.blocks[b].ref_count for b in block_ids]


def run_to_end(engine, limit=1000):
    for _ in range(limit):
        if engine.is_finished():
            return
        engine.step()
    raise AssertionError(f"not finished in {limit} steps")


# task A

def test_fork_shares_the_prompt_blocks():
    """(a) After the prefill step, n - 1 children wait in prompted with a copy of the parent's block_table; every
    shared block has n users. Children hold the prompt only and have one token left to compute."""
    engine = make_engine()
    parent = add(engine, 600)  # 3 blocks, the last one 88 tokens
    engine.step()
    sched = engine.scheduler
    assert engine.step_records[-1].is_prefill and engine.step_records[-1].num_tokens == 600
    children = list(sched.prompted)
    assert len(children) == 3, f"want 3 children in prompted, got {len(children)}"
    for child in children:
        assert child.block_table == parent.block_table and child.block_table is not parent.block_table
        assert child.token_ids == parent.prompt_token_ids and child.num_tokens == 600
        assert child.num_cached_tokens == 599, "the child computes its own last prompt token, to sample its own first output"
        assert child.n == 1 and child.status == SequenceStatus.WAITING
        assert child.arrival_time == parent.arrival_time
    assert ref_counts(engine, parent.block_table) == [4, 4, 4]


def test_no_fork_without_n():
    engine = make_engine()
    add(engine, 600, n=1)
    engine.step()
    assert not engine.scheduler.prompted


# task B

def test_cow_copies_only_the_last_block():
    """(b) Entering running, each child gets a fresh last block copied from the parent's; full blocks stay shared."""
    engine = make_engine()
    parent = add(engine, 600)
    engine.step()                      # prefill + fork
    children = list(engine.scheduler.prompted)
    shared, last = parent.block_table[:2], parent.block_table[2]
    engine.step()                      # decode: admit the children, then run all 4
    record = engine.step_records[-1]
    assert not record.is_prefill and sorted(record.seq_ids) == sorted([parent.seq_id] + [c.seq_id for c in children])
    assert not engine.scheduler.prompted
    new_blocks = [c.block_table[2] for c in children]
    for child in children:
        assert child.block_table[:2] == shared
    assert len(set(new_blocks)) == 3 and last not in new_blocks
    assert engine.model_runner.clones == [[(last, b) for b in new_blocks]], "one clone_blocks call, (src, dst) per child"
    assert engine.model_runner.calls[-2:] == ["clone_blocks", "run"], "copy before the forward that writes the copies"
    assert ref_counts(engine, shared) == [4, 4] and ref_counts(engine, [last]) == [1]


def test_cow_even_when_the_prompt_fills_its_blocks():
    """A 512-token prompt fills 2 blocks, but the child recomputes token 511, which lives in block 2: still a copy."""
    engine = make_engine()
    parent = add(engine, 512)
    engine.step()
    engine.step()
    assert [src for src, dst in engine.model_runner.clones[0]] == [parent.block_table[1]] * 3


def test_children_wait_in_prompted_until_a_block_is_free():
    """Entering running needs a block for the copy; without one the child stays in prompted. Nobody is lost."""
    engine = make_engine(num_blocks=4)
    parent = add(engine, 600)          # 3 of 4 blocks
    engine.step()
    engine.step()                      # one free block: one child gets in
    assert len(engine.scheduler.prompted) == 2
    run_to_end(engine)
    assert len(engine.request_records) == 4
    assert len(engine.scheduler.block_manager.free_block_ids) == 4


def test_either_side_can_finish_first():
    """(c)(d) Parent aborted after the fork, then a child: the others still finish, and every block comes back."""
    engine = make_engine()
    parent = add(engine, 600, max_tokens=20)
    engine.step()
    engine.step()
    children = [s for s in engine.scheduler.running if s is not parent]
    engine.abort_request([parent.seq_id])
    engine.step()
    engine.abort_request([children[0].seq_id])
    run_to_end(engine)
    reasons = Counter(r.finish_reason for r in engine.request_records)
    assert reasons == {"abort": 2, "length": 2}
    assert all(r.num_output_tokens == 20 for r in engine.request_records if r.finish_reason == "length")
    assert len(engine.scheduler.block_manager.free_block_ids) == 64


def test_n4_prefills_once():
    """The doc experiment in miniature: 4 prompts with n=4 -> 4 prompts' worth of prefill, 16 outputs."""
    engine = make_engine()
    for _ in range(4):
        add(engine, 600)
    run_to_end(engine)
    assert sum(r.num_tokens for r in engine.step_records if r.is_prefill) == 4 * 600
    assert len(engine.request_records) == 16
    assert len(engine.scheduler.block_manager.free_block_ids) == 64
