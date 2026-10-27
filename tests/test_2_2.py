"""2.2 acceptance: a new prompt no longer stalls every decode. Prefill is tried only every `interval` decode steps
and under the KV watermark; once it starts it keeps going; when it schedules nothing the step is a decode.

Pure CPU: LLMEngine.step() runs against a fake model runner, the scheduler and block manager are real.
"""
from itertools import count
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BLOCK = 256
EOS = 7
TOKENS = count(1000)  # a different token per request, so prefix caching never shares blocks between them


class FakeRunner:
    last_timing = dict(cpu_prepare=0., cpu_launch=0., cpu_sample_wait=0., gpu_forward=0., gpu_sample=0.)

    def call(self, method, seqs, is_prefill):
        return [1] * len(seqs)


def make_engine(interval, watermark=1.0, num_blocks=64, max_num_batched_tokens=16384):
    Sequence.block_size = BLOCK
    config = SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=max_num_batched_tokens, max_model_len=4096,
                             eos=EOS, kvcache_block_size=BLOCK, num_kvcache_blocks=num_blocks,
                             prefill_attempt_interval=interval, prefill_watermark=watermark)
    engine = LLMEngine.__new__(LLMEngine)  # skip loading a model
    engine.scheduler = Scheduler(config)
    engine.model_runner = FakeRunner()
    engine.step_records, engine.request_records, engine.trace = [], [], None
    return engine


def add(engine, prompt_len, max_tokens=100) -> Sequence:
    seq = Sequence([next(TOKENS)] * prompt_len, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
    engine.scheduler.add(seq)
    return seq


def step(engine):
    """One engine step. Returns 'P' / 'D' and the seq_ids it ran."""
    n = len(engine.step_records)
    engine.step()
    assert len(engine.step_records) == n + 1, "an empty step: nothing was scheduled"
    record = engine.step_records[-1]
    return "P" if record.is_prefill else "D", record.seq_ids


def kinds(engine, n):
    return "".join(step(engine)[0] for _ in range(n))


# task A

def test_new_prompt_waits_for_the_attempt_point():
    """(a) Doc experiment in miniature: a prompt that arrives mid-decode waits for decode_step % interval == 0."""
    engine = make_engine(interval=4)
    add(engine, 100)
    assert kinds(engine, 2) == "PD"
    assert engine.scheduler.decode_step == 1
    late = add(engine, 100)
    assert kinds(engine, 3) == "DDD", "waiting is not empty, but this is not an attempt point"
    kind, ids = step(engine)
    assert kind == "P" and ids == [late.seq_id]
    assert engine.scheduler.decode_step == 0, "a prefill step resets decode_step"


def test_prefill_keeps_going_once_started():
    """decode_step == 0 right after a prefill is again an attempt point, so the queue drains back to back."""
    engine = make_engine(interval=4, max_num_batched_tokens=512)
    add(engine, 100)
    assert kinds(engine, 2) == "PD"
    for _ in range(3):
        add(engine, 512)  # each fills a prefill step on its own
    assert kinds(engine, 3) == "DDD"
    assert kinds(engine, 4) == "PPPD"


def test_no_prefill_above_watermark():
    """(b) KV usage is counted in blocks: 5 of 8 used > 0.5, so the new prompt waits even at attempt points."""
    engine = make_engine(interval=1, watermark=0.5, num_blocks=8)
    big = add(engine, 5 * BLOCK - 10, max_tokens=6)
    assert kinds(engine, 1) == "P"
    small = add(engine, 100)
    assert kinds(engine, 5) == "DDDDD"
    assert small in engine.scheduler.waiting
    assert big.is_finished
    kind, ids = step(engine)  # nothing is running any more
    assert kind == "P" and ids == [small.seq_id]


def test_idle_engine_does_not_wait_for_the_attempt_point():
    """Nothing is running: there is no decode to protect, prefill right away."""
    engine = make_engine(interval=4)
    first = add(engine, 100, max_tokens=3)
    assert kinds(engine, 3) == "PDD"
    assert first.is_finished and engine.scheduler.decode_step == 2
    second = add(engine, 100)
    kind, ids = step(engine)
    assert kind == "P" and ids == [second.seq_id]


def test_prefill_that_schedules_nothing_falls_back_to_decode():
    """(c) At an attempt point the waiting prompt does not fit yet: run a decode, never an empty step."""
    engine = make_engine(interval=1, num_blocks=8)
    running = add(engine, 6 * BLOCK - 10, max_tokens=4)
    assert kinds(engine, 1) == "P"
    blocked = add(engine, 3 * BLOCK)
    kind, ids = step(engine)
    assert kind == "D" and ids == [running.seq_id]
    assert blocked in engine.scheduler.waiting


# task B

def test_chunked_seq_goes_last():
    """Any seq may be chunked, but only the last one in the batch. It stays WAITING and finishes in the next prefill."""
    engine = make_engine(interval=70, max_num_batched_tokens=1024)
    small, long = add(engine, 256), add(engine, 1024)
    kind, ids = step(engine)
    assert kind == "P" and ids == [small.seq_id, long.seq_id]
    assert engine.step_records[-1].num_tokens == 1024
    assert long in engine.scheduler.waiting and long.num_cached_tokens == 768
    kind, ids = step(engine)
    assert kind == "P" and ids == [long.seq_id], "decode_step was reset, so the chunk continues right away"
    assert long in engine.scheduler.running
