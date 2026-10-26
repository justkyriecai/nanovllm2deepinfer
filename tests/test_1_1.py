import os

import pytest

from nanovllm.engine.stats import RequestRecord, StepRecord

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
CPU_FIELDS = ["cpu_schedule", "cpu_prepare", "cpu_launch", "cpu_sample_wait", "cpu_postprocess"]
GPU_FIELDS = ["gpu_forward", "gpu_sample"]


def make_step(**kw):
    base = dict(is_prefill=False, num_seqs=4, num_tokens=4, wall=0.010,
                cpu_schedule=0.001, cpu_prepare=0.001, cpu_launch=0.0005,
                cpu_sample_wait=0.006, cpu_postprocess=0.001,
                gpu_forward=0.006, gpu_sample=0.002)
    base.update(kw)
    return StepRecord(**base)


def test_gpu_busy():
    assert make_step().gpu_busy == pytest.approx(0.8)
    assert make_step(gpu_forward=0.0, gpu_sample=0.0).gpu_busy == 0.0


def test_request_record():
    r = RequestRecord(seq_id=0, num_prompt_tokens=10, num_output_tokens=3,
                      arrival_time=1.0, token_times=[1.5, 1.6, 1.8], num_preemptions=0)
    assert r.ttft == pytest.approx(0.5)
    assert r.tbt == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("name", ["steady_decode", "mixed_long_short", "kv_pressure", "many_small"])
def test_workloads_deterministic(name):
    from bench.workloads import SCENARIOS
    make = SCENARIOS[name]
    assert make(seed=0) == make(seed=0)
    assert make(seed=0) != make(seed=1)


def test_mixed_long_short_shape():
    from bench.workloads import mixed_long_short
    reqs = mixed_long_short(seed=0)
    longs = [r for r in reqs if r.prompt_len == 3072]
    assert len(reqs) - len(longs) == 128
    assert all(r.arrival_step == 0 for r in reqs if r.prompt_len != 3072)
    steps = sorted(r.arrival_step for r in longs)
    assert all(b - a == 20 for a, b in zip(steps, steps[1:]))


def test_percentile():
    from bench.metrics import percentile
    assert percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)
    assert percentile([4, 1, 3, 2], 0) == 1
    assert percentile([4, 1, 3, 2], 100) == 4
    assert percentile([7], 99) == 7


def test_summarize():
    from bench.metrics import format_report, summarize_requests, summarize_steps
    steps = [make_step(is_prefill=True, wall=0.020, gpu_forward=0.018, gpu_sample=0.0),
             make_step(preempted=[3]),
             make_step(wall=0.030, gpu_forward=0.010, gpu_sample=0.0)]
    s = summarize_steps(steps)
    assert s["prefill"]["steps"] == 1 and s["decode"]["steps"] == 2
    # time-weighted: total GPU time over total wall time, not the mean of ratios
    assert s["decode"]["gpu_busy"] == pytest.approx((0.008 + 0.010) / (0.010 + 0.030))
    assert s["decode"]["wall"]["max"] == pytest.approx(0.030)
    assert s["preemptions"] == 1

    reqs = [RequestRecord(0, 10, 3, 0.0, [1.0, 1.1, 1.3], 0),
            RequestRecord(1, 10, 2, 0.0, [2.0, 2.4], 2)]
    r = summarize_requests(reqs, duration=4.0)
    assert r["requests"] == 2 and r["output_tokens"] == 5
    assert r["throughput"] == pytest.approx(5 / 4.0)
    assert r["ttft"]["max"] == pytest.approx(2.0)
    assert r["tbt"]["max"] == pytest.approx(0.4)
    assert r["preempted_requests"] == 1
    assert "gpu busy" in format_report("x", s, r)


def test_trace_roundtrip(tmp_path):
    from bench.trace import TraceWriter, read_trace
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    writer(make_step(seq_ids=[1, 2], finished=[2]))
    writer(make_step(is_prefill=True, preempted=[1]))
    writer.close()
    rows = read_trace(path)
    assert len(rows) == 2
    assert rows[0]["seq_ids"] == [1, 2] and rows[0]["finished"] == [2]
    assert rows[1]["is_prefill"] and rows[1]["preempted"] == [1]


def test_greedy_match():
    from bench.correctness import greedy_match
    assert greedy_match([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert greedy_match([1, 2, 3], [1, 2, 3]) == 3
    assert greedy_match([5, 2], [1, 2]) == 0
    assert greedy_match([1, 2], [1, 2, 3]) == 2


def test_teacher_forced_nll():
    torch = pytest.importorskip("torch")
    from bench.correctness import teacher_forced_nll
    vocab = 8

    class Fake:
        """Position t always predicts token (t + 1) % vocab with prob ~1."""
        device = "cpu"

        def __call__(self, input_ids):
            n = input_ids.size(1)
            logits = torch.full((1, n, vocab), -1e4)
            logits[0, torch.arange(n), (torch.arange(n) + 1) % vocab] = 0.0
            return type("Out", (), {"logits": logits})()

    # prompt occupies positions 0..2, output tokens sit at positions 3, 4
    assert teacher_forced_nll(Fake(), [0, 1, 2], [3, 4]) == pytest.approx(0.0, abs=1e-3)
    # one wrong output token out of two: mean nll ~ 1e4 / 2
    assert teacher_forced_nll(Fake(), [0, 1, 2], [3, 7]) == pytest.approx(5e3, rel=1e-2)


def test_compare():
    from bench.correctness import compare
    golden = [{"prompt": "a", "greedy_match": 32, "nll": 0.5}, {"prompt": "b", "greedy_match": 20, "nll": 0.7}]
    ok = [{"prompt": "a", "greedy_match": 30, "nll": 0.55}, {"prompt": "b", "greedy_match": 10, "nll": 0.7}]
    assert compare(ok, golden) == []
    low_match = [{"prompt": "a", "greedy_match": 3, "nll": 0.5}, {"prompt": "b", "greedy_match": 5, "nll": 0.7}]
    assert len(compare(low_match, golden)) == 1
    high_nll = [{"prompt": "a", "greedy_match": 32, "nll": 2.0}, {"prompt": "b", "greedy_match": 20, "nll": 0.7}]
    assert len(compare(high_nll, golden)) == 1
    other_prompts = [{"prompt": "x", "greedy_match": 32, "nll": 0.5}, {"prompt": "b", "greedy_match": 20, "nll": 0.7}]
    assert compare(other_prompts, golden)


@pytest.fixture(scope="module")
def llm():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not os.path.isdir(MODEL):
        pytest.skip("needs a GPU and Qwen3-0.6B")
    from nanovllm import LLM
    return LLM(MODEL, enforce_eager=False, max_model_len=4096)


def test_step_records(llm):
    from nanovllm import SamplingParams
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
    llm.step_records.clear()
    llm.request_records.clear()
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16)
    llm.generate([[1] * 200, [2] * 300], sp, use_tqdm=False)

    records = llm.step_records
    assert any(r.is_prefill for r in records) and any(not r.is_prefill for r in records)
    for r in records:
        for f in CPU_FIELDS + GPU_FIELDS:
            assert getattr(r, f) >= 0, f
        assert r.gpu_forward > 0
        assert 0 < r.gpu_busy <= 1
        # the 5 CPU stages run back to back inside one step
        assert sum(getattr(r, f) for f in CPU_FIELDS) <= r.wall * 1.01
        # GPU time is in seconds, same unit as wall
        assert r.gpu_forward < r.wall

    requests = sorted(llm.request_records, key=lambda r: r.num_prompt_tokens)
    assert [r.num_prompt_tokens for r in requests] == [200, 300]
    for r in requests:
        assert r.num_output_tokens == 16
        assert len(r.token_times) == 16
        assert r.ttft > 0
        assert len(r.tbt) == 15 and all(t > 0 for t in r.tbt)
        assert r.num_preemptions == 0
    assert sum(len(s.finished) for s in records) == 2


def test_drive(llm):
    from bench.run import drive
    from bench.workloads import Request
    llm.step_records.clear()
    llm.request_records.clear()
    duration = drive(llm, [Request(200, 8, arrival_step=0), Request(300, 8, arrival_step=3)], seed=0)
    assert duration > 0
    assert len(llm.request_records) == 2
    # the second request arrives at step 3, so it can't be in the first 3 engine steps
    late = max(llm.request_records, key=lambda r: r.num_prompt_tokens).seq_id
    assert all(late not in s.seq_ids for s in llm.step_records[:3])
    for r in llm.request_records:
        assert r.ttft > 0 and len(r.tbt) == 7
