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
