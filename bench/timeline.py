"""Capture a kernel-level timeline of a few prefill and a few decode steps.

    python -m bench.timeline --scenario steady_decode

Writes out/<scenario>_prefill.json and out/<scenario>_decode.json (Chrome trace format).
Open them at https://ui.perfetto.dev. The profiler adds CPU overhead per op, so launch-bound
gaps look a bit wider than they really are; the shape of the timeline is what matters.
"""
import argparse
import os
from random import Random

from bench.run import warmup
from bench.workloads import SCENARIOS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="steady_decode")
    p.add_argument("--steps", type=int, default=3, help="steps captured per phase")
    args = p.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from nanovllm import LLM, SamplingParams
    llm = LLM(os.path.expanduser(args.model), enforce_eager=False, max_model_len=4096)
    requests = SCENARIOS[args.scenario](0)
    warmup(llm, requests)

    # every request arrives at once, so the first steps are prefill and decode follows
    rng = Random(0)
    for r in requests:
        prompt = [rng.randint(0, 10000) for _ in range(r.prompt_len)]
        llm.add_request(prompt, SamplingParams(temperature=r.temperature, ignore_eos=True, max_tokens=r.max_tokens))

    def capture(phase):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(args.steps):
                with record_function(f"{phase} step {i}"):
                    llm.step()
            torch.cuda.synchronize()
        path = f"out/{args.scenario}_{phase}.json"
        prof.export_chrome_trace(path)
        print(f"wrote {path}")

    os.makedirs("out", exist_ok=True)
    capture("prefill")
    while llm.step_records[-1].is_prefill:
        llm.step()
    for _ in range(20):  # let decode settle
        llm.step()
    capture("decode")


if __name__ == "__main__":
    main()
