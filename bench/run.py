import argparse
import json
import os
from random import Random
from time import perf_counter

from bench.metrics import format_report, summarize_requests, summarize_steps
from bench.trace import TraceWriter
from bench.workloads import SCENARIOS


def drive(llm, requests, seed: int = 0) -> float:
    """Feed requests by arrival_step, step the engine until all are done. Returns wall seconds."""
    from nanovllm import SamplingParams
    rng = Random(seed)
    pending = sorted(requests, key=lambda r: r.arrival_step)
    prompts = [[rng.randint(0, 10000) for _ in range(r.prompt_len)] for r in pending]
    i = step = 0
    t0 = perf_counter()
    while i < len(pending) or not llm.is_finished():
        while i < len(pending) and pending[i].arrival_step <= step:
            r = pending[i]
            sp = SamplingParams(temperature=r.temperature, ignore_eos=True, max_tokens=r.max_tokens)
            llm.add_request(prompts[i], sp)
            i += 1
        if not llm.is_finished():
            llm.step()
        step += 1
    return perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="steady_decode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace", help="write per-step JSONL here")
    p.add_argument("--json", help="write the summary here")
    args = p.parse_args()

    from nanovllm import LLM, SamplingParams
    llm = LLM(os.path.expanduser(args.model), enforce_eager=False, max_model_len=4096)
    llm.generate(["warmup"], SamplingParams(max_tokens=16), use_tqdm=False)
    llm.step_records.clear()
    llm.request_records.clear()

    writer = None
    if args.trace:
        os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
        writer = llm.trace = TraceWriter(args.trace)
    duration = drive(llm, SCENARIOS[args.scenario](args.seed), args.seed)
    if writer:
        writer.close()

    steps = summarize_steps(llm.step_records)
    requests = summarize_requests(llm.request_records, duration)
    print(format_report(args.scenario, steps, requests))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"scenario": args.scenario, "steps": steps, "requests": requests}, f, indent=2)


if __name__ == "__main__":
    main()
