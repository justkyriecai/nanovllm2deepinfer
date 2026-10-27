"""Boundary requests for 2.1. One case per process, since a case may crash the engine.

    python -m bench.edge --case a          # prompt larger than the whole KV cache
    python -m bench.edge --case b          # cancel a running request by hand
    python -m bench.edge --case b --abort  # cancel it through LLMEngine.abort_request (after 2.1)
    python -m bench.edge --case c --max-tokens 1
    python -m bench.edge --case c --max-tokens 2
"""
import argparse
import os
import traceback

from nanovllm import LLM, SamplingParams

MAX_MODEL_LEN = 4096


def report(llm, tag):
    bm = llm.scheduler.block_manager
    print(f"[{tag}] waiting={[s.seq_id for s in llm.scheduler.waiting]} "
          f"running={[s.seq_id for s in llm.scheduler.running]} "
          f"free_blocks={len(bm.free_block_ids)}/{len(bm.blocks)} used={sorted(bm.used_block_ids)}")


def run(llm, max_steps=200, on_step=None):
    for step in range(max_steps):
        if llm.is_finished():
            break
        n = len(llm.step_records)
        outputs, _ = llm.step()
        rec = llm.step_records[-1] if len(llm.step_records) > n else None
        kind = ("prefill" if rec.is_prefill else "decode ") if rec else "(idle) "
        finished = [(seq_id, reason) for seq_id, _, reason in outputs]
        print(f"step {step:3d} {kind} seqs={rec.seq_ids if rec else []} finished={finished}")
        if on_step:
            on_step(step)
    report(llm, "end")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["a", "b", "c"], required=True)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--abort", action="store_true")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    args = p.parse_args()
    llm = LLM(os.path.expanduser(args.model), max_model_len=MAX_MODEL_LEN, max_num_seqs=8)
    num_blocks = len(llm.scheduler.block_manager.blocks)
    block_size = llm.scheduler.block_size
    print(f"num_kvcache_blocks={num_blocks} block_size={block_size} capacity={num_blocks * block_size} tokens")
    sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True)

    try:
        if args.case == "a":
            llm.add_request([1] * 100, sp(8))
            llm.add_request([2] * (num_blocks * block_size + 1), sp(8))
            llm.add_request([3] * 100, sp(8))
            run(llm)
        elif args.case == "b":
            for i in range(3):
                llm.add_request([10 + i] * 300, sp(32))

            def cancel(step):
                if step == 4:
                    victim = llm.scheduler.running[1]
                    if args.abort:
                        print(f"--- cancel seq {victim.seq_id}: abort_request")
                        llm.abort_request([victim.seq_id])
                    else:
                        print(f"--- cancel seq {victim.seq_id}: running.remove(seq)")
                        llm.scheduler.running.remove(victim)
                    report(llm, "after cancel")
            run(llm, on_step=cancel)
        else:
            llm.add_request([4] * MAX_MODEL_LEN, sp(args.max_tokens))  # prompt_len == max_model_len
            run(llm)
    except Exception:
        traceback.print_exc()
        report(llm, "crash")


if __name__ == "__main__":
    main()
