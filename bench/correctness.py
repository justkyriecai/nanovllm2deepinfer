"""Is the engine still computing the right thing? HF transformers is the judge.

    python -m bench.correctness --write tests/golden/qwen3_0_6b.json    # once, on known-good code
    python -m bench.correctness --golden tests/golden/qwen3_0_6b.json   # after every model change

Per prompt, two numbers:
  greedy_match  how many leading tokens nano's near-greedy output shares with HF's greedy output
  nll           mean negative log-likelihood of nano's output, teacher-forced under HF
"""
import argparse
import json
import os
import sys

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, Neil Armstrong",
    "The three primary colors are",
    "Water boils at a temperature of",
    "To make a cup of tea, first",
    "The largest planet in our solar system is",
    "Once upon a time, in a small village,",
]
MAX_TOKENS = 32
TEMPERATURE = 1e-3  # nano forbids 0; this is greedy up to bf16 ties


def greedy_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def teacher_forced_nll(model, prompt_ids: list[int], output_ids: list[int]) -> float:
    """Feed prompt + output through `model` in one forward pass, return the mean
    -log p(output token | everything before it) over the output tokens only.
    `model(input_ids).logits` has shape [1, seq_len, vocab]; `model.device` says where inputs go."""
    import torch
    ids = torch.tensor([prompt_ids + output_ids], device=model.device)
    logits = model(ids).logits[0].float()
    start = len(prompt_ids)
    # logits at position t predict token t+1, so output token i is scored at start - 1 + i
    pred = logits[start - 1 : start - 1 + len(output_ids)]
    logp = torch.log_softmax(pred, dim=-1)
    target = torch.tensor(output_ids, device=logp.device)
    return -logp[torch.arange(len(output_ids)), target].mean().item()


def compare(results: list[dict], golden: list[dict]) -> list[str]:
    """Return a list of failure messages, empty if results are within tolerance of golden.

    Rules, on the mean over all prompts (single bf16 token flips are expected, so don't judge per prompt):
      - mean greedy_match must be at least half of golden's
      - mean nll must not exceed golden's by more than 0.3
    Also fail if the prompts differ from golden's (the golden file is for another prompt set)."""
    if [r["prompt"] for r in results] != [g["prompt"] for g in golden]:
        return ["prompts differ from the golden file"]
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    failures = []
    match, golden_match = mean(results, "greedy_match"), mean(golden, "greedy_match")
    if match < golden_match / 2:
        failures.append(f"mean greedy_match {match:.1f} < half of golden {golden_match:.1f}")
    nll, golden_nll = mean(results, "nll"), mean(golden, "nll")
    if nll > golden_nll + 0.3:
        failures.append(f"mean nll {nll:.3f} > golden {golden_nll:.3f} + 0.3")
    return failures


def nano_outputs(model_path: str, prompt_ids: list[list[int]]) -> list[list[int]]:
    from nanovllm import LLM, SamplingParams
    # leave room for the HF judge on the same GPU
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=TEMPERATURE, ignore_eos=True, max_tokens=MAX_TOKENS)
    return [o["token_ids"] for o in llm.generate(prompt_ids, sp, use_tqdm=False)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", help="write results as the new golden file")
    g.add_argument("--golden", help="compare against this golden file, exit 1 on failure")
    args = p.parse_args()
    model_path = os.path.expanduser(args.model)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = [tok.encode(text) for text in PROMPTS]
    nano = nano_outputs(model_path, prompt_ids)

    hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    results = []
    with torch.no_grad():
        for text, ids, out in zip(PROMPTS, prompt_ids, nano):
            ref = hf.generate(torch.tensor([ids], device=hf.device), do_sample=False,
                              max_new_tokens=MAX_TOKENS, min_new_tokens=MAX_TOKENS)[0, len(ids):].tolist()
            results.append({"prompt": text, "greedy_match": greedy_match(out, ref),
                            "nll": teacher_forced_nll(hf, ids, out), "output": tok.decode(out)})

    for r in results:
        print(f"match={r['greedy_match']:>2}/{MAX_TOKENS}  nll={r['nll']:.3f}  {r['prompt']!r} -> {r['output']!r}")

    if args.write:
        os.makedirs(os.path.dirname(args.write) or ".", exist_ok=True)
        with open(args.write, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.write}")
        return
    with open(args.golden) as f:
        failures = compare(results, json.load(f))
    for msg in failures:
        print("FAIL", msg)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
