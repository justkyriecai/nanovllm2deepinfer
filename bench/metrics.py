STAGES = ["wall", "cpu_schedule", "cpu_prepare", "cpu_launch", "cpu_sample_wait",
          "cpu_postprocess", "gpu_forward", "gpu_sample"]


def percentile(xs, p: float) -> float:
    """Linear interpolation between closest ranks, p in [0, 100]."""
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(xs) -> dict:
    if not xs:
        return {}
    return {"mean": sum(xs) / len(xs), "p50": percentile(xs, 50),
            "p99": percentile(xs, 99), "max": max(xs)}


def summarize_steps(steps) -> dict:
    out = {}
    for kind, group in (("prefill", [s for s in steps if s.is_prefill]),
                        ("decode", [s for s in steps if not s.is_prefill])):
        if not group:
            continue
        gpu = sum(s.gpu_forward + s.gpu_sample for s in group)
        out[kind] = {
            "steps": len(group),
            "gpu_busy": gpu / sum(s.wall for s in group),
            **{f: dist([getattr(s, f) for s in group]) for f in STAGES},
        }
    out["preemptions"] = sum(len(s.preempted) for s in steps)
    return out


def summarize_requests(requests, duration: float) -> dict:
    output_tokens = sum(r.num_output_tokens for r in requests)
    return {
        "requests": len(requests),
        "output_tokens": output_tokens,
        "throughput": output_tokens / duration,
        "ttft": dist([r.ttft for r in requests if r.token_times]),  # ignored / early-aborted ones never got a token
        "tbt": dist([t for r in requests for t in r.tbt]),
        "preempted_requests": sum(r.num_preemptions > 0 for r in requests),
    }


def format_report(scenario: str, steps: dict, requests: dict) -> str:
    ms = lambda d: "  ".join(f"{k}={v * 1e3:.2f}" for k, v in d.items())
    lines = [f"== {scenario} ==",
             f"requests={requests['requests']}  output_tokens={requests['output_tokens']}  "
             f"throughput={requests['throughput']:.1f} tok/s",
             f"TTFT ms: {ms(requests['ttft'])}",
             f"TBT  ms: {ms(requests['tbt'])}",
             f"preemptions={steps['preemptions']}  preempted_requests={requests['preempted_requests']}"]
    for kind in ("prefill", "decode"):
        if kind not in steps:
            continue
        s = steps[kind]
        lines.append(f"-- {kind}: {s['steps']} steps, gpu busy {s['gpu_busy']:.1%}")
        lines += [f"   {f:<16} {ms(s[f])}" for f in STAGES]
    return "\n".join(lines)
