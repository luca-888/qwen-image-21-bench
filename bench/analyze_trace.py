"""Offline (CPU-only) analysis of a vllm-omni torch-profiler trace for Qwen-Image 2.1.

GPU time is attributed to a pipeline phase when the kernel's launching CUDA runtime call
falls inside that phase's Python frame. Profiled runs are slower than unprofiled ones, so
use the proportions here and take absolute latency from the timing runs.

    python3 bench/analyze_trace.py results/<run>/trace/trace_rank0.json.gz [--json out.json]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics

PHASES = {
    "encode_prompt": r"pipeline_qwen_image_21\.py\(\d+\): encode_prompt$",
    "diffuse": r"cfg_parallel\.py\(\d+\): diffuse$",
    "decode_latents": r"pipeline_qwen_image_21\.py\(\d+\): _decode_latents$",
    "post_process": r"pipeline_qwen_image_21\.py\(\d+\): post_process_func$",
    "pipeline_forward": r"pipeline_qwen_image_21\.py\(\d+\): forward$",
}
# Python frames as `file.py(line): forward`; line numbers are for vllm-omni 3dc35694.
STEP = r"qwen_image_21_transformer\.py\(1021\): forward$"  # Transformer2DModel.forward, once per step
MODULES = {
    "block": r"qwen_image_21_transformer\.py\(653\): forward$",       # QwenImage21TransformerBlock
    "attention": r"qwen_image_21_transformer\.py\(490\): forward$",   # QwenImage21Attention
    "swiglu_mlp": r"qwen_image_21_transformer\.py\(292\): forward$",  # QwenImage21SwiGLUFeedForward
}


def kernel_category(name: str) -> str:
    n = name.lower()
    if "flash" in n or "fmha" in n or "attn" in n:
        return "attention"
    if any(s in n for s in ("gemm", "nvjet", "xmma", "cutlass")):
        return "gemm"
    if "conv" in n or "implicit" in n:
        return "conv"
    if "norm" in n:
        return "norm"
    if "reduce" in n:
        return "reduce"
    if "elementwise" in n or "catarray" in n or "copy" in n:
        return "elementwise/copy"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        events = json.load(f)["traceEvents"]

    launch_ts = {e["args"]["correlation"]: e["ts"] for e in events
                 if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    gpu = sorted(
        ((launch_ts[e["args"]["correlation"]], e["dur"], e["name"], e["cat"]) for e in events
         if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
         and e.get("args", {}).get("correlation") in launch_ts),
    )
    launches = [g[0] for g in gpu]
    prefix = [0.0]
    for g in gpu:
        prefix.append(prefix[-1] + g[1])

    def gpu_ms(a: float, b: float) -> float:
        i, j = bisect.bisect_left(launches, a), bisect.bisect_right(launches, b)
        return (prefix[j] - prefix[i]) / 1e3

    def gpu_count(a: float, b: float) -> int:
        return bisect.bisect_right(launches, b) - bisect.bisect_left(launches, a)

    py = [e for e in events if e.get("cat") == "python_function"]

    def frames(pat: str) -> list[tuple[float, float]]:
        r = re.compile(pat)
        return sorted((e["ts"], e["ts"] + e["dur"]) for e in py if r.search(e["name"]))

    out: dict = {"phases": {}, "steps": {}, "modules": {}, "kernels": {}}

    for name, pat in PHASES.items():
        fr = frames(pat)
        out["phases"][name] = {
            "n": len(fr),
            "cpu_wall_ms": sum(b - a for a, b in fr) / 1e3,
            "gpu_busy_ms": sum(gpu_ms(a, b) for a, b in fr),
        }

    steps = frames(STEP)
    # gpu_kernels counts kernels, memcpys, and memsets launched inside the step's Python frame.
    per = [{"cpu_wall_ms": (b - a) / 1e3, "gpu_busy_ms": gpu_ms(a, b), "gpu_kernels": gpu_count(a, b)}
           for a, b in steps]
    if per:
        rest = per[1:] or per
        out["steps"] = {
            "n": len(per),
            "first": per[0],
            "rest_mean_gpu_busy_ms": statistics.fmean(p["gpu_busy_ms"] for p in rest),
            "rest_mean_cpu_wall_ms": statistics.fmean(p["cpu_wall_ms"] for p in rest),
            "rest_mean_gpu_kernels": statistics.fmean(p["gpu_kernels"] for p in rest),
            "per_step": per,
        }

    for name, pat in MODULES.items():
        fr = frames(pat)
        out["modules"][name] = {"n": len(fr), "gpu_busy_ms": sum(gpu_ms(a, b) for a, b in fr)}

    cats = collections.defaultdict(lambda: [0, 0.0])
    top = collections.defaultdict(lambda: [0, 0.0])
    for _, dur, name, _ in gpu:
        c = cats[kernel_category(name)]
        c[0] += 1
        c[1] += dur
        t = top[name[:120]]
        t[0] += 1
        t[1] += dur
    total = sum(v[1] for v in cats.values()) or 1.0
    out["kernels"]["by_category"] = {
        k: {"n": n, "ms": t / 1e3, "pct": 100 * t / total}
        for k, (n, t) in sorted(cats.items(), key=lambda x: -x[1][1])
    }
    out["kernels"]["top"] = [
        {"name": k, "n": n, "ms": t / 1e3, "pct": 100 * t / total}
        for k, (n, t) in sorted(top.items(), key=lambda x: -x[1][1])[:15]
    ]
    out["kernels"]["total_gpu_ms"] = total / 1e3
    out["kernels"]["count"] = len(gpu)

    # ---- human-readable report
    print("== Phases (profiled run; use proportions)")
    pf = out["phases"].get("pipeline_forward", {}).get("gpu_busy_ms") or 1.0
    for k, v in out["phases"].items():
        print(f"  {k:18s} cpu {v['cpu_wall_ms']:9.1f} ms   gpu {v['gpu_busy_ms']:9.1f} ms  "
              f"({100 * v['gpu_busy_ms'] / pf:5.1f}% of forward GPU)")
    if per:
        s = out["steps"]
        print(f"== Denoise steps: n={s['n']}  first gpu {s['first']['gpu_busy_ms']:.1f} ms  "
              f"rest mean gpu {s['rest_mean_gpu_busy_ms']:.1f} ms / cpu {s['rest_mean_cpu_wall_ms']:.1f} ms / "
              f"{s['rest_mean_gpu_kernels']:.0f} kernels")
    blk = out["modules"].get("block", {}).get("gpu_busy_ms") or 1.0
    print("== Modules (GPU, share of all transformer-block time)")
    for k, v in out["modules"].items():
        print(f"  {k:12s} n={v['n']:5d}  {v['gpu_busy_ms']:9.1f} ms  {100 * v['gpu_busy_ms'] / blk:5.1f}%")
    print("== Kernel categories")
    for k, v in out["kernels"]["by_category"].items():
        print(f"  {k:18s} {v['ms']:9.1f} ms  {v['pct']:5.1f}%  n={v['n']}")
    print("== Top kernels")
    for t in out["kernels"]["top"][:8]:
        print(f"  {t['ms']:9.1f} ms {t['pct']:5.1f}%  n={t['n']:6d}  {t['name']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
