"""Compare the A0 GPU-idle follow-up runs (modal_app.py::idle_check).

For each run: request wall and diffuse time, NVML clocks/power during the request, and for
traced runs the kernel-duration sum, busy-interval union and per-step numbers.

    python3 bench/compare_idle.py results/<stamp>_a0idle_t2i_h200_{time,trace,tracelight,time2} [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STEPS = 50
# NVML clock event reasons (bitmask).
REASONS = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
           0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def clock_stats(clocks: list[dict], a: float, b: float) -> dict:
    xs = [c for c in clocks if a <= c["t"] <= b]
    if not xs:
        return {"n": 0}

    def mean(k):
        v = [c[k] for c in xs if c.get(k) is not None]
        return statistics.fmean(v) if v else None

    reason_share = {}
    for bit, name in REASONS.items():
        hits = sum(1 for c in xs if c.get("reasons") is not None and c["reasons"] & bit)
        if hits:
            reason_share[name] = hits / len(xs)
    return {"n": len(xs), "sm_mhz": mean("sm_mhz"), "mem_mhz": mean("mem_mhz"), "power_w": mean("power_w"),
            "temp_c": mean("temp_c"), "reason_share": reason_share}


def diffuse_s(row: dict) -> float | None:
    for k, v in (row.get("stage_durations_s") or {}).items():
        if k.endswith("diffuse"):
            return v
    return None


def summarize(run: Path) -> dict:
    rows = [r for r in load_jsonl(run / "runs.jsonl") if not r.get("error")]
    clocks = load_jsonl(run / "clocks.jsonl")
    target = [r for r in rows if r["phase"] in ("measured", "trace")]
    out: dict = {"run": run.name, "requests": []}
    for r in target:
        d = diffuse_s(r)
        out["requests"].append({
            "phase": r["phase"], "wall_s": r["wall_s"], "diffuse_s": d,
            "diffuse_per_step_ms": d * 1e3 / STEPS if d else None,
            "clocks": clock_stats(clocks, r["t_start"], r["t_end"]),
        })
    ta = run / "trace_analysis.json"
    if ta.exists():
        t = json.loads(ta.read_text())
        k = t["kernels"]
        out["trace"] = {
            "kernel_sum_ms": k["total_gpu_ms"], "busy_union_ms": k.get("busy_union_ms"),
            "gpu_span_ms": k.get("gpu_span_ms"), "kernel_count": k["count"],
            "by_category_ms": {c: v["ms"] for c, v in k["by_category"].items()},
            "diffuse_kernel_sum_ms": t["phases"].get("diffuse", {}).get("gpu_busy_ms") or None,
            "step_rest_mean_gpu_ms": t["steps"].get("rest_mean_gpu_busy_ms"),
            "step_rest_mean_cpu_ms": t["steps"].get("rest_mean_cpu_wall_ms"),
            "step_rest_mean_kernels": t["steps"].get("rest_mean_gpu_kernels"),
        }
    return out


def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = [summarize(Path(r)) for r in args.runs]

    print(f"{'run':40s} {'phase':9s} {'wall s':>7s} {'diff/step':>9s} {'SM MHz':>7s} {'mem MHz':>7s} "
          f"{'W':>6s} {'°C':>5s}  reasons")
    for r in res:
        for q in r["requests"]:
            c = q["clocks"]
            rs = ", ".join(f"{k} {v:.0%}" for k, v in (c.get("reason_share") or {}).items())
            print(f"{r['run'][-40:]:40s} {q['phase']:9s} {q['wall_s']:7.3f} {fmt(q['diffuse_per_step_ms']):>9s} "
                  f"{fmt(c.get('sm_mhz'), 0):>7s} {fmt(c.get('mem_mhz'), 0):>7s} {fmt(c.get('power_w')):>6s} "
                  f"{fmt(c.get('temp_c'), 0):>5s}  {rs}")

    untimed = [q for r in res if "trace" not in r for q in r["requests"] if q["phase"] == "measured"]
    base_wall = statistics.fmean(q["wall_s"] for q in untimed) * 1e3 if untimed else None
    base_step = statistics.fmean(q["diffuse_per_step_ms"] for q in untimed if q["diffuse_per_step_ms"]) \
        if untimed else None
    print(f"\nUntraced baseline: wall {fmt(base_wall)} ms/request, diffuse {fmt(base_step, 2)} ms/step")
    for r in res:
        t = r.get("trace")
        if not t:
            continue
        q = r["requests"][0]
        print(f"\n== {r['run']}")
        print(f"  traced request wall {q['wall_s'] * 1e3:.1f} ms (x{q['wall_s'] * 1e3 / base_wall:.3f} of untraced)")
        print(f"  kernel duration sum {t['kernel_sum_ms']:.1f} ms | busy union {fmt(t['busy_union_ms'])} ms | "
              f"GPU span {fmt(t['gpu_span_ms'])} ms | {t['kernel_count']} events")
        print(f"  in-trace GPU busy (union/span): {fmt(100 * t['busy_union_ms'] / t['gpu_span_ms'])}%")
        print(f"  A0-style idle, whole request (untraced wall - kernel sum): "
              f"{fmt(100 * (1 - t['kernel_sum_ms'] / base_wall))}%")
        if t["step_rest_mean_gpu_ms"]:
            print(f"  steps 1-49: gpu {t['step_rest_mean_gpu_ms']:.2f} ms, cpu {t['step_rest_mean_cpu_ms']:.2f} ms, "
                  f"{t['step_rest_mean_kernels']:.0f} kernels")
            print(f"  A0-style idle, per step (untraced diffuse/50 - step gpu): "
                  f"{fmt(100 * (1 - t['step_rest_mean_gpu_ms'] / base_step))}%")
        print("  by category ms: " + ", ".join(f"{k} {v:.0f}" for k, v in t["by_category_ms"].items()))

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
