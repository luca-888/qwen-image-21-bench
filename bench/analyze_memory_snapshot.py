"""Offline (CPU-only, stdlib) analysis of a torch CUDA memory snapshot from a vllm-omni trace run.

The snapshot holds the allocator state at the end of the profiled request plus a bounded
history of alloc/free events (vllm-omni records the last 100,000). Allocated memory is
reconstructed by replaying that history backwards from the end state, so the result covers
only the recorded window; the window length is reported.

    python3 bench/analyze_memory_snapshot.py <memory_snapshot_rank0.pickle> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pickle

GIB = 1024**3
# innermost-first search of the Python stack of the allocation that sets the peak
PHASES = [
    ("vae.decode", ("_decode_latents",)),
    ("diffuse", ("diffuse",)),
    ("vae.encode", ("_encode_vae_image",)),
    ("encode_prompt", ("encode_prompt",)),
    ("post_process", ("post_process_func",)),
]


def phase_of(frames: list[dict]) -> str:
    names = {f["name"] for f in frames}
    for phase, markers in PHASES:
        if any(m in names for m in markers):
            return phase
    return "other"


def analyze(path: str) -> dict:
    with open(path, "rb") as f:
        snap = pickle.load(f)
    end_allocated = sum(
        b["size"] for s in snap["segments"] for b in s["blocks"] if b["state"] == "active_allocated"
    )
    end_reserved = sum(s["total_size"] for s in snap["segments"])
    events = [e for trace in snap["device_traces"] for e in trace]
    events.sort(key=lambda e: e["time_us"])

    # allocated[i] is the level right after events[i]; walk back from the end state
    level, after = end_allocated, [0] * len(events)
    for i in range(len(events) - 1, -1, -1):
        after[i] = level
        action, size = events[i]["action"], events[i]["size"]
        if action == "alloc":
            level -= size
        elif action == "free_completed":
            level += size
    start_allocated = level

    peak_i = max(range(len(events)), key=after.__getitem__)
    # the largest allocation still being made on the way up to the peak names the phase
    peak_alloc = next(e for e in reversed(events[: peak_i + 1]) if e["action"] == "alloc")
    by_phase: dict[str, int] = {}
    for i, e in enumerate(events):
        if e["action"] == "alloc":
            p = phase_of(e["frames"])
            by_phase[p] = max(by_phase.get(p, 0), after[i])

    t0 = events[0]["time_us"]
    return {
        "events": len(events),
        "window_s": (events[-1]["time_us"] - t0) / 1e6,
        "window_first_phase": phase_of(events[0]["frames"]),
        "allocated_gib": {
            "window_start": start_allocated / GIB,
            "window_min": min(min(after), start_allocated) / GIB,
            "peak": after[peak_i] / GIB,
            "end": end_allocated / GIB,
            "peak_minus_end": (after[peak_i] - end_allocated) / GIB,
        },
        "reserved_end_gib": end_reserved / GIB,
        "peak": {
            "phase": phase_of(peak_alloc["frames"]),
            "s_into_window": (events[peak_i]["time_us"] - t0) / 1e6,
            "alloc_size_gib": peak_alloc["size"] / GIB,
            "innermost_frames": [f"{f['name']} ({f['filename'].rsplit('/', 1)[-1]}:{f['line']})"
                                 for f in peak_alloc["frames"][:6]],
        },
        "max_allocated_gib_by_phase": {p: v / GIB for p, v in sorted(by_phase.items(), key=lambda kv: -kv[1])},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--json", help="also write the result to this file")
    args = ap.parse_args()
    result = analyze(args.snapshot)
    text = json.dumps(result, indent=1)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
