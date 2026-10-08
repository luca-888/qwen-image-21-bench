"""A0 serving path: client latency vs engine time, to expose response encoding/transport cost.

Starts `vllm serve ... --omni` in-process, waits for readiness, then sends sequential
/v1/images/generations requests (warmup → feasibility → measured). Afterwards runs the shared
diffusion serving benchmark at concurrency 1 as a cross-check. Writes:
    env.json, server.log, runs.jsonl, summary.json, images/, benchmark_c1.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from a0_offline import GpuMemSampler, TEAPOT, collect_env  # noqa: E402

PORT = 8091


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--vllm-omni-sha", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default=TEAPOT)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--feasibility", type=int, default=1)
    p.add_argument("--measured", type=int, default=2)
    p.add_argument("--ready-timeout", type=int, default=1800)
    args = p.parse_args()
    args.mode, args.image, args.trace = "t2i-serving", [], False
    return args


def wait_ready(proc: subprocess.Popen, timeout: int) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("server not ready")


def request(args: argparse.Namespace) -> dict:
    body = json.dumps({
        "model": args.model,
        "prompt": args.prompt,
        "size": f"{args.width}x{args.height}",
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "seed": args.seed,
        "response_format": "b64_json",
        "return_stage_metrics": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/images/generations", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        t_headers = time.perf_counter()
        raw = r.read()
    t_body = time.perf_counter()
    payload = json.loads(raw)
    t_json = time.perf_counter()
    b64 = payload["data"][0]["b64_json"]
    png = base64.b64decode(b64)
    t_b64 = time.perf_counter()
    from PIL import Image
    Image.open(io.BytesIO(png)).load()
    t_png = time.perf_counter()
    meta = {k: v for k, v in payload.items() if k != "data"}
    meta["data0_keys"] = [k for k in payload["data"][0] if k != "b64_json"]
    return {
        "client_total_s": t_body - t0,
        "time_to_headers_s": t_headers - t0,
        "body_transfer_s": t_body - t_headers,
        "client_json_parse_s": t_json - t_body,
        "client_b64_decode_s": t_b64 - t_json,
        "client_png_decode_s": t_png - t_b64,
        "response_bytes": len(raw),
        "png_bytes": len(png),
        "image_sha256": hashlib.sha256(png).hexdigest(),
        "_png": png,
        "response_meta": meta,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(collect_env(args), indent=2))

    cmd = ["vllm", "serve", args.model, "--omni", "--port", str(PORT),
           "--revision", args.model_revision, "--enable-diffusion-pipeline-profiler"]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    (out / "server_cmd.txt").write_text(" ".join(cmd) + "\n")
    log = (out / "server.log").open("w")
    t_launch = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    summary: dict = {}
    try:
        with GpuMemSampler() as init_mem:
            summary["process_to_ready_s"] = wait_ready(proc, args.ready_timeout)
        summary["ready_device_used_peak_mib"] = init_mem.peak_mib

        phases = ["warmup"] * args.warmup + ["feasibility"] * args.feasibility + ["measured"] * args.measured
        counters: dict[str, int] = {}
        rows = []
        with (out / "runs.jsonl").open("w") as runs_f:
            for phase in phases:
                idx = counters.get(phase, 0)
                counters[phase] = idx + 1
                row: dict = {"phase": phase, "index": idx}
                try:
                    with GpuMemSampler() as mem:
                        row.update(request(args))
                    row["device_used_peak_mib"] = mem.peak_mib
                    (out / "images" / f"{phase}_{idx}.png").write_bytes(row.pop("_png"))
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"
                    row["traceback"] = traceback.format_exc()
                print(json.dumps(row), flush=True)
                runs_f.write(json.dumps(row) + "\n")
                runs_f.flush()
                rows.append(row)

        measured = [r for r in rows if r["phase"] == "measured" and not r.get("error")]
        for key in ["client_total_s", "time_to_headers_s", "body_transfer_s", "client_json_parse_s",
                    "client_b64_decode_s", "client_png_decode_s", "response_bytes", "device_used_peak_mib"]:
            xs = [r[key] for r in measured]
            if xs:
                summary[key] = {"n": len(xs), "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs),
                                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0}
        summary["failures"] = sum(1 for r in rows if r.get("error"))
        summary["image_sha256_identical"] = len({r.get("image_sha256") for r in measured}) <= 1

        # Cross-check with the shared benchmark (concurrency 1). Failures are recorded, not fatal.
        bench = ["python3", "/opt/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py",
                 "--port", str(PORT), "--model", args.model, "--dataset", "random", "--task", "t2i",
                 "--num-prompts", "3", "--max-concurrency", "1", "--warmup-requests", "1",
                 "--width", str(args.width), "--height", str(args.height),
                 "--num-inference-steps", str(args.steps), "--seed", str(args.seed),
                 "--return-stage-metrics", "--output-file", str(out / "benchmark_c1.json")]
        (out / "benchmark_cmd.txt").write_text(" ".join(bench) + "\n")
        r = subprocess.run(bench, capture_output=True, text=True, timeout=1800)
        (out / "benchmark_c1.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        summary["benchmark_c1_returncode"] = r.returncode
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        summary["server_lifetime_s"] = time.perf_counter() - t_launch
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
