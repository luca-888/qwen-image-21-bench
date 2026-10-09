"""A0: single-request baseline + per-stage breakdown for Qwen-Image 2.1.

Runs inside one process / one engine so warmups and measured runs share the same
prepared state. Writes:
    env.json        pinned revisions, package versions, GPU identity
    runs.jsonl      one row per request (phase, wall time, stage durations, memory)
    summary.json    mean/min/max/stdev over measured runs
    images/         every output image (+ sha256 in runs.jsonl)
    trace/          torch-profiler artifacts (only with --trace)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import platform
import statistics
import subprocess
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path

TEAPOT = "A ceramic teapot on a wooden table"
EDIT_PROMPT = "Let this mascot dance under the moon, surrounded by floating stars"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--vllm-omni-sha", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["t2i", "edit"], default="t2i")
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", nargs="*", default=[])
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--feasibility", type=int, default=1)
    p.add_argument("--measured", type=int, default=2)
    p.add_argument("--trace", action="store_true",
                   help="Diagnostic run: torch-profile one request after warmup. Not for timing.")
    p.add_argument("--stages", action="store_true",
                   help="Diagnostic run: time every step of a request (bench/stagehook patches the "
                        "pipeline's profiler targets; modal_app parses the log into stage_log.json). "
                        "A local modification of the pinned checkout, so not a timing baseline.")
    p.add_argument("--mem", action="store_true",
                   help="Diagnostic run: engine DEBUG logging, so each request logs its reserved and "
                        "allocated peaks (parsed into engine_peak_memory.json by modal_app). Not for timing.")
    return p.parse_args()


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_env(args: argparse.Namespace) -> dict:
    import torch

    return {
        "vllm_omni_sha": args.vllm_omni_sha,
        "vllm_omni_checkout": sh("git -C /opt/vllm-omni rev-parse HEAD"),
        "vllm_omni_dirty": sh("git -C /opt/vllm-omni status --porcelain"),
        "model": args.model,
        "model_revision": args.model_revision,
        "packages": {n: pkg(n) for n in
                     ["vllm", "vllm-omni", "torch", "transformers", "diffusers", "flash-attn", "triton"]},
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "nvidia_smi": sh("nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,"
                         "memory.total,clocks.max.sm,power.limit --format=csv"),
        "nvidia_smi_topo": sh("nvidia-smi topo -m"),
        "hostname": platform.node(),
        "cpu": sh("lscpu | grep 'Model name'"),
        # Requested values, not resolved ones: neither script passes these, so the engine uses its
        # defaults (platform-default attention backend, prefix KV stored in the native dtype).
        "runtime_config": {"diffusion_attention_config": None, "prefix_kv_cache_dtype": None,
                           "diffusion_kv_cache_dtype": None},
        "workload": {k: getattr(args, k) for k in
                     ["mode", "prompt", "image", "height", "width", "steps", "seed",
                      "true_cfg_scale", "enforce_eager", "warmup", "feasibility", "measured", "trace", "mem", "stages"]},
    }


class GpuMemSampler:
    """Polls device-used memory via nvidia-smi (whole device, including the CUDA context)."""

    def __init__(self, interval: float = 0.1):
        self.interval, self.peak_mib, self._stop = interval, 0, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
            try:
                self.peak_mib = max(self.peak_mib, max(int(x) for x in out.split()))
            except ValueError:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def main() -> None:
    args = parse_args()
    if args.mem:
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # before vllm is imported; inherited by engine workers
    if args.prompt is None:
        args.prompt = TEAPOT if args.mode == "t2i" else EDIT_PROMPT
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(collect_env(args), indent=2))

    import torch
    from PIL import Image

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras.qwen_image_21 import build_image_to_image_prompt

    omni_kwargs = {
        "model": args.model,
        "revision": args.model_revision,
        "enable_diffusion_pipeline_profiler": True,
    }
    if args.enforce_eager:
        omni_kwargs["enforce_eager"] = True
    if args.trace:
        omni_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(out / "trace"),
            "torch_profiler_use_gzip": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": True,
            "torch_profiler_with_memory": True,
            "active_iterations": 1,
        }

    t0 = time.perf_counter()
    with GpuMemSampler() as init_mem:
        omni = Omni(**omni_kwargs)
    init_s = time.perf_counter() - t0
    # vllm does not configure log output for the offline entrypoint, so surface the two loggers
    # the diagnostic runs read: the runner's per-request peak line (DEBUG) and the profiler's
    # per-stage lines (INFO), which include `forward` and `_decode_latents`.
    surfaced = ([("vllm_omni.diffusion.worker.diffusion_model_runner", logging.DEBUG)] if args.mem else []) + \
               ([("vllm_omni.diffusion.profiler.diffusion_pipeline_profiler", logging.INFO)] if args.stages else [])
    for name, level in surfaced:
        log = logging.getLogger(name)
        log.setLevel(level)
        if not log.hasHandlers():
            log.addHandler(logging.StreamHandler())

    if args.mode == "t2i":
        prompt = {"prompt": args.prompt, "modalities": ["image"]}
    else:
        imgs = [Image.open(p).convert("RGB") for p in args.image]
        prompt = build_image_to_image_prompt(
            prompt=args.prompt, negative_prompt=None,
            input_image=imgs[0] if len(imgs) == 1 else imgs,
            height=args.height, width=args.width,
        )

    def make_params():
        return OmniDiffusionSamplingParams(
            height=args.height, width=args.width, seed=args.seed,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.steps, num_outputs_per_prompt=1,
        )

    runs_f = (out / "runs.jsonl").open("w")
    phases = (["warmup"] * args.warmup
              + ([] if args.trace else ["feasibility"] * args.feasibility + ["measured"] * args.measured)
              + (["trace"] if args.trace else []))
    counters: dict[str, int] = {}

    for phase in phases:
        idx = counters.get(phase, 0)
        counters[phase] = idx + 1
        row: dict = {"phase": phase, "index": idx}
        if phase == "trace":
            omni.start_profile()
        try:
            with GpuMemSampler() as mem:
                t = time.perf_counter()
                outputs = omni.generate(prompt, sampling_params_list=[make_params()])
                row["wall_s"] = time.perf_counter() - t
            row["device_used_peak_mib"] = mem.peak_mib
            o = outputs[0]
            row["stage_durations_s"] = dict(getattr(o, "stage_durations", {}) or {})
            row["engine_peak_memory_mb"] = getattr(o, "peak_memory_mb", None)
            # The runner resets torch's peak counters before each request; non-zero here only if
            # the diffusion worker shares this process.
            row["torch_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            row["torch_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            row["error"] = getattr(o, "error", None)
            images = getattr(o, "images", None) or []
            if images:
                buf = io.BytesIO()
                images[0].save(buf, format="PNG")
                row["image_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
                (out / "images" / f"{phase}_{idx}.png").write_bytes(buf.getvalue())
        except Exception as e:  # noqa: BLE001 - failures/OOMs stay in the record
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
        if phase == "trace":
            row["profile_result"] = str(omni.stop_profile())
        print(json.dumps(row), flush=True)
        runs_f.write(json.dumps(row) + "\n")
        runs_f.flush()

    measured = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
    measured = [r for r in measured if r["phase"] == "measured" and not r.get("error")]

    def stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0}

    stage_names = sorted({k for r in measured for k in r["stage_durations_s"]})
    summary = {
        "engine_init_s": init_s,
        "engine_init_device_used_peak_mib": init_mem.peak_mib,
        "wall_s": stats([r["wall_s"] for r in measured]),
        "device_used_peak_mib": stats([r["device_used_peak_mib"] for r in measured]),
        "stages_s": {s: stats([r["stage_durations_s"].get(s, 0.0) for r in measured]) for s in stage_names},
        "image_sha256_identical": len({r.get("image_sha256") for r in measured}) <= 1,
        "failures": sum(1 for l in (out / "runs.jsonl").read_text().splitlines() if json.loads(l).get("error")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
