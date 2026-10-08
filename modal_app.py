"""Modal entrypoints for Qwen-Image 2.1 benchmarks (vllm-omni#8586).

Usage:
    modal run modal_app.py::download                      # prefetch weights (CPU only)
    modal run modal_app.py::a0 --mode t2i                 # timing run, H200
    modal run modal_app.py::a0 --mode t2i --trace         # separate torch-profiler run
    modal run modal_app.py::a0 --mode edit --image inputs/qwen_bear.png
    modal volume get qwen21-results <run_id> results/      # pull results locally
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

# ---- Pinned revisions ------------------------------------------------------
# Issue baseline: vllm-omni main@3dc35694. Base image matches docker/Dockerfile.cuda
# at that commit.
VLLM_OMNI_SHA = "3dc35694b3d458fc1461a8fe71832ac12e8c8cea"
BASE_IMAGE = "vllm/vllm-openai:v0.31.0"
MODEL_ID = "Qwen/Qwen-Image-2.1"
# HF main as of 2026-10-08. #7945 used b3179ad355be050328e483a9dfdd9e60cd62adfa.
MODEL_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"

HF_CACHE = "/cache/hf"
RESULTS = "/results"

hf_volume = modal.Volume.from_name("qwen21-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen21-results", create_if_missing=True)

# Set HF_SECRET=1 locally if you created a Modal secret named "huggingface" with HF_TOKEN.
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("HF_SECRET") else []

image = (
    # Modal detects the interpreter via `python`; the vllm image only ships `python3`
    # (docker/Dockerfile.cuda adds the same symlink).
    modal.Image.from_registry(
        BASE_IMAGE, setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/bin/python"]
    )
    .entrypoint([])
    .apt_install("git", "jq")
    .run_commands(
        "git clone https://github.com/vllm-project/vllm-omni /opt/vllm-omni",
        f"cd /opt/vllm-omni && git checkout {VLLM_OMNI_SHA}",
        'cd /opt/vllm-omni && uv pip install --python "$(which python3)" --no-cache-dir .',
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(Path(__file__).parent / "bench", "/root/bench")
    .add_local_dir(Path(__file__).parent / "inputs", "/root/inputs")
)

app = modal.App("qwen-image-21-bench", image=image)
VOLUMES = {HF_CACHE: hf_volume, RESULTS: results_volume}


@app.function(volumes=VOLUMES, secrets=SECRETS, timeout=3600, cpu=4)
def prefetch() -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    hf_volume.commit()
    return path


@app.function(volumes=VOLUMES, secrets=SECRETS, timeout=3 * 3600, gpu="H200")
def run_bench(args: list[str], run_id: str, script: str = "a0_offline.py") -> str:
    out_dir = f"{RESULTS}/{run_id}"
    cmd = [
        "python3", f"/root/bench/{script}",
        "--model", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--vllm-omni-sha", VLLM_OMNI_SHA,
        "--out", out_dir,
        *args,
    ]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        # Keep partial results (failures/OOMs must stay visible).
        results_volume.commit()
    return out_dir


@app.function(timeout=900, cpu=2)
def precheck() -> None:
    """Cheap CPU-only check that imports and CLI parse before paying for a GPU."""
    subprocess.run([
        "python3", "-c",
        "from vllm_omni.entrypoints.omni import Omni;"
        "from vllm_omni.inputs.data import OmniDiffusionSamplingParams;"
        "from vllm_omni.model_extras.qwen_image_21 import build_image_to_image_prompt;"
        "print('imports ok')",
    ], check=True)
    subprocess.run(["python3", "/root/bench/a0_offline.py", "--help"], check=True, capture_output=True)


PUBLISHED_EAGER_S = 7.34  # recipe: H200 BF16 eager steady-state
STOP_DEVIATION = 0.15     # README stop condition


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bench(gpu: str, args: list[str], run_id: str) -> str:
    fn = run_bench if gpu == "H200" else run_bench.with_options(gpu=gpu)
    return fn.remote(args, run_id)


@app.function(volumes={RESULTS: results_volume}, timeout=8 * 3600, cpu=1)
def pipeline(gpu: str = "H200", with_edit: bool = True) -> dict:
    """precheck → prefetch → t2i timing → (stop check) → t2i trace → edit timing → edit trace.

    Runs remotely so it keeps going after the local client disconnects (`modal run --detach`).
    Any failure stops the remaining GPU steps.
    """
    log: dict = {"gpu": gpu, "steps": []}
    tag = gpu.replace(":", "x").lower()
    base = ["--warmup", "1", "--feasibility", "1", "--measured", "2", "--enforce-eager"]

    precheck.remote()
    log["steps"].append("precheck ok")
    prefetch.remote()
    log["steps"].append("prefetch ok")

    t2i = f"{_stamp()}_a0_t2i_{tag}_eager_time"
    _bench(gpu, ["--mode", "t2i", *base], t2i)
    results_volume.reload()
    summary = json.loads(Path(f"{RESULTS}/{t2i}/summary.json").read_text())
    mean = (summary.get("wall_s") or {}).get("mean")
    log["t2i_time"] = {"run_id": t2i, "wall_mean_s": mean, "failures": summary.get("failures")}
    if mean is None or summary.get("failures"):
        log["stopped"] = "t2i timing had failures"
        return log
    if gpu == "H200" and abs(mean - PUBLISHED_EAGER_S) / PUBLISHED_EAGER_S > STOP_DEVIATION:
        log["stopped"] = f"t2i mean {mean:.2f}s deviates >{STOP_DEVIATION:.0%} from {PUBLISHED_EAGER_S}s"
        return log

    t2i_trace = f"{_stamp()}_a0_t2i_{tag}_eager_trace"
    _bench(gpu, ["--mode", "t2i", *base, "--trace"], t2i_trace)
    log["t2i_trace"] = t2i_trace

    if with_edit:
        # Reference image from the recipe's editing example (inputs/qwen_bear.png).
        ref = "/root/inputs/qwen_bear.png"
        edit = f"{_stamp()}_a0_edit_{tag}_eager_time"
        _bench(gpu, ["--mode", "edit", "--image", ref, *base], edit)
        log["edit_time"] = edit
        edit_trace = f"{_stamp()}_a0_edit_{tag}_eager_trace"
        _bench(gpu, ["--mode", "edit", "--image", ref, *base, "--trace"], edit_trace)
        log["edit_trace"] = edit_trace

    Path(f"{RESULTS}/pipeline_{_stamp()}.json").write_text(json.dumps(log, indent=2))
    results_volume.commit()
    return log


@app.local_entrypoint()
def all(gpu: str = "H200", with_edit: bool = True):
    log = pipeline.remote(gpu, with_edit)
    print(json.dumps(log, indent=2))
    print("fetch with: modal volume get qwen21-results / results/")


@app.local_entrypoint()
def serving(gpu: str = "H200", eager: bool = True):
    """A0 serving path: process-to-ready, client latency vs engine time, response size/decoding."""
    run_id = f"{_stamp()}_a0_serving_t2i_{gpu.replace(':', 'x').lower()}_{'eager' if eager else 'default'}"
    args = ["--warmup", "1", "--feasibility", "1", "--measured", "2"] + (["--enforce-eager"] if eager else [])
    fn = run_bench if gpu == "H200" else run_bench.with_options(gpu=gpu)
    print(fn.remote(args, run_id, "a0_serving.py"))


@app.local_entrypoint()
def download():
    print(prefetch.remote())


@app.local_entrypoint()
def a0(
    mode: str = "t2i",
    gpu: str = "H200",
    trace: bool = False,
    eager: bool = True,
    image: str = "",
    warmup: int = 1,
    feasibility: int = 1,
    measured: int = 2,
    prompt: str = "",
    tag: str = "",
):
    run_id = "_".join([
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "a0", mode, gpu.replace(":", "x").lower(),
        "eager" if eager else "default",
        "trace" if trace else "time",
    ] + ([tag] if tag else []))
    args = ["--mode", mode, "--warmup", str(warmup),
            "--feasibility", str(feasibility), "--measured", str(measured)]
    if eager:
        args.append("--enforce-eager")
    if prompt:
        args += ["--prompt", prompt]
    if trace:
        args.append("--trace")
    if mode == "edit":
        if not image:
            raise SystemExit("--image is required for --mode edit")
        # Ship the reference image through the results volume so it is archived with the run.
        ref = Path(image)
        with results_volume.batch_upload(force=True) as batch:
            batch.put_file(str(ref), f"/{run_id}/inputs/{ref.name}")
        args += ["--image", f"{RESULTS}/{run_id}/inputs/{ref.name}"]

    out = _bench(gpu, args, run_id)
    print(f"done: {out}\nfetch with: modal volume get qwen21-results {run_id} results/")
