"""Modal entrypoints for Qwen-Image 2.1 benchmarks (vllm-omni#8586).

Usage:
    modal run modal_app.py::download                      # prefetch weights (CPU only)
    modal run modal_app.py::a0 --mode t2i                 # timing run, H200
    modal run modal_app.py::a0 --mode t2i --trace         # separate torch-profiler run
    modal run modal_app.py::a0 --mode edit --image results/<run>/images/measured_0.png
    modal volume get qwen21-results <run_id> results/      # pull results locally
"""

from __future__ import annotations

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
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install("git", "jq")
    .run_commands(
        "git clone https://github.com/vllm-project/vllm-omni /opt/vllm-omni",
        f"cd /opt/vllm-omni && git checkout {VLLM_OMNI_SHA}",
        'cd /opt/vllm-omni && uv pip install --python "$(which python3)" --no-cache-dir .',
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(Path(__file__).parent / "bench", "/root/bench")
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
def run_bench(args: list[str], run_id: str) -> str:
    out_dir = f"{RESULTS}/{run_id}"
    cmd = [
        "python3", "/root/bench/a0_offline.py",
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
):
    run_id = "_".join([
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "a0", mode, gpu.replace(":", "x").lower(),
        "eager" if eager else "default",
        "trace" if trace else "time",
    ])
    args = ["--mode", mode, "--warmup", str(warmup),
            "--feasibility", str(feasibility), "--measured", str(measured)]
    if eager:
        args.append("--enforce-eager")
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

    out = run_bench.with_options(gpu=gpu).remote(args, run_id)
    print(f"done: {out}\nfetch with: modal volume get qwen21-results {run_id} results/")
