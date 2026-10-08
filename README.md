# qwen-image-21-bench

Reproducible baselines and profiles for **Qwen-Image 2.1** on vLLM-Omni, run on [Modal](https://modal.com).
Contribution to [vllm-project/vllm-omni#8586](https://github.com/vllm-project/vllm-omni/issues/8586), item **A0**.

## A0 plan (declared before measurement)

| | |
|---|---|
| **Hypothesis** | The published eager BF16 result (H200, 7.34 s/image, 36.9 GB peak) reproduces within ±5% at `main@3dc35694`, and denoising (`diffuse`) dominates end-to-end time. |
| **Isolated variable** | None for the baseline. The only later change is the workload: text-to-image, then single-reference editing. |
| **Fixed controls** | vllm-omni `3dc35694`, base image `vllm/vllm-openai:v0.31.0`, model `Qwen/Qwen-Image-2.1@d26bb612`, 1× H200, BF16, eager (`enforce_eager=True`), 1024×1024, 50 steps, seed 42, `true_cfg_scale=1.0`, prefix KV cache at default, no offload, no quantization, no VAE tiling. |
| **Run budget** | Per configuration: 1 warmup (excluded), 1 feasibility run, 2 measured runs, all in the same engine process. Torch-profiler traces come from a **separate** run and are not used for timing. |
| **Success criterion** | A per-stage time breakdown with run-to-run variation and peak device memory, with commands, raw results, traces, and images attached, and the largest measured cost identified. |
| **Stop condition** | Measured eager time deviates >15% from 7.34 s. In that case, stop and report the environment differences before profiling further. |

### Stages

`--enable-diffusion-pipeline-profiler` times `tokenizer.forward`, `text_encoder.forward`, `vae.encode`, `diffuse`, and `vae.decode`.
The torch trace splits `diffuse` into prefix prefill and per-step decode.
Response encoding is measured on the serving path (follow-up).

## Layout

```
modal_app.py        Modal image (pinned SHA), volumes, entrypoints
bench/a0_offline.py runs inside the container; one engine, warmup → feasibility → measured
results/<run_id>/   pulled from the Modal volume: env.json, runs.jsonl, summary.json, images/, trace/
```

Large traces (>50 MB) are published as GitHub Release assets, not committed.

## Run

```bash
pip install modal && modal setup
modal run modal_app.py::download                 # cache weights in the Volume once
modal run modal_app.py::a0 --mode t2i            # timing
modal run modal_app.py::a0 --mode t2i --trace    # torch-profiler trace (diagnostic only)
modal volume get qwen21-results <run_id> results/
```

Editing with a single reference image (for example, the T2I output from a previous run):

```bash
modal run modal_app.py::a0 --mode edit --image results/<t2i_run>/images/measured_0.png
```

Other GPUs: `--gpu H100`, `--gpu A100-80GB`, `--gpu L40S`. Don't pool results across GPU types.
Gated or rate-limited downloads: create a Modal secret `huggingface` with `HF_TOKEN`, then prefix commands with `HF_SECRET=1`.

## Results

| Run | GPU | Mode | Wall s (mean ± sd) | Largest stage | Peak device MiB | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |
