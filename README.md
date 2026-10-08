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
Response encoding is measured on the serving path (`bench/a0_serving.py`).

## Layout

```
modal_app.py           Modal image (pinned SHA), volumes, entrypoints
bench/a0_offline.py    runs inside the container; one engine, warmup → feasibility → measured
bench/a0_serving.py    runs inside the container; `vllm serve`, sequential requests, shared serving benchmark
bench/analyze_trace.py offline analysis of a torch-profiler trace → trace_analysis.json
bench/plot_trace.py    draws figures/*.svg from two trace_analysis.json files
inputs/                reference image for editing (the recipe's qwen_bear.png)
figures/               figures used in A0_REPORT.md
results/<run_id>/      pulled from the Modal volume: env.json, runs.jsonl, summary.json, images/, trace/
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

Editing with a single reference image (the recipe's `qwen_bear.png`, committed under `inputs/`):

```bash
modal run modal_app.py::a0 --mode edit --image inputs/qwen_bear.png
```

Serving path, and the full A0 set (T2I and edit, timing and trace) in one go:

```bash
modal run --detach modal_app.py::serving
modal run --detach modal_app.py::all
```

Other GPUs: `--gpu H100`, `--gpu A100-80GB`, `--gpu L40S`. Don't pool results across GPU types.
Gated or rate-limited downloads: create a Modal secret `huggingface` with `HF_TOKEN`, then prefix commands with `HF_SECRET=1`.

## Results

| Run | GPU | Mode | Wall s (mean ± sd) | Largest stage | Peak device MiB | Notes |
|---|---|---|---|---|---|---|
| `20261008T100913Z_a0_t2i_h200_eager_time` | H200 | T2I, eager | 7.387 ± 0.014 (published 7.34) | `diffuse` 7.180 s (97.2%) | 39 925 (engine 39 030) | 4/4 outputs bit-identical |
| `20261008T125447Z_a0_edit_h200_eager_time_bear` | H200 | Edit, 1 ref (`qwen_bear.png`), eager | 8.363 ± 0.001 | `diffuse` 7.999 s (95.6%) | 41 121 | 4/4 outputs bit-identical |
| `20261008T104032Z_a0_serving_t2i_h200_eager` | H200 | T2I serving, eager | client 7.453 (engine 7.342) | `diffuse`; response encoding ≈0.105 s | 39 799 | pixel-identical to offline |

Full write-up: [A0_REPORT.md](A0_REPORT.md). Traces and memory snapshots: release [`a0-h200-20261008`](https://github.com/luca-888/qwen-image-21-bench/releases/tag/a0-h200-20261008).
