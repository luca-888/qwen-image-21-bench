# A0: Qwen-Image 2.1 single-request baseline and per-stage profile (H200)

Item **A0** of [vllm-project/vllm-omni#8586](https://github.com/vllm-project/vllm-omni/issues/8586).
The plan (hypothesis, controls, run budget, stop condition) was declared in [README.md](README.md) before measurement.

## Setup

| | |
|---|---|
| Code | vllm-omni `3dc35694b3d4` (clean checkout, no local changes), base image `vllm/vllm-openai:v0.31.0` |
| Packages | vllm 0.31.0, torch 2.13.0+cu130, transformers 5.14.1, diffusers 0.40.0, triton 3.7.1 |
| Model | `Qwen/Qwen-Image-2.1@d26bb61231c3` |
| Hardware | 1× NVIDIA H200 141 GB (Modal), driver 580.95.05; GPU UUIDs in each `env.json` |
| Workload | BF16, eager (`enforce_eager=True`), 1024×1024, 50 steps, seed 42, `true_cfg_scale=1.0`. Default prefix KV cache; no offload, quantization, or VAE tiling |
| T2I prompt | "A ceramic teapot on a wooden table" |
| Edit | Same settings plus one reference image (the T2I output); prompt "Let this mascot dance under the moon, surrounded by floating stars" |
| Budget | Per config: 1 warmup (excluded), 1 feasibility run, 2 measured runs, all in one engine process. Traces come from separate runs and are not used for timing |

## 1. Latency: the baseline reproduces

| | T2I | Edit (1 ref) |
|---|---:|---:|
| **Wall time / image** (measured, n=2) | **7.387 s ± 0.014** | **8.464 s ± 0.032** |
| Published (recipe, H200 eager BF16) | 7.34 s → **+0.6%** | — |
| `diffuse` (50 steps) | 7.180 s (97.2%) | 8.112 s (95.8%) |
| `text_encoder.forward` | 0.040 s (0.5%) | 0.076 s (0.9%) |
| `vae.encode` | — | 0.033 s (0.4%) |
| `vae.decode` | 0.111 s (1.5%) | 0.112 s (1.3%) |
| Engine init (excluded) | 34.9 s | — |
| Failures / OOM | 0 | 0 |
| Output determinism | 4/4 runs bit-identical | 4/4 bit-identical |

Stage times come from `--enable-diffusion-pipeline-profiler` (synchronized). Raw rows: `results/*_time/runs.jsonl`.

## 2. Serving path and response encoding (T2I)

`vllm serve ... --omni --enforce-eager`, sequential `/v1/images/generations` requests with `response_format=b64_json`, measured n=2:

| | |
|---|---:|
| Process-to-ready (including weight load) | 72.1 s |
| Client total latency | **7.453 s** |
| Engine `stage_gen_time_ms` | 7.342 s (142.8 ms per denoise step) |
| **Server: generation done → response sent** (PNG encode + base64 + JSON) | **≈ 0.105 s (1.4%)** |
| Response body transfer (5.6 MB JSON, 4.2 MB PNG) | 0.006 s |
| Client JSON / base64 / PNG decode | 0.009 / 0.012 / 0.020 s |

Serving outputs are **pixel-identical** to offline outputs. The PNG bytes differ only because of the PNG encoder.

Cross-check with `benchmarks/diffusion/diffusion_benchmark_serving.py` (`--dataset random --max-concurrency 1`, 3 prompts): mean latency 7.918 s against engine `stage_0_gen_ms` 7.375 s. That benchmark uses `/v1/chat/completions` with random prompts, so the request-path overhead there (≈0.54 s) is not directly comparable. It is flagged for follow-up.

## 3. Where the time goes inside denoising (torch-profiler traces)

These proportions come from profiled runs, which are slower. Analysis: `python3 bench/analyze_trace.py <trace>`.

**Per step, GPU busy time vs wall time**

| | T2I | Edit |
|---|---:|---:|
| GPU busy, step 0 | 128.5 ms | **280.2 ms** (includes prefix prefill of the reference image) |
| GPU busy, steps 1–49 (mean) | 132.0 ms | 146.7 ms (+11%) |
| Unprofiled wall time per step (`diffuse`/50) | 143.6 ms | 162.2 ms |
| Kernels per step | ≈1,700 | ≈1,720 |

- In T2I eager mode, about **8% of each step is GPU idle** (143.6 ms wall vs 132.0 ms of kernels). The cause is launch/host overhead from about 1,700 small kernels per step. This is the headroom for CUDA Graph and compile (A2).
- With a short T2I prompt, **prefix prefill is negligible**: step 0 costs the same as the other steps.
- With one reference image, prefill adds **about 134 ms once** (step 0). Every later step is **about 15 ms slower** because attention runs over the longer cached prefix. Edit attention kernel time doubles (626 → 1,248 ms over 50 steps). Together these account for most of the +0.93 s `diffuse` difference.

**GPU time by kernel class** (whole request)

| | T2I | Edit |
|---|---:|---:|
| GEMM (almost all `nvjet_sm90_tst_256x128`) | **54.2%** | 49.2% |
| Elementwise / copy / cat / dtype casts | **32.8%** | 31.1% |
| FlashAttention-3 (SM90) | 9.3% | 16.3% |
| Reduce + norm | 3.3% | ~3% |

**GPU time by module** (share of transformer-block time; T2I / Edit)

- Attention module (QKV/out projections, Q/K norm, RoPE, attention): 48.8% / 53.5%
- SwiGLU MLP: 40.7% / 36.9%
- Remainder (AdaLN modulation, residuals, gating): ~10%

## 4. Memory

| | T2I | Edit |
|---|---:|---:|
| **Peak allocated** (memory snapshot) | **36.86 GiB**. Matches the published 36.9 GB | 36.86 GiB |
| Peak reserved (= engine `peak_memory_mb`) | 38.12 GiB (39,030 MiB) | 39.28 GiB (40,226 MiB) |
| Device used (nvidia-smi, includes CUDA context) | 39,925 MiB | 41,123 MiB |
| Resident after request (weights, etc.) | 30.29 GiB | 30.29 GiB |
| **Peak location** | **VAE decode**: +6.56 GiB transient | VAE decode |

The engine's `peak_memory_mb` reports **reserved** memory. The recipe's 36.9 GB matches **allocated** memory, so the two numbers do not conflict.
The memory history covers only the last ~3 s of the profiled request (`max_entries=100000`). The peak is therefore confirmed within VAE decode, but transient memory during early denoising steps is outside the recorded window.

## 5. Conclusion: largest measured cost

1. **Denoising is 96–97% of end-to-end latency.** Within it:
   - GEMM is about half of GPU time.
   - **Elementwise/copy kernels are about a third**, which points to fusion (A1). This is H200 eager mode. #7945 measured a fusion-only gain on A100 with offload, a different configuration.
   - Launch overhead is about 8% of wall time in eager mode, which points to graph/compile (A2).
2. **For editing**, the extra cost is the reference-image prefix: a one-time ~134 ms prefill plus ~15 ms per step of longer attention. That is the workload for A3/B2.
3. **Peak memory is set by VAE decode**, not by denoising: +6.6 GiB over resident weights. VAE tiling is the lever here (C2).
4. Text encoding, VAE, and response encoding are each ≤1.5% of latency.

## Artifacts

- Timing runs: `results/20261008T100913Z_a0_t2i_h200_eager_time`, `results/20261008T103632Z_a0_edit_h200_eager_time`
- Serving: `results/20261008T104032Z_a0_serving_t2i_h200_eager` (`server.log`, `benchmark_c1.json`)
- Trace runs: `results/20261008T101117Z_a0_t2i_h200_eager_trace`, `results/20261008T103802Z_a0_edit_h200_eager_trace` (`trace_analysis.json`). Full `trace_rank0.json.gz` and the memory snapshot are published as release assets, not committed.
- Reproduce: `modal run --detach modal_app.py::all` and `modal run --detach modal_app.py::serving`
