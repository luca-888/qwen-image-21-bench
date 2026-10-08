## A0 results: H200 baseline and per-stage profile

Setup:
- Code: `main@3dc35694` (clean checkout), `vllm/vllm-openai:v0.31.0`, torch 2.13.0+cu130, transformers 5.14.1
- Model: `Qwen/Qwen-Image-2.1@d26bb612`
- Hardware: 1× H200 141 GB
- Workload: BF16, eager, 1024×1024, 50 steps, seed 42, `true_cfg_scale=1.0`
- Runs: 1 warmup (excluded), 1 feasibility run and 2 measured runs per configuration, in one engine process. Traces come from separate runs.

Full report, commands, raw results and images: https://github.com/luca-888/qwen-image-21-bench/blob/main/A0_REPORT.md

**Latency.** The published eager BF16 number reproduces.

| | T2I | Edit (1 ref) |
|---|---:|---:|
| Wall time / image | **7.387 s ± 0.014** (published 7.34 s, +0.6%) | **8.464 s ± 0.032** |
| `diffuse` (50 steps) | 7.180 s (97.2%) | 8.112 s (95.8%) |
| Text encoder / VAE encode / VAE decode | 0.040 / – / 0.111 s | 0.076 / 0.033 / 0.112 s |
| Failures / OOM | 0 | 0 |

All runs were bit-identical. Serving output is pixel-identical to offline output.

**Serving** (`/v1/images/generations`, b64 PNG):
- Process-to-ready: 72.1 s
- Client latency: 7.453 s. Engine time: 7.342 s.
- Response encoding (PNG + base64 + JSON): **≈0.105 s (1.4%)**. Transfer: 6 ms.

**Largest cost: denoising, 96–97% of latency.** From the torch-profiler traces (proportions only):
- **GPU kernel time:** GEMM 54%, **elementwise/copy 33%**, FlashAttention-3 9%. This is fusion headroom (A1).
- **Per-step idle time:** in T2I eager mode a step spends 132 ms of GPU time, but its wall time is 143.6 ms. So about **8% of each step is GPU idle** while ~1,700 kernels per step are launched (A2).
- **Prefix prefill:** negligible for short T2I prompts. With one reference image, prefill takes **~134 ms once**, and each later step is ~15 ms slower because attention covers the longer prefix. Attention kernel time doubles (A3/B2).

**Memory.**
- Peak **allocated** is 36.86 GiB, which matches the published 36.9 GB. The engine's `peak_memory_mb` (39,030) is **reserved** memory.
- The peak occurs in **VAE decode**: +6.6 GiB transient on top of ~30.3 GiB resident (C2).

The full traces and the memory snapshot are attached to the release. The traces open in Perfetto.

The shared benchmark (`/v1/chat/completions`, random prompts, c=1) shows ~0.54 s between client latency and engine time. That is much larger than on the images endpoint. I haven't isolated it yet and will look into it next unless someone already knows the cause.

cc @ztang2370 (B1, c=1 baseline): same workload here on H200. Comparing the A100 stage split would be useful.
cc @Dmaner (C1): this run can serve as the H200 BF16 eager reference; I'm happy to align env and commit.
