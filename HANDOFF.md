# A0 交接说明（2026-10-08）

vllm-omni#8586 的 A0 项：Qwen-Image 2.1 baseline 复现，并按阶段做 profiling。
**所有 GPU 任务都已经提交到 Modal**，用的是 `--detach` 模式，不依赖提交它们的那台机器。剩下的工作都在 CPU 上：收集结果、分析、写报告。

## 1. 环境

- Modal workspace：`oscarelpalomino`（本机 profile 是 `luca2`）。必须在这个 workspace 里操作，Volume 和镜像缓存都在这里。
- Volume：`qwen21-hf-cache`（权重）、`qwen21-results`（结果）
- 固定版本：vllm-omni `3dc35694`，基础镜像 `vllm/vllm-openai:v0.31.0`，模型 `Qwen/Qwen-Image-2.1@d26bb612`，1× H200，BF16，eager，1024×1024，50 步，seed 42，`true_cfg_scale=1.0`
- 新机器上的准备：

```bash
git clone https://github.com/luca-888/qwen-image-21-bench && cd qwen-image-21-bench
pip install modal && modal setup        # 授权时选 oscarelpalomino
```

- **如果本机配了 HTTP 代理**，`modal volume get` 会失败，报错提示缺 `aiohttp-socks`。解决办法：执行 `unset HTTP_PROXY HTTPS_PROXY`，或者安装 `pip install 'modal[api-proxy-support]'`。

## 2. 已提交的 GPU 任务

| Modal App | 内容 | 结果目录（在 `qwen21-results` Volume 里） | 提交时的状态 |
|---|---|---|---|
| [ap-UI0zjVnYBIVv877X9O2jPz](https://modal.com/apps/oscarelpalomino/main/ap-UI0zjVnYBIVv877X9O2jPz) | `all` 流程：t2i 计时 → t2i trace → edit 计时 → edit trace | `20261008T100913Z_a0_t2i_h200_eager_time` | ✅ 完成 |
| | | `20261008T101117Z_a0_t2i_h200_eager_trace` | ✅ 完成（包含显存快照） |
| | | `20261008T103632Z_a0_edit_h200_eager_time` | ✅ 完成 |
| | | `20261008T103802Z_a0_edit_h200_eager_trace` | ✅ 完成（19:04，含显存快照） |
| | | `pipeline_<时间>.json` | ❌ 没有写入：所有运行结束后，本机日志连接断开导致总控函数被取消。测试数据完整，不受影响 |
| [ap-atPKBkZUIkraA7fnx9OFDx](https://modal.com/apps/oscarelpalomino/main/ap-atPKBkZUIkraA7fnx9OFDx) | `serving`：起 server，统计 process-to-ready、客户端延迟、响应大小和解码耗时，再跑官方 benchmark（并发 1）交叉验证 | `20261008T104032Z_a0_serving_t2i_h200_eager` | ✅ 完成，0 失败，benchmark 返回码 0 |

检查进度：

```bash
modal app list
modal volume ls qwen21-results
```

两个 App 都变成 `stopped` 时，说明全部跑完了。

## 3. 收集结果

```bash
unset HTTP_PROXY HTTPS_PROXY
modal volume get qwen21-results / results/ --force
```

每个结果目录里有：`env.json`、`runs.jsonl`（每个请求一行，失败的也在里面）、`summary.json`、`images/`。trace 目录还有 `trace/<session>/trace_rank0.json.gz`、`ops_rank0.xlsx`、`memory_snapshot_rank0.pickle`、`profiler_out_0.txt`。serving 目录还有 `server.log`、`benchmark_c1.json` 和 `benchmark_c1.log`。

## 4. 已有结果（H200，eager BF16，正式测量 n=2）

| | t2i | edit（1 张参考图，用 t2i 的输出当参考） |
|---|---:|---:|
| 每张图耗时 | **7.387 s ± 0.014**（历史数据 7.34 s，**+0.6%**，复现成功） | **8.464 s ± 0.032** |
| `diffuse`（50 步） | 7.180 s（97.2%） | 8.112 s（95.8%） |
| text encoder | 0.040 s | 0.076 s |
| VAE encode | — | 0.033 s |
| VAE decode | 0.111 s | 0.112 s |
| 显存峰值（nvidia-smi 整卡） | 39.9 GB | 41.1 GB |
| 显存峰值（引擎报告的 `peak_memory_mb`） | 39.0 GB（历史数据 36.9 GB，**相差 2.1 GB，原因待查**） | 见 runs.jsonl |
| 输出是否稳定 | 4 次运行 SHA256 完全相同 | 完全相同 |
| 引擎初始化 | 34.9 s | — |

t2i 的 trace 分析结果（`bench/analyze_trace.py`；profiler 会让整体变慢，只看比例）：

- 去噪每一步：GPU 实际计算 132 ms，而计时运行中每步实际耗时 143.6 ms，所以 eager 模式下约 **8% 的时间是 kernel 启动造成的 GPU 空闲**（每步约 1700 个 kernel）。这部分对应 A2（CUDA Graph / compile）。
- GPU 时间按 kernel 类型：GEMM 54%（几乎全是 `nvjet_sm90`），**逐元素运算和拷贝 33%**，FlashAttention-3 9%，reduce 和 norm 3%。这部分对应 A1（kernel 融合）。
- 每个 block 内部：attention 模块（包括 QKV 投影和 RoPE）占 49%，SwiGLU MLP 占 41%，norm 和 modulation 等约 10%。
- 短 prompt 的 t2i 里，prefix prefill 可以忽略：第 0 步和后面的步耗时一样。
- edit 比 t2i 多 1.08 s，其中 0.93 s 在 `diffuse` 里，推测是参考图让 prefix 变长，每步 attention 的 KV 也跟着变长。**需要用 edit 的 trace 来验证**。

serving 路径（t2i，eager，正式测量 n=2；`20261008T104032Z_a0_serving_t2i_h200_eager`）：

| | 数值 |
|---|---:|
| process-to-ready（包括加载权重） | 72.1 s |
| 客户端总延迟 | **7.453 s** |
| 引擎 `stage_gen_time_ms` | 7.342 s（每步去噪 142.8 ms） |
| **生成结束到响应发出（PNG 编码 + base64 + JSON，服务端）** | **约 0.105 s（1.4%）** |
| 网络传输响应体（5.6 MB JSON，里面是 4.2 MB 的 PNG） | 0.006 s |
| 客户端解析 JSON、解 base64、解 PNG | 0.009 s、0.012 s、0.020 s |
| 显存峰值（nvidia-smi） | 39.8 GB |

→ 响应编码约 0.1 s，和 VAE decode 是同一个量级，但跟去噪比可以忽略。serving 输出的 PNG 和离线结果的 SHA256 不一样，这是两边 PNG 编码方式不同造成的，**要解码成像素后再比较**，确认画面内容一致。

> 更新：edit trace 分析、显存分析和 serving 分析都已完成，见 [A0_REPORT.md](A0_REPORT.md)。剩下的只有 Release 上传和 issue 评论。

## 5. 剩下的工作（全部在 CPU 上）

- [ ] 等两个 App 跑完，拉取全部结果。如果有失败，先看 `runs.jsonl` 和 `server.log` 里的报错。
- [ ] 分析 edit 的 trace：`python3 bench/analyze_trace.py results/<edit_trace>/trace/*/trace_rank0.json.gz --json results/<edit_trace>/trace_analysis.json`。重点看第 0 步（prefill）和后续各步的差别，以及 attention 占比相比 t2i 的变化。
- [ ] 查清显存差异（39.0 GB 对 36.9 GB）：用 https://pytorch.org/memory_viz 打开 `memory_snapshot_rank0.pickle`，看峰值时刻显存都被哪些张量占着（权重、prefix KV、激活、VAE），再和 recipe 的测量口径对比。也要考虑 torch cu130 和 cu129 的差异。
- [ ] serving：初步结果见上面的表。还要检查 `benchmark_c1.json` 的结果是否一致，并把 serving 和离线的图片解码成像素后比较。
- [ ] 把结果表填进 README 的 Results 部分。提交小文件（json 和图片）；trace（`.json.gz`）和 `.pickle` 已经在 `.gitignore` 里排除了，上传到 GitHub Release：

```bash
gh release create a0-h200-20261008 results/*/trace/*/trace_rank0.json.gz --title "A0 H200 traces"
```

  如果多个文件重名，先重命名再上传。
- [ ] 在 issue #8586 里贴结果：环境、各阶段耗时表（带波动）、显存峰值、最大开销（去噪，以及其中 GEMM、逐元素运算和启动空闲的占比）、仓库和 Release 的链接。最好 @ztang2370（B1）和 @Dmaner（C1），方便对齐 baseline。
- [ ] 收尾：确认没有残留的容器（`modal container list`）。A0 收尾后，如果不再需要，可以删除 `qwen21-hf-cache` 节省存储费用。

## 6. 文件说明

- `modal_app.py`：镜像、Volume 和入口。`all` 是完整的 A0 离线流程，`serving` 是 serving 测试，`a0` 用于单独跑某一项，`download` 只下载权重。
- `bench/a0_offline.py`：离线测试脚本，在同一个引擎里依次跑 warmup、可行性测试、正式测量。
- `bench/a0_serving.py`：serving 测试脚本。
- `bench/analyze_trace.py`：离线 trace 分析，只需要 CPU。
