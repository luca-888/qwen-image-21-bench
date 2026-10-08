"""Diagnostic hook for `a0_offline.py --stages`; inert unless QWEN21_BENCH_STAGES is set.

vllm-omni's pipeline profiler times a fixed list of methods and offers no option to extend
it. With the variable set, this patches the Qwen-Image 2.1 pipeline module as it is imported
(in every Python process, so engine workers are covered) to:
  - extend QwenImage21Pipeline._PROFILER_TARGETS to every step of a request, and
  - time the pre- and post-process functions, logging them in the profiler's own line format.
This is a local modification of the pinned checkout; runs that use it are not timing baselines.
"""

import os
import sys

TARGET = "vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21"
STAGE_TARGETS = [
    "_prepare_generation_context", "encode_prompt", "text_encoder.forward",
    "prepare_latents", "_encode_vae_image", "vae.encode", "prepare_timesteps",
    "diffuse", "_decode_latents", "vae.decode",
]


def _patch(module) -> None:
    import functools
    import time

    module.QwenImage21Pipeline._PROFILER_TARGETS = list(STAGE_TARGETS)

    def timed_factory(factory, label):
        @functools.wraps(factory)
        def make(*args, **kwargs):
            func = factory(*args, **kwargs)

            @functools.wraps(func)
            def timed(*a, **kw):
                start = time.perf_counter()
                try:
                    return func(*a, **kw)
                finally:
                    print(f"[DiffusionPipelineProfiler] QwenImage21Pipeline.{label} took "
                          f"{time.perf_counter() - start:.6f}s", file=sys.stderr, flush=True)

            return timed

        return make

    # the engine calls these two outside QwenImage21Pipeline.forward
    module.get_qwen_image_21_pre_process_func = timed_factory(module.get_qwen_image_21_pre_process_func, "pre_process")
    module.get_qwen_image_21_post_process_func = timed_factory(module.get_qwen_image_21_post_process_func, "post_process")
    print(f"[stagehook] patched {TARGET} in pid {os.getpid()}", file=sys.stderr, flush=True)


if os.environ.get("QWEN21_BENCH_STAGES"):
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
