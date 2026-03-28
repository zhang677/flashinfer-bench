"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.
"""

from __future__ import annotations

import logging
import statistics
from multiprocessing import Lock
from typing import Any, List

import torch
from flashinfer.testing import bench_gpu_time_with_cuda_event, bench_gpu_time_with_cupti

from flashinfer_bench.compile import Runnable

logger = logging.getLogger(__name__)

# Global lock to serialize all CUPTI profiling calls.
# CUPTI does not support concurrent profiling from multiple threads in the
# same process, so we must serialize across all devices (not just per-device).
_cupti_lock = Lock()


def time_runnable(fn: Runnable, args: List[Any], warmup: int, iters: int, device: str) -> float:
    """Time the execution of a value-returning style Runnable kernel.

    Uses CUPTI activity tracing for precise hardware-level kernel timing,
    with automatic fallback to CUDA events if CUPTI is unavailable.

    Parameters
    ----------
    fn : Runnable
        The kernel function to benchmark (must be value-returning style).
    args : List[Any]
        List of arguments in definition order.
    warmup : int
        Number of warmup iterations before timing.
    iters : int
        Number of timing iterations to average over.
    device : str
        The CUDA device to run the benchmark on.

    Returns
    -------
    float
        The median execution time in milliseconds.
    """
    with _cupti_lock:
        with torch.cuda.device(device):
            try:
                times = bench_gpu_time_with_cupti(
                    fn=fn,
                    dry_run_iters=warmup,
                    repeat_iters=iters,
                    input_args=tuple(args),
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                )
            except ValueError as e:
                logger.warning("CUPTI profiling failed (%s), falling back to cuda events", e)
                times = bench_gpu_time_with_cuda_event(
                    fn=fn,
                    dry_run_iters=warmup,
                    repeat_iters=iters,
                    input_args=tuple(args),
                    cold_l2_cache=True,
                )
            return statistics.median(times)
