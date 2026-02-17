"""CUDA-aware timing context manager for accurate GPU timing."""

import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer():
    """Context manager that returns accurate elapsed time, using CUDA events if available.

    Usage:
        with cuda_timer() as timer:
            # do work
        elapsed = timer.elapsed  # seconds
    """
    result = _TimerResult()

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        try:
            yield result
        finally:
            end_event.record()
            torch.cuda.synchronize()
            result.elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> s
    else:
        start = time.perf_counter()
        try:
            yield result
        finally:
            result.elapsed = time.perf_counter() - start


class _TimerResult:
    """Holds the elapsed time from a timer context manager."""

    def __init__(self):
        self.elapsed: float = 0.0


def get_peak_memory_mb() -> float:
    """Get peak GPU memory usage in MB, or 0 if CUDA is unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory():
    """Reset the peak memory tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
