import time
from contextlib import contextmanager
from typing import Dict, Any

import torch


class LatencyProfiler:
    """
    Lightweight latency profiler.
    Uses CUDA events for GPU timing when requested and available,
    and perf_counter for CPU sections.
    """

    def __init__(self, enabled: bool = True, use_cuda_events: bool = True) -> None:
        self.enabled = bool(enabled)
        self.use_cuda_events = bool(use_cuda_events) and torch.cuda.is_available()
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def track(self, name: str, use_cuda: bool = False):
        if not self.enabled:
            yield
            return
        if use_cuda and self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            duration = start.elapsed_time(end) / 1000.0
        else:
            start = time.perf_counter()
            yield
            duration = time.perf_counter() - start
        self.totals[name] = self.totals.get(name, 0.0) + duration
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> Dict[str, Any]:
        return {
            "totals_s": dict(self.totals),
            "counts": dict(self.counts),
        }
