"""Metric tracking for speculative decoding evaluation."""

import math
import time
from dataclasses import dataclass, field


@dataclass
class DecodingMetrics:
    """Metrics for a single decoding run."""
    total_tokens: int = 0
    total_steps: int = 0
    accepted_tokens: int = 0
    draft_tokens: int = 0
    latency_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    # Per-phase timing (populated when profile=True)
    draft_time: float = 0.0
    verify_time: float = 0.0
    sampling_time: float = 0.0
    overhead_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.draft_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.latency_seconds == 0.0:
            return 0.0
        return self.total_tokens / self.latency_seconds

    @property
    def avg_accepted_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.accepted_tokens / self.total_steps

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_steps": self.total_steps,
            "accepted_tokens": self.accepted_tokens,
            "draft_tokens": self.draft_tokens,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_second": self.tokens_per_second,
            "avg_accepted_per_step": self.avg_accepted_per_step,
            "latency_seconds": self.latency_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "draft_time": self.draft_time,
            "verify_time": self.verify_time,
            "sampling_time": self.sampling_time,
            "overhead_time": self.overhead_time,
        }


class MetricsTracker:
    """Accumulates metrics across multiple decoding runs."""

    def __init__(self):
        self.runs: list[DecodingMetrics] = []
        self.domain_runs: dict[str, list[DecodingMetrics]] = {}

    def add_run(self, metrics: DecodingMetrics, domain: str | None = None):
        self.runs.append(metrics)
        if domain is not None:
            if domain not in self.domain_runs:
                self.domain_runs[domain] = []
            self.domain_runs[domain].append(metrics)

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    def mean_acceptance_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.acceptance_rate for r in self.runs) / len(self.runs)

    def mean_tokens_per_second(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.tokens_per_second for r in self.runs) / len(self.runs)

    def mean_latency(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.latency_seconds for r in self.runs) / len(self.runs)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.runs)

    def latency_percentiles(self) -> dict:
        """Compute p50, p95, p99 latency across runs."""
        if not self.runs:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        latencies = sorted(r.latency_seconds for r in self.runs)
        n = len(latencies)

        def _percentile(p: float) -> float:
            idx = p / 100.0 * (n - 1)
            lo = int(math.floor(idx))
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return latencies[lo] * (1 - frac) + latencies[hi] * frac

        return {
            "p50": _percentile(50),
            "p95": _percentile(95),
            "p99": _percentile(99),
        }

    def std_tokens_per_second(self) -> float:
        """Standard deviation of tokens/second across runs."""
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_tokens_per_second()
        variance = sum((r.tokens_per_second - mean) ** 2 for r in self.runs) / (len(self.runs) - 1)
        return math.sqrt(variance)

    def std_acceptance_rate(self) -> float:
        """Standard deviation of acceptance rate across runs."""
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_acceptance_rate()
        variance = sum((r.acceptance_rate - mean) ** 2 for r in self.runs) / (len(self.runs) - 1)
        return math.sqrt(variance)

    def domain_summary(self) -> dict:
        """Per-domain mean metrics."""
        result = {}
        for domain, runs in self.domain_runs.items():
            n = len(runs)
            if n == 0:
                continue
            result[domain] = {
                "num_runs": n,
                "mean_acceptance_rate": sum(r.acceptance_rate for r in runs) / n,
                "mean_tokens_per_second": sum(r.tokens_per_second for r in runs) / n,
                "mean_latency_seconds": sum(r.latency_seconds for r in runs) / n,
            }
        return result

    def summary(self) -> dict:
        percentiles = self.latency_percentiles()
        return {
            "num_runs": self.num_runs,
            "mean_acceptance_rate": self.mean_acceptance_rate(),
            "mean_tokens_per_second": self.mean_tokens_per_second(),
            "mean_latency_seconds": self.mean_latency(),
            "total_tokens": self.total_tokens(),
            "std_tokens_per_second": self.std_tokens_per_second(),
            "std_acceptance_rate": self.std_acceptance_rate(),
            "latency_p50": percentiles["p50"],
            "latency_p95": percentiles["p95"],
            "latency_p99": percentiles["p99"],
            "domain_summary": self.domain_summary(),
        }
