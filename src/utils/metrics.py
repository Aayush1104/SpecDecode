"""Metric tracking for speculative decoding evaluation."""

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
        }


class MetricsTracker:
    """Accumulates metrics across multiple decoding runs."""

    def __init__(self):
        self.runs: list[DecodingMetrics] = []

    def add_run(self, metrics: DecodingMetrics):
        self.runs.append(metrics)

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

    def summary(self) -> dict:
        return {
            "num_runs": self.num_runs,
            "mean_acceptance_rate": self.mean_acceptance_rate(),
            "mean_tokens_per_second": self.mean_tokens_per_second(),
            "mean_latency_seconds": self.mean_latency(),
            "total_tokens": self.total_tokens(),
        }
