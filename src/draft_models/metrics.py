"""Training metrics for draft model training."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainingMetrics:
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_mb: float = 0.0
    kd_loss: Optional[float] = None
    lm_loss: Optional[float] = None
    grad_norm: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
