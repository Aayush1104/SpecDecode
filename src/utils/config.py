"""YAML config loader with dataclass-based configs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    target_model: str = "gpt2"
    draft_model: str = "distilgpt2"
    target_dtype: str = "float16"
    draft_dtype: str = "float16"
    device: str = "auto"
    backend: str = "huggingface"  # "huggingface" or "vllm"


@dataclass
class DecodingConfig:
    speculation_length: int = 5
    temperature: float = 1.0
    max_new_tokens: int = 128
    top_k: int = 0  # 0 means disabled
    top_p: float = 1.0  # 1.0 means disabled


@dataclass
class EvalConfig:
    datasets: list[str] = field(default_factory=lambda: ["humaneval"])
    num_samples: int = 100
    output_dir: str = "results"
    seed: int = 42
    warmup_steps: int = 3


@dataclass
class LoggingConfig:
    level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "specdecode"
    wandb_run_name: Optional[str] = None


@dataclass
class DataConfig:
    domain: str = "code"
    dataset_name: str = "bigcode/the-stack-dedup"
    dataset_subset: Optional[str] = None
    streaming: bool = True
    max_tokens: int = 10_000_000
    max_seq_length: int = 2048
    val_split: float = 0.05
    num_workers: int = 4
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/draft"
    num_train_steps: int = 50_000
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    save_steps: int = 5000
    eval_steps: int = 1000
    log_steps: int = 100
    resume_from: Optional[str] = None


@dataclass
class DistillationConfig:
    enabled: bool = False
    teacher_model: str = "Qwen/Qwen2.5-7B"
    teacher_dtype: str = "bfloat16"
    alpha: float = 0.5
    temperature: float = 2.0


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


def _apply_dict_to_dataclass(dc, d: dict):
    """Recursively apply dict values to a dataclass instance."""
    for key, value in d.items():
        if not hasattr(dc, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict_to_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = ExperimentConfig()
    if raw:
        _apply_dict_to_dataclass(config, raw)
    return config


def merge_configs(*paths: str | Path) -> ExperimentConfig:
    """Load and merge multiple config files (later files override earlier)."""
    config = ExperimentConfig()
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw:
            _apply_dict_to_dataclass(config, raw)
    return config
