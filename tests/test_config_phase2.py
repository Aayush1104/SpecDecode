"""Tests for Phase 2 config extensions (DataConfig, TrainingConfig, DistillationConfig)."""

import pytest
import yaml

from src.utils.config import (
    DataConfig,
    DistillationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
)


class TestDataConfigDefaults:
    """Test DataConfig default values."""

    def test_defaults(self):
        dc = DataConfig()
        assert dc.domain == "code"
        assert dc.dataset_name == "transformersbook/codeparrot"
        assert dc.dataset_subset is None
        assert dc.streaming is True
        assert dc.max_tokens == 10_000_000
        assert dc.max_seq_length == 2048
        assert dc.val_split == 0.05
        assert dc.num_workers == 4
        assert dc.seed == 42


class TestTrainingConfigDefaults:
    """Test TrainingConfig default values."""

    def test_defaults(self):
        tc = TrainingConfig()
        assert tc.output_dir == "checkpoints/draft"
        assert tc.num_train_steps == 50_000
        assert tc.per_device_batch_size == 4
        assert tc.gradient_accumulation_steps == 4
        assert tc.learning_rate == 3e-4
        assert tc.lr_scheduler == "cosine"
        assert tc.warmup_steps == 1000
        assert tc.weight_decay == 0.01
        assert tc.max_grad_norm == 1.0
        assert tc.dtype == "bfloat16"
        assert tc.save_steps == 5000
        assert tc.eval_steps == 1000
        assert tc.log_steps == 100
        assert tc.resume_from is None


class TestDistillationConfigDefaults:
    """Test DistillationConfig default values."""

    def test_defaults(self):
        dc = DistillationConfig()
        assert dc.enabled is False
        assert dc.teacher_model == "Qwen/Qwen2.5-7B"
        assert dc.teacher_dtype == "bfloat16"
        assert dc.alpha == 0.5
        assert dc.temperature == 2.0


class TestExperimentConfigPhase2:
    """Test that ExperimentConfig includes new Phase 2 fields."""

    def test_has_new_fields(self):
        config = ExperimentConfig()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.distillation, DistillationConfig)

    def test_backward_compat_with_base_yaml(self):
        """Phase 1 configs should still load without errors."""
        config = load_config("configs/base.yaml")
        assert config.model.target_model == "gpt2"
        assert config.model.draft_model == "distilgpt2"
        # New fields get defaults
        assert config.data.domain == "code"
        assert config.training.num_train_steps == 50_000
        assert config.distillation.enabled is False

    def test_load_training_config_yaml(self, tmp_path):
        """Load a YAML with training-specific fields."""
        config_data = {
            "model": {"draft_model": "Qwen/Qwen2.5-1.5B"},
            "data": {
                "domain": "chat",
                "max_seq_length": 1024,
            },
            "training": {
                "output_dir": "checkpoints/test",
                "num_train_steps": 100,
                "learning_rate": 1e-4,
            },
            "distillation": {
                "enabled": True,
                "alpha": 0.7,
                "temperature": 3.0,
            },
        }
        config_file = tmp_path / "train.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)

        assert config.model.draft_model == "Qwen/Qwen2.5-1.5B"
        assert config.data.domain == "chat"
        assert config.data.max_seq_length == 1024
        assert config.data.streaming is True  # default preserved
        assert config.training.output_dir == "checkpoints/test"
        assert config.training.num_train_steps == 100
        assert config.training.learning_rate == 1e-4
        assert config.distillation.enabled is True
        assert config.distillation.alpha == 0.7
        assert config.distillation.temperature == 3.0

    def test_load_actual_train_code_yaml(self):
        """Load the actual train_code.yaml config."""
        config = load_config("configs/train_code.yaml")
        assert config.model.target_model == "Qwen/Qwen2.5-7B"
        assert config.model.draft_model == "Qwen/Qwen2.5-1.5B"
        assert config.data.domain == "code"
        assert config.training.output_dir == "checkpoints/draft-code"
        assert config.distillation.enabled is False

    def test_load_actual_distill_code_yaml(self):
        """Load the actual distill_code.yaml config."""
        config = load_config("configs/distill_code.yaml")
        assert config.distillation.enabled is True
        assert config.distillation.alpha == 0.5
        assert config.distillation.temperature == 2.0
        assert config.training.output_dir == "checkpoints/draft-code-distilled"

    def test_unknown_training_key_raises(self, tmp_path):
        """Unknown key inside training section should raise."""
        config_data = {"training": {"nonexistent_key": 42}}
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="Unknown config key"):
            load_config(config_file)
