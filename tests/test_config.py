"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ExperimentConfig,
    load_config,
    merge_configs,
    ModelConfig,
    DecodingConfig,
)


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_valid_config(self, tmp_path):
        """Load a valid config file."""
        config_data = {
            "model": {
                "target_model": "gpt2-medium",
                "draft_model": "distilgpt2",
            },
            "decoding": {
                "speculation_length": 7,
                "temperature": 0.8,
            },
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)

        assert config.model.target_model == "gpt2-medium"
        assert config.model.draft_model == "distilgpt2"
        assert config.decoding.speculation_length == 7
        assert config.decoding.temperature == 0.8
        # Defaults should be preserved
        assert config.decoding.max_new_tokens == 128

    def test_load_empty_config(self, tmp_path):
        """Empty config should use all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)
        default = ExperimentConfig()

        assert config.model.target_model == default.model.target_model
        assert config.decoding.speculation_length == default.decoding.speculation_length

    def test_load_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_unknown_key_raises(self, tmp_path):
        """Unknown config keys should raise ValueError."""
        config_data = {"model": {"nonexistent_field": "value"}}
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="Unknown config key"):
            load_config(config_file)

    def test_load_base_config(self):
        """Load the actual base.yaml config file."""
        config = load_config("configs/base.yaml")
        assert config.model.target_model == "gpt2"
        assert config.model.draft_model == "distilgpt2"
        assert config.decoding.speculation_length == 5


class TestMergeConfigs:
    """Tests for merging multiple config files."""

    def test_merge_override(self, tmp_path):
        """Later configs should override earlier ones."""
        base = {"model": {"target_model": "gpt2"}, "decoding": {"temperature": 1.0}}
        override = {"decoding": {"temperature": 0.5}}

        base_file = tmp_path / "base.yaml"
        override_file = tmp_path / "override.yaml"
        base_file.write_text(yaml.dump(base))
        override_file.write_text(yaml.dump(override))

        config = merge_configs(base_file, override_file)

        assert config.model.target_model == "gpt2"  # from base
        assert config.decoding.temperature == 0.5  # overridden


class TestDefaults:
    """Test default values."""

    def test_model_config_defaults(self):
        mc = ModelConfig()
        assert mc.target_model == "gpt2"
        assert mc.backend == "huggingface"

    def test_decoding_config_defaults(self):
        dc = DecodingConfig()
        assert dc.speculation_length == 5
        assert dc.temperature == 1.0
        assert dc.max_new_tokens == 128
