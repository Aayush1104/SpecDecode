"""Tests for the adaptive routing system (Phase 3)."""

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from src.routing.features import FeatureExtractor
from src.routing.model import RouterMLP
from src.utils.config import ExperimentConfig, load_config, RouterConfig


# ──────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────


class TestRouterConfig:
    """Tests for RouterConfig defaults and YAML loading."""

    def test_defaults(self):
        """RouterConfig should have sensible defaults."""
        rc = RouterConfig()
        assert rc.enabled is False
        assert rc.embedding_dim == 768
        assert rc.hidden_dim == 256
        assert rc.num_epochs == 5
        assert rc.learning_rate == 1e-3
        assert rc.batch_size == 64
        assert isinstance(rc.draft_models, dict)
        assert "code" in rc.draft_models

    def test_experiment_config_has_router(self):
        """ExperimentConfig should include a RouterConfig field."""
        config = ExperimentConfig()
        assert hasattr(config, "router")
        assert isinstance(config.router, RouterConfig)
        assert config.router.enabled is False

    def test_yaml_loading(self, tmp_path):
        """Loading a YAML with router section should populate RouterConfig."""
        config_data = {
            "router": {
                "enabled": True,
                "hidden_dim": 512,
                "num_epochs": 10,
            },
        }
        config_file = tmp_path / "router_test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.router.enabled is True
        assert config.router.hidden_dim == 512
        assert config.router.num_epochs == 10
        # Defaults preserved for unset fields
        assert config.router.embedding_dim == 768

    def test_backward_compat_base_yaml(self):
        """configs/base.yaml (no router section) should load with defaults."""
        config = load_config("configs/base.yaml")
        assert config.router.enabled is False
        assert config.router.embedding_dim == 768
        assert config.model.target_model == "gpt2"

    def test_unknown_router_key_raises(self, tmp_path):
        """Unknown key inside router section should raise ValueError."""
        config_data = {
            "router": {
                "nonexistent_field": "value",
            },
        }
        config_file = tmp_path / "bad_router.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="Unknown config key"):
            load_config(config_file)


# ──────────────────────────────────────────────────────────────
# MLP tests
# ──────────────────────────────────────────────────────────────


class TestRouterMLP:
    """Tests for the RouterMLP classifier."""

    def test_forward_shape(self):
        """Forward pass should return correct logit shape."""
        model = RouterMLP(input_dim=10, hidden_dim=8, num_drafts=3)
        x = torch.randn(4, 10)
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_predict(self):
        """Predict should return integer indices."""
        model = RouterMLP(input_dim=10, hidden_dim=8, num_drafts=3)
        x = torch.randn(4, 10)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64
        assert all(0 <= p.item() < 3 for p in preds)

    def test_predict_with_confidence(self):
        """predict_with_confidence should return indices and probabilities."""
        model = RouterMLP(input_dim=10, hidden_dim=8, num_drafts=3)
        x = torch.randn(4, 10)
        indices, confidences = model.predict_with_confidence(x)
        assert indices.shape == (4,)
        assert confidences.shape == (4,)
        # Confidences should be valid probabilities
        assert all(0.0 <= c.item() <= 1.0 for c in confidences)

    def test_save_load_roundtrip(self, tmp_path):
        """Save → load should produce identical outputs."""
        model = RouterMLP(input_dim=10, hidden_dim=8, num_drafts=3)
        model.eval()
        x = torch.randn(2, 10)
        original_out = model(x)

        path = str(tmp_path / "router.pt")
        model.save(path)
        loaded = RouterMLP.load(path)
        loaded_out = loaded(x)

        assert torch.allclose(original_out, loaded_out, atol=1e-6)

    def test_config_preservation(self, tmp_path):
        """Loaded model should preserve architecture config."""
        model = RouterMLP(input_dim=20, hidden_dim=16, num_drafts=5, dropout=0.2)
        path = str(tmp_path / "router.pt")
        model.save(path)
        loaded = RouterMLP.load(path)

        assert loaded.input_dim == 20
        assert loaded.hidden_dim == 16
        assert loaded.num_drafts == 5
        assert loaded.dropout_rate == 0.2


# ──────────────────────────────────────────────────────────────
# Domain hint tests (no embedding model needed)
# ──────────────────────────────────────────────────────────────


class TestDomainHints:
    """Tests for the domain hint heuristics in FeatureExtractor."""

    def test_code_detection(self):
        """Code-like prompts should have high code score."""
        code_prompt = 'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n    return n'
        hints = FeatureExtractor._compute_domain_hints(code_prompt)
        # hints = [code, math, conversation, factuality]
        assert hints[0] > 0.5  # code score should be high
        assert hints[0] >= hints[2]  # code > conversation

    def test_math_detection(self):
        """Math prompts should have high math score."""
        math_prompt = "Solve the equation: calculate the sum of 15 + 23 and find the product"
        hints = FeatureExtractor._compute_domain_hints(math_prompt)
        assert hints[1] > 0.5  # math score should be high

    def test_conversation_detection(self):
        """Conversational prompts should have high conversation score."""
        conv_prompt = "Please explain to me how quantum computing works and describe its benefits"
        hints = FeatureExtractor._compute_domain_hints(conv_prompt)
        assert hints[2] > 0.5  # conversation score should be high

    def test_factuality_detection(self):
        """Factual queries should have high factuality score."""
        fact_prompt = "What is the capital of France? Is it true that Paris is the largest city?"
        hints = FeatureExtractor._compute_domain_hints(fact_prompt)
        assert hints[3] > 0.5  # factuality score should be high

    def test_scores_bounded(self):
        """All scores should be in [0, 1]."""
        prompts = [
            "def foo(): pass",
            "What is 2+2?",
            "Tell me a story",
            "",
            "a" * 10000,
        ]
        for prompt in prompts:
            hints = FeatureExtractor._compute_domain_hints(prompt)
            assert len(hints) == 4
            for score in hints:
                assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────
# Trainer test with synthetic data
# ──────────────────────────────────────────────────────────────


class TestRouterTrainer:
    """Test router training with synthetic separable data."""

    def test_training_improves_accuracy(self, tmp_path):
        """Training on clearly separable data should achieve >90% accuracy."""
        from src.routing.data_collection import RoutingSample
        from src.routing.trainer import RouterTrainer

        torch.manual_seed(42)

        # Generate synthetic separable data: 3 classes with distinct feature clusters
        num_per_class = 200
        feature_dim = 10
        samples = []

        for class_id in range(3):
            for _ in range(num_per_class):
                # Each class has features centered at different locations
                features = torch.randn(feature_dim).tolist()
                # Make features[class_id * 3] and features[class_id * 3 + 1] large
                # to create clear separation
                offset = class_id * 3
                for j in range(min(3, feature_dim - offset)):
                    features[offset + j] += 5.0

                samples.append(RoutingSample(
                    prompt=f"sample_{class_id}",
                    domain="synthetic",
                    features=features,
                    acceptance_rates={"a": 0.0, "b": 0.0, "c": 0.0},
                    best_draft=["a", "b", "c"][class_id],
                    best_draft_id=class_id,
                ))

        # Configure training
        config = ExperimentConfig()
        config.router.hidden_dim = 32
        config.router.num_epochs = 10
        config.router.learning_rate = 1e-2
        config.router.batch_size = 32
        config.router.router_checkpoint = str(tmp_path / "router.pt")

        trainer = RouterTrainer(config)
        model = trainer.train(samples=samples)

        # Evaluate on the training data (should be very accurate given clear separation)
        all_features = torch.tensor([s.features for s in samples], dtype=torch.float32)
        all_labels = torch.tensor([s.best_draft_id for s in samples])

        model.eval()
        with torch.no_grad():
            preds = model.predict(all_features.to(trainer.device))
            accuracy = (preds.cpu() == all_labels).float().mean().item()

        assert accuracy > 0.90, f"Expected >90% accuracy, got {accuracy:.1%}"

    def test_model_saves_correctly(self, tmp_path):
        """Trained model should be loadable from checkpoint."""
        from src.routing.data_collection import RoutingSample
        from src.routing.trainer import RouterTrainer

        torch.manual_seed(0)
        feature_dim = 5
        samples = []
        for class_id in range(2):
            for _ in range(50):
                features = torch.randn(feature_dim).tolist()
                features[class_id * 2] += 3.0
                samples.append(RoutingSample(
                    prompt="test",
                    domain="test",
                    features=features,
                    acceptance_rates={"x": 0.0, "y": 0.0},
                    best_draft=["x", "y"][class_id],
                    best_draft_id=class_id,
                ))

        config = ExperimentConfig()
        config.router.hidden_dim = 16
        config.router.num_epochs = 3
        config.router.batch_size = 16
        config.router.router_checkpoint = str(tmp_path / "router.pt")

        trainer = RouterTrainer(config)
        trainer.train(samples=samples)

        # Load from checkpoint
        loaded = RouterMLP.load(str(tmp_path / "router.pt"))
        assert loaded.input_dim == feature_dim
        assert loaded.num_drafts == 2
