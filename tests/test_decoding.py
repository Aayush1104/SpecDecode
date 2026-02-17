"""End-to-end tests for speculative and standard decoding.

Uses GPT-2 (target) and distilgpt2 (draft) for integration testing.
"""

import pytest
import torch

from src.speculative.backends import HuggingFaceBackend
from src.speculative.decoding import speculative_decode, standard_decode
from src.utils.metrics import DecodingMetrics


class TestStandardDecode:
    """Tests for standard autoregressive decoding."""

    @pytest.fixture(scope="class")
    def model(self):
        return HuggingFaceBackend("distilgpt2", dtype="float32", device="cpu")

    def test_generates_correct_length(self, model):
        """Should generate exactly max_new_tokens tokens."""
        input_ids = torch.tensor([[50256]])
        max_new = 10

        output_ids, metrics = standard_decode(
            model, input_ids, max_new_tokens=max_new, temperature=1.0
        )

        assert output_ids.shape[1] == 1 + max_new  # prompt + generated
        assert metrics.total_tokens == max_new

    def test_returns_valid_metrics(self, model):
        """Metrics should be populated correctly."""
        input_ids = torch.tensor([[50256]])

        _, metrics = standard_decode(
            model, input_ids, max_new_tokens=5, temperature=1.0
        )

        assert isinstance(metrics, DecodingMetrics)
        assert metrics.total_tokens == 5
        assert metrics.total_steps == 5
        assert metrics.latency_seconds >= 0

    def test_greedy_is_deterministic(self, model):
        """Greedy decoding should produce same output."""
        input_ids = torch.tensor([[50256]])

        out1, _ = standard_decode(model, input_ids, max_new_tokens=10, temperature=0)
        out2, _ = standard_decode(model, input_ids, max_new_tokens=10, temperature=0)

        assert torch.equal(out1, out2)

    def test_output_contains_prompt(self, model):
        """Output should start with the input prompt."""
        input_ids = torch.tensor([[100, 200, 300]])

        output_ids, _ = standard_decode(
            model, input_ids, max_new_tokens=5, temperature=1.0
        )

        assert torch.equal(output_ids[:, :3], input_ids)


class TestSpeculativeDecode:
    """Tests for speculative decoding."""

    @pytest.fixture(scope="class")
    def models(self):
        target = HuggingFaceBackend("gpt2", dtype="float32", device="cpu")
        draft = HuggingFaceBackend("distilgpt2", dtype="float32", device="cpu")
        return target, draft

    def test_generates_tokens(self, models):
        """Should generate some tokens."""
        target, draft = models
        input_ids = torch.tensor([[50256]])

        output_ids, metrics = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=10,
            speculation_length=3,
            temperature=1.0,
        )

        # Should have generated tokens (at least some)
        assert output_ids.shape[1] > 1
        assert metrics.total_tokens > 0

    def test_returns_valid_metrics(self, models):
        """Metrics should include acceptance rate info."""
        target, draft = models
        input_ids = torch.tensor([[50256]])

        _, metrics = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=10,
            speculation_length=3,
            temperature=1.0,
        )

        assert isinstance(metrics, DecodingMetrics)
        assert metrics.total_steps > 0
        assert metrics.draft_tokens > 0
        assert 0 <= metrics.acceptance_rate <= 1.0
        assert metrics.latency_seconds >= 0

    def test_output_contains_prompt(self, models):
        """Output should start with the input prompt."""
        target, draft = models
        input_ids = torch.tensor([[100, 200, 300]])

        output_ids, _ = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=5,
            speculation_length=2,
            temperature=1.0,
        )

        assert torch.equal(output_ids[:, :3], input_ids)

    def test_speculation_length_1(self, models):
        """Should work with speculation_length=1."""
        target, draft = models
        input_ids = torch.tensor([[50256]])

        output_ids, metrics = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=5,
            speculation_length=1,
            temperature=1.0,
        )

        assert output_ids.shape[1] > 1

    def test_same_model_high_acceptance(self):
        """Using the same model as target and draft should give high acceptance."""
        model = HuggingFaceBackend("distilgpt2", dtype="float32", device="cpu")
        input_ids = torch.tensor([[50256]])

        _, metrics = speculative_decode(
            model, model, input_ids,
            max_new_tokens=20,
            speculation_length=5,
            temperature=0,  # greedy for determinism
        )

        # Same model should accept almost everything
        assert metrics.acceptance_rate > 0.8
