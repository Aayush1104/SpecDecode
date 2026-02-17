"""Tests for model backends.

These tests use small models (GPT-2 family) to verify backend correctness.
They require model downloads and are marked accordingly.
"""

import pytest
import torch

from src.speculative.backends import (
    HuggingFaceBackend,
    VLLMBackend,
    create_backend,
)


# Mark tests that require model downloads
requires_model = pytest.mark.skipif(
    not torch.cuda.is_available() and True,  # Always run on CPU for small models
    reason="",
)


class TestHuggingFaceBackend:
    """Tests for the HuggingFace backend."""

    @pytest.fixture(scope="class")
    def backend(self):
        """Create a small GPT-2 backend for testing."""
        return HuggingFaceBackend("distilgpt2", dtype="float32", device="cpu")

    def test_forward_output_shapes(self, backend):
        """Forward pass should return correct shapes."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits, kv = backend.forward(input_ids)

        assert logits.shape[0] == 1  # batch
        assert logits.shape[1] == 5  # seq_len
        assert logits.shape[2] == backend.get_vocab_size()  # vocab
        assert kv is not None

    def test_forward_with_kv_cache(self, backend):
        """Forward with KV cache should only process new tokens."""
        input_ids = torch.tensor([[1, 2, 3]])
        logits1, kv = backend.forward(input_ids)

        new_ids = torch.tensor([[4]])
        logits2, kv2 = backend.forward(new_ids, past_key_values=kv)

        assert logits2.shape[1] == 1  # only new token

    def test_generate_with_logits(self, backend):
        """Generate should produce tokens and logits."""
        input_ids = torch.tensor([[1, 2, 3]])
        tokens, logits, kv = backend.generate_with_logits(
            input_ids, num_tokens=5, temperature=1.0
        )

        assert tokens.shape == (1, 5)
        assert logits.shape == (1, 5, backend.get_vocab_size())
        assert kv is not None

    def test_generate_greedy(self, backend):
        """Greedy generation (temperature=0) should be deterministic."""
        input_ids = torch.tensor([[50256]])  # EOS token
        tokens1, _, _ = backend.generate_with_logits(
            input_ids, num_tokens=5, temperature=0
        )
        tokens2, _, _ = backend.generate_with_logits(
            input_ids, num_tokens=5, temperature=0
        )
        assert torch.equal(tokens1, tokens2)

    def test_vocab_size(self, backend):
        """Vocab size should be positive."""
        assert backend.get_vocab_size() > 0


class TestVLLMBackend:
    """Tests for the vLLM backend stub."""

    def test_raises_not_implemented(self):
        """vLLM backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            VLLMBackend("gpt2")


class TestCreateBackend:
    """Tests for the backend factory function."""

    def test_create_huggingface(self):
        """Factory should create HuggingFace backend."""
        backend = create_backend("distilgpt2", backend_type="huggingface", dtype="float32", device="cpu")
        assert isinstance(backend, HuggingFaceBackend)

    def test_create_vllm_raises(self):
        """Factory with vLLM should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_backend("gpt2", backend_type="vllm")

    def test_create_unknown_raises(self):
        """Factory with unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("gpt2", backend_type="nonexistent")
