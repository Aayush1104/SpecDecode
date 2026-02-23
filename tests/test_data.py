"""Tests for the data pipeline (no network required)."""

import pytest
import torch

from src.utils.config import DataConfig
from src.draft_models.data import (
    DomainDataPipeline,
    InMemoryDataset,
    PackedDataset,
    DOMAIN_REGISTRY,
)


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # Return deterministic token IDs based on text length
        return list(range(1, len(text) + 1))


class TestPackedDataset:
    """Tests for the PackedDataset chunking logic."""

    def test_chunks_correct_length(self):
        """Each chunk should be exactly max_seq_length."""
        tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 5  # 50 tokens total
        ds = PackedDataset(iter(tokens), max_seq_length=8)
        chunks = list(ds)
        for chunk in chunks:
            assert chunk.shape == (8,)
            assert chunk.dtype == torch.long

    def test_leftover_tokens_discarded(self):
        """Tokens that don't fill a complete chunk should be discarded."""
        tokens = [[1, 2, 3, 4, 5]]  # 5 tokens, seq_length=3 → 1 chunk + 2 leftover
        ds = PackedDataset(iter(tokens), max_seq_length=3)
        chunks = list(ds)
        assert len(chunks) == 1
        assert chunks[0].tolist() == [1, 2, 3]

    def test_empty_input(self):
        """Empty token stream should produce no chunks."""
        ds = PackedDataset(iter([]), max_seq_length=4)
        chunks = list(ds)
        assert len(chunks) == 0

    def test_exact_multiple(self):
        """Token count that's exact multiple of seq_length."""
        tokens = [[1, 2, 3, 4, 5, 6]]
        ds = PackedDataset(iter(tokens), max_seq_length=3)
        chunks = list(ds)
        assert len(chunks) == 2
        assert chunks[0].tolist() == [1, 2, 3]
        assert chunks[1].tolist() == [4, 5, 6]


class TestInMemoryDataset:
    """Tests for InMemoryDataset."""

    def test_len_and_getitem(self):
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        ds = InMemoryDataset(tensors)
        assert len(ds) == 2
        assert ds[0].tolist() == [1, 2, 3]
        assert ds[1].tolist() == [4, 5, 6]


class TestDomainRegistry:
    """Test domain registry completeness."""

    def test_code_domain_registered(self):
        assert "code" in DOMAIN_REGISTRY
        assert DOMAIN_REGISTRY["code"]["text_field"] == "content"

    def test_chat_domain_registered(self):
        assert "chat" in DOMAIN_REGISTRY
        assert DOMAIN_REGISTRY["chat"]["text_field"] == "chosen"

    def test_reasoning_domain_registered(self):
        assert "reasoning" in DOMAIN_REGISTRY
        assert "concat_fields" in DOMAIN_REGISTRY["reasoning"]


class TestDomainDataPipeline:
    """Tests for DomainDataPipeline with synthetic data."""

    def test_extract_text_simple(self):
        """Text extraction from a single field."""
        config = DataConfig(domain="code")
        tokenizer = FakeTokenizer()
        pipeline = DomainDataPipeline(config, tokenizer)
        domain_info = {"text_field": "content"}
        text = pipeline._extract_text({"content": "hello world"}, domain_info)
        assert text == "hello world"

    def test_extract_text_concat(self):
        """Text extraction with concat fields."""
        config = DataConfig(domain="reasoning")
        tokenizer = FakeTokenizer()
        pipeline = DomainDataPipeline(config, tokenizer)
        domain_info = {"concat_fields": ["question", "answer"]}
        text = pipeline._extract_text(
            {"question": "What is 2+2?", "answer": "4"}, domain_info
        )
        assert "What is 2+2?" in text
        assert "4" in text

    def test_extract_text_missing_field(self):
        """Missing field should return empty string."""
        config = DataConfig(domain="code")
        tokenizer = FakeTokenizer()
        pipeline = DomainDataPipeline(config, tokenizer)
        domain_info = {"text_field": "content"}
        text = pipeline._extract_text({}, domain_info)
        assert text == ""

    def test_get_domain_info_known(self):
        """Known domain should return registry info."""
        config = DataConfig(domain="code")
        tokenizer = FakeTokenizer()
        pipeline = DomainDataPipeline(config, tokenizer)
        info = pipeline._get_domain_info()
        assert info["dataset_name"] == "transformersbook/codeparrot"
        assert info["subset"] is None

    def test_get_domain_info_custom(self):
        """Unknown domain should fall back to config values."""
        config = DataConfig(domain="custom", dataset_name="my/dataset")
        tokenizer = FakeTokenizer()
        pipeline = DomainDataPipeline(config, tokenizer)
        info = pipeline._get_domain_info()
        assert info["dataset_name"] == "my/dataset"
        assert info["text_field"] == "text"

    def test_tokenize_stream_respects_max_tokens(self):
        """Token stream should stop after max_tokens."""
        config = DataConfig(max_tokens=20)
        tokenizer = FakeTokenizer()
        pipeline = DomainDataPipeline(config, tokenizer)

        # Each "example" has 5 chars → 5 tokens + 1 EOS = 6 tokens each
        examples = [{"text": "hello"} for _ in range(10)]
        domain_info = {"text_field": "text"}

        all_tokens = list(
            pipeline._tokenize_stream(iter(examples), domain_info, max_tokens=20)
        )
        total = sum(len(t) for t in all_tokens)
        # Should stop at or slightly after 20 tokens
        assert total >= 6  # At least one example
        assert total <= 30  # Not too many
