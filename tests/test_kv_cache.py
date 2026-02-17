"""Tests for KV cache management."""

import pytest
import torch
from transformers import DynamicCache

from src.speculative.kv_cache import (
    get_kv_cache_length,
    trim_kv_cache,
    validate_kv_cache,
)


class TestTrimKVCache:
    """Tests for KV cache trimming."""

    def test_trim_tuple_format(self):
        """Trim legacy tuple-format KV cache."""
        num_layers = 4
        batch_size = 1
        num_heads = 8
        seq_len = 20
        head_dim = 64

        kv = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

        trimmed = trim_kv_cache(kv, target_length=10)

        assert len(trimmed) == num_layers
        for key, value in trimmed:
            assert key.shape == (batch_size, num_heads, 10, head_dim)
            assert value.shape == (batch_size, num_heads, 10, head_dim)

    def test_trim_preserves_content(self):
        """Trimming should preserve the first target_length entries."""
        key = torch.arange(20).float().reshape(1, 1, 20, 1)
        value = torch.arange(20).float().reshape(1, 1, 20, 1) * 2
        kv = ((key, value),)

        trimmed = trim_kv_cache(kv, target_length=5)

        expected_key = torch.arange(5).float().reshape(1, 1, 5, 1)
        expected_value = torch.arange(5).float().reshape(1, 1, 5, 1) * 2
        assert torch.equal(trimmed[0][0], expected_key)
        assert torch.equal(trimmed[0][1], expected_value)

    def test_trim_dynamic_cache(self):
        """Trim DynamicCache format."""
        cache = DynamicCache()
        num_layers = 4
        batch_size = 1
        num_heads = 8
        seq_len = 15
        head_dim = 64

        for layer in range(num_layers):
            key = torch.randn(batch_size, num_heads, seq_len, head_dim)
            value = torch.randn(batch_size, num_heads, seq_len, head_dim)
            cache.update(key, value, layer)

        trimmed = trim_kv_cache(cache, target_length=7)

        assert isinstance(trimmed, DynamicCache)
        assert len(trimmed) == num_layers
        for layer in range(num_layers):
            assert trimmed.layers[layer].keys.shape[2] == 7
            assert trimmed.layers[layer].values.shape[2] == 7

    def test_trim_none(self):
        """Trimming None should return None."""
        assert trim_kv_cache(None, 5) is None

    def test_trim_to_zero(self):
        """Trimming to length 0 should produce empty sequence dim."""
        key = torch.randn(1, 2, 10, 4)
        value = torch.randn(1, 2, 10, 4)
        kv = ((key, value),)

        trimmed = trim_kv_cache(kv, target_length=0)
        assert trimmed[0][0].shape[2] == 0

    def test_trim_to_same_length(self):
        """Trimming to current length should be a no-op."""
        key = torch.randn(1, 2, 10, 4)
        value = torch.randn(1, 2, 10, 4)
        kv = ((key, value),)

        trimmed = trim_kv_cache(kv, target_length=10)
        assert torch.equal(trimmed[0][0], key)


class TestValidateKVCache:
    """Tests for KV cache validation."""

    def test_valid_tuple_cache(self):
        """Correctly validate a tuple-format cache."""
        kv = tuple(
            (torch.randn(1, 4, 10, 8), torch.randn(1, 4, 10, 8))
            for _ in range(3)
        )
        assert validate_kv_cache(kv, expected_length=10) is True
        assert validate_kv_cache(kv, expected_length=5) is False

    def test_valid_dynamic_cache(self):
        """Correctly validate a DynamicCache."""
        cache = DynamicCache()
        for layer in range(3):
            cache.update(
                torch.randn(1, 4, 10, 8),
                torch.randn(1, 4, 10, 8),
                layer,
            )
        assert validate_kv_cache(cache, expected_length=10) is True
        assert validate_kv_cache(cache, expected_length=5) is False

    def test_none_cache(self):
        """None cache should be valid for expected_length=0."""
        assert validate_kv_cache(None, expected_length=0) is True
        assert validate_kv_cache(None, expected_length=5) is False


class TestGetKVCacheLength:
    """Tests for getting KV cache sequence length."""

    def test_tuple_format(self):
        kv = ((torch.randn(1, 2, 15, 4), torch.randn(1, 2, 15, 4)),)
        assert get_kv_cache_length(kv) == 15

    def test_dynamic_cache(self):
        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 12, 4), torch.randn(1, 2, 12, 4), 0)
        assert get_kv_cache_length(cache) == 12

    def test_none(self):
        assert get_kv_cache_length(None) == 0
