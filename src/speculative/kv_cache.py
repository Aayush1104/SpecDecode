"""KV cache management for speculative decoding.

Handles trimming KV caches after rejection (discard entries for rejected tokens)
and validation. Supports HuggingFace DynamicCache format.
"""

import torch
from transformers import DynamicCache


def trim_kv_cache(past_key_values, target_length: int):
    """Trim KV cache to target_length sequence positions.

    After rejection at position j, we need to discard KV entries for
    positions beyond the accepted prefix.

    Args:
        past_key_values: KV cache (DynamicCache or tuple of (key, value) tuples)
        target_length: Number of sequence positions to keep

    Returns:
        Trimmed KV cache in the same format
    """
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        return _trim_dynamic_cache(past_key_values, target_length)

    # Legacy tuple format: tuple of (key, value) per layer
    # key/value shape: (batch, num_heads, seq_len, head_dim)
    trimmed = []
    for layer_kv in past_key_values:
        key, value = layer_kv[0], layer_kv[1]
        trimmed_key = key[:, :, :target_length, :]
        trimmed_value = value[:, :, :target_length, :]
        trimmed.append((trimmed_key, trimmed_value))
    return tuple(trimmed)


def _trim_dynamic_cache(cache: DynamicCache, target_length: int) -> DynamicCache:
    """Trim a DynamicCache to target_length in-place."""
    cache.crop(target_length)
    return cache


def get_kv_cache_length(past_key_values) -> int:
    """Get the sequence length stored in a KV cache.

    Returns 0 if past_key_values is None.
    """
    if past_key_values is None:
        return 0

    if isinstance(past_key_values, DynamicCache):
        if len(past_key_values) == 0:
            return 0
        return past_key_values.layers[0].keys.shape[2]

    # Legacy tuple format
    if len(past_key_values) == 0:
        return 0
    return past_key_values[0][0].shape[2]


def validate_kv_cache(past_key_values, expected_length: int) -> bool:
    """Check that all layers in the KV cache have the expected sequence length.

    Args:
        past_key_values: KV cache to validate
        expected_length: Expected sequence dimension size

    Returns:
        True if valid, False otherwise
    """
    if past_key_values is None:
        return expected_length == 0

    if isinstance(past_key_values, DynamicCache):
        for layer_idx in range(len(past_key_values)):
            layer = past_key_values.layers[layer_idx]
            key = layer.keys
            value = layer.values
            if key.shape[2] != expected_length or value.shape[2] != expected_length:
                return False
        return True

    # Legacy tuple format
    for layer_kv in past_key_values:
        key, value = layer_kv[0], layer_kv[1]
        if key.shape[2] != expected_length or value.shape[2] != expected_length:
            return False
    return True
