#!/usr/bin/env python3
"""Focused benchmark to test optimized speculative decoding."""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from src.speculative.backends import create_backend
from src.speculative.decoding import speculative_decode, standard_decode


def run_test(target, draft, tokenizer, prompt, max_tokens, K, temperature, label=""):
    """Run a single comparison test."""
    device = target._device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Standard decoding
    torch.cuda.synchronize()
    std_out, std_metrics = standard_decode(target, input_ids, max_new_tokens=max_tokens,
                                           temperature=temperature)
    torch.cuda.synchronize()

    # Speculative decoding
    torch.cuda.synchronize()
    spec_out, spec_metrics = speculative_decode(target, draft, input_ids,
                                                 max_new_tokens=max_tokens,
                                                 speculation_length=K,
                                                 temperature=temperature,
                                                 profile=True)
    torch.cuda.synchronize()

    speedup = spec_metrics.tokens_per_second / std_metrics.tokens_per_second if std_metrics.tokens_per_second > 0 else 0

    print(f"  {label:<40} | Std: {std_metrics.tokens_per_second:6.1f} tok/s | "
          f"Spec: {spec_metrics.tokens_per_second:6.1f} tok/s | "
          f"Accept: {spec_metrics.acceptance_rate:5.1%} | "
          f"Speedup: {speedup:.2f}x {'**' if speedup > 1.0 else ''}")
    if spec_metrics.draft_time > 0:
        print(f"    Draft: {spec_metrics.draft_time:.3f}s | Verify: {spec_metrics.verify_time:.3f}s | "
              f"Overhead: {spec_metrics.overhead_time:.3f}s | Sampling: {spec_metrics.sampling_time:.3f}s")

    return {
        "label": label, "K": K, "temp": temperature, "max_tokens": max_tokens,
        "std_tps": std_metrics.tokens_per_second,
        "spec_tps": spec_metrics.tokens_per_second,
        "acceptance": spec_metrics.acceptance_rate,
        "speedup": speedup,
        "draft_time": spec_metrics.draft_time,
        "verify_time": spec_metrics.verify_time,
        "overhead_time": spec_metrics.overhead_time,
    }


if __name__ == "__main__":
    prompt_code = "Write a Python function that implements binary search on a sorted list and returns the index of the target element:"
    prompt_story = "Once upon a time in a distant kingdom, there lived a young princess who discovered a hidden library beneath the castle. Inside she found"

    # Load models once
    print("Loading Qwen2.5-7B (target)...")
    target = create_backend("Qwen/Qwen2.5-7B", dtype="bfloat16", device="cuda:0")

    print("Loading Qwen2.5-0.5B (draft)...")
    draft_small = create_backend("Qwen/Qwen2.5-0.5B", dtype="bfloat16", device="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    print("Warming up...")
    device = target._device
    warmup_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = standard_decode(target, warmup_ids, max_new_tokens=8, temperature=0)
        _ = speculative_decode(target, draft_small, warmup_ids, max_new_tokens=8,
                               speculation_length=5, temperature=0)
    torch.cuda.synchronize()

    print("\n" + "=" * 120)
    print("OPTIMIZED BENCHMARK: Qwen2.5-7B + Qwen2.5-0.5B")
    print("=" * 120)

    results = []

    # Test various K values at temp=0
    for K in [3, 4, 5, 6, 7, 8]:
        results.append(run_test(
            target, draft_small, tokenizer,
            prompt_code, max_tokens=128, K=K, temperature=0,
            label=f"code, temp=0, K={K}, 128tok",
        ))

    # Test temp=0 with more tokens
    for max_tok in [64, 128, 256, 512]:
        results.append(run_test(
            target, draft_small, tokenizer,
            prompt_code, max_tokens=max_tok, K=5, temperature=0,
            label=f"code, temp=0, K=5, {max_tok}tok",
        ))

    # Story prompt
    results.append(run_test(
        target, draft_small, tokenizer,
        prompt_story, max_tokens=256, K=5, temperature=0,
        label="story, temp=0, K=5, 256tok",
    ))

    # Now try with 1.5B draft
    print("\nLoading Qwen2.5-1.5B (draft)...")
    del draft_small
    gc.collect()
    torch.cuda.empty_cache()
    draft_med = create_backend("Qwen/Qwen2.5-1.5B", dtype="bfloat16", device="cuda:0")

    # Warmup
    with torch.no_grad():
        _ = speculative_decode(target, draft_med, warmup_ids, max_new_tokens=8,
                               speculation_length=5, temperature=0)
    torch.cuda.synchronize()

    print(f"\n{'='*120}")
    print("Qwen2.5-7B + Qwen2.5-1.5B")
    print("=" * 120)

    for K in [3, 5, 7]:
        results.append(run_test(
            target, draft_med, tokenizer,
            prompt_code, max_tokens=128, K=K, temperature=0,
            label=f"1.5B draft, code, temp=0, K={K}, 128tok",
        ))

    results.append(run_test(
        target, draft_med, tokenizer,
        prompt_code, max_tokens=256, K=5, temperature=0,
        label="1.5B draft, code, temp=0, K=5, 256tok",
    ))

    # Summary
    print(f"\n\n{'='*120}")
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Config':<45} {'Std tok/s':>10} {'Spec tok/s':>10} {'Accept':>8} {'Speedup':>8}")
    print("-" * 90)
    for r in results:
        marker = " **" if r["speedup"] > 1.0 else ""
        print(f"{r['label']:<45} {r['std_tps']:>10.1f} {r['spec_tps']:>10.1f} "
              f"{r['acceptance']:>7.1%} {r['speedup']:>7.2f}x{marker}")
