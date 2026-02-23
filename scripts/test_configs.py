#!/usr/bin/env python3
"""Test different configurations to find where speculative decoding wins."""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from src.speculative.backends import create_backend
from src.speculative.decoding import speculative_decode, standard_decode


def run_test(target_name, draft_name, prompt, max_tokens, K, temperature, dtype="bfloat16"):
    """Run a single comparison test."""
    print(f"\n{'='*70}")
    print(f"Target: {target_name}")
    print(f"Draft:  {draft_name}")
    print(f"K={K}, max_tokens={max_tokens}, temp={temperature}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'='*70}")

    target = create_backend(target_name, dtype=dtype, device="cuda:0")
    draft = create_backend(draft_name, dtype=dtype, device="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = standard_decode(target, input_ids, max_new_tokens=8, temperature=temperature)
        _ = speculative_decode(target, draft, input_ids, max_new_tokens=8,
                               speculation_length=K, temperature=temperature)

    torch.cuda.synchronize()

    # Standard decoding
    print("Running standard decoding...")
    torch.cuda.synchronize()
    std_out, std_metrics = standard_decode(target, input_ids, max_new_tokens=max_tokens,
                                           temperature=temperature)
    torch.cuda.synchronize()
    std_text = tokenizer.decode(std_out[0, input_ids.shape[1]:], skip_special_tokens=True)

    # Speculative decoding
    print("Running speculative decoding...")
    torch.cuda.synchronize()
    spec_out, spec_metrics = speculative_decode(target, draft, input_ids,
                                                 max_new_tokens=max_tokens,
                                                 speculation_length=K,
                                                 temperature=temperature,
                                                 profile=True)
    torch.cuda.synchronize()
    spec_text = tokenizer.decode(spec_out[0, input_ids.shape[1]:], skip_special_tokens=True)

    speedup = spec_metrics.tokens_per_second / std_metrics.tokens_per_second if std_metrics.tokens_per_second > 0 else 0

    print(f"\nResults:")
    print(f"  Standard:    {std_metrics.tokens_per_second:6.1f} tok/s | {std_metrics.latency_seconds:.3f}s | {std_metrics.total_tokens} tokens")
    print(f"  Speculative: {spec_metrics.tokens_per_second:6.1f} tok/s | {spec_metrics.latency_seconds:.3f}s | {spec_metrics.total_tokens} tokens")
    print(f"  Acceptance:  {spec_metrics.acceptance_rate:.1%}")
    print(f"  Speedup:     {speedup:.2f}x {'<-- WINNER' if speedup > 1.0 else ''}")
    if spec_metrics.draft_time > 0:
        print(f"  Draft time:  {spec_metrics.draft_time:.3f}s | Verify: {spec_metrics.verify_time:.3f}s | Overhead: {spec_metrics.overhead_time:.3f}s")

    print(f"\n  Standard output: {std_text[:100]}...")
    print(f"  Spec output:     {spec_text[:100]}...")

    # Cleanup
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "target": target_name, "draft": draft_name,
        "K": K, "temp": temperature, "max_tokens": max_tokens,
        "std_tps": std_metrics.tokens_per_second,
        "spec_tps": spec_metrics.tokens_per_second,
        "acceptance": spec_metrics.acceptance_rate,
        "speedup": speedup,
    }


if __name__ == "__main__":
    prompt_code = "Write a Python function that implements binary search on a sorted list and returns the index of the target element:"
    prompt_story = "Once upon a time in a distant kingdom, there lived a young princess who discovered a hidden library beneath the castle. Inside she found"

    results = []

    # Test 1: Qwen 7B + 0.5B, temp=0 (greedy - highest acceptance)
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B",
        prompt_code, max_tokens=128, K=5, temperature=0,
    ))

    # Test 2: Qwen 7B + 0.5B, temp=0.5
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B",
        prompt_code, max_tokens=128, K=5, temperature=0.5,
    ))

    # Test 3: Qwen 7B + 1.5B, temp=0 (better draft)
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-1.5B",
        prompt_code, max_tokens=128, K=5, temperature=0,
    ))

    # Test 4: Qwen 7B + 1.5B, temp=0.5
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-1.5B",
        prompt_code, max_tokens=128, K=5, temperature=0.5,
    ))

    # Test 5: Qwen 7B + 1.5B, temp=0, K=3 (less speculation)
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-1.5B",
        prompt_code, max_tokens=128, K=3, temperature=0,
    ))

    # Test 6: Qwen 7B + 1.5B, temp=0, story prompt
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-1.5B",
        prompt_story, max_tokens=128, K=5, temperature=0,
    ))

    # Test 7: Qwen 7B + 1.5B, temp=0, longer output
    results.append(run_test(
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-1.5B",
        prompt_code, max_tokens=256, K=5, temperature=0,
    ))

    # Summary
    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Target':<20} {'Draft':<20} {'K':>3} {'Temp':>5} {'Std tok/s':>10} {'Spec tok/s':>10} {'Accept':>8} {'Speedup':>8}")
    print("-" * 90)
    for r in results:
        marker = " **" if r["speedup"] > 1.0 else ""
        print(f"{r['target']:<20} {r['draft']:<20} {r['K']:>3} {r['temp']:>5.1f} "
              f"{r['std_tps']:>10.1f} {r['spec_tps']:>10.1f} {r['acceptance']:>7.1%} {r['speedup']:>7.2f}x{marker}")
