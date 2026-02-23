#!/usr/bin/env python3
"""Benchmark with torch.compile to reduce per-call overhead."""

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
    device = target._device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    torch.cuda.synchronize()
    std_out, std_metrics = standard_decode(target, input_ids, max_new_tokens=max_tokens,
                                           temperature=temperature)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    spec_out, spec_metrics = speculative_decode(target, draft, input_ids,
                                                 max_new_tokens=max_tokens,
                                                 speculation_length=K,
                                                 temperature=temperature,
                                                 profile=True)
    torch.cuda.synchronize()

    speedup = spec_metrics.tokens_per_second / std_metrics.tokens_per_second if std_metrics.tokens_per_second > 0 else 0

    print(f"  {label:<45} | Std: {std_metrics.tokens_per_second:6.1f} tok/s | "
          f"Spec: {spec_metrics.tokens_per_second:6.1f} tok/s | "
          f"Accept: {spec_metrics.acceptance_rate:5.1%} | "
          f"Speedup: {speedup:.2f}x {'**' if speedup > 1.0 else ''}")
    if spec_metrics.draft_time > 0:
        print(f"    Draft: {spec_metrics.draft_time:.3f}s | Verify: {spec_metrics.verify_time:.3f}s | "
              f"Overhead: {spec_metrics.overhead_time:.3f}s | Sampling: {spec_metrics.sampling_time:.3f}s")

    return speedup, spec_metrics


if __name__ == "__main__":
    prompt = "Write a Python function that implements binary search on a sorted list and returns the index of the target element:"

    # Load models
    print("Loading Qwen2.5-7B target...")
    target = create_backend("Qwen/Qwen2.5-7B", dtype="bfloat16", device="cuda:0")

    print("Loading Qwen2.5-0.5B draft...")
    draft = create_backend("Qwen/Qwen2.5-0.5B", dtype="bfloat16", device="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warmup without compile
    print("\nWarming up (no compile)...")
    device = target._device
    warmup_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = standard_decode(target, warmup_ids, max_new_tokens=8, temperature=0)
        _ = speculative_decode(target, draft, warmup_ids, max_new_tokens=8,
                               speculation_length=5, temperature=0)
    torch.cuda.synchronize()

    print("\n=== BASELINE (no compile) ===")
    run_test(target, draft, tokenizer, prompt, 128, 5, 0, "baseline K=5, 128tok")

    # Compile the models
    print("\nCompiling target model...")
    target.model = torch.compile(target.model, mode="reduce-overhead")
    print("Compiling draft model...")
    draft.model = torch.compile(draft.model, mode="reduce-overhead")

    # Warmup compiled models (first few calls compile/trace)
    print("Warming up compiled models (this takes a while)...")
    with torch.no_grad():
        for _ in range(3):
            _ = standard_decode(target, warmup_ids, max_new_tokens=16, temperature=0)
            _ = speculative_decode(target, draft, warmup_ids, max_new_tokens=16,
                                   speculation_length=5, temperature=0)
    torch.cuda.synchronize()

    print("\n=== WITH torch.compile (reduce-overhead) ===")
    results = []
    for K in [3, 5, 7]:
        run_test(target, draft, tokenizer, prompt, 128, K, 0,
                 f"compiled K={K}, 128tok")

    run_test(target, draft, tokenizer, prompt, 256, 5, 0,
             "compiled K=5, 256tok")
    run_test(target, draft, tokenizer, prompt, 512, 5, 0,
             "compiled K=5, 512tok")
