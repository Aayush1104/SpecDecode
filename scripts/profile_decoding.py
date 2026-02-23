#!/usr/bin/env python3
"""Profile speculative decoding to get per-phase timing breakdown.

Runs speculative decoding with profile=True on a small sample
and prints a detailed timing breakdown.

Usage:
    python3 scripts/profile_decoding.py --config configs/benchmark_profile.yaml
    python3 scripts/profile_decoding.py --config configs/benchmark_profile.yaml --torch-profile
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.evaluation.datasets import load_dataset_prompts
from src.speculative.backends import create_backend
from src.speculative.decoding import speculative_decode
from src.utils.config import load_config
from src.utils.logging import setup_logging
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Profile speculative decoding")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_profile.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to profile",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Also run PyTorch profiler for detailed trace",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(level=config.logging.level)

    # Load models
    logger.info("Loading target model: %s", config.model.target_model)
    target = create_backend(
        config.model.target_model,
        backend_type=config.model.backend,
        dtype=config.model.target_dtype,
        device=config.model.device,
    )

    draft_name = config.benchmark.generic_draft_model
    logger.info("Loading draft model: %s", draft_name)
    draft = create_backend(
        draft_name,
        backend_type=config.model.backend,
        dtype=config.model.draft_dtype,
        device=config.model.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a small dataset
    dataset_name = config.eval.datasets[0] if config.eval.datasets else "humaneval"
    samples = load_dataset_prompts(
        dataset_name, num_samples=args.num_samples, seed=config.eval.seed
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Profile runs
    logger.info("Running %d profiled samples...", len(samples))
    total_draft = 0.0
    total_verify = 0.0
    total_sampling = 0.0
    total_overhead = 0.0
    total_latency = 0.0

    for i, sample in enumerate(samples):
        input_ids = tokenizer.encode(
            sample["prompt"], return_tensors="pt"
        ).to(device)

        _, metrics = speculative_decode(
            target_model=target,
            draft_model=draft,
            input_ids=input_ids,
            max_new_tokens=config.decoding.max_new_tokens,
            speculation_length=config.decoding.speculation_length,
            temperature=config.decoding.temperature,
            profile=True,
        )

        total_draft += metrics.draft_time
        total_verify += metrics.verify_time
        total_sampling += metrics.sampling_time
        total_overhead += metrics.overhead_time
        total_latency += metrics.latency_seconds

        logger.info(
            "  Sample %d: %.3fs total, draft=%.3fs verify=%.3fs "
            "sampling=%.3fs overhead=%.3fs (%.1f tok/s)",
            i + 1, metrics.latency_seconds,
            metrics.draft_time, metrics.verify_time,
            metrics.sampling_time, metrics.overhead_time,
            metrics.tokens_per_second,
        )

    # Summary
    print("\n" + "=" * 60)
    print("  PROFILING SUMMARY")
    print("=" * 60)
    print(f"  Samples:          {len(samples)}")
    print(f"  Total latency:    {total_latency:.3f}s")
    print(f"  ---")
    print(f"  Draft phase:      {total_draft:.3f}s ({total_draft/total_latency*100:.1f}%)")
    print(f"  Verify phase:     {total_verify:.3f}s ({total_verify/total_latency*100:.1f}%)")
    print(f"  Sampling phase:   {total_sampling:.3f}s ({total_sampling/total_latency*100:.1f}%)")
    print(f"  Overhead:         {total_overhead:.3f}s ({total_overhead/total_latency*100:.1f}%)")
    profiled_total = total_draft + total_verify + total_sampling + total_overhead
    unaccounted = total_latency - profiled_total
    print(f"  Unaccounted:      {unaccounted:.3f}s ({unaccounted/total_latency*100:.1f}%)")
    print("=" * 60)

    # Optional: PyTorch profiler trace
    if args.torch_profile and samples:
        logger.info("Running PyTorch profiler...")
        input_ids = tokenizer.encode(
            samples[0]["prompt"], return_tensors="pt"
        ).to(device)

        trace_dir = str(Path(config.benchmark.output_dir) / "profile_trace")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ] + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            speculative_decode(
                target_model=target,
                draft_model=draft,
                input_ids=input_ids,
                max_new_tokens=config.decoding.max_new_tokens,
                speculation_length=config.decoding.speculation_length,
                temperature=config.decoding.temperature,
                profile=True,
            )

        # Print key averages
        print("\nPyTorch Profiler Key Averages:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        # Export trace
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        trace_path = str(Path(trace_dir) / "trace.json")
        prof.export_chrome_trace(trace_path)
        logger.info("Trace exported to %s", trace_path)


if __name__ == "__main__":
    main()
