"""Command-line interface for specdecode.

Usage:
    specdecode generate  --model <target> --draft <draft> --prompt "..."
    specdecode benchmark --config configs/benchmark.yaml
    specdecode profile   --config configs/benchmark_profile.yaml
    specdecode evaluate  --config configs/eval_full.yaml
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="specdecode",
        description="Speculative decoding for 2-3x LLM inference speedup",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── generate ─────────────────────────────────────────────
    gen = subparsers.add_parser(
        "generate",
        help="Generate text using speculative decoding",
    )
    gen.add_argument("--model", type=str, required=True, help="Target model name or path")
    gen.add_argument("--draft", type=str, required=True, help="Draft model name or path")
    gen.add_argument("--prompt", type=str, required=True, help="Input prompt")
    gen.add_argument("--K", type=int, default=5, help="Speculation length (default: 5)")
    gen.add_argument("--max-tokens", type=int, default=128, help="Max new tokens (default: 128)")
    gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    gen.add_argument("--dtype", type=str, default="float16", help="Model dtype (default: float16)")
    gen.add_argument("--device", type=str, default="auto", help="Device (default: auto)")

    # ── benchmark ────────────────────────────────────────────
    bench = subparsers.add_parser(
        "benchmark",
        help="Run the benchmark suite",
    )
    bench.add_argument("--config", type=str, default="configs/benchmark.yaml", help="Config file path")
    bench.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated list of experiments to run (e.g. baseline,generic_draft)",
    )

    # ── profile ──────────────────────────────────────────────
    prof = subparsers.add_parser(
        "profile",
        help="Run profiling with timing breakdown",
    )
    prof.add_argument("--config", type=str, default="configs/benchmark_profile.yaml", help="Config file path")
    prof.add_argument("--num-samples", type=int, default=None, help="Override number of samples")
    prof.add_argument("--torch-profile", action="store_true", help="Enable PyTorch profiler")

    # ── evaluate ─────────────────────────────────────────────
    ev = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on datasets",
    )
    ev.add_argument("--config", type=str, default="configs/eval_full.yaml", help="Config file path")
    ev.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    ev.add_argument("--num-samples", type=int, default=None, help="Override number of samples")

    return parser


def cmd_generate(args):
    """Run speculative decoding on a single prompt."""
    import torch
    from transformers import AutoTokenizer

    from src.speculative.backends import create_backend
    from src.speculative.decoding import speculative_decode

    print(f"Loading target model: {args.model}")
    target = create_backend(args.model, dtype=args.dtype, device=args.device)

    print(f"Loading draft model: {args.draft}")
    draft = create_backend(args.draft, dtype=args.dtype, device=args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    device = next(target.model.parameters()).device
    input_ids = input_ids.to(device)

    print(f"Generating (K={args.K}, max_tokens={args.max_tokens}, temp={args.temperature})...")
    output_ids, metrics = speculative_decode(
        target_model=target,
        draft_model=draft,
        input_ids=input_ids,
        max_new_tokens=args.max_tokens,
        speculation_length=args.K,
        temperature=args.temperature,
        profile=True,
    )

    generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(generated)
    print("\n" + "-" * 60)
    print("Metrics:")
    print(f"  Tokens generated:   {metrics.total_tokens}")
    print(f"  Tokens/second:      {metrics.tokens_per_second:.1f}")
    print(f"  Acceptance rate:    {metrics.acceptance_rate:.1%}")
    print(f"  Latency:            {metrics.latency_seconds:.3f}s")
    print(f"  Peak memory:        {metrics.peak_memory_mb:.1f} MB")
    if metrics.draft_time > 0:
        print(f"  Draft time:         {metrics.draft_time:.3f}s")
        print(f"  Verify time:        {metrics.verify_time:.3f}s")
        print(f"  Sampling time:      {metrics.sampling_time:.3f}s")
        print(f"  Overhead time:      {metrics.overhead_time:.3f}s")
    print("-" * 60)


def cmd_benchmark(args):
    """Run the benchmark suite."""
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer

    from src.evaluation.benchmark import BenchmarkRunner
    from src.evaluation.results import save_results
    from src.evaluation.visualization import generate_all_charts
    from src.speculative.backends import create_backend
    from src.utils.config import load_config
    from src.utils.logging import setup_logging

    config = load_config(args.config)
    if args.experiments:
        config.benchmark.experiments = [e.strip() for e in args.experiments.split(",")]

    logger = setup_logging(level=config.logging.level)

    logger.info("Loading target model: %s", config.model.target_model)
    target = create_backend(
        config.model.target_model,
        backend_type=config.model.backend,
        dtype=config.model.target_dtype,
        device=config.model.device,
    )

    draft_model = None
    if "generic_draft" in config.benchmark.experiments or "ablation_K" in config.benchmark.experiments:
        logger.info("Loading generic draft: %s", config.benchmark.generic_draft_model)
        draft_model = create_backend(
            config.benchmark.generic_draft_model,
            backend_type=config.model.backend,
            dtype=config.model.draft_dtype,
            device=config.model.device,
        )

    specialized_drafts = {}
    if "specialized_drafts" in config.benchmark.experiments or "adaptive_routing" in config.benchmark.experiments:
        for name, path in config.router.draft_models.items():
            logger.info("Loading specialized draft '%s': %s", name, path)
            specialized_drafts[name] = create_backend(
                path,
                backend_type=config.model.backend,
                dtype=config.model.draft_dtype,
                device=config.model.device,
            )

    router_model = None
    feature_extractor = None
    if "adaptive_routing" in config.benchmark.experiments and config.router.enabled:
        from src.routing.features import FeatureExtractor
        from src.routing.model import RouterMLP

        logger.info("Loading router: %s", config.router.router_checkpoint)
        router_model = RouterMLP.load(config.router.router_checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        router_model = router_model.to(device)

        logger.info("Loading embedding model: %s", config.router.embedding_model)
        feature_extractor = FeatureExtractor(
            embedding_model_name=config.router.embedding_model,
            embedding_dim=config.router.embedding_dim,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    runner = BenchmarkRunner(
        config=config,
        tokenizer=tokenizer,
        target_model=target,
        draft_model=draft_model,
        specialized_drafts=specialized_drafts,
        router_model=router_model,
        feature_extractor=feature_extractor,
    )

    logger.info("Starting benchmark suite")
    results = runner.run_all()

    output_dir = config.benchmark.output_dir
    for exp_name, exp_results in results.items():
        save_results(exp_results, output_dir, prefix=exp_name)

    charts_dir = str(Path(output_dir) / "charts")
    logger.info("Generating charts in %s", charts_dir)
    generate_all_charts(results, charts_dir)

    logger.info("Benchmark complete. Results saved to %s", output_dir)


def cmd_profile(args):
    """Run profiling with timing breakdown."""
    import torch
    from transformers import AutoTokenizer

    from src.evaluation.datasets import load_dataset_prompts
    from src.speculative.backends import create_backend
    from src.speculative.decoding import speculative_decode
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.utils.metrics import MetricsTracker

    config = load_config(args.config)
    config.benchmark.profile = True

    if args.num_samples is not None:
        config.eval.num_samples = args.num_samples

    logger = setup_logging(level=config.logging.level)

    logger.info("Loading target model: %s", config.model.target_model)
    target = create_backend(
        config.model.target_model,
        backend_type=config.model.backend,
        dtype=config.model.target_dtype,
        device=config.model.device,
    )

    logger.info("Loading draft model: %s", config.model.draft_model)
    draft = create_backend(
        config.model.draft_model,
        backend_type=config.model.backend,
        dtype=config.model.draft_dtype,
        device=config.model.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = config.eval.datasets[0] if config.eval.datasets else "humaneval"
    samples = load_dataset_prompts(dataset_name, num_samples=config.eval.num_samples)

    tracker = MetricsTracker()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_torch_profile = args.torch_profile
    profiler_ctx = None
    if use_torch_profile:
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                *([] if not torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CUDA]),
            ],
            record_shapes=True,
        )
        profiler_ctx.__enter__()

    for i, sample in enumerate(samples):
        input_ids = tokenizer.encode(sample["prompt"], return_tensors="pt").to(device)

        _, metrics = speculative_decode(
            target_model=target,
            draft_model=draft,
            input_ids=input_ids,
            max_new_tokens=config.decoding.max_new_tokens,
            speculation_length=config.decoding.speculation_length,
            temperature=config.decoding.temperature,
            profile=True,
        )
        tracker.add_run(metrics, domain=sample.get("domain"))

        logger.info(
            "Sample %d/%d: %.1f tok/s, accept %.1f%%, draft=%.3fs verify=%.3fs",
            i + 1, len(samples),
            metrics.tokens_per_second,
            metrics.acceptance_rate * 100,
            metrics.draft_time,
            metrics.verify_time,
        )

    if use_torch_profile and profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        print("\nPyTorch Profiler Summary:")
        print(profiler_ctx.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    summary = tracker.summary()
    print("\n" + "=" * 60)
    print("Profiling Summary")
    print("=" * 60)
    print(f"  Samples:            {summary['num_runs']}")
    print(f"  Mean tokens/sec:    {summary['mean_tokens_per_second']:.1f}")
    print(f"  Mean acceptance:    {summary['mean_acceptance_rate']:.1%}")
    print(f"  Mean latency:       {summary['mean_latency_seconds']:.3f}s")
    print(f"  Latency p50:        {summary['latency_p50']:.3f}s")
    print(f"  Latency p95:        {summary['latency_p95']:.3f}s")
    print(f"  Latency p99:        {summary['latency_p99']:.3f}s")

    # Aggregate phase timing
    total_draft = sum(r.draft_time for r in tracker.runs)
    total_verify = sum(r.verify_time for r in tracker.runs)
    total_sampling = sum(r.sampling_time for r in tracker.runs)
    total_overhead = sum(r.overhead_time for r in tracker.runs)
    total_time = total_draft + total_verify + total_sampling + total_overhead

    if total_time > 0:
        print("\n  Phase Breakdown:")
        print(f"    Draft:            {total_draft:.3f}s ({total_draft/total_time:.1%})")
        print(f"    Verify:           {total_verify:.3f}s ({total_verify/total_time:.1%})")
        print(f"    Sampling:         {total_sampling:.3f}s ({total_sampling/total_time:.1%})")
        print(f"    Overhead:         {total_overhead:.3f}s ({total_overhead/total_time:.1%})")

    if summary.get("domain_summary"):
        print("\n  Per-Domain:")
        for domain, ds in summary["domain_summary"].items():
            print(f"    {domain}: {ds['mean_tokens_per_second']:.1f} tok/s, "
                  f"accept {ds['mean_acceptance_rate']:.1%}")

    print("=" * 60)


def cmd_evaluate(args):
    """Run evaluation on datasets."""
    import torch
    from transformers import AutoTokenizer

    from src.evaluation.evaluator import Evaluator
    from src.evaluation.results import save_results
    from src.speculative.backends import create_backend
    from src.utils.config import load_config
    from src.utils.logging import setup_logging

    config = load_config(args.config)
    logger = setup_logging(level=config.logging.level)

    logger.info("Loading target model: %s", config.model.target_model)
    target = create_backend(
        config.model.target_model,
        backend_type=config.model.backend,
        dtype=config.model.target_dtype,
        device=config.model.device,
    )

    logger.info("Loading draft model: %s", config.model.draft_model)
    draft = create_backend(
        config.model.draft_model,
        backend_type=config.model.backend,
        dtype=config.model.draft_dtype,
        device=config.model.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    evaluator = Evaluator(config, tokenizer)
    datasets = [args.dataset] if args.dataset else config.eval.datasets

    for dataset_name in datasets:
        logger.info("Evaluating on %s", dataset_name)

        logger.info("Running baseline...")
        baseline = evaluator.evaluate_standard(
            target, dataset_name, num_samples=args.num_samples,
        )

        logger.info("Running speculative...")
        speculative = evaluator.evaluate_speculative(
            target, draft, dataset_name, num_samples=args.num_samples,
        )

        comparison = evaluator.compare(baseline, speculative)

        print("\n" + "=" * 60)
        print(f"  Results: {dataset_name}")
        print("=" * 60)
        print(f"  Baseline:     {comparison['baseline_tokens_per_second']:.1f} tok/s")
        print(f"  Speculative:  {comparison['speculative_tokens_per_second']:.1f} tok/s")
        print(f"  Speedup:      {comparison['speedup']:.2f}x")
        print(f"  Acceptance:   {comparison['acceptance_rate']:.1%}")
        print("=" * 60)

        save_results(
            {"baseline": baseline, "speculative": speculative, "comparison": comparison},
            config.eval.output_dir,
            prefix=f"eval_{dataset_name}",
        )

    logger.info("Evaluation complete")


def main():
    """Main entry point for the specdecode CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "profile": cmd_profile,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
