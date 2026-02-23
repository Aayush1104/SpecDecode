#!/usr/bin/env python3
"""Run the comprehensive benchmark suite.

Loads all models (target, generic draft, specialized drafts, router),
runs all configured experiments, saves results, and generates charts.

Usage:
    python3 scripts/run_benchmark.py --config configs/benchmark.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.results import save_results
from src.evaluation.visualization import generate_all_charts
from src.speculative.backends import create_backend
from src.utils.config import load_config
from src.utils.logging import setup_logging
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Override which experiments to run (e.g. baseline generic_draft)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.experiments:
        config.benchmark.experiments = args.experiments

    logger = setup_logging(
        level=config.logging.level,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
    )

    # Load target model
    logger.info("Loading target model: %s", config.model.target_model)
    target = create_backend(
        config.model.target_model,
        backend_type=config.model.backend,
        dtype=config.model.target_dtype,
        device=config.model.device,
    )

    # Load generic draft model
    draft_model = None
    if "generic_draft" in config.benchmark.experiments or "ablation_K" in config.benchmark.experiments:
        draft_name = config.benchmark.generic_draft_model
        logger.info("Loading generic draft model: %s", draft_name)
        draft_model = create_backend(
            draft_name,
            backend_type=config.model.backend,
            dtype=config.model.draft_dtype,
            device=config.model.device,
        )

    # Load specialized draft models
    specialized_drafts = {}
    if ("specialized_drafts" in config.benchmark.experiments
            or "adaptive_routing" in config.benchmark.experiments):
        for name, path in config.router.draft_models.items():
            logger.info("Loading specialized draft '%s' from: %s", name, path)
            specialized_drafts[name] = create_backend(
                path,
                backend_type=config.model.backend,
                dtype=config.model.draft_dtype,
                device=config.model.device,
            )

    # Load router
    router_model = None
    feature_extractor = None
    if "adaptive_routing" in config.benchmark.experiments and config.router.enabled:
        from src.routing.features import FeatureExtractor
        from src.routing.model import RouterMLP

        logger.info("Loading router from: %s", config.router.router_checkpoint)
        router_model = RouterMLP.load(config.router.router_checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        router_model = router_model.to(device)

        logger.info("Loading embedding model: %s", config.router.embedding_model)
        feature_extractor = FeatureExtractor(
            embedding_model_name=config.router.embedding_model,
            embedding_dim=config.router.embedding_dim,
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run benchmark
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

    # Save results
    output_dir = config.benchmark.output_dir
    for exp_name, exp_results in results.items():
        save_results(exp_results, output_dir, prefix=exp_name)

    # Generate charts
    charts_dir = str(Path(output_dir) / "charts")
    logger.info("Generating charts in %s", charts_dir)
    generate_all_charts(results, charts_dir)

    logger.info("Benchmark complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
