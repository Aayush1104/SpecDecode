#!/usr/bin/env python3
"""Run speculative decoding on benchmarks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import Evaluator
from src.evaluation.results import print_summary_table, save_results
from src.speculative.backends import create_backend
from src.utils.config import load_config
from src.utils.logging import setup_logging
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding evaluation")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset (default: from config)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override number of samples",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(
        level=config.logging.level,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
    )

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
    all_results = {}

    for dataset_name in datasets:
        logger.info("Evaluating on %s", dataset_name)
        results = evaluator.evaluate_speculative(
            target,
            draft,
            dataset_name,
            num_samples=args.num_samples,
        )
        all_results[dataset_name] = results
        save_results(
            results, config.eval.output_dir, prefix=f"speculative_{dataset_name}"
        )

    logger.info("Speculative decoding evaluation complete")
    return all_results


if __name__ == "__main__":
    main()
