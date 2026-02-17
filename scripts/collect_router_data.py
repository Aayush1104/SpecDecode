#!/usr/bin/env python3
"""Collect router training data by benchmarking all draft models.

Usage:
    python3 scripts/collect_router_data.py --config configs/router_collect.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing.data_collection import collect_router_data, save_router_data
from src.routing.features import FeatureExtractor
from src.speculative.backends import create_backend
from src.utils.config import load_config
from src.utils.logging import setup_logging
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Collect router training data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_collect.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
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

    # Load all draft models
    draft_models = {}
    for name, path in config.router.draft_models.items():
        logger.info("Loading draft model '%s' from: %s", name, path)
        draft_models[name] = create_backend(
            path,
            backend_type=config.model.backend,
            dtype=config.model.draft_dtype,
            device=config.model.device,
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Feature extractor
    logger.info("Loading embedding model: %s", config.router.embedding_model)
    feature_extractor = FeatureExtractor(
        embedding_model_name=config.router.embedding_model,
        embedding_dim=config.router.embedding_dim,
    )

    # Collect data
    samples = collect_router_data(
        target_model=target,
        draft_models=draft_models,
        feature_extractor=feature_extractor,
        config=config,
        tokenizer=tokenizer,
    )

    # Save
    save_router_data(samples, config.router.training_data_path)
    logger.info("Data collection complete. %d samples saved.", len(samples))


if __name__ == "__main__":
    main()
