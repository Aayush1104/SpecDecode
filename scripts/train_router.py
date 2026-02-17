#!/usr/bin/env python3
"""Train the adaptive router MLP.

Usage:
    python3 scripts/train_router.py --config configs/router_train.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing.trainer import RouterTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train router MLP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_train.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override training data path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(
        level=config.logging.level,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
    )

    trainer = RouterTrainer(config)
    model = trainer.train(data_path=args.data_path)

    logger.info(
        "Router training complete. Model saved to %s",
        config.router.router_checkpoint,
    )


if __name__ == "__main__":
    main()
