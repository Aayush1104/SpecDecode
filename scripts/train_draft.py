#!/usr/bin/env python3
"""Train a draft model with optional knowledge distillation.

Usage:
    accelerate launch scripts/train_draft.py --config configs/train_code.yaml
    accelerate launch --config_file accelerate_config.yaml scripts/train_draft.py --config configs/distill_code.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.draft_models.data import DomainDataPipeline
from src.draft_models.distiller import DistillationTrainer
from src.draft_models.trainer import DraftModelTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train draft model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(
        level=config.logging.level,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
    )

    # Always use target model's tokenizer (shared tokenizer family)
    logger.info("Loading tokenizer from: %s", config.model.target_model)
    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build data pipeline
    logger.info("Building data pipeline for domain: %s", config.data.domain)
    pipeline = DomainDataPipeline(config.data, tokenizer)
    train_dl, val_dl = pipeline.create_dataloaders(
        batch_size=config.training.per_device_batch_size
    )

    # Load draft (student) model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    draft_dtype = dtype_map.get(config.training.dtype, torch.bfloat16)

    logger.info("Loading draft model: %s", config.model.draft_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        config.model.draft_model, torch_dtype=draft_dtype
    )

    if config.distillation.enabled:
        # Load teacher model
        teacher_dtype = dtype_map.get(config.distillation.teacher_dtype, torch.bfloat16)
        teacher_name = config.distillation.teacher_model
        logger.info("Loading teacher model: %s", teacher_name)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=teacher_dtype,
            device_map="auto",
        )

        trainer = DistillationTrainer(
            config=config,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
    else:
        trainer = DraftModelTrainer(
            config=config,
            model=student_model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )

    logger.info("Starting training for %d steps", config.training.num_train_steps)
    trainer.train()
    logger.info("Done!")


if __name__ == "__main__":
    main()
