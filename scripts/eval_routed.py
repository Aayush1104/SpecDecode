#!/usr/bin/env python3
"""Evaluate speculative decoding with adaptive routing.

Usage:
    python3 scripts/eval_routed.py --config configs/eval_routed.yaml
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.evaluation.datasets import load_dataset_prompts
from src.evaluation.results import save_results
from src.routing.features import FeatureExtractor
from src.routing.model import RouterMLP
from src.routing.router import route_and_decode
from src.speculative.backends import create_backend
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.metrics import MetricsTracker
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate with adaptive routing")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_routed.yaml",
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

    # Load trained router
    logger.info("Loading router from: %s", config.router.router_checkpoint)
    router_model = RouterMLP.load(config.router.router_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_model = router_model.to(device)

    draft_names = list(draft_models.keys())
    all_results = {}

    for dataset_name in config.eval.datasets:
        logger.info("Evaluating on %s", dataset_name)
        prompts = load_dataset_prompts(
            dataset_name,
            num_samples=config.eval.num_samples,
            seed=config.eval.seed,
        )

        tracker = MetricsTracker()
        outputs = []
        draft_usage = Counter()

        for i, sample in enumerate(prompts):
            input_ids = tokenizer.encode(
                sample["prompt"], return_tensors="pt"
            ).to(device)

            output_ids, metrics, selected_draft = route_and_decode(
                target_model=target,
                draft_models=draft_models,
                router_model=router_model,
                input_ids=input_ids,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                max_new_tokens=config.decoding.max_new_tokens,
                speculation_length=config.decoding.speculation_length,
                temperature=config.decoding.temperature,
                draft_names=draft_names,
            )

            generated_text = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append({
                "prompt": sample["prompt"],
                "generated": generated_text,
                "domain": sample.get("domain", "unknown"),
                "selected_draft": selected_draft,
                "metrics": metrics.to_dict(),
            })
            tracker.add_run(metrics)
            draft_usage[selected_draft] += 1

            if (i + 1) % 10 == 0:
                logger.info(
                    "  [%s] %d/%d, avg %.1f tok/s, accept rate %.1f%%",
                    dataset_name, i + 1, len(prompts),
                    tracker.mean_tokens_per_second(),
                    tracker.mean_acceptance_rate() * 100,
                )

        # Log draft usage distribution
        logger.info(
            "Draft usage for %s: %s",
            dataset_name,
            dict(draft_usage),
        )

        results = {
            "method": "routed_speculative",
            "dataset": dataset_name,
            "summary": tracker.summary(),
            "draft_usage": dict(draft_usage),
            "outputs": outputs,
        }
        all_results[dataset_name] = results
        save_results(results, config.eval.output_dir, prefix=f"routed_{dataset_name}")

    logger.info("Routed evaluation complete")
    return all_results


if __name__ == "__main__":
    main()
