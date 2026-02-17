"""Evaluation runner for speculative decoding."""

from typing import Callable, Optional

import torch

from src.evaluation.datasets import load_dataset_prompts
from src.speculative.backends import ModelBackend
from src.speculative.decoding import speculative_decode, standard_decode
from src.utils.config import ExperimentConfig
from src.utils.logging import get_logger
from src.utils.metrics import DecodingMetrics, MetricsTracker

logger = get_logger()


class Evaluator:
    """Runs evaluation comparing speculative vs standard decoding."""

    def __init__(self, config: ExperimentConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def evaluate_standard(
        self,
        model: ModelBackend,
        dataset_name: str,
        num_samples: Optional[int] = None,
    ) -> dict:
        """Run standard autoregressive decoding on a dataset."""
        samples = load_dataset_prompts(
            dataset_name,
            num_samples=num_samples or self.config.eval.num_samples,
            seed=self.config.eval.seed,
        )

        tracker = MetricsTracker()
        outputs = []

        for i, sample in enumerate(samples):
            input_ids = self.tokenizer.encode(
                sample["prompt"], return_tensors="pt"
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = input_ids.to(device)

            output_ids, metrics = standard_decode(
                model=model,
                input_ids=input_ids,
                max_new_tokens=self.config.decoding.max_new_tokens,
                temperature=self.config.decoding.temperature,
            )

            generated_text = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append({
                "prompt": sample["prompt"],
                "generated": generated_text,
                "domain": sample["domain"],
                "metrics": metrics.to_dict(),
            })
            tracker.add_run(metrics)

            if (i + 1) % 10 == 0:
                logger.info(
                    "Standard decoding: %d/%d, avg %.1f tok/s",
                    i + 1, len(samples), tracker.mean_tokens_per_second(),
                )

        return {
            "method": "standard",
            "dataset": dataset_name,
            "summary": tracker.summary(),
            "outputs": outputs,
        }

    def evaluate_speculative(
        self,
        target_model: ModelBackend,
        draft_model: ModelBackend,
        dataset_name: str,
        num_samples: Optional[int] = None,
    ) -> dict:
        """Run speculative decoding on a dataset."""
        samples = load_dataset_prompts(
            dataset_name,
            num_samples=num_samples or self.config.eval.num_samples,
            seed=self.config.eval.seed,
        )

        tracker = MetricsTracker()
        outputs = []

        for i, sample in enumerate(samples):
            input_ids = self.tokenizer.encode(
                sample["prompt"], return_tensors="pt"
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = input_ids.to(device)

            output_ids, metrics = speculative_decode(
                target_model=target_model,
                draft_model=draft_model,
                input_ids=input_ids,
                max_new_tokens=self.config.decoding.max_new_tokens,
                speculation_length=self.config.decoding.speculation_length,
                temperature=self.config.decoding.temperature,
            )

            generated_text = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append({
                "prompt": sample["prompt"],
                "generated": generated_text,
                "domain": sample["domain"],
                "metrics": metrics.to_dict(),
            })
            tracker.add_run(metrics)

            if (i + 1) % 10 == 0:
                logger.info(
                    "Speculative decoding: %d/%d, avg %.1f tok/s, accept rate %.1f%%",
                    i + 1, len(samples),
                    tracker.mean_tokens_per_second(),
                    tracker.mean_acceptance_rate() * 100,
                )

        return {
            "method": "speculative",
            "dataset": dataset_name,
            "summary": tracker.summary(),
            "outputs": outputs,
        }

    def compare(
        self, baseline_results: dict, speculative_results: dict
    ) -> dict:
        """Compare baseline and speculative results."""
        baseline_summary = baseline_results["summary"]
        spec_summary = speculative_results["summary"]

        baseline_tps = baseline_summary["mean_tokens_per_second"]
        spec_tps = spec_summary["mean_tokens_per_second"]
        speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0.0

        baseline_lat = baseline_summary["mean_latency_seconds"]
        spec_lat = spec_summary["mean_latency_seconds"]
        latency_ratio = spec_lat / baseline_lat if baseline_lat > 0 else 0.0

        return {
            "dataset": baseline_results["dataset"],
            "baseline_tokens_per_second": baseline_tps,
            "speculative_tokens_per_second": spec_tps,
            "speedup": speedup,
            "baseline_latency": baseline_lat,
            "speculative_latency": spec_lat,
            "latency_ratio": latency_ratio,
            "acceptance_rate": spec_summary.get("mean_acceptance_rate", 0.0),
            "num_samples": baseline_summary["num_runs"],
        }
