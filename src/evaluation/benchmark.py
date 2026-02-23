"""Comprehensive benchmark runner for speculative decoding experiments."""

from typing import Optional

import torch

from src.evaluation.datasets import load_dataset_prompts
from src.evaluation.quality import compute_quality_metrics
from src.speculative.backends import ModelBackend
from src.speculative.decoding import speculative_decode, standard_decode
from src.utils.config import ExperimentConfig
from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker

logger = get_logger()


class BenchmarkRunner:
    """Runs all 5 experiment types from the benchmark suite.

    Experiments:
        1. baseline: Standard autoregressive decoding (target only)
        2. generic_draft: Speculative decoding with generic draft (e.g. distilgpt2)
        3. specialized_drafts: Each specialized draft model separately
        4. adaptive_routing: Router selects best draft per prompt
        5. ablation_K: Vary speculation length across configured values
    """

    def __init__(
        self,
        config: ExperimentConfig,
        tokenizer,
        target_model: ModelBackend,
        draft_model: Optional[ModelBackend] = None,
        specialized_drafts: Optional[dict[str, ModelBackend]] = None,
        router_model=None,
        feature_extractor=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.draft_model = draft_model
        self.specialized_drafts = specialized_drafts or {}
        self.router_model = router_model
        self.feature_extractor = feature_extractor

    def run_all(self) -> dict[str, dict]:
        """Run all configured experiments, return combined results."""
        experiments = self.config.benchmark.experiments
        all_results = {}

        for dataset_name in self.config.eval.datasets:
            logger.info("Loading dataset: %s", dataset_name)
            dataset = load_dataset_prompts(
                dataset_name,
                num_samples=self.config.eval.num_samples,
                seed=self.config.eval.seed,
            )

            for exp_name in experiments:
                runner = {
                    "baseline": self.run_baseline,
                    "generic_draft": self.run_generic_draft,
                    "specialized_drafts": self.run_specialized_drafts,
                    "adaptive_routing": self.run_adaptive_routing,
                    "ablation_K": self.run_ablation_K,
                }
                if exp_name not in runner:
                    logger.warning("Unknown experiment: %s, skipping", exp_name)
                    continue

                logger.info("Running experiment: %s on %s", exp_name, dataset_name)
                result = runner[exp_name](dataset)

                # Compute quality metrics
                if result.get("outputs"):
                    try:
                        quality = compute_quality_metrics(result["outputs"], dataset_name)
                        result["quality"] = quality
                    except Exception as e:
                        logger.warning("Quality metrics failed for %s: %s", exp_name, e)

                key = exp_name if exp_name != "ablation_K" else exp_name
                all_results[key] = result

        return all_results

    def run_baseline(self, dataset: list[dict]) -> dict:
        """Experiment 1: Standard autoregressive decoding on target model."""
        tracker = MetricsTracker()
        outputs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for i, sample in enumerate(dataset):
            if i < self.config.benchmark.warmup_samples:
                # Warmup runs - don't record
                input_ids = self.tokenizer.encode(
                    sample["prompt"], return_tensors="pt"
                ).to(device)
                standard_decode(
                    model=self.target_model,
                    input_ids=input_ids,
                    max_new_tokens=min(16, self.config.decoding.max_new_tokens),
                    temperature=self.config.decoding.temperature,
                )
                continue

            input_ids = self.tokenizer.encode(
                sample["prompt"], return_tensors="pt"
            ).to(device)

            output_ids, metrics = standard_decode(
                model=self.target_model,
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
                "domain": sample.get("domain", "unknown"),
                "reference": sample.get("reference", ""),
                "metrics": metrics.to_dict(),
            })
            tracker.add_run(metrics, domain=sample.get("domain"))

            if (i + 1) % 10 == 0:
                logger.info(
                    "Baseline: %d/%d, avg %.1f tok/s",
                    i + 1, len(dataset), tracker.mean_tokens_per_second(),
                )

        return {
            "method": "baseline",
            "dataset": dataset[0].get("domain", "mixed") if dataset else "unknown",
            "summary": tracker.summary(),
            "outputs": outputs,
        }

    def run_generic_draft(self, dataset: list[dict]) -> dict:
        """Experiment 2: Speculative decoding with generic draft model."""
        if self.draft_model is None:
            logger.warning("No generic draft model provided, skipping generic_draft")
            return {"method": "generic_draft", "summary": {}, "outputs": []}

        return self._run_speculative(
            dataset=dataset,
            draft_model=self.draft_model,
            method_name="generic_draft",
        )

    def run_specialized_drafts(self, dataset: list[dict]) -> dict:
        """Experiment 3: Each specialized draft model separately."""
        if not self.specialized_drafts:
            logger.warning("No specialized drafts provided, skipping")
            return {"method": "specialized_drafts", "summary": {}, "outputs": []}

        all_outputs = []
        combined_tracker = MetricsTracker()

        for draft_name, draft_backend in self.specialized_drafts.items():
            logger.info("Running specialized draft: %s", draft_name)
            result = self._run_speculative(
                dataset=dataset,
                draft_model=draft_backend,
                method_name=f"specialized_{draft_name}",
            )
            # Tag outputs with draft name
            for out in result.get("outputs", []):
                out["draft_model"] = draft_name
                all_outputs.append(out)
                # Re-add to combined tracker
                from src.utils.metrics import DecodingMetrics
                m = out["metrics"]
                dm = DecodingMetrics(
                    total_tokens=m["total_tokens"],
                    total_steps=m["total_steps"],
                    accepted_tokens=m["accepted_tokens"],
                    draft_tokens=m["draft_tokens"],
                    latency_seconds=m["latency_seconds"],
                )
                combined_tracker.add_run(dm, domain=out.get("domain"))

        return {
            "method": "specialized_drafts",
            "summary": combined_tracker.summary(),
            "outputs": all_outputs,
        }

    def run_adaptive_routing(self, dataset: list[dict]) -> dict:
        """Experiment 4: Router selects best draft per prompt."""
        if not self.router_model or not self.feature_extractor or not self.specialized_drafts:
            logger.warning("Router/feature_extractor/specialized_drafts not provided, skipping")
            return {"method": "adaptive_routing", "summary": {}, "outputs": []}

        from src.routing.router import route_and_decode

        tracker = MetricsTracker()
        outputs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        profile = self.config.benchmark.profile

        for i, sample in enumerate(dataset):
            if i < self.config.benchmark.warmup_samples:
                continue

            input_ids = self.tokenizer.encode(
                sample["prompt"], return_tensors="pt"
            ).to(device)

            output_ids, metrics, selected_draft = route_and_decode(
                target_model=self.target_model,
                draft_models=self.specialized_drafts,
                router_model=self.router_model,
                input_ids=input_ids,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.decoding.max_new_tokens,
                speculation_length=self.config.decoding.speculation_length,
                temperature=self.config.decoding.temperature,
                profile=profile,
            )

            generated_text = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append({
                "prompt": sample["prompt"],
                "generated": generated_text,
                "domain": sample.get("domain", "unknown"),
                "reference": sample.get("reference", ""),
                "selected_draft": selected_draft,
                "metrics": metrics.to_dict(),
            })
            tracker.add_run(metrics, domain=sample.get("domain"))

            if (i + 1) % 10 == 0:
                logger.info(
                    "Adaptive routing: %d/%d, avg %.1f tok/s",
                    i + 1, len(dataset), tracker.mean_tokens_per_second(),
                )

        return {
            "method": "adaptive_routing",
            "summary": tracker.summary(),
            "outputs": outputs,
        }

    def run_ablation_K(self, dataset: list[dict]) -> dict:
        """Experiment 5: Vary speculation_length across configured values."""
        if self.draft_model is None:
            logger.warning("No draft model provided for ablation, skipping")
            return {"method": "ablation_K", "summary": {}, "outputs": []}

        k_results = {}
        for k in self.config.benchmark.speculation_lengths:
            logger.info("Ablation: K=%d", k)
            result = self._run_speculative(
                dataset=dataset,
                draft_model=self.draft_model,
                method_name=f"ablation_K={k}",
                speculation_length=k,
            )
            k_results[f"K={k}"] = result

        return {
            "method": "ablation_K",
            **k_results,
        }

    def _run_speculative(
        self,
        dataset: list[dict],
        draft_model: ModelBackend,
        method_name: str,
        speculation_length: Optional[int] = None,
    ) -> dict:
        """Shared helper for running speculative decoding experiments."""
        tracker = MetricsTracker()
        outputs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        K = speculation_length or self.config.decoding.speculation_length
        profile = self.config.benchmark.profile

        for i, sample in enumerate(dataset):
            if i < self.config.benchmark.warmup_samples:
                continue

            input_ids = self.tokenizer.encode(
                sample["prompt"], return_tensors="pt"
            ).to(device)

            output_ids, metrics = speculative_decode(
                target_model=self.target_model,
                draft_model=draft_model,
                input_ids=input_ids,
                max_new_tokens=self.config.decoding.max_new_tokens,
                speculation_length=K,
                temperature=self.config.decoding.temperature,
                profile=profile,
            )

            generated_text = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append({
                "prompt": sample["prompt"],
                "generated": generated_text,
                "domain": sample.get("domain", "unknown"),
                "reference": sample.get("reference", ""),
                "metrics": metrics.to_dict(),
            })
            tracker.add_run(metrics, domain=sample.get("domain"))

            if (i + 1) % 10 == 0:
                logger.info(
                    "%s: %d/%d, avg %.1f tok/s, accept %.1f%%",
                    method_name, i + 1, len(dataset),
                    tracker.mean_tokens_per_second(),
                    tracker.mean_acceptance_rate() * 100,
                )

        return {
            "method": method_name,
            "dataset": dataset[0].get("domain", "mixed") if dataset else "unknown",
            "summary": tracker.summary(),
            "outputs": outputs,
        }
