"""Router training data collection.

Benchmarks all draft models on validation prompts and records which draft
achieves the highest acceptance rate per prompt.
"""

import json
from dataclasses import asdict, dataclass

import torch

from src.evaluation.datasets import load_dataset_prompts
from src.speculative.backends import ModelBackend
from src.speculative.decoding import speculative_decode
from src.utils.config import ExperimentConfig
from src.utils.logging import get_logger

logger = get_logger()


@dataclass
class RoutingSample:
    """A single training sample for the router."""

    prompt: str
    domain: str
    features: list[float]
    acceptance_rates: dict[str, float]  # draft_name → acceptance_rate
    best_draft: str
    best_draft_id: int


def collect_router_data(
    target_model: ModelBackend,
    draft_models: dict[str, ModelBackend],
    feature_extractor,
    config: ExperimentConfig,
    tokenizer,
) -> list[RoutingSample]:
    """Collect router training data by benchmarking all draft models.

    For each prompt in the configured datasets, runs speculative_decode()
    with every draft model and records which draft achieved the highest
    acceptance rate.

    Args:
        target_model: The target model backend
        draft_models: Dict mapping draft name → ModelBackend
        feature_extractor: FeatureExtractor instance
        config: Experiment configuration
        tokenizer: Tokenizer for encoding prompts

    Returns:
        List of RoutingSample instances
    """
    draft_names = list(draft_models.keys())
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_new_tokens = config.router.collection_max_new_tokens
    speculation_length = config.decoding.speculation_length
    temperature = config.decoding.temperature

    for dataset_name in config.router.collection_datasets:
        logger.info("Collecting data from dataset: %s", dataset_name)
        prompts = load_dataset_prompts(
            dataset_name,
            num_samples=config.router.collection_num_samples,
            seed=config.eval.seed,
        )

        for i, sample in enumerate(prompts):
            prompt_text = sample["prompt"]
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

            # Extract features
            features = feature_extractor.extract(prompt_text, tokenizer=tokenizer)
            features_list = features.cpu().tolist()

            # Benchmark each draft model
            acceptance_rates = {}
            for draft_name, draft_model in draft_models.items():
                try:
                    _, metrics = speculative_decode(
                        target_model=target_model,
                        draft_model=draft_model,
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        speculation_length=speculation_length,
                        temperature=temperature,
                    )
                    acceptance_rates[draft_name] = metrics.acceptance_rate
                except Exception as e:
                    logger.warning(
                        "Draft %s failed on sample %d: %s", draft_name, i, e
                    )
                    acceptance_rates[draft_name] = 0.0

            # Determine best draft
            best_draft = max(acceptance_rates, key=acceptance_rates.get)
            best_draft_id = draft_names.index(best_draft)

            samples.append(RoutingSample(
                prompt=prompt_text,
                domain=sample.get("domain", "unknown"),
                features=features_list,
                acceptance_rates=acceptance_rates,
                best_draft=best_draft,
                best_draft_id=best_draft_id,
            ))

            if (i + 1) % 10 == 0:
                logger.info(
                    "  [%s] %d/%d samples collected",
                    dataset_name, i + 1, len(prompts),
                )

    logger.info("Collected %d total routing samples", len(samples))
    return samples


def save_router_data(samples: list[RoutingSample], path: str):
    """Save routing samples to a JSON file.

    Args:
        samples: List of RoutingSample instances
        path: Output file path
    """
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in samples]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d routing samples to %s", len(samples), path)


def load_router_data(path: str) -> list[RoutingSample]:
    """Load routing samples from a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        List of RoutingSample instances
    """
    with open(path) as f:
        data = json.load(f)
    samples = [RoutingSample(**item) for item in data]
    logger.info("Loaded %d routing samples from %s", len(samples), path)
    return samples
