"""Dataset loading for evaluation benchmarks."""

from typing import Optional

from src.utils.logging import get_logger

logger = get_logger()


def load_dataset_prompts(
    dataset_name: str,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Load prompts from a benchmark dataset.

    Args:
        dataset_name: One of "humaneval", "gsm8k", "mt_bench", "truthfulqa"
        num_samples: Maximum number of samples to load (None for all)
        seed: Random seed for sampling

    Returns:
        List of dicts with keys: prompt, reference (optional), domain
    """
    loaders = {
        "humaneval": _load_humaneval,
        "gsm8k": _load_gsm8k,
        "mt_bench": _load_mt_bench,
        "truthfulqa": _load_truthfulqa,
    }

    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}"
        )

    samples = loaders[dataset_name](seed=seed)

    if num_samples is not None and num_samples < len(samples):
        import random
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)

    logger.info("Loaded %d samples from %s", len(samples), dataset_name)
    return samples


def _load_humaneval(seed: int = 42) -> list[dict]:
    """Load HumanEval code generation prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        samples = []
        for item in ds:
            samples.append({
                "prompt": item["prompt"],
                "reference": item.get("canonical_solution", ""),
                "domain": "code",
                "task_id": item.get("task_id", ""),
            })
        return samples
    except Exception as e:
        logger.warning("Failed to load HumanEval: %s. Using fallback prompts.", e)
        return _fallback_prompts("code")


def _load_gsm8k(seed: int = 42) -> list[dict]:
    """Load GSM8K math reasoning prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        samples = []
        for item in ds:
            samples.append({
                "prompt": f"Question: {item['question']}\nAnswer:",
                "reference": item.get("answer", ""),
                "domain": "math",
            })
        return samples
    except Exception as e:
        logger.warning("Failed to load GSM8K: %s. Using fallback prompts.", e)
        return _fallback_prompts("math")


def _load_mt_bench(seed: int = 42) -> list[dict]:
    """Load MT-Bench conversation prompts (subset)."""
    # MT-Bench is typically loaded from a local file or specific source.
    # Use a curated subset of general prompts as fallback.
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        samples = []
        for item in ds:
            prompt = item["prompt"][0] if isinstance(item["prompt"], list) else item["prompt"]
            samples.append({
                "prompt": prompt,
                "reference": "",
                "domain": "conversation",
            })
        return samples
    except Exception as e:
        logger.warning("Failed to load MT-Bench: %s. Using fallback prompts.", e)
        return _fallback_prompts("conversation")


def _load_truthfulqa(seed: int = 42) -> list[dict]:
    """Load TruthfulQA prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        samples = []
        for item in ds:
            samples.append({
                "prompt": f"Question: {item['question']}\nAnswer:",
                "reference": item.get("best_answer", ""),
                "domain": "factuality",
            })
        return samples
    except Exception as e:
        logger.warning("Failed to load TruthfulQA: %s. Using fallback prompts.", e)
        return _fallback_prompts("factuality")


def _fallback_prompts(domain: str) -> list[dict]:
    """Provide minimal fallback prompts when dataset loading fails."""
    fallbacks = {
        "code": [
            {"prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n", "reference": "", "domain": "code"},
            {"prompt": "def binary_search(arr, target):\n    \"\"\"Search for target in sorted array.\"\"\"\n", "reference": "", "domain": "code"},
            {"prompt": "def merge_sort(arr):\n    \"\"\"Sort array using merge sort.\"\"\"\n", "reference": "", "domain": "code"},
        ],
        "math": [
            {"prompt": "Question: What is 15 * 23?\nAnswer:", "reference": "345", "domain": "math"},
            {"prompt": "Question: If a train travels 60 mph for 2.5 hours, how far does it go?\nAnswer:", "reference": "150 miles", "domain": "math"},
            {"prompt": "Question: What is the square root of 144?\nAnswer:", "reference": "12", "domain": "math"},
        ],
        "conversation": [
            {"prompt": "Write a short poem about the ocean.", "reference": "", "domain": "conversation"},
            {"prompt": "Explain quantum computing in simple terms.", "reference": "", "domain": "conversation"},
            {"prompt": "What are the benefits of regular exercise?", "reference": "", "domain": "conversation"},
        ],
        "factuality": [
            {"prompt": "Question: What is the capital of France?\nAnswer:", "reference": "Paris", "domain": "factuality"},
            {"prompt": "Question: Who wrote Romeo and Juliet?\nAnswer:", "reference": "William Shakespeare", "domain": "factuality"},
            {"prompt": "Question: What is the speed of light?\nAnswer:", "reference": "Approximately 299,792,458 meters per second", "domain": "factuality"},
        ],
    }
    return fallbacks.get(domain, fallbacks["conversation"])
