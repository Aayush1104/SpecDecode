"""Results storage, loading, and visualization."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger()


def save_results(results: dict, output_dir: str, prefix: str = "") -> Path:
    """Save results to a JSON file.

    Args:
        results: Results dictionary to save
        output_dir: Directory to save to
        prefix: Optional prefix for the filename

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json" if prefix else f"results_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", filepath)
    return filepath


def load_results(filepath: str | Path) -> dict:
    """Load results from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def print_summary_table(comparison: dict):
    """Print a formatted comparison summary to console."""
    print("\n" + "=" * 60)
    print(f"  Evaluation Results: {comparison['dataset']}")
    print("=" * 60)
    print(f"  Samples:            {comparison['num_samples']}")
    print(f"  Acceptance Rate:    {comparison['acceptance_rate']:.1%}")
    print("-" * 60)
    print(f"  {'Metric':<25} {'Baseline':>12} {'Speculative':>12}")
    print("-" * 60)
    print(
        f"  {'Tokens/sec':<25} "
        f"{comparison['baseline_tokens_per_second']:>12.1f} "
        f"{comparison['speculative_tokens_per_second']:>12.1f}"
    )
    print(
        f"  {'Latency (s)':<25} "
        f"{comparison['baseline_latency']:>12.3f} "
        f"{comparison['speculative_latency']:>12.3f}"
    )
    print("-" * 60)
    print(f"  Speedup:            {comparison['speedup']:.2f}x")
    print("=" * 60 + "\n")


def plot_comparison(comparison: dict, output_path: Optional[str] = None):
    """Generate comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # Throughput comparison
        methods = ["Baseline", "Speculative"]
        throughputs = [
            comparison["baseline_tokens_per_second"],
            comparison["speculative_tokens_per_second"],
        ]
        colors = [sns.color_palette()[0], sns.color_palette()[2]]
        axes[0].bar(methods, throughputs, color=colors)
        axes[0].set_ylabel("Tokens/second")
        axes[0].set_title("Throughput")

        # Latency comparison
        latencies = [
            comparison["baseline_latency"],
            comparison["speculative_latency"],
        ]
        axes[1].bar(methods, latencies, color=colors)
        axes[1].set_ylabel("Seconds")
        axes[1].set_title("Mean Latency")

        # Acceptance rate
        axes[2].bar(
            ["Acceptance Rate"],
            [comparison["acceptance_rate"]],
            color=colors[1],
        )
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("Rate")
        axes[2].set_title("Draft Token Acceptance")

        fig.suptitle(
            f"Speculative Decoding: {comparison['speedup']:.2f}x Speedup "
            f"({comparison['dataset']})",
            fontsize=14,
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Plot saved to %s", output_path)
        else:
            plt.show()

        plt.close(fig)

    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot generation")
