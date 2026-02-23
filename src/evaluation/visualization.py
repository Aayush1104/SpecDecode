"""Visualization suite for benchmark results."""

import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger()


def plot_speedup_by_domain(results: dict, output_path: str):
    """Bar chart: speedup per domain (code/math/conv/fact)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # Collect domain summaries from different methods
    domains = []
    speedups = []
    methods = []

    baseline_domains = _extract_domain_metrics(results.get("baseline", {}))

    for method_name, method_results in results.items():
        if method_name == "baseline":
            continue
        method_domains = _extract_domain_metrics(method_results)
        for domain, metrics in method_domains.items():
            base_tps = baseline_domains.get(domain, {}).get("mean_tokens_per_second", 0)
            method_tps = metrics.get("mean_tokens_per_second", 0)
            if base_tps > 0:
                domains.append(domain)
                speedups.append(method_tps / base_tps)
                methods.append(method_name)

    if not domains:
        logger.warning("No domain data available for speedup chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("husl", len(set(methods)))
    sns.barplot(x=domains, y=speedups, hue=methods, ax=ax, palette=palette)
    ax.set_ylabel("Speedup vs Baseline")
    ax.set_xlabel("Domain")
    ax.set_title("Speedup by Domain")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved speedup by domain chart to %s", output_path)


def plot_acceptance_over_samples(results: dict, output_path: str):
    """Line chart: running acceptance rate across samples."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, method_results in results.items():
        if method_name == "baseline":
            continue
        outputs = method_results.get("outputs", [])
        if not outputs:
            continue

        running_rates = []
        total_accepted = 0
        total_draft = 0
        for out in outputs:
            m = out.get("metrics", {})
            total_accepted += m.get("accepted_tokens", 0)
            total_draft += m.get("draft_tokens", 0)
            rate = total_accepted / total_draft if total_draft > 0 else 0
            running_rates.append(rate)

        ax.plot(range(1, len(running_rates) + 1), running_rates, label=method_name)

    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Running Acceptance Rate")
    ax.set_title("Acceptance Rate Over Samples")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved acceptance over samples chart to %s", output_path)


def plot_quality_vs_speed(results: dict, output_path: str):
    """Scatter: quality score vs tokens/sec for each method."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    names = []
    speeds = []
    qualities = []

    for method_name, method_results in results.items():
        summary = method_results.get("summary", {})
        quality = method_results.get("quality", {})

        tps = summary.get("mean_tokens_per_second", 0)
        score = quality.get("score", 0) if quality else 0

        names.append(method_name)
        speeds.append(tps)
        qualities.append(score)

    ax.scatter(speeds, qualities, s=100, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (speeds[i], qualities[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Tokens/second")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality vs Speed Trade-off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved quality vs speed chart to %s", output_path)


def plot_speedup_vs_K(ablation_results: dict, output_path: str):
    """Line chart: speedup for different speculation lengths K."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    k_values = []
    speedups = []
    acceptance_rates = []

    for k_str, k_data in sorted(ablation_results.items()):
        try:
            k = int(k_str.replace("K=", ""))
        except (ValueError, AttributeError):
            continue
        summary = k_data.get("summary", {})
        k_values.append(k)
        speedups.append(summary.get("mean_tokens_per_second", 0))
        acceptance_rates.append(summary.get("mean_acceptance_rate", 0))

    if not k_values:
        logger.warning("No ablation data available")
        return

    ax.plot(k_values, speedups, "o-", label="Tokens/sec", color="tab:blue")
    ax.set_xlabel("Speculation Length (K)")
    ax.set_ylabel("Tokens/second", color="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(k_values, acceptance_rates, "s--", label="Accept Rate", color="tab:orange")
    ax2.set_ylabel("Acceptance Rate", color="tab:orange")

    ax.set_title("Performance vs Speculation Length")
    ax.set_xticks(k_values)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved speedup vs K chart to %s", output_path)


def plot_latency_distribution(results: dict, output_path: str):
    """Box plot: latency distribution per method."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []

    for method_name, method_results in results.items():
        outputs = method_results.get("outputs", [])
        for out in outputs:
            latency = out.get("metrics", {}).get("latency_seconds", 0)
            data.append(latency)
            labels.append(method_name)

    if not data:
        logger.warning("No latency data available")
        return

    import pandas as pd
    df = pd.DataFrame({"Latency (s)": data, "Method": labels})
    sns.boxplot(x="Method", y="Latency (s)", data=df, ax=ax)
    ax.set_title("Latency Distribution by Method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved latency distribution chart to %s", output_path)


def plot_draft_usage(routed_results: dict, output_path: str):
    """Bar chart: router draft selection distribution."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    draft_counts: dict[str, int] = {}
    outputs = routed_results.get("outputs", [])
    for out in outputs:
        draft = out.get("selected_draft", "unknown")
        draft_counts[draft] = draft_counts.get(draft, 0) + 1

    if not draft_counts:
        logger.warning("No draft usage data available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(draft_counts.keys())
    counts = list(draft_counts.values())

    colors = sns.color_palette("Set2", len(names))
    ax.bar(names, counts, color=colors)
    ax.set_xlabel("Draft Model")
    ax.set_ylabel("Times Selected")
    ax.set_title("Router Draft Model Selection Distribution")

    # Add count labels on bars
    for i, (name, count) in enumerate(zip(names, counts)):
        ax.text(i, count + 0.5, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved draft usage chart to %s", output_path)


def generate_all_charts(benchmark_results: dict, output_dir: str):
    """Generate all applicable charts from benchmark results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        logger.warning("matplotlib not available, skipping chart generation")
        return

    # Speedup by domain (needs baseline + at least one other method)
    if "baseline" in benchmark_results and len(benchmark_results) > 1:
        try:
            plot_speedup_by_domain(
                benchmark_results, str(output_dir / "speedup_by_domain.png")
            )
        except Exception as e:
            logger.warning("Failed to generate speedup by domain chart: %s", e)

    # Acceptance over samples
    try:
        plot_acceptance_over_samples(
            benchmark_results, str(output_dir / "acceptance_over_samples.png")
        )
    except Exception as e:
        logger.warning("Failed to generate acceptance chart: %s", e)

    # Quality vs speed
    try:
        plot_quality_vs_speed(
            benchmark_results, str(output_dir / "quality_vs_speed.png")
        )
    except Exception as e:
        logger.warning("Failed to generate quality vs speed chart: %s", e)

    # Speedup vs K (ablation)
    if "ablation_K" in benchmark_results:
        try:
            plot_speedup_vs_K(
                benchmark_results["ablation_K"],
                str(output_dir / "speedup_vs_K.png"),
            )
        except Exception as e:
            logger.warning("Failed to generate speedup vs K chart: %s", e)

    # Latency distribution
    try:
        plot_latency_distribution(
            benchmark_results, str(output_dir / "latency_distribution.png")
        )
    except Exception as e:
        logger.warning("Failed to generate latency distribution chart: %s", e)

    # Draft usage (adaptive routing)
    if "adaptive_routing" in benchmark_results:
        try:
            plot_draft_usage(
                benchmark_results["adaptive_routing"],
                str(output_dir / "draft_usage.png"),
            )
        except Exception as e:
            logger.warning("Failed to generate draft usage chart: %s", e)

    logger.info("Charts saved to %s", output_dir)


def _extract_domain_metrics(method_results: dict) -> dict:
    """Extract per-domain metrics from method results."""
    domain_metrics: dict[str, dict] = {}
    outputs = method_results.get("outputs", [])

    for out in outputs:
        domain = out.get("domain", "unknown")
        if domain not in domain_metrics:
            domain_metrics[domain] = {"tps_sum": 0.0, "count": 0}
        m = out.get("metrics", {})
        domain_metrics[domain]["tps_sum"] += m.get("tokens_per_second", 0)
        domain_metrics[domain]["count"] += 1

    result = {}
    for domain, data in domain_metrics.items():
        if data["count"] > 0:
            result[domain] = {
                "mean_tokens_per_second": data["tps_sum"] / data["count"],
            }
    return result
