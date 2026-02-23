"""Programmatic analysis of benchmark results."""

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger()


def analyze_benchmark(results: dict) -> dict:
    """Full analysis of benchmark results.

    Args:
        results: Dict mapping method names to their result dicts.
                 Each result dict should have 'summary', 'outputs', and optionally 'quality'.

    Returns:
        Analysis dict with keys: best_draft_per_domain, routing_vs_best_single,
        quality_speed_tradeoffs, failure_cases, optimal_K.
    """
    return {
        "best_draft_per_domain": _best_draft_per_domain(results),
        "routing_vs_best_single": _routing_improvement(results),
        "quality_speed_tradeoffs": _quality_speed_analysis(results),
        "failure_cases": _find_failure_cases(results),
        "optimal_K": _find_optimal_K(results),
    }


def generate_report(analysis: dict, output_path: str):
    """Write a text summary report.

    Args:
        analysis: Output from analyze_benchmark().
        output_path: Path to write the report file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 70)
    lines.append("  SPECULATIVE DECODING BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Best draft per domain
    lines.append("1. BEST DRAFT MODEL PER DOMAIN")
    lines.append("-" * 40)
    best_per_domain = analysis.get("best_draft_per_domain", {})
    if best_per_domain:
        for domain, info in best_per_domain.items():
            lines.append(
                f"  {domain}: {info.get('best_method', 'N/A')} "
                f"({info.get('tokens_per_second', 0):.1f} tok/s, "
                f"accept rate {info.get('acceptance_rate', 0):.1%})"
            )
    else:
        lines.append("  No domain-specific data available.")
    lines.append("")

    # Routing vs best single
    lines.append("2. ADAPTIVE ROUTING vs BEST SINGLE DRAFT")
    lines.append("-" * 40)
    routing = analysis.get("routing_vs_best_single", {})
    if routing.get("routing_tps") is not None:
        lines.append(f"  Routing:       {routing['routing_tps']:.1f} tok/s")
        lines.append(f"  Best single:   {routing['best_single_tps']:.1f} tok/s ({routing['best_single_name']})")
        lines.append(f"  Improvement:   {routing.get('improvement_pct', 0):.1f}%")
    else:
        lines.append("  No routing data available.")
    lines.append("")

    # Quality-speed tradeoffs
    lines.append("3. QUALITY vs SPEED TRADEOFFS")
    lines.append("-" * 40)
    tradeoffs = analysis.get("quality_speed_tradeoffs", [])
    if tradeoffs:
        lines.append(f"  {'Method':<25} {'tok/s':>8} {'Quality':>8}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*8}")
        for t in tradeoffs:
            lines.append(
                f"  {t['method']:<25} {t['tokens_per_second']:>8.1f} {t['quality_score']:>8.3f}"
            )
    else:
        lines.append("  No quality data available.")
    lines.append("")

    # Optimal K
    lines.append("4. OPTIMAL SPECULATION LENGTH (K)")
    lines.append("-" * 40)
    optimal = analysis.get("optimal_K", {})
    if optimal.get("best_K") is not None:
        lines.append(f"  Optimal K:     {optimal['best_K']}")
        lines.append(f"  Best tok/s:    {optimal['best_tps']:.1f}")
        lines.append("  All K values tested:")
        for k_info in optimal.get("all_K", []):
            lines.append(
                f"    K={k_info['K']}: {k_info['tokens_per_second']:.1f} tok/s, "
                f"accept rate {k_info['acceptance_rate']:.1%}"
            )
    else:
        lines.append("  No ablation data available.")
    lines.append("")

    # Failure cases
    lines.append("5. FAILURE CASES (Low Acceptance Rate)")
    lines.append("-" * 40)
    failures = analysis.get("failure_cases", [])
    if failures:
        for f in failures[:10]:  # Show top 10
            lines.append(
                f"  [{f.get('method', '?')}] accept={f.get('acceptance_rate', 0):.1%} "
                f"prompt={f.get('prompt', '')[:60]}..."
            )
    else:
        lines.append("  No significant failure cases found.")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text)

    logger.info("Report written to %s", output_path)
    return report_text


def _best_draft_per_domain(results: dict) -> dict:
    """Find best draft model per domain based on tokens/second."""
    # Collect per-domain metrics for specialized draft methods
    domain_methods: dict[str, list[dict]] = {}

    for method_name, method_results in results.items():
        if method_name == "baseline":
            continue
        outputs = method_results.get("outputs", [])
        # Group by domain
        domain_data: dict[str, list[dict]] = {}
        for out in outputs:
            domain = out.get("domain", "unknown")
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(out.get("metrics", {}))

        for domain, metrics_list in domain_data.items():
            if domain not in domain_methods:
                domain_methods[domain] = []
            n = len(metrics_list)
            mean_tps = sum(m.get("tokens_per_second", 0) for m in metrics_list) / n if n else 0
            mean_ar = sum(m.get("acceptance_rate", 0) for m in metrics_list) / n if n else 0
            domain_methods[domain].append({
                "method": method_name,
                "tokens_per_second": mean_tps,
                "acceptance_rate": mean_ar,
            })

    best_per_domain = {}
    for domain, method_list in domain_methods.items():
        best = max(method_list, key=lambda x: x["tokens_per_second"])
        best_per_domain[domain] = {
            "best_method": best["method"],
            "tokens_per_second": best["tokens_per_second"],
            "acceptance_rate": best["acceptance_rate"],
        }

    return best_per_domain


def _routing_improvement(results: dict) -> dict:
    """Compare adaptive routing against best single draft."""
    routing_results = results.get("adaptive_routing", {})
    routing_summary = routing_results.get("summary", {})
    routing_tps = routing_summary.get("mean_tokens_per_second")

    if routing_tps is None:
        return {"routing_tps": None, "best_single_tps": None,
                "best_single_name": None, "improvement_pct": None}

    # Find best single-draft method
    best_single_tps = 0
    best_single_name = None

    for method_name, method_results in results.items():
        if method_name in ("baseline", "adaptive_routing", "ablation_K"):
            continue
        summary = method_results.get("summary", {})
        tps = summary.get("mean_tokens_per_second", 0)
        if tps > best_single_tps:
            best_single_tps = tps
            best_single_name = method_name

    improvement = ((routing_tps - best_single_tps) / best_single_tps * 100
                   if best_single_tps > 0 else 0)

    return {
        "routing_tps": routing_tps,
        "best_single_tps": best_single_tps,
        "best_single_name": best_single_name,
        "improvement_pct": improvement,
    }


def _quality_speed_analysis(results: dict) -> list[dict]:
    """Analyze quality vs speed tradeoffs for each method."""
    tradeoffs = []
    for method_name, method_results in results.items():
        summary = method_results.get("summary", {})
        quality = method_results.get("quality", {})
        tradeoffs.append({
            "method": method_name,
            "tokens_per_second": summary.get("mean_tokens_per_second", 0),
            "quality_score": quality.get("score", 0) if quality else 0,
            "acceptance_rate": summary.get("mean_acceptance_rate", 0),
        })

    # Sort by tokens/second descending
    tradeoffs.sort(key=lambda x: x["tokens_per_second"], reverse=True)
    return tradeoffs


def _find_failure_cases(results: dict) -> list[dict]:
    """Find samples with low acceptance rates (potential failure cases)."""
    failures = []
    for method_name, method_results in results.items():
        if method_name == "baseline":
            continue
        outputs = method_results.get("outputs", [])
        for out in outputs:
            metrics = out.get("metrics", {})
            acceptance = metrics.get("acceptance_rate", 1.0)
            if acceptance < 0.2:  # Less than 20% acceptance
                failures.append({
                    "method": method_name,
                    "prompt": out.get("prompt", ""),
                    "domain": out.get("domain", "unknown"),
                    "acceptance_rate": acceptance,
                    "tokens_per_second": metrics.get("tokens_per_second", 0),
                })

    # Sort by acceptance rate ascending
    failures.sort(key=lambda x: x["acceptance_rate"])
    return failures


def _find_optimal_K(results: dict) -> dict:
    """Find the optimal speculation length from ablation results."""
    ablation = results.get("ablation_K", {})
    if not ablation:
        return {"best_K": None, "best_tps": None, "all_K": []}

    all_k = []
    for k_str, k_data in ablation.items():
        try:
            k = int(k_str.replace("K=", ""))
        except (ValueError, AttributeError):
            continue
        summary = k_data.get("summary", {})
        all_k.append({
            "K": k,
            "tokens_per_second": summary.get("mean_tokens_per_second", 0),
            "acceptance_rate": summary.get("mean_acceptance_rate", 0),
        })

    if not all_k:
        return {"best_K": None, "best_tps": None, "all_K": []}

    all_k.sort(key=lambda x: x["K"])
    best = max(all_k, key=lambda x: x["tokens_per_second"])

    return {
        "best_K": best["K"],
        "best_tps": best["tokens_per_second"],
        "all_K": all_k,
    }
