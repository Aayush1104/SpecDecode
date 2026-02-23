#!/usr/bin/env python3
"""Analyze benchmark results and generate report + visualizations.

Loads saved benchmark results, runs analysis, generates report and charts.

Usage:
    python3 scripts/analyze_results.py --results-dir results/benchmark/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.analysis import analyze_benchmark, generate_report
from src.evaluation.visualization import generate_all_charts
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmark/",
        help="Directory containing benchmark result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for report and charts (default: results-dir)",
    )
    args = parser.parse_args()

    logger = setup_logging(level="INFO")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    # Load all result files
    benchmark_results = {}
    for filepath in sorted(results_dir.glob("*.json")):
        logger.info("Loading %s", filepath.name)
        with open(filepath) as f:
            data = json.load(f)
        # Use method name as key, strip timestamp from filename
        method = data.get("method", filepath.stem.rsplit("_", 1)[0])
        benchmark_results[method] = data

    if not benchmark_results:
        logger.error("No result files found in %s", results_dir)
        sys.exit(1)

    logger.info("Loaded %d result files", len(benchmark_results))

    # Run analysis
    logger.info("Running analysis...")
    analysis = analyze_benchmark(benchmark_results)

    # Generate report
    report_path = str(output_dir / "benchmark_report.txt")
    report_text = generate_report(analysis, report_path)
    print(report_text)

    # Generate charts
    charts_dir = str(output_dir / "charts")
    logger.info("Generating charts in %s", charts_dir)
    generate_all_charts(benchmark_results, charts_dir)

    logger.info("Analysis complete. Report: %s, Charts: %s", report_path, charts_dir)


if __name__ == "__main__":
    main()
