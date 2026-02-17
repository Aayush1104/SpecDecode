#!/usr/bin/env python3
"""Compare baseline and speculative decoding results."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.results import load_results, plot_comparison, print_summary_table
from src.evaluation.evaluator import Evaluator
from src.utils.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Compare decoding results")
    parser.add_argument(
        "--baseline", type=str, required=True,
        help="Path to baseline results JSON",
    )
    parser.add_argument(
        "--speculative", type=str, required=True,
        help="Path to speculative results JSON",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Path to save comparison plot (optional)",
    )
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    speculative = load_results(args.speculative)

    config = ExperimentConfig()
    evaluator = Evaluator(config, tokenizer=None)
    comparison = evaluator.compare(baseline, speculative)

    print_summary_table(comparison)

    if args.plot:
        plot_comparison(comparison, output_path=args.plot)


if __name__ == "__main__":
    main()
