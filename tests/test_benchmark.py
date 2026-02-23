"""Tests for Phase 4: Optimization & Comprehensive Evaluation.

Covers: extended metrics, profiling, quality metrics, benchmark config,
visualization, and analysis.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from src.utils.config import BenchmarkConfig, ExperimentConfig, load_config
from src.utils.metrics import DecodingMetrics, MetricsTracker


# ──────────────────────────────────────────────────────────────
# Extended Metrics Tests
# ──────────────────────────────────────────────────────────────


class TestDecodingMetricsExtended:
    """Tests for per-phase timing fields in DecodingMetrics."""

    def test_phase_timing_defaults(self):
        """Per-phase timing fields should default to 0."""
        m = DecodingMetrics()
        assert m.draft_time == 0.0
        assert m.verify_time == 0.0
        assert m.sampling_time == 0.0
        assert m.overhead_time == 0.0

    def test_phase_timing_in_to_dict(self):
        """to_dict() should include per-phase timing fields."""
        m = DecodingMetrics(
            draft_time=0.1, verify_time=0.2,
            sampling_time=0.05, overhead_time=0.03,
        )
        d = m.to_dict()
        assert d["draft_time"] == 0.1
        assert d["verify_time"] == 0.2
        assert d["sampling_time"] == 0.05
        assert d["overhead_time"] == 0.03

    def test_backward_compat_existing_fields(self):
        """Existing fields and properties should still work."""
        m = DecodingMetrics(
            total_tokens=10, total_steps=5,
            accepted_tokens=8, draft_tokens=10,
            latency_seconds=1.0,
        )
        assert m.acceptance_rate == 0.8
        assert m.tokens_per_second == 10.0
        assert m.avg_accepted_per_step == 1.6


class TestMetricsTrackerExtended:
    """Tests for percentiles, std dev, and domain tracking."""

    def _make_runs(self, n=20):
        """Create n DecodingMetrics with varied latencies."""
        runs = []
        for i in range(n):
            runs.append(DecodingMetrics(
                total_tokens=10 + i,
                total_steps=5,
                accepted_tokens=8 + (i % 3),
                draft_tokens=10,
                latency_seconds=0.5 + i * 0.1,
            ))
        return runs

    def test_latency_percentiles(self):
        """Percentiles should be computed correctly."""
        tracker = MetricsTracker()
        for m in self._make_runs(100):
            tracker.add_run(m)

        p = tracker.latency_percentiles()
        assert "p50" in p
        assert "p95" in p
        assert "p99" in p
        assert p["p50"] <= p["p95"] <= p["p99"]
        assert p["p50"] > 0

    def test_latency_percentiles_empty(self):
        """Percentiles should be 0 for empty tracker."""
        tracker = MetricsTracker()
        p = tracker.latency_percentiles()
        assert p["p50"] == 0.0

    def test_latency_percentiles_single_run(self):
        """Percentiles should be equal for single run."""
        tracker = MetricsTracker()
        tracker.add_run(DecodingMetrics(latency_seconds=1.5))
        p = tracker.latency_percentiles()
        assert p["p50"] == 1.5
        assert p["p95"] == 1.5
        assert p["p99"] == 1.5

    def test_std_tokens_per_second(self):
        """Standard deviation should be > 0 for varied data."""
        tracker = MetricsTracker()
        for m in self._make_runs(20):
            tracker.add_run(m)

        std = tracker.std_tokens_per_second()
        assert std > 0

    def test_std_tokens_per_second_single(self):
        """Std dev should be 0 for single run."""
        tracker = MetricsTracker()
        tracker.add_run(DecodingMetrics(total_tokens=10, latency_seconds=1.0))
        assert tracker.std_tokens_per_second() == 0.0

    def test_std_acceptance_rate(self):
        """Standard deviation of acceptance rate should work."""
        tracker = MetricsTracker()
        tracker.add_run(DecodingMetrics(accepted_tokens=8, draft_tokens=10))
        tracker.add_run(DecodingMetrics(accepted_tokens=4, draft_tokens=10))
        std = tracker.std_acceptance_rate()
        assert std > 0

    def test_domain_tracking(self):
        """Domain-based tracking should accumulate correctly."""
        tracker = MetricsTracker()
        tracker.add_run(
            DecodingMetrics(total_tokens=10, latency_seconds=1.0, accepted_tokens=8, draft_tokens=10),
            domain="code",
        )
        tracker.add_run(
            DecodingMetrics(total_tokens=20, latency_seconds=2.0, accepted_tokens=15, draft_tokens=20),
            domain="code",
        )
        tracker.add_run(
            DecodingMetrics(total_tokens=15, latency_seconds=1.5, accepted_tokens=10, draft_tokens=15),
            domain="math",
        )

        assert len(tracker.domain_runs) == 2
        assert len(tracker.domain_runs["code"]) == 2
        assert len(tracker.domain_runs["math"]) == 1

    def test_domain_summary(self):
        """domain_summary() should return per-domain means."""
        tracker = MetricsTracker()
        tracker.add_run(
            DecodingMetrics(total_tokens=10, latency_seconds=1.0, accepted_tokens=8, draft_tokens=10),
            domain="code",
        )
        tracker.add_run(
            DecodingMetrics(total_tokens=20, latency_seconds=2.0, accepted_tokens=16, draft_tokens=20),
            domain="code",
        )
        summary = tracker.domain_summary()
        assert "code" in summary
        assert summary["code"]["num_runs"] == 2
        assert abs(summary["code"]["mean_acceptance_rate"] - 0.8) < 1e-6

    def test_add_run_without_domain(self):
        """add_run() without domain should still work (backward compat)."""
        tracker = MetricsTracker()
        tracker.add_run(DecodingMetrics(total_tokens=10))
        assert tracker.num_runs == 1
        assert len(tracker.domain_runs) == 0

    def test_summary_extended(self):
        """summary() should include new fields."""
        tracker = MetricsTracker()
        for m in self._make_runs(10):
            tracker.add_run(m, domain="code")

        s = tracker.summary()
        # Existing fields
        assert "num_runs" in s
        assert "mean_acceptance_rate" in s
        assert "mean_tokens_per_second" in s
        assert "mean_latency_seconds" in s
        assert "total_tokens" in s
        # New fields
        assert "std_tokens_per_second" in s
        assert "std_acceptance_rate" in s
        assert "latency_p50" in s
        assert "latency_p95" in s
        assert "latency_p99" in s
        assert "domain_summary" in s
        assert "code" in s["domain_summary"]


# ──────────────────────────────────────────────────────────────
# Profiling Tests
# ──────────────────────────────────────────────────────────────


class TestProfiling:
    """Tests for profile=True in speculative_decode."""

    @pytest.fixture(scope="class")
    def models(self):
        from src.speculative.backends import HuggingFaceBackend
        target = HuggingFaceBackend("gpt2", dtype="float32", device="cpu")
        draft = HuggingFaceBackend("distilgpt2", dtype="float32", device="cpu")
        return target, draft

    def test_profile_populates_timing(self, models):
        """profile=True should populate per-phase timing fields."""
        from src.speculative.decoding import speculative_decode
        target, draft = models
        input_ids = torch.tensor([[50256]])

        _, metrics = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=10,
            speculation_length=3,
            temperature=1.0,
            profile=True,
        )

        assert metrics.draft_time > 0
        assert metrics.verify_time > 0
        assert metrics.sampling_time > 0
        assert metrics.overhead_time > 0

    def test_profile_false_no_timing(self, models):
        """profile=False should leave timing fields at 0."""
        from src.speculative.decoding import speculative_decode
        target, draft = models
        input_ids = torch.tensor([[50256]])

        _, metrics = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=10,
            speculation_length=3,
            temperature=1.0,
            profile=False,
        )

        assert metrics.draft_time == 0.0
        assert metrics.verify_time == 0.0
        assert metrics.sampling_time == 0.0
        assert metrics.overhead_time == 0.0

    def test_profile_does_not_affect_output(self, models):
        """Profiling should not change decoding behavior (greedy)."""
        from src.speculative.decoding import speculative_decode
        target, draft = models
        input_ids = torch.tensor([[50256]])

        torch.manual_seed(42)
        out1, m1 = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=10, speculation_length=3,
            temperature=0, profile=False,
        )

        torch.manual_seed(42)
        out2, m2 = speculative_decode(
            target, draft, input_ids,
            max_new_tokens=10, speculation_length=3,
            temperature=0, profile=True,
        )

        assert torch.equal(out1, out2)
        assert m1.total_tokens == m2.total_tokens


# ──────────────────────────────────────────────────────────────
# Quality Metrics Tests
# ──────────────────────────────────────────────────────────────


class TestQualityMetrics:
    """Tests for domain-specific quality evaluators."""

    def test_code_syntax_valid(self):
        from src.evaluation.quality import compute_quality_metrics
        outputs = [
            {"prompt": "", "generated": "def foo():\n    return 42\n", "domain": "code"},
            {"prompt": "", "generated": "x = [1, 2, 3]", "domain": "code"},
        ]
        result = compute_quality_metrics(outputs, "humaneval")
        assert result["metric"] == "syntax_validity"
        assert result["score"] == 1.0
        assert result["valid_count"] == 2

    def test_code_syntax_invalid(self):
        from src.evaluation.quality import compute_quality_metrics
        outputs = [
            {"prompt": "", "generated": "def foo(:\n", "domain": "code"},
        ]
        result = compute_quality_metrics(outputs, "humaneval")
        assert result["score"] == 0.0

    def test_math_exact_match(self):
        from src.evaluation.quality import compute_quality_metrics
        outputs = [
            {"prompt": "", "generated": "The answer is 42.", "domain": "math", "reference": "42"},
            {"prompt": "", "generated": "I think it's 100", "domain": "math", "reference": "50"},
        ]
        result = compute_quality_metrics(outputs, "gsm8k")
        assert result["metric"] == "exact_match"
        assert result["correct_count"] == 1
        assert result["score"] == 0.5

    def test_math_no_reference(self):
        from src.evaluation.quality import compute_quality_metrics
        outputs = [
            {"prompt": "", "generated": "The answer is 42.", "domain": "math", "reference": ""},
        ]
        result = compute_quality_metrics(outputs, "gsm8k")
        assert result["correct_count"] == 0

    def test_conversation_quality(self):
        from src.evaluation.quality import compute_quality_metrics
        outputs = [
            {"prompt": "", "generated": "This is a response with some words", "domain": "conversation"},
            {"prompt": "", "generated": "Another different reply here", "domain": "conversation"},
        ]
        result = compute_quality_metrics(outputs, "mt_bench")
        assert result["metric"] == "response_quality"
        assert result["mean_length_words"] > 0
        assert 0 <= result["score"] <= 1.0

    def test_factuality_match(self):
        from src.evaluation.quality import compute_quality_metrics
        outputs = [
            {"prompt": "", "generated": "The capital of France is Paris.", "domain": "factuality", "reference": "Paris"},
            {"prompt": "", "generated": "I don't know", "domain": "factuality", "reference": "Paris"},
        ]
        result = compute_quality_metrics(outputs, "truthfulqa")
        assert result["metric"] == "reference_match"
        assert result["match_count"] == 1
        assert result["score"] == 0.5

    def test_empty_outputs(self):
        from src.evaluation.quality import compute_quality_metrics
        result = compute_quality_metrics([], "humaneval")
        assert result["total"] == 0


# ──────────────────────────────────────────────────────────────
# BenchmarkConfig Tests
# ──────────────────────────────────────────────────────────────


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig defaults and YAML loading."""

    def test_defaults(self):
        """BenchmarkConfig should have sensible defaults."""
        bc = BenchmarkConfig()
        assert "baseline" in bc.experiments
        assert "adaptive_routing" in bc.experiments
        assert 5 in bc.speculation_lengths
        assert bc.generic_draft_model == "distilgpt2"
        assert bc.profile is False
        assert bc.warmup_samples == 3

    def test_experiment_config_has_benchmark(self):
        """ExperimentConfig should include a BenchmarkConfig field."""
        config = ExperimentConfig()
        assert hasattr(config, "benchmark")
        assert isinstance(config.benchmark, BenchmarkConfig)

    def test_yaml_loading(self, tmp_path):
        """Loading a YAML with benchmark section should populate BenchmarkConfig."""
        config_data = {
            "benchmark": {
                "experiments": ["baseline", "generic_draft"],
                "profile": True,
                "warmup_samples": 5,
            },
        }
        config_file = tmp_path / "bench_test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.benchmark.experiments == ["baseline", "generic_draft"]
        assert config.benchmark.profile is True
        assert config.benchmark.warmup_samples == 5
        # Defaults preserved
        assert config.benchmark.generic_draft_model == "distilgpt2"

    def test_backward_compat_base_yaml(self):
        """configs/base.yaml (no benchmark section) should load with defaults."""
        config = load_config("configs/base.yaml")
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert "baseline" in config.benchmark.experiments

    def test_benchmark_yaml_loads(self):
        """configs/benchmark.yaml should load without errors."""
        config = load_config("configs/benchmark.yaml")
        assert config.benchmark.profile is False
        assert "ablation_K" in config.benchmark.experiments
        assert config.model.target_model == "Qwen/Qwen2.5-7B"

    def test_benchmark_profile_yaml_loads(self):
        """configs/benchmark_profile.yaml should load with profile=True."""
        config = load_config("configs/benchmark_profile.yaml")
        assert config.benchmark.profile is True


# ──────────────────────────────────────────────────────────────
# Visualization Tests
# ──────────────────────────────────────────────────────────────


class TestVisualization:
    """Tests that chart generation functions don't crash."""

    @pytest.fixture
    def sample_results(self):
        """Create minimal benchmark results for testing."""
        return {
            "baseline": {
                "method": "baseline",
                "summary": {"mean_tokens_per_second": 10.0, "mean_acceptance_rate": 0},
                "outputs": [
                    {"domain": "code", "metrics": {"tokens_per_second": 10, "latency_seconds": 1.0,
                                                    "accepted_tokens": 0, "draft_tokens": 0}},
                    {"domain": "math", "metrics": {"tokens_per_second": 12, "latency_seconds": 0.8,
                                                    "accepted_tokens": 0, "draft_tokens": 0}},
                ],
                "quality": {"score": 0.8},
            },
            "generic_draft": {
                "method": "generic_draft",
                "summary": {"mean_tokens_per_second": 18.0, "mean_acceptance_rate": 0.7},
                "outputs": [
                    {"domain": "code", "metrics": {"tokens_per_second": 18, "latency_seconds": 0.6,
                                                    "accepted_tokens": 7, "draft_tokens": 10}},
                    {"domain": "math", "metrics": {"tokens_per_second": 20, "latency_seconds": 0.5,
                                                    "accepted_tokens": 8, "draft_tokens": 10}},
                ],
                "quality": {"score": 0.75},
            },
            "adaptive_routing": {
                "method": "adaptive_routing",
                "summary": {"mean_tokens_per_second": 22.0, "mean_acceptance_rate": 0.8},
                "outputs": [
                    {"domain": "code", "selected_draft": "code_draft",
                     "metrics": {"tokens_per_second": 22, "latency_seconds": 0.5,
                                 "accepted_tokens": 8, "draft_tokens": 10}},
                    {"domain": "math", "selected_draft": "reasoning_draft",
                     "metrics": {"tokens_per_second": 24, "latency_seconds": 0.4,
                                 "accepted_tokens": 9, "draft_tokens": 10}},
                ],
                "quality": {"score": 0.82},
            },
            "ablation_K": {
                "method": "ablation_K",
                "K=3": {"summary": {"mean_tokens_per_second": 15, "mean_acceptance_rate": 0.85}},
                "K=5": {"summary": {"mean_tokens_per_second": 18, "mean_acceptance_rate": 0.75}},
                "K=7": {"summary": {"mean_tokens_per_second": 16, "mean_acceptance_rate": 0.65}},
            },
        }

    def test_generate_all_charts(self, sample_results, tmp_path):
        """generate_all_charts should create PNG files."""
        from src.evaluation.visualization import generate_all_charts
        output_dir = str(tmp_path / "charts")
        generate_all_charts(sample_results, output_dir)

        # Check that at least some chart files were created
        chart_dir = Path(output_dir)
        assert chart_dir.exists()
        png_files = list(chart_dir.glob("*.png"))
        assert len(png_files) > 0

    def test_speedup_by_domain(self, sample_results, tmp_path):
        """Speedup by domain chart should be created."""
        from src.evaluation.visualization import plot_speedup_by_domain
        output_path = str(tmp_path / "speedup_domain.png")
        plot_speedup_by_domain(sample_results, output_path)
        assert Path(output_path).exists()

    def test_acceptance_over_samples(self, sample_results, tmp_path):
        """Acceptance chart should be created."""
        from src.evaluation.visualization import plot_acceptance_over_samples
        output_path = str(tmp_path / "acceptance.png")
        plot_acceptance_over_samples(sample_results, output_path)
        assert Path(output_path).exists()

    def test_quality_vs_speed(self, sample_results, tmp_path):
        """Quality vs speed chart should be created."""
        from src.evaluation.visualization import plot_quality_vs_speed
        output_path = str(tmp_path / "quality_speed.png")
        plot_quality_vs_speed(sample_results, output_path)
        assert Path(output_path).exists()

    def test_speedup_vs_K(self, sample_results, tmp_path):
        """Speedup vs K chart should be created."""
        from src.evaluation.visualization import plot_speedup_vs_K
        output_path = str(tmp_path / "speedup_k.png")
        plot_speedup_vs_K(sample_results["ablation_K"], output_path)
        assert Path(output_path).exists()

    def test_latency_distribution(self, sample_results, tmp_path):
        """Latency distribution chart should be created."""
        from src.evaluation.visualization import plot_latency_distribution
        output_path = str(tmp_path / "latency_dist.png")
        plot_latency_distribution(sample_results, output_path)
        assert Path(output_path).exists()

    def test_draft_usage(self, sample_results, tmp_path):
        """Draft usage chart should be created."""
        from src.evaluation.visualization import plot_draft_usage
        output_path = str(tmp_path / "draft_usage.png")
        plot_draft_usage(sample_results["adaptive_routing"], output_path)
        assert Path(output_path).exists()


# ──────────────────────────────────────────────────────────────
# Analysis Tests
# ──────────────────────────────────────────────────────────────


class TestAnalysis:
    """Tests for the analysis module."""

    @pytest.fixture
    def sample_results(self):
        """Create benchmark results for analysis testing."""
        return {
            "baseline": {
                "method": "baseline",
                "summary": {"mean_tokens_per_second": 10.0, "mean_acceptance_rate": 0},
                "outputs": [
                    {"domain": "code", "metrics": {"tokens_per_second": 10, "acceptance_rate": 0}},
                ],
            },
            "generic_draft": {
                "method": "generic_draft",
                "summary": {"mean_tokens_per_second": 18.0, "mean_acceptance_rate": 0.7},
                "outputs": [
                    {"domain": "code", "metrics": {"tokens_per_second": 18, "acceptance_rate": 0.7}},
                ],
                "quality": {"score": 0.75},
            },
            "adaptive_routing": {
                "method": "adaptive_routing",
                "summary": {"mean_tokens_per_second": 22.0, "mean_acceptance_rate": 0.8},
                "outputs": [
                    {"domain": "code", "metrics": {"tokens_per_second": 22, "acceptance_rate": 0.8}},
                ],
                "quality": {"score": 0.82},
            },
            "ablation_K": {
                "method": "ablation_K",
                "K=3": {"summary": {"mean_tokens_per_second": 15, "mean_acceptance_rate": 0.85}},
                "K=5": {"summary": {"mean_tokens_per_second": 18, "mean_acceptance_rate": 0.75}},
                "K=7": {"summary": {"mean_tokens_per_second": 16, "mean_acceptance_rate": 0.65}},
            },
        }

    def test_analyze_benchmark_keys(self, sample_results):
        """analyze_benchmark() should return all expected keys."""
        from src.evaluation.analysis import analyze_benchmark
        analysis = analyze_benchmark(sample_results)

        assert "best_draft_per_domain" in analysis
        assert "routing_vs_best_single" in analysis
        assert "quality_speed_tradeoffs" in analysis
        assert "failure_cases" in analysis
        assert "optimal_K" in analysis

    def test_routing_improvement(self, sample_results):
        """Routing improvement should be computed correctly."""
        from src.evaluation.analysis import analyze_benchmark
        analysis = analyze_benchmark(sample_results)

        routing = analysis["routing_vs_best_single"]
        assert routing["routing_tps"] == 22.0
        assert routing["best_single_tps"] == 18.0
        assert routing["improvement_pct"] > 0

    def test_optimal_K(self, sample_results):
        """Should find optimal K from ablation results."""
        from src.evaluation.analysis import analyze_benchmark
        analysis = analyze_benchmark(sample_results)

        optimal = analysis["optimal_K"]
        assert optimal["best_K"] == 5  # K=5 has highest tps
        assert optimal["best_tps"] == 18

    def test_quality_speed_tradeoffs(self, sample_results):
        """Quality-speed tradeoffs should be sorted by speed."""
        from src.evaluation.analysis import analyze_benchmark
        analysis = analyze_benchmark(sample_results)

        tradeoffs = analysis["quality_speed_tradeoffs"]
        assert len(tradeoffs) > 0
        # Should be sorted descending by tokens_per_second
        for i in range(len(tradeoffs) - 1):
            assert tradeoffs[i]["tokens_per_second"] >= tradeoffs[i + 1]["tokens_per_second"]

    def test_failure_cases_low_acceptance(self, sample_results):
        """Should find samples with low acceptance rates."""
        from src.evaluation.analysis import analyze_benchmark
        # Add a low-acceptance sample
        sample_results["generic_draft"]["outputs"].append({
            "domain": "code",
            "prompt": "some hard prompt",
            "metrics": {"tokens_per_second": 5, "acceptance_rate": 0.1},
        })
        analysis = analyze_benchmark(sample_results)

        failures = analysis["failure_cases"]
        assert len(failures) > 0
        assert failures[0]["acceptance_rate"] < 0.2

    def test_generate_report(self, sample_results, tmp_path):
        """generate_report() should write a text file."""
        from src.evaluation.analysis import analyze_benchmark, generate_report
        analysis = analyze_benchmark(sample_results)
        report_path = str(tmp_path / "report.txt")
        report_text = generate_report(analysis, report_path)

        assert Path(report_path).exists()
        assert "BENCHMARK REPORT" in report_text
        assert len(report_text) > 100

    def test_analyze_empty_results(self):
        """analyze_benchmark with empty results should not crash."""
        from src.evaluation.analysis import analyze_benchmark
        analysis = analyze_benchmark({})
        assert "best_draft_per_domain" in analysis
        assert analysis["optimal_K"]["best_K"] is None
