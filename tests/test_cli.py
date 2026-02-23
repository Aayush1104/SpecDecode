"""Tests for the CLI module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.cli import build_parser, main


class TestBuildParser:
    """Tests for CLI argument parsing."""

    def test_parser_creates_subcommands(self):
        parser = build_parser()
        # Parser should have subcommands
        assert parser._subparsers is not None

    def test_no_args_command_is_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    # ── generate subcommand ──────────────────────────────────

    def test_generate_required_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate", "--model", "gpt2", "--draft", "distilgpt2",
            "--prompt", "Hello world",
        ])
        assert args.command == "generate"
        assert args.model == "gpt2"
        assert args.draft == "distilgpt2"
        assert args.prompt == "Hello world"

    def test_generate_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate", "--model", "gpt2", "--draft", "distilgpt2",
            "--prompt", "test",
        ])
        assert args.K == 5
        assert args.max_tokens == 128
        assert args.temperature == 1.0
        assert args.dtype == "float16"
        assert args.device == "auto"

    def test_generate_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate", "--model", "gpt2-medium", "--draft", "gpt2",
            "--prompt", "test", "--K", "3", "--max-tokens", "64",
            "--temperature", "0.5", "--dtype", "bfloat16", "--device", "cpu",
        ])
        assert args.K == 3
        assert args.max_tokens == 64
        assert args.temperature == 0.5
        assert args.dtype == "bfloat16"
        assert args.device == "cpu"

    def test_generate_missing_model_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--draft", "distilgpt2", "--prompt", "hi"])

    def test_generate_missing_draft_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--model", "gpt2", "--prompt", "hi"])

    def test_generate_missing_prompt_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--model", "gpt2", "--draft", "distilgpt2"])

    # ── benchmark subcommand ─────────────────────────────────

    def test_benchmark_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"
        assert args.config == "configs/benchmark.yaml"
        assert args.experiments is None

    def test_benchmark_custom_config(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_benchmark_experiments_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "benchmark", "--experiments", "baseline,generic_draft",
        ])
        assert args.experiments == "baseline,generic_draft"

    # ── profile subcommand ───────────────────────────────────

    def test_profile_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["profile"])
        assert args.command == "profile"
        assert args.config == "configs/benchmark_profile.yaml"
        assert args.num_samples is None
        assert args.torch_profile is False

    def test_profile_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "profile", "--config", "custom.yaml",
            "--num-samples", "10", "--torch-profile",
        ])
        assert args.config == "custom.yaml"
        assert args.num_samples == 10
        assert args.torch_profile is True

    # ── evaluate subcommand ──────────────────────────────────

    def test_evaluate_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate"])
        assert args.command == "evaluate"
        assert args.config == "configs/eval_full.yaml"
        assert args.dataset is None
        assert args.num_samples is None

    def test_evaluate_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "evaluate", "--config", "my_eval.yaml",
            "--dataset", "humaneval", "--num-samples", "50",
        ])
        assert args.config == "my_eval.yaml"
        assert args.dataset == "humaneval"
        assert args.num_samples == 50


class TestMainEntryPoint:
    """Tests for the main() function dispatch."""

    def test_no_command_prints_help_and_exits(self):
        with patch("sys.argv", ["specdecode"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("src.cli.cmd_generate")
    def test_generate_dispatches(self, mock_cmd):
        with patch("sys.argv", [
            "specdecode", "generate",
            "--model", "gpt2", "--draft", "distilgpt2", "--prompt", "hi",
        ]):
            main()
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert args.model == "gpt2"

    @patch("src.cli.cmd_benchmark")
    def test_benchmark_dispatches(self, mock_cmd):
        with patch("sys.argv", ["specdecode", "benchmark"]):
            main()
        mock_cmd.assert_called_once()

    @patch("src.cli.cmd_profile")
    def test_profile_dispatches(self, mock_cmd):
        with patch("sys.argv", ["specdecode", "profile"]):
            main()
        mock_cmd.assert_called_once()

    @patch("src.cli.cmd_evaluate")
    def test_evaluate_dispatches(self, mock_cmd):
        with patch("sys.argv", ["specdecode", "evaluate"]):
            main()
        mock_cmd.assert_called_once()


class TestCmdGenerate:
    """Tests for the generate subcommand with mocked backends."""

    @patch("transformers.AutoTokenizer")
    @patch("src.speculative.backends.create_backend")
    @patch("src.speculative.decoding.speculative_decode")
    def test_generate_calls_speculative_decode(
        self, mock_decode, mock_create, mock_tokenizer_cls
    ):
        import torch
        from src.utils.metrics import DecodingMetrics

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tok.decode.return_value = "generated text"
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok

        # Mock backend — need model.parameters() to return a real device
        mock_param = torch.nn.Parameter(torch.zeros(1))
        mock_backend = MagicMock()
        mock_backend.model.parameters.return_value = iter([mock_param])
        mock_create.return_value = mock_backend

        # Mock decode output
        metrics = DecodingMetrics(
            total_tokens=10, total_steps=5,
            accepted_tokens=8, draft_tokens=10,
            latency_seconds=1.0, peak_memory_mb=100.0,
            draft_time=0.1, verify_time=0.5,
            sampling_time=0.05, overhead_time=0.05,
        )
        mock_decode.return_value = (torch.tensor([[1, 2, 3, 4, 5]]), metrics)

        # Build args
        from src.cli import cmd_generate
        parser = build_parser()
        args = parser.parse_args([
            "generate", "--model", "gpt2", "--draft", "distilgpt2",
            "--prompt", "Hello", "--K", "3", "--max-tokens", "10",
            "--device", "cpu",
        ])

        # Should not raise
        cmd_generate(args)

        # Verify speculative_decode was called
        mock_decode.assert_called_once()
        call_kwargs = mock_decode.call_args
        assert call_kwargs[1]["speculation_length"] == 3
        assert call_kwargs[1]["max_new_tokens"] == 10
