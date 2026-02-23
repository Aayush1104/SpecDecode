# SpecDecode

**Speculative decoding for faster LLM inference**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

SpecDecode speeds up LLM inference by having a small draft model propose tokens that a larger target model verifies in parallel. The output distribution stays identical to the target model alone. An adaptive router picks the best draft model for each prompt.

## Architecture

```
                    ┌─────────────────────────────┐
                    │         Input Prompt         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Adaptive Router         │
                    │  (MLP + Sentence Embeddings) │
                    └──┬─────────┬─────────┬──────┘
                       │         │         │
              ┌────────▼──┐ ┌───▼────┐ ┌──▼────────┐
              │Code Draft │ │Chat    │ │Reasoning  │
              │Model      │ │Draft   │ │Draft      │
              └────────┬──┘ └───┬────┘ └──┬────────┘
                       │         │         │
                    ┌──▼─────────▼─────────▼──────┐
                    │   Speculative Decode Loop    │
                    │  ┌─────────────────────────┐ │
                    │  │ 1. Draft K tokens       │ │
                    │  │ 2. Verify with target   │ │
                    │  │ 3. Rejection sampling   │ │
                    │  │ 4. Accept/reject + bonus│ │
                    │  └─────────────────────────┘ │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Generated Output         │
                    │  (identical distribution to  │
                    │   target model alone)        │
                    └─────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/Aayush1104/specdecode.git
cd specdecode

# Install in development mode
pip install -e .

# Optional
pip install -e ".[demo]"   # Gradio demo
pip install -e ".[dev]"    # Testing (pytest)
pip install -e ".[flash]"  # Flash Attention
```

## Quick Start

### Python API

```python
import torch
from src.speculative.backends import create_backend
from src.speculative.decoding import speculative_decode
from transformers import AutoTokenizer

target = create_backend("Qwen/Qwen2.5-7B", dtype="bfloat16")
draft = create_backend("Qwen/Qwen2.5-0.5B", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

input_ids = tokenizer.encode("def fibonacci(n):", return_tensors="pt").cuda()
output_ids, metrics = speculative_decode(target, draft, input_ids, max_new_tokens=128)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
print(f"Speedup: {metrics.tokens_per_second:.1f} tok/s, acceptance: {metrics.acceptance_rate:.0%}")
```

### CLI

```bash
# Generate text
specdecode generate --model Qwen/Qwen2.5-7B --draft Qwen/Qwen2.5-0.5B --prompt "Hello, world"

# Run benchmark suite
specdecode benchmark --config configs/benchmark.yaml

# Run profiling
specdecode profile --config configs/benchmark_profile.yaml --num-samples 5

# Evaluate on datasets
specdecode evaluate --config configs/eval_full.yaml --dataset humaneval
```

### Gradio Demo

```bash
pip install -e ".[demo]"
python3 demo/app.py
# Open http://localhost:7860
```

## Usage Guide

There are three stages to getting the most out of SpecDecode. You can skip straight to benchmarking if you just want to use off-the-shelf models. Training draft models and the router is only needed if you want domain-specialized performance.

### 1. Benchmarking with Pre-trained Models

The simplest way to use SpecDecode is with existing HuggingFace models. Pick a target and a smaller draft model from the same family.

```bash
# Quick test with the CLI
specdecode generate \
  --model Qwen/Qwen2.5-7B \
  --draft Qwen/Qwen2.5-0.5B \
  --prompt "Write a Python function that sorts a list" \
  --max-tokens 128 \
  --K 5

# Full benchmark suite
specdecode benchmark --config configs/benchmark.yaml

# Run specific experiments only
specdecode benchmark --config configs/benchmark.yaml --experiments baseline,generic_draft

# Benchmark with profiling to see where time is spent
specdecode benchmark --config configs/benchmark_profile.yaml
```

Good model pairings to try (target + draft, same tokenizer family)

| Target | Draft | Notes |
|--------|-------|-------|
| Qwen/Qwen2.5-14B-Instruct | Qwen/Qwen2.5-0.5B | Best speedup on larger targets |
| Qwen/Qwen2.5-7B | Qwen/Qwen2.5-0.5B | Good baseline pairing |
| Qwen/Qwen2.5-7B | Qwen/Qwen2.5-1.5B | Higher acceptance, slower draft |

Speculative decoding helps most when the target model is large relative to the draft. On fast hardware like H100s, a 7B target may already be fast enough that the draft overhead cancels out the gains. With 14B+ targets, you should see real speedups.

### 2. Training Domain-Specialized Draft Models

Fine-tuning a draft model on domain-specific data improves its acceptance rate. A code-tuned draft will match the target better on code prompts, so more tokens get accepted and you get more speedup.

**Fine-tuning** trains the draft on domain data with standard language modeling loss.

```bash
accelerate launch scripts/train_draft.py --config configs/train_code.yaml
```

The default config (`configs/train_code.yaml`) trains Qwen2.5-1.5B on Python code from the `transformersbook/codeparrot` dataset. Edit the config to change the model, dataset, or training parameters.

**Knowledge distillation** trains the draft to match the target model's output distribution directly. This tends to produce higher acceptance rates than plain fine-tuning because the draft learns to mimic the target's behavior, not just the data.

```bash
accelerate launch scripts/train_draft.py --config configs/distill_code.yaml
```

Distillation loads both the teacher (target) and student (draft) models, so it uses more GPU memory. Run with `--num_processes=1` if you hit OOM errors.

Key config options in `configs/train_code.yaml`

| Field | What it does | Default |
|-------|-------------|---------|
| `model.draft_model` | Base model to fine-tune | Qwen/Qwen2.5-1.5B |
| `model.target_model` | Target model (for distillation) | Qwen/Qwen2.5-7B |
| `data.domain` | Training domain (code, chat, reasoning) | code |
| `data.dataset_name` | HuggingFace dataset | transformersbook/codeparrot |
| `training.num_train_steps` | Total training steps | 50000 |
| `training.per_device_batch_size` | Batch size per GPU | 4 |
| `distillation.enabled` | Use knowledge distillation | false |
| `logging.use_wandb` | Log to Weights & Biases | false |

Checkpoints are saved to `training.output_dir` (default `checkpoints/draft-code/`). The best checkpoint by validation loss goes in the `best/` subdirectory.

Once training finishes, you can use your trained draft directly.

```python
target = create_backend("Qwen/Qwen2.5-7B", dtype="bfloat16")
draft = create_backend("checkpoints/draft-code/best", dtype="bfloat16")
```

### 3. Training the Adaptive Router

The router is an MLP that picks the best draft model for each prompt. You need at least two trained draft models for this to be useful. If you only have one, skip this step.

**Step 1. Collect performance data.** This runs speculative decoding with each draft model on a set of prompts and records which draft had the best acceptance rate.

```bash
python3 scripts/collect_router_data.py --config configs/router_collect.yaml
```

Before running, update `configs/router_collect.yaml` with the paths to your trained draft models.

```yaml
router:
  draft_models:
    code: "checkpoints/draft-code/best"
    chat: "checkpoints/draft-chat/best"
    reasoning: "Qwen/Qwen2.5-0.5B"  # or another trained checkpoint
```

This saves training data to `data/router_training.json`.

**Step 2. Train the router MLP.**

```bash
python3 scripts/train_router.py --config configs/router_train.yaml
```

The router trains quickly since it's just a small feedforward network over sentence embeddings.

**Using the router at inference time.**

```python
from src.routing.router import AdaptiveRouter

router = AdaptiveRouter.from_config(config)
output_ids, metrics = router.route_and_decode(target, draft_models, input_ids)
```

## How It Works

### The Algorithm

Speculative decoding works in iterations. Each iteration has four steps.

1. The draft model generates K tokens one at a time (fast, small model)
2. The target model scores all K tokens in a single forward pass (one slow call instead of K)
3. Each draft token is accepted with probability min(1, p_target/p_draft)
4. On rejection at position j, a replacement token is sampled from norm(max(0, p_target - p_draft)). If all K tokens pass, a bonus token is sampled from the target's logits at position K.

The math guarantees the output distribution is identical to running the target model alone. You get the same quality with fewer target model calls.

### Why It's Faster

The target model is the bottleneck. Each forward pass loads billions of parameters from memory, which takes the same time whether you're scoring 1 token or K tokens (it's memory-bandwidth bound, not compute bound). By batching K draft tokens into one verification call, you get up to K+1 tokens per target forward pass instead of 1.

The speedup depends on how often draft tokens get accepted. Higher acceptance means more tokens per iteration.

### KV Cache Management

The decode loop maintains a careful invariant. At each iteration start, the draft KV cache is one position ahead of the target's (the target hasn't seen the bonus token yet). The bonus token gets folded into the next iteration's verification pass, which saves one target forward call per iteration.

After rejection sampling, both caches are trimmed back to remove rejected positions, and the draft model processes the bonus token to set up the next iteration.

### Adaptive Routing

Different prompts work better with different draft models. Code prompts get higher acceptance rates with a code-specialized draft. Math prompts do better with a reasoning draft.

The router is a small MLP that takes sentence embeddings (768-dim from `all-mpnet-base-v2`) plus a few extra features (prompt length, domain hint scores) and predicts which draft model will have the highest acceptance rate. It adds negligible latency since the classification happens once per prompt.

## Project Structure

```
specdecode/
├── src/
│   ├── speculative/           # Core speculative decoding
│   │   ├── decoding.py        # Main decode loop + standard baseline
│   │   ├── backends.py        # ModelBackend ABC + HuggingFace impl
│   │   ├── rejection_sampling.py  # Token acceptance/rejection
│   │   └── kv_cache.py        # KV cache trimming utilities
│   ├── draft_models/          # Draft model training
│   │   ├── trainer.py         # Fine-tuning trainer
│   │   ├── distiller.py       # Knowledge distillation
│   │   └── data.py            # Domain data pipeline
│   ├── routing/               # Adaptive routing
│   │   ├── router.py          # AdaptiveRouter + route_and_decode()
│   │   ├── model.py           # RouterMLP classifier
│   │   ├── features.py        # Feature extraction (sentence embeddings)
│   │   └── trainer.py         # Router training loop
│   ├── evaluation/            # Evaluation and benchmarking
│   │   ├── benchmark.py       # BenchmarkRunner (5 experiment types)
│   │   ├── evaluator.py       # Standard vs speculative evaluator
│   │   ├── datasets.py        # Dataset loaders (HumanEval, GSM8K, etc.)
│   │   ├── quality.py         # Domain-specific quality metrics
│   │   ├── visualization.py   # Chart generation
│   │   └── analysis.py        # Benchmark analysis and report
│   ├── utils/                 # Shared utilities
│   │   ├── config.py          # YAML config with dataclasses
│   │   ├── metrics.py         # DecodingMetrics and MetricsTracker
│   │   ├── timing.py          # CUDA-aware timing
│   │   └── logging.py         # Structured logging + WandB
│   └── cli.py                 # CLI entry point
├── demo/
│   └── app.py                 # Gradio demo
├── configs/                   # YAML configuration files
├── scripts/                   # Training and data collection scripts
├── tests/                     # Test suite
├── docs/                      # Documentation
│   ├── technical_report.md    # Technical report
│   └── model_card_template.md # HuggingFace model card template
└── pyproject.toml
```

## Benchmark Experiments

The benchmark suite (`specdecode benchmark`) runs 5 experiments.

| # | Experiment | What it does |
|---|-----------|-------------|
| 1 | `baseline` | Standard autoregressive decoding (target only) |
| 2 | `generic_draft` | Speculative decoding with a generic small model |
| 3 | `specialized_drafts` | Each domain-specialized draft model on its own |
| 4 | `adaptive_routing` | Router picks the best draft model per prompt |
| 5 | `ablation_K` | Tests speculation length K at 3, 5, and 7 |

Results include throughput (tokens/sec), acceptance rate, latency percentiles, and domain-specific quality metrics.

## Configuration

All parameters live in YAML files. See `configs/` for examples.

| Config | Purpose |
|--------|---------|
| `configs/base.yaml` | Minimal config |
| `configs/benchmark.yaml` | Full benchmark suite |
| `configs/benchmark_profile.yaml` | Benchmarking with profiling |
| `configs/eval_full.yaml` | Multi-dataset evaluation |
| `configs/train_code.yaml` | Code draft model training |
| `configs/distill_code.yaml` | Knowledge distillation |
| `configs/router_collect.yaml` | Router data collection |
| `configs/router_train.yaml` | Router training |

## Citation

```bibtex
@software{specdecode2024,
  title={SpecDecode: Speculative Decoding with Adaptive Routing},
  author={Aayush},
  year={2024},
  url={https://github.com/Aayush1104/specdecode}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
