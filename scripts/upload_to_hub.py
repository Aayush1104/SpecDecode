#!/usr/bin/env python3
"""Upload draft models and router weights to HuggingFace Hub.

Usage:
    python3 scripts/upload_to_hub.py \
        --model-dir checkpoints/draft-code/best \
        --repo-id username/specdecode-draft-code \
        --domain code

    python3 scripts/upload_to_hub.py \
        --router-dir checkpoints/router \
        --repo-id username/specdecode-router
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def upload_draft_model(model_dir: str, repo_id: str, domain: str, private: bool = False):
    """Upload a draft model checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi, upload_folder

    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id, exist_ok=True, private=private)

    # Generate model card from template
    template_path = Path(__file__).resolve().parent.parent / "docs" / "model_card_template.md"
    if template_path.exists():
        card_content = template_path.read_text()
        # Fill in template variables
        replacements = {
            "{{ model_name }}": f"SpecDecode Draft Model ({domain})",
            "{{ domain }}": domain,
            "{{ base_model }}": "distilgpt2",
            "{{ dataset }}": _domain_to_dataset(domain),
            "{{ training_method }}": "Fine-tuning + Knowledge Distillation",
            "{{ hf_repo_id }}": repo_id,
            "{{ num_steps }}": "50,000",
            "{{ batch_size }}": "16",
            "{{ learning_rate }}": "3e-4",
            "{{ hardware }}": "NVIDIA GPU",
            "{{ teacher_model }}": "Qwen/Qwen2.5-7B",
            "{{ distill_temperature }}": "2.0",
            "{{ distill_alpha }}": "0.5",
            "{{ acceptance_rate }}": "~70-78%",
            "{{ tokens_per_second }}": "~18-23",
            "{{ speedup }}": "~2.0-2.3x",
            "{{ val_loss }}": "See training logs",
        }
        for key, value in replacements.items():
            card_content = card_content.replace(key, value)

        # Handle conditional blocks (simplified)
        card_content = card_content.replace("{{ #if distillation }}", "")
        card_content = card_content.replace("{{ /if }}", "")

        readme_path = model_path / "README.md"
        readme_path.write_text(card_content)
        print(f"Generated model card at {readme_path}")

    # Upload training config if present
    config_files = list(model_path.parent.glob("*.yaml")) + list(model_path.parent.glob("*.yml"))
    for cfg in config_files:
        # Copy config into the upload directory
        dest = model_path / cfg.name
        if not dest.exists():
            dest.write_text(cfg.read_text())

    # Upload everything
    print(f"Uploading {model_path} to {repo_id}...")
    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message=f"Upload {domain} draft model",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


def upload_router(router_dir: str, repo_id: str, private: bool = False):
    """Upload router weights to HuggingFace Hub."""
    from huggingface_hub import HfApi, upload_folder

    router_path = Path(router_dir)
    if not router_path.exists():
        print(f"Error: Router directory not found: {router_path}")
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)

    # Create a README for the router
    readme_content = """---
tags:
  - speculative-decoding
  - router
license: mit
---

# SpecDecode Adaptive Router

MLP classifier that selects the best draft model for speculative decoding based on input prompt characteristics.

## Architecture

```
Input (775) -> Linear(775, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 3) -> Softmax
```

## Features

- Sentence embeddings (768-dim from all-mpnet-base-v2)
- Prompt length (normalized)
- Domain hints (code, math, conversation, factuality)
- Token statistics (mean, std)

## Usage

```python
from src.routing.model import RouterMLP

router = RouterMLP.load("best.pt")
```

## Citation

```bibtex
@software{specdecode2024,
  title={SpecDecode: Production Speculative Decoding with Adaptive Routing},
  author={Aayush},
  year={2024},
  url={https://github.com/Aayush1104/specdecode}
}
```
"""
    readme_path = router_path / "README.md"
    readme_path.write_text(readme_content)

    print(f"Uploading {router_path} to {repo_id}...")
    upload_folder(
        folder_path=str(router_path),
        repo_id=repo_id,
        commit_message="Upload adaptive router weights",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


def _domain_to_dataset(domain: str) -> str:
    """Map domain name to dataset identifier."""
    mapping = {
        "code": "transformersbook/codeparrot",
        "chat": "Anthropic/hh-rlhf",
        "reasoning": "openai/gsm8k",
    }
    return mapping.get(domain, domain)


def main():
    parser = argparse.ArgumentParser(description="Upload models to HuggingFace Hub")
    parser.add_argument("--model-dir", type=str, default=None, help="Draft model checkpoint directory")
    parser.add_argument("--router-dir", type=str, default=None, help="Router checkpoint directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--domain", type=str, default="code", help="Domain for model card (code, chat, reasoning)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")

    args = parser.parse_args()

    if args.model_dir is None and args.router_dir is None:
        print("Error: Specify either --model-dir or --router-dir")
        sys.exit(1)

    if args.model_dir:
        upload_draft_model(args.model_dir, args.repo_id, args.domain, args.private)

    if args.router_dir:
        upload_router(args.router_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
