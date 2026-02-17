"""Data pipeline for draft model training."""

from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

import datasets as hf_datasets

from src.utils.config import DataConfig


# Domain → (dataset_name, subset, text_field, concat_fields)
DOMAIN_REGISTRY = {
    "code": {
        "dataset_name": "bigcode/the-stack-dedup",
        "subset": "data/python",
        "text_field": "content",
    },
    "chat": {
        "dataset_name": "Anthropic/hh-rlhf",
        "subset": None,
        "text_field": "chosen",
    },
    "reasoning": {
        "dataset_name": "openai/gsm8k",
        "subset": "main",
        "concat_fields": ["question", "answer"],
    },
}


class PackedDataset(IterableDataset):
    """Wraps a token iterator into fixed-length chunks."""

    def __init__(self, token_iterator, max_seq_length: int):
        self.token_iterator = token_iterator
        self.max_seq_length = max_seq_length

    def __iter__(self):
        buffer = []
        for tokens in self.token_iterator:
            buffer.extend(tokens)
            while len(buffer) >= self.max_seq_length:
                chunk = buffer[: self.max_seq_length]
                buffer = buffer[self.max_seq_length :]
                yield torch.tensor(chunk, dtype=torch.long)


class InMemoryDataset(torch.utils.data.Dataset):
    """Simple in-memory dataset for validation."""

    def __init__(self, chunks: list[torch.Tensor]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class DomainDataPipeline:
    """Loads, tokenizes, and chunks domain data for training."""

    def __init__(self, data_config: DataConfig, tokenizer):
        self.config = data_config
        self.tokenizer = tokenizer

    def _get_domain_info(self) -> dict:
        """Get dataset info for the configured domain."""
        domain = self.config.domain
        if domain in DOMAIN_REGISTRY:
            info = DOMAIN_REGISTRY[domain].copy()
            # Allow config overrides
            if self.config.dataset_name != "bigcode/the-stack-dedup":
                info["dataset_name"] = self.config.dataset_name
            if self.config.dataset_subset is not None:
                info["subset"] = self.config.dataset_subset
            return info
        # Custom domain: use config values directly
        return {
            "dataset_name": self.config.dataset_name,
            "subset": self.config.dataset_subset,
            "text_field": "text",
        }

    def _extract_text(self, example: dict, domain_info: dict) -> str:
        """Extract text from a dataset example."""
        if "concat_fields" in domain_info:
            parts = [
                str(example.get(f, "")) for f in domain_info["concat_fields"]
            ]
            return "\n".join(parts)
        field = domain_info.get("text_field", "text")
        return str(example.get(field, ""))

    def _load_streaming_dataset(self, domain_info: dict):
        """Load a streaming HF dataset."""
        kwargs = {"streaming": self.config.streaming}
        if domain_info.get("subset"):
            kwargs["name"] = domain_info["subset"]
        ds = hf_datasets.load_dataset(
            domain_info["dataset_name"], split="train", **kwargs
        )
        return ds

    def _tokenize_stream(self, dataset, domain_info: dict, max_tokens: int):
        """Yield token lists from a streaming dataset up to max_tokens."""
        total_tokens = 0
        for example in dataset:
            if total_tokens >= max_tokens:
                break
            text = self._extract_text(example, domain_info)
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue
            # Add EOS between documents
            tokens.append(self.tokenizer.eos_token_id)
            total_tokens += len(tokens)
            yield tokens

    def _build_val_set(
        self, domain_info: dict
    ) -> list[torch.Tensor]:
        """Build a fixed in-memory validation set."""
        val_tokens = int(self.config.max_tokens * self.config.val_split)
        dataset = self._load_streaming_dataset(domain_info)
        if self.config.streaming:
            dataset = dataset.shuffle(
                seed=self.config.seed, buffer_size=1000
            )

        chunks = []
        buffer = []
        total = 0
        for tokens in self._tokenize_stream(dataset, domain_info, val_tokens):
            buffer.extend(tokens)
            total += len(tokens)
            while len(buffer) >= self.config.max_seq_length:
                chunk = buffer[: self.config.max_seq_length]
                buffer = buffer[self.config.max_seq_length :]
                chunks.append(torch.tensor(chunk, dtype=torch.long))
        return chunks

    def _train_token_iterator(self, domain_info: dict):
        """Yield token lists for training (infinite)."""
        while True:
            dataset = self._load_streaming_dataset(domain_info)
            if self.config.streaming:
                dataset = dataset.shuffle(
                    seed=self.config.seed, buffer_size=10000
                )
            yield from self._tokenize_stream(
                dataset, domain_info, self.config.max_tokens
            )

    def create_dataloaders(
        self, batch_size: int, num_workers: Optional[int] = None
    ) -> tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders.

        Args:
            batch_size: Batch size for both dataloaders.
            num_workers: Number of workers (default: from config).

        Returns:
            (train_dataloader, val_dataloader)
        """
        workers = num_workers if num_workers is not None else self.config.num_workers
        domain_info = self._get_domain_info()

        # Build validation set (materialized in memory)
        val_chunks = self._build_val_set(domain_info)
        val_dataset = InMemoryDataset(val_chunks) if val_chunks else InMemoryDataset([torch.zeros(self.config.max_seq_length, dtype=torch.long)])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # In-memory, no need for workers
        )

        # Build training set (streaming)
        train_dataset = PackedDataset(
            self._train_token_iterator(domain_info),
            self.config.max_seq_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  # Streaming datasets don't support multi-worker
        )

        return train_loader, val_loader
