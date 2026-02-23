"""Model backend abstraction for speculative decoding."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def generate_with_logits(
        self,
        input_ids: Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        past_key_values=None,
    ) -> tuple[Tensor, Tensor, object]:
        """Generate tokens autoregressively and return logits.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            past_key_values: Optional KV cache

        Returns:
            (generated_tokens, logits, kv_cache)
            - generated_tokens: (batch, num_tokens)
            - logits: (batch, num_tokens, vocab_size)
            - kv_cache: updated KV cache
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        past_key_values=None,
    ) -> tuple[Tensor, object]:
        """Run forward pass and return logits.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            past_key_values: Optional KV cache

        Returns:
            (logits, kv_cache)
            - logits: (batch, seq_len, vocab_size)
            - kv_cache: updated KV cache
        """
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...


class HuggingFaceBackend(ModelBackend):
    """Backend using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        # Resolve local paths for from_pretrained
        local_path = Path(model_name)
        load_name = str(local_path) if local_path.exists() else model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            load_name,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device == "auto" else None,
        )
        if self.device != "auto":
            self.model = self.model.to(self.device)
        self.model.eval()
        self._device = next(self.model.parameters()).device

        # For local checkpoints, tokenizer may be in the same dir or parent
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_name)
        except OSError:
            # Fallback: try parent directories for tokenizer
            if local_path.exists():
                for parent in [local_path.parent, local_path.parent.parent]:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(str(parent))
                        break
                    except OSError:
                        continue
                else:
                    raise OSError(
                        f"Could not find tokenizer for local checkpoint '{model_name}'. "
                        f"Ensure tokenizer files are saved alongside the model."
                    )
            else:
                raise
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    @torch.no_grad()
    def generate_with_logits(
        self,
        input_ids: Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        past_key_values=None,
    ) -> tuple[Tensor, Tensor, object]:
        input_ids = input_ids.to(self._device)

        all_tokens = []
        all_logits = []
        current_ids = input_ids

        for _ in range(num_tokens):
            outputs = self.model(
                current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1:, :]  # (batch, 1, vocab)
            past_key_values = outputs.past_key_values

            if temperature > 0:
                probs = torch.softmax(logits[:, 0, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            else:
                next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)

            all_tokens.append(next_token)
            all_logits.append(logits)
            current_ids = next_token

        tokens = torch.cat(all_tokens, dim=1)  # (batch, num_tokens)
        logits = torch.cat(all_logits, dim=1)   # (batch, num_tokens, vocab)

        return tokens, logits, past_key_values

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        past_key_values=None,
    ) -> tuple[Tensor, object]:
        input_ids = input_ids.to(self._device)

        outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size


class VLLMBackend(ModelBackend):
    """Backend using vLLM. Stub for future implementation."""

    def __init__(self, model_name: str, **kwargs):
        raise NotImplementedError(
            "vLLM backend is not yet implemented. Use 'huggingface' backend."
        )

    def generate_with_logits(self, input_ids, num_tokens, temperature=1.0, past_key_values=None):
        raise NotImplementedError

    def forward(self, input_ids, past_key_values=None):
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        raise NotImplementedError


def create_backend(
    model_name: str,
    backend_type: str = "huggingface",
    dtype: str = "float16",
    device: str = "auto",
) -> ModelBackend:
    """Factory function to create a model backend."""
    if backend_type == "huggingface":
        return HuggingFaceBackend(model_name, dtype=dtype, device=device)
    elif backend_type == "vllm":
        return VLLMBackend(model_name)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
