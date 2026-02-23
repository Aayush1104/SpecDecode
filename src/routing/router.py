"""Adaptive router that selects the best draft model per prompt."""

import torch
from torch import Tensor

from src.routing.features import FeatureExtractor
from src.routing.model import RouterMLP
from src.speculative.backends import ModelBackend
from src.speculative.decoding import speculative_decode
from src.utils.metrics import DecodingMetrics


class AdaptiveRouter:
    """Selects the best draft model for a given prompt using a trained MLP."""

    def __init__(
        self,
        target_model: ModelBackend,
        draft_models: dict[str, ModelBackend],
        router_model: RouterMLP,
        feature_extractor: FeatureExtractor,
        tokenizer,
    ):
        self.target_model = target_model
        self.draft_models = draft_models
        self.draft_names = list(draft_models.keys())
        self.router_model = router_model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def select_draft(self, prompt: str) -> tuple[str, ModelBackend, float]:
        """Select the best draft model for a prompt.

        Args:
            prompt: The input prompt text

        Returns:
            (draft_name, draft_backend, confidence)
        """
        features = self.feature_extractor.extract(prompt, tokenizer=self.tokenizer)
        features = features.unsqueeze(0)  # (1, dim)

        # Move to router model device
        device = next(self.router_model.parameters()).device
        features = features.to(device)

        idx, confidence = self.router_model.predict_with_confidence(features)
        draft_idx = idx.item()
        draft_name = self.draft_names[draft_idx]
        draft_backend = self.draft_models[draft_name]

        return draft_name, draft_backend, confidence.item()


def route_and_decode(
    target_model: ModelBackend,
    draft_models: dict[str, ModelBackend],
    router_model: RouterMLP,
    input_ids: Tensor,
    feature_extractor: FeatureExtractor,
    tokenizer,
    max_new_tokens: int = 128,
    speculation_length: int = 5,
    temperature: float = 1.0,
    draft_names: list[str] | None = None,
    profile: bool = False,
) -> tuple[Tensor, DecodingMetrics, str]:
    """Route a prompt to the best draft model and run speculative decoding.

    1. Decode input_ids to text for feature extraction
    2. Extract features → router selects draft
    3. Call speculative_decode() with selected draft
    4. Return (output_ids, metrics, selected_draft_name)

    Args:
        target_model: The target model backend
        draft_models: Dict mapping draft name → ModelBackend
        router_model: Trained RouterMLP classifier
        input_ids: Input prompt tokens (batch, prompt_len)
        feature_extractor: FeatureExtractor instance
        tokenizer: Tokenizer for decoding input_ids to text
        max_new_tokens: Maximum tokens to generate
        speculation_length: Draft tokens per speculation step
        temperature: Sampling temperature
        draft_names: Optional ordered list of draft names (if None, uses dict keys)

    Returns:
        (output_ids, metrics, selected_draft_name)
    """
    if draft_names is None:
        draft_names = list(draft_models.keys())

    # Decode input_ids to text for feature extraction
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Extract features and select draft
    features = feature_extractor.extract(prompt_text, tokenizer=tokenizer)
    features = features.unsqueeze(0)

    device = next(router_model.parameters()).device
    features = features.to(device)

    idx, confidence = router_model.predict_with_confidence(features)
    draft_idx = idx.item()
    selected_name = draft_names[draft_idx]
    selected_draft = draft_models[selected_name]

    # Run speculative decoding with the selected draft
    output_ids, metrics = speculative_decode(
        target_model=target_model,
        draft_model=selected_draft,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        speculation_length=speculation_length,
        temperature=temperature,
        profile=profile,
    )

    return output_ids, metrics, selected_name
