"""Core rejection sampling for speculative decoding.

Implements the rejection sampling scheme that preserves the target model's
output distribution exactly, as described in:
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
"""

from typing import Optional

import torch
from torch import Tensor


def rejection_sample(
    target_logits: Tensor,
    draft_logits: Tensor,
    draft_tokens: Tensor,
    temperature: float = 1.0,
    bonus_logits: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Perform rejection sampling to accept/reject draft tokens.

    For each position i in [0, K):
    1. Compute p = target_prob[draft_token_i], q = draft_prob[draft_token_i]
    2. Accept with probability min(1, p/q)
    3. On first rejection at position j, sample bonus token from norm(max(0, p - q))

    Args:
        target_logits: Target model logits (batch, K, vocab_size) for verification
        draft_logits: Draft model logits (batch, K, vocab_size) for verification
        draft_tokens: Draft tokens (batch, K)
        temperature: Sampling temperature
        bonus_logits: Target model logits for the bonus position when all K accepted
                      (batch, 1, vocab_size). If None, uses target_logits[:, K-1, :].

    Returns:
        accepted_tokens: Tokens to keep (batch, max_accepted+1), padded with -1
        num_accepted: Number of accepted draft tokens per batch element (batch,)
        bonus_tokens: Bonus token (batch,)
    """
    batch_size, K, vocab_size = target_logits.shape
    device = target_logits.device

    # Apply temperature and compute probabilities
    if temperature > 0:
        target_probs = torch.softmax(target_logits / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
    else:
        # Greedy: put all probability on argmax
        target_probs = torch.zeros_like(target_logits)
        target_probs.scatter_(-1, target_logits.argmax(dim=-1, keepdim=True), 1.0)
        draft_probs = torch.zeros_like(draft_logits)
        draft_probs.scatter_(-1, draft_logits.argmax(dim=-1, keepdim=True), 1.0)

    # Compute bonus probs for the all-accepted case
    if bonus_logits is not None:
        if temperature > 0:
            bonus_probs = torch.softmax(bonus_logits[:, 0, :] / temperature, dim=-1)
        else:
            bonus_probs = torch.zeros(batch_size, vocab_size, device=device)
            bonus_probs.scatter_(-1, bonus_logits[:, 0, :].argmax(dim=-1, keepdim=True), 1.0)
    else:
        bonus_probs = None

    # Gather probabilities of draft tokens under both models
    draft_tokens_expanded = draft_tokens.unsqueeze(-1)  # (batch, K, 1)
    p = target_probs.gather(-1, draft_tokens_expanded).squeeze(-1)  # (batch, K)
    q = draft_probs.gather(-1, draft_tokens_expanded).squeeze(-1)  # (batch, K)

    # Acceptance probability: min(1, p/q)
    # Handle q=0: if q=0 and p>0, accept (p/q -> inf, so min(1,inf) = 1).
    # If both are 0, reject.
    ratio = torch.where(
        q > 0, p / q,
        torch.where(p > 0, torch.ones_like(p), torch.zeros_like(p))
    )
    accept_prob = torch.clamp(ratio, max=1.0)

    # Sample uniform random for acceptance test
    r = torch.rand_like(accept_prob)
    accepted = r < accept_prob  # (batch, K) boolean

    # Find first rejection: contiguous accepted prefix
    accepted_cumulative = accepted.cumprod(dim=1)  # (batch, K)
    num_accepted = accepted_cumulative.sum(dim=1).long()  # (batch,)

    # Build output tokens
    accepted_tokens = torch.full(
        (batch_size, K + 1), -1, dtype=draft_tokens.dtype, device=device
    )
    for b in range(batch_size):
        n = num_accepted[b].item()
        if n > 0:
            accepted_tokens[b, :n] = draft_tokens[b, :n]

    # Sample bonus tokens
    bonus_tokens = _sample_bonus_tokens(
        target_probs, draft_probs, num_accepted, K, bonus_probs
    )

    for b in range(batch_size):
        n = num_accepted[b].item()
        accepted_tokens[b, n] = bonus_tokens[b]

    return accepted_tokens, num_accepted, bonus_tokens


def _sample_bonus_tokens(
    target_probs: Tensor,
    draft_probs: Tensor,
    num_accepted: Tensor,
    K: int,
    bonus_probs: Optional[Tensor],
) -> Tensor:
    """Sample bonus tokens from the adjusted distribution.

    If rejected at position j: sample from norm(max(0, p_target - p_draft)) at position j.
    If all K accepted: sample from bonus_probs (target distribution at position K).
    """
    batch_size = target_probs.shape[0]
    device = target_probs.device

    bonus_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        n = num_accepted[b].item()
        if n < K:
            # Rejection at position n: sample from norm(max(0, p_target - p_draft))
            p_target = target_probs[b, n]
            p_draft = draft_probs[b, n]
            adjusted = torch.clamp(p_target - p_draft, min=0.0)
            adjusted_sum = adjusted.sum()
            if adjusted_sum > 0:
                adjusted = adjusted / adjusted_sum
                bonus_tokens[b] = torch.multinomial(adjusted, num_samples=1).squeeze()
            else:
                bonus_tokens[b] = torch.multinomial(p_target, num_samples=1).squeeze()
        else:
            # All K tokens accepted — sample from bonus distribution
            if bonus_probs is not None:
                bonus_tokens[b] = torch.multinomial(
                    bonus_probs[b], num_samples=1
                ).squeeze()
            else:
                # Fallback: use target distribution at last position
                bonus_tokens[b] = torch.multinomial(
                    target_probs[b, K - 1], num_samples=1
                ).squeeze()

    return bonus_tokens
