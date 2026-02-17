"""Main speculative decoding loop and standard decoding baseline."""

import torch
from torch import Tensor

from src.speculative.backends import ModelBackend
from src.speculative.kv_cache import trim_kv_cache
from src.speculative.rejection_sampling import rejection_sample
from src.utils.metrics import DecodingMetrics
from src.utils.timing import cuda_timer, get_peak_memory_mb, reset_peak_memory


def _sample_token(logits: Tensor, temperature: float) -> Tensor:
    """Sample a single token from logits. Returns (batch, 1)."""
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        return logits.argmax(dim=-1, keepdim=True)


def speculative_decode(
    target_model: ModelBackend,
    draft_model: ModelBackend,
    input_ids: Tensor,
    max_new_tokens: int,
    speculation_length: int = 5,
    temperature: float = 1.0,
) -> tuple[Tensor, DecodingMetrics]:
    """Run speculative decoding.

    Invariant maintained at each iteration start:
    - Both KV caches are at the same length (kv_len)
    - next_target_logit holds target's logit for the next position
    - next_draft_logit holds draft's logit for the next position

    Algorithm per iteration:
    1. Draft model generates K tokens (first from cached logit, rest autoregressively)
    2. Target model scores all K tokens in one forward pass
    3. Rejection sampling accepts/rejects draft tokens
    4. KV caches trimmed and bonus token forwarded to restore invariant

    Args:
        target_model: The large target model
        draft_model: The small draft model
        input_ids: Input prompt tokens (batch, prompt_len)
        max_new_tokens: Maximum number of new tokens to generate
        speculation_length: Number of draft tokens per step (K)
        temperature: Sampling temperature

    Returns:
        (output_ids, metrics) where output_ids includes the prompt
    """
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    metrics = DecodingMetrics()
    reset_peak_memory()

    # Process prompt through both models
    target_out, target_kv = target_model.forward(input_ids)
    draft_out, draft_kv = draft_model.forward(input_ids)

    # Both KV caches at prompt_len
    kv_len = prompt_len

    # Cache logits for the next generation position
    next_target_logit = target_out[:, -1:, :]  # (batch, 1, vocab)
    next_draft_logit = draft_out[:, -1:, :]    # (batch, 1, vocab)

    generated = []

    with cuda_timer() as timer:
        while len(generated) < max_new_tokens:
            K = min(speculation_length, max_new_tokens - len(generated))
            metrics.total_steps += 1

            # === DRAFT PHASE ===
            # Generate K tokens from the draft model
            draft_token_list = []
            draft_logit_list = [next_draft_logit]  # logit that generates draft_token[0]

            # First draft token from cached logit (no forward needed)
            t = _sample_token(next_draft_logit[:, 0, :], temperature)
            draft_token_list.append(t)

            # Generate remaining K-1 tokens autoregressively
            for _i in range(1, K):
                step_out, draft_kv = draft_model.forward(t, past_key_values=draft_kv)
                draft_logit_list.append(step_out[:, -1:, :])
                t = _sample_token(step_out[:, -1, :], temperature)
                draft_token_list.append(t)

            draft_tokens = torch.cat(draft_token_list, dim=1)   # (batch, K)
            draft_logits = torch.cat(draft_logit_list, dim=1)   # (batch, K, vocab)
            metrics.draft_tokens += K

            # State: draft_kv at kv_len + K - 1
            # (K-1 forwards; first token was from cached logit)

            # === VERIFICATION PHASE ===
            # Target scores all K draft tokens in one forward pass
            target_step_out, target_kv = target_model.forward(
                draft_tokens, past_key_values=target_kv
            )
            # target_kv now at kv_len + K
            # target_step_out[:, i, :] = target logit after seeing context + draft_tokens[:i+1]
            #   = prediction for the position AFTER draft_tokens[i]

            # Build verification logits (K logits, one per draft token):
            # draft_token[0] verified by next_target_logit (cached, at position kv_len)
            # draft_token[i>0] verified by target_step_out[:, i-1, :] (at position kv_len+i)
            if K > 1:
                target_verify = torch.cat(
                    [next_target_logit, target_step_out[:, :K-1, :]], dim=1
                )
            else:
                target_verify = next_target_logit  # (batch, 1, vocab)

            # Bonus logit for the all-accepted case (at position kv_len + K)
            bonus_logits = target_step_out[:, K-1:K, :]  # (batch, 1, vocab)

            # === REJECTION SAMPLING ===
            _, num_accepted, bonus_tokens = rejection_sample(
                target_logits=target_verify,
                draft_logits=draft_logits,
                draft_tokens=draft_tokens,
                temperature=temperature,
                bonus_logits=bonus_logits,
            )

            n_acc = num_accepted[0].item()
            metrics.accepted_tokens += n_acc

            # === COLLECT TOKENS ===
            for i in range(n_acc):
                generated.append(draft_tokens[:, i:i+1])
            generated.append(bonus_tokens[:1].unsqueeze(1))

            # === KV CACHE CLEANUP ===
            # Goal: restore invariant — both caches at kv_len + n_acc + 1,
            # with logits for the next position.

            # If all K accepted, draft_kv is missing draft_token[K-1]; forward it
            if n_acc == K:
                _, draft_kv = draft_model.forward(
                    draft_tokens[:, K-1:K], past_key_values=draft_kv
                )
                # draft_kv now at kv_len + K (same as target_kv)

            # Trim both caches to kv_len + n_acc (context through last accepted token)
            trim_to = kv_len + n_acc
            target_kv = trim_kv_cache(target_kv, trim_to)
            draft_kv = trim_kv_cache(draft_kv, trim_to)

            # Forward bonus token through both models to update caches and get next logits
            bonus_input = bonus_tokens[:1].unsqueeze(1)  # (batch, 1)
            target_bonus_out, target_kv = target_model.forward(
                bonus_input, past_key_values=target_kv
            )
            draft_bonus_out, draft_kv = draft_model.forward(
                bonus_input, past_key_values=draft_kv
            )

            # Update state
            next_target_logit = target_bonus_out[:, -1:, :]
            next_draft_logit = draft_bonus_out[:, -1:, :]
            kv_len = trim_to + 1  # = kv_len + n_acc + 1

    metrics.latency_seconds = timer.elapsed
    metrics.total_tokens = len(generated)
    metrics.peak_memory_mb = get_peak_memory_mb()

    # Build output
    if generated:
        all_generated = torch.cat(generated[:max_new_tokens], dim=1)
        output_ids = torch.cat([input_ids, all_generated], dim=1)
    else:
        output_ids = input_ids

    return output_ids, metrics


def standard_decode(
    model: ModelBackend,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[Tensor, DecodingMetrics]:
    """Standard autoregressive decoding (baseline for comparison).

    Args:
        model: The model to decode with
        input_ids: Input prompt tokens (batch, prompt_len)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature

    Returns:
        (output_ids, metrics) where output_ids includes the prompt
    """
    metrics = DecodingMetrics()
    reset_peak_memory()

    generated_tokens = []
    past_key_values = None

    with cuda_timer() as timer:
        current_ids = input_ids

        for step in range(max_new_tokens):
            logits, past_key_values = model.forward(
                current_ids, past_key_values=past_key_values
            )

            next_logits = logits[:, -1, :]
            next_token = _sample_token(next_logits, temperature)

            generated_tokens.append(next_token)
            current_ids = next_token
            metrics.total_steps += 1

    metrics.latency_seconds = timer.elapsed
    metrics.total_tokens = len(generated_tokens)
    metrics.peak_memory_mb = get_peak_memory_mb()

    if generated_tokens:
        all_generated = torch.cat(generated_tokens, dim=1)
        output_ids = torch.cat([input_ids, all_generated], dim=1)
    else:
        output_ids = input_ids

    return output_ids, metrics
