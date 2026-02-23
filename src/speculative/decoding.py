"""Main speculative decoding loop and standard decoding baseline."""

import time

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
    profile: bool = False,
) -> tuple[Tensor, DecodingMetrics]:
    """Run speculative decoding.

    Optimized algorithm that merges the bonus token's target forward into the
    next iteration's verification pass, reducing target model calls from 2 to 1
    per iteration.

    Invariant at iteration start:
    - draft_kv at kv_len, target_kv at kv_len - 1 (one behind, missing bonus)
    - pending_bonus: the bonus token that target hasn't processed yet
    - next_draft_logit: draft's logit for position kv_len

    Exception: first iteration has both caches at kv_len and next_target_logit
    cached from prompt processing.

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

    # Validate vocabulary compatibility
    target_vocab = target_model.get_vocab_size()
    draft_vocab = draft_model.get_vocab_size()
    max_input_id = input_ids.max().item()
    if max_input_id >= draft_vocab:
        raise ValueError(
            f"Input contains token ID {max_input_id} but draft model vocabulary "
            f"size is only {draft_vocab}. Both models must share the same "
            f"tokenizer. Use a draft model from the same model family "
            f"(e.g., Qwen2.5-0.5B as draft for Qwen2.5-7B)."
        )
    # Allow small padding differences (e.g., 152064 vs 151936 in Qwen family)
    # but flag large mismatches that indicate different tokenizers
    if abs(target_vocab - draft_vocab) > max(1024, target_vocab * 0.02):
        raise ValueError(
            f"Vocabulary size mismatch: target has {target_vocab} tokens, "
            f"draft has {draft_vocab} tokens. Both models must share the same "
            f"tokenizer (e.g., GPT-2 family together, Qwen family together)."
        )

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
    n_generated = 0
    first_iteration = True
    pending_bonus = None  # bonus token target hasn't seen yet

    with cuda_timer() as timer:
        while n_generated < max_new_tokens:
            K = min(speculation_length, max_new_tokens - n_generated)
            metrics.total_steps += 1

            # === DRAFT PHASE ===
            if profile:
                _t0 = time.perf_counter()

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

            if profile:
                metrics.draft_time += time.perf_counter() - _t0

            # State: draft_kv at kv_len + K - 1
            # (K-1 forwards; first token was from cached logit)

            # === VERIFICATION PHASE ===
            if profile:
                _t0 = time.perf_counter()

            if first_iteration:
                # First iteration: target_kv at kv_len, we have next_target_logit
                # Forward K draft tokens through target
                target_step_out, target_kv = target_model.forward(
                    draft_tokens, past_key_values=target_kv
                )
                # target_kv now at kv_len + K

                # Build verification logits:
                # draft_token[0] verified by next_target_logit (cached)
                # draft_token[i>0] verified by target_step_out[:, i-1, :]
                if K > 1:
                    target_verify = torch.cat(
                        [next_target_logit, target_step_out[:, :K-1, :]], dim=1
                    )
                else:
                    target_verify = next_target_logit  # (batch, 1, vocab)

                # Bonus logit for the all-accepted case
                bonus_logits = target_step_out[:, K-1:K, :]  # (batch, 1, vocab)
                first_iteration = False
            else:
                # Subsequent iterations: target_kv at kv_len - 1 (missing bonus)
                # Forward [bonus, draft_tokens] through target in ONE call
                verify_input = torch.cat(
                    [pending_bonus[:1].unsqueeze(1), draft_tokens], dim=1
                )  # (batch, K+1)
                target_step_out, target_kv = target_model.forward(
                    verify_input, past_key_values=target_kv
                )
                # target_kv now at (kv_len - 1) + K + 1 = kv_len + K
                # target_step_out[:, 0, :] = prediction after seeing bonus → verifies draft_token[0]
                # target_step_out[:, i, :] = prediction after bonus + draft[:i] → verifies draft_token[i]
                # target_step_out[:, K, :] = bonus logit for all-accepted

                # Verification logits (K logits)
                target_verify = target_step_out[:, :K, :]  # (batch, K, vocab)

                # Bonus logit for the all-accepted case
                bonus_logits = target_step_out[:, K:K+1, :]  # (batch, 1, vocab)

            if profile:
                metrics.verify_time += time.perf_counter() - _t0

            # === REJECTION SAMPLING ===
            if profile:
                _t0 = time.perf_counter()
            _, num_accepted, bonus_tokens = rejection_sample(
                target_logits=target_verify,
                draft_logits=draft_logits,
                draft_tokens=draft_tokens,
                temperature=temperature,
                bonus_logits=bonus_logits,
            )

            n_acc = num_accepted[0].item()
            metrics.accepted_tokens += n_acc

            if profile:
                metrics.sampling_time += time.perf_counter() - _t0

            # === COLLECT TOKENS ===
            if n_acc > 0:
                generated.append(draft_tokens[:, :n_acc])
            generated.append(bonus_tokens[:1].unsqueeze(1))
            n_generated += n_acc + 1

            # === KV CACHE CLEANUP ===
            if profile:
                _t0 = time.perf_counter()

            # If all K accepted, draft_kv is missing draft_token[K-1]; forward it
            if n_acc == K:
                _, draft_kv = draft_model.forward(
                    draft_tokens[:, K-1:K], past_key_values=draft_kv
                )
                # draft_kv now at kv_len + K

            # Trim both caches:
            # - target_kv from kv_len + K to kv_len + n_acc
            # - draft_kv from kv_len + K - 1 (or K if all accepted) to kv_len + n_acc
            trim_to = kv_len + n_acc
            target_kv = trim_kv_cache(target_kv, trim_to)
            draft_kv = trim_kv_cache(draft_kv, trim_to)

            # Forward bonus through draft only (target deferred to next verify)
            bonus_input = bonus_tokens[:1].unsqueeze(1)  # (batch, 1)
            draft_bonus_out, draft_kv = draft_model.forward(
                bonus_input, past_key_values=draft_kv
            )

            # Update state
            # target_kv stays at kv_len + n_acc (one behind draft)
            next_draft_logit = draft_bonus_out[:, -1:, :]
            pending_bonus = bonus_tokens
            kv_len = trim_to + 1  # = kv_len + n_acc + 1 (draft's kv_len)

            if profile:
                metrics.overhead_time += time.perf_counter() - _t0

    metrics.latency_seconds = timer.elapsed
    metrics.total_tokens = n_generated
    metrics.peak_memory_mb = get_peak_memory_mb()

    # Build output
    if generated:
        all_generated = torch.cat(generated, dim=1)
        # Trim to exactly max_new_tokens
        if all_generated.shape[1] > max_new_tokens:
            all_generated = all_generated[:, :max_new_tokens]
            metrics.total_tokens = max_new_tokens
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
