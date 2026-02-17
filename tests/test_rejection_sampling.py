"""Tests for rejection sampling — the mathematical core of speculative decoding.

These tests verify that the rejection sampling scheme preserves the target
model's output distribution exactly.
"""

import pytest
import torch
from collections import Counter

from src.speculative.rejection_sampling import rejection_sample


class TestRejectionSamplingBasic:
    """Basic correctness tests for rejection sampling."""

    def test_output_shapes(self):
        """Check that output tensors have correct shapes."""
        batch_size, K, vocab_size = 2, 5, 100
        target_logits = torch.randn(batch_size, K, vocab_size)
        draft_logits = torch.randn(batch_size, K, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (batch_size, K))

        accepted, num_accepted, bonus = rejection_sample(
            target_logits, draft_logits, draft_tokens
        )

        assert accepted.shape == (batch_size, K + 1)
        assert num_accepted.shape == (batch_size,)
        assert bonus.shape == (batch_size,)

    def test_num_accepted_range(self):
        """num_accepted should be in [0, K]."""
        K = 5
        target_logits = torch.randn(1, K, 50)
        draft_logits = torch.randn(1, K, 50)
        draft_tokens = torch.randint(0, 50, (1, K))

        for _ in range(20):
            _, num_accepted, _ = rejection_sample(
                target_logits, draft_logits, draft_tokens
            )
            assert 0 <= num_accepted[0].item() <= K

    def test_bonus_token_always_valid(self):
        """Bonus token should always be a valid vocab index."""
        vocab_size = 50
        target_logits = torch.randn(1, 3, vocab_size)
        draft_logits = torch.randn(1, 3, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (1, 3))

        for _ in range(20):
            _, _, bonus = rejection_sample(
                target_logits, draft_logits, draft_tokens
            )
            assert 0 <= bonus[0].item() < vocab_size

    def test_accepted_tokens_match_draft(self):
        """Accepted tokens should be the draft tokens (in order)."""
        vocab_size = 50
        K = 5
        target_logits = torch.randn(1, K, vocab_size)
        draft_logits = torch.randn(1, K, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (1, K))

        accepted, num_accepted, _ = rejection_sample(
            target_logits, draft_logits, draft_tokens
        )

        n = num_accepted[0].item()
        if n > 0:
            assert torch.equal(accepted[0, :n], draft_tokens[0, :n])


class TestAllAccept:
    """When draft and target have identical distributions, all should be accepted."""

    def test_identical_logits_high_acceptance(self):
        """With identical logits, acceptance rate should be very high."""
        K = 5
        vocab_size = 50
        logits = torch.randn(1, K, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (1, K))

        # With identical distributions, acceptance prob = min(1, p/q) = 1
        total_accepted = 0
        num_trials = 100
        for _ in range(num_trials):
            _, num_accepted, _ = rejection_sample(
                logits.clone(), logits.clone(), draft_tokens
            )
            total_accepted += num_accepted[0].item()

        # Should accept all K tokens every time
        assert total_accepted == K * num_trials


class TestAllReject:
    """When draft and target are very different, rejection rate should be high."""

    def test_very_different_distributions(self):
        """When distributions are very different, acceptance should be low."""
        K = 5
        vocab_size = 50

        # Target puts all mass on token 0, draft on token 1
        target_logits = torch.full((1, K, vocab_size), -100.0)
        target_logits[:, :, 0] = 100.0  # target strongly prefers token 0

        draft_logits = torch.full((1, K, vocab_size), -100.0)
        draft_logits[:, :, 1] = 100.0  # draft strongly prefers token 1

        # Draft always picks token 1
        draft_tokens = torch.ones(1, K, dtype=torch.long)

        total_accepted = 0
        num_trials = 50
        for _ in range(num_trials):
            _, num_accepted, _ = rejection_sample(
                target_logits, draft_logits, draft_tokens
            )
            total_accepted += num_accepted[0].item()

        # Should reject almost all tokens (draft picks token target doesn't like)
        acceptance_rate = total_accepted / (K * num_trials)
        assert acceptance_rate < 0.1


class TestTemperature:
    """Test temperature handling."""

    def test_zero_temperature_greedy(self):
        """Temperature=0 should behave like greedy decoding."""
        K = 3
        vocab_size = 10

        target_logits = torch.randn(1, K, vocab_size)
        draft_logits = torch.randn(1, K, vocab_size)

        # Draft picks the greedy token from target
        target_greedy = target_logits.argmax(dim=-1)  # (1, K)
        draft_tokens = target_greedy

        accepted, num_accepted, bonus = rejection_sample(
            target_logits, draft_logits, draft_tokens, temperature=0.0
        )

        # If draft matches target's greedy choice, should be accepted
        assert num_accepted[0].item() == K

    def test_high_temperature_still_valid(self):
        """High temperature should still produce valid outputs."""
        K = 3
        vocab_size = 20
        target_logits = torch.randn(1, K, vocab_size)
        draft_logits = torch.randn(1, K, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (1, K))

        accepted, num_accepted, bonus = rejection_sample(
            target_logits, draft_logits, draft_tokens, temperature=10.0
        )

        assert 0 <= num_accepted[0].item() <= K
        assert 0 <= bonus[0].item() < vocab_size


class TestEdgeCases:
    """Edge cases and numerical stability."""

    def test_single_token(self):
        """K=1 should work correctly."""
        vocab_size = 50
        target_logits = torch.randn(1, 1, vocab_size)
        draft_logits = torch.randn(1, 1, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (1, 1))

        accepted, num_accepted, bonus = rejection_sample(
            target_logits, draft_logits, draft_tokens
        )
        assert accepted.shape == (1, 2)  # K+1
        assert num_accepted[0].item() in [0, 1]

    def test_large_logits_numerical_stability(self):
        """Very large logits should not cause NaN or Inf."""
        K = 3
        vocab_size = 50
        target_logits = torch.randn(1, K, vocab_size) * 1000
        draft_logits = torch.randn(1, K, vocab_size) * 1000
        draft_tokens = torch.randint(0, vocab_size, (1, K))

        accepted, num_accepted, bonus = rejection_sample(
            target_logits, draft_logits, draft_tokens
        )

        assert not torch.any(torch.isnan(accepted.float()))
        assert not torch.any(torch.isnan(bonus.float()))

    def test_batch_size_greater_than_one(self):
        """Multiple batch elements should be handled independently."""
        batch_size = 4
        K = 3
        vocab_size = 30

        target_logits = torch.randn(batch_size, K, vocab_size)
        draft_logits = torch.randn(batch_size, K, vocab_size)
        draft_tokens = torch.randint(0, vocab_size, (batch_size, K))

        accepted, num_accepted, bonus = rejection_sample(
            target_logits, draft_logits, draft_tokens
        )

        assert accepted.shape == (batch_size, K + 1)
        assert num_accepted.shape == (batch_size,)
        assert bonus.shape == (batch_size,)

        for b in range(batch_size):
            assert 0 <= num_accepted[b].item() <= K
            assert 0 <= bonus[b].item() < vocab_size


class TestDistributionCorrectness:
    """Statistical tests to verify the output distribution matches the target."""

    def test_distribution_matches_target(self):
        """Over many samples, rejection sampling should match target distribution.

        Uses a small vocabulary and many samples to statistically verify
        that the first generated token follows the target distribution.
        """
        torch.manual_seed(42)
        vocab_size = 5
        K = 1  # Single token for clear distribution test
        num_samples = 5000

        # Define clear target distribution
        target_logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0, -2.0]]])
        target_probs = torch.softmax(target_logits[0, 0], dim=-1)

        # Draft distribution is different
        draft_logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]])  # uniform

        # Count token frequencies from rejection sampling
        counts = Counter()
        for _ in range(num_samples):
            # Draft samples from uniform
            draft_probs = torch.softmax(draft_logits[0, 0], dim=-1)
            draft_token = torch.multinomial(draft_probs, 1).unsqueeze(0)

            accepted, num_accepted, bonus = rejection_sample(
                target_logits, draft_logits, draft_token
            )

            # The output token is either the accepted draft token or the bonus
            n = num_accepted[0].item()
            if n > 0:
                output_token = accepted[0, 0].item()
            else:
                output_token = bonus[0].item()
            counts[output_token] += 1

        # Compute empirical distribution
        empirical_probs = torch.zeros(vocab_size)
        for token, count in counts.items():
            empirical_probs[token] = count / num_samples

        # Check that empirical distribution is close to target
        # Using a loose tolerance due to sampling variance
        for i in range(vocab_size):
            assert abs(empirical_probs[i].item() - target_probs[i].item()) < 0.05, (
                f"Token {i}: empirical={empirical_probs[i]:.3f}, "
                f"target={target_probs[i]:.3f}"
            )
