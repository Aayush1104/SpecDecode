"""Feature extraction for the adaptive router."""

import re

import torch
from torch import Tensor


class FeatureExtractor:
    """Extracts feature vectors from prompts for router classification.

    Features (total dim = embedding_dim + 7):
    - Sentence embedding (e.g. 768-dim from all-mpnet-base-v2)
    - Prompt length (normalized)
    - 4 domain-hint scores (code/math/conversation/factuality)
    - Token ID mean (normalized)
    - Token ID std (normalized)
    """

    def __init__(self, embedding_model_name: str, embedding_dim: int, device: str = "auto"):
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.embedding_dim = embedding_dim
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

    @staticmethod
    def _compute_domain_hints(text: str) -> list[float]:
        """Compute heuristic domain scores for a prompt.

        Returns [code_score, math_score, conversation_score, factuality_score]
        each in [0, 1].
        """
        text_lower = text.lower()

        # Code hints
        code_patterns = [
            r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\breturn\b',
            r'\bfunction\b', r'\bconst\b', r'\bvar\b', r'\blet\b',
            r'[{}\[\]]', r'->',  r'::', r'==', r'!=',
            r'"""', r"'''", r'//', r'/\*',
        ]
        code_hits = sum(1 for p in code_patterns if re.search(p, text))
        code_score = min(code_hits / 5.0, 1.0)

        # Math hints
        math_patterns = [
            r'\d+\s*[+\-*/]\s*\d+', r'\bsolve\b', r'\bcalculate\b',
            r'\bequation\b', r'\bmath\b', r'\balgebra\b',
            r'\bsum\b', r'\bproduct\b', r'\bintegral\b',
            r'\bderivative\b', r'\bprobability\b', r'\bpercent\b',
        ]
        math_hits = sum(1 for p in math_patterns if re.search(p, text_lower))
        math_score = min(math_hits / 3.0, 1.0)

        # Conversation hints
        conv_patterns = [
            r'\bwrite\b', r'\bexplain\b', r'\btell me\b',
            r'\bwhat do you think\b', r'\bhow would you\b',
            r'\bplease\b', r'\bhelp me\b', r'\bopinion\b',
            r'\bdiscuss\b', r'\bdescribe\b',
        ]
        conv_hits = sum(1 for p in conv_patterns if re.search(p, text_lower))
        conv_score = min(conv_hits / 3.0, 1.0)

        # Factuality hints
        fact_patterns = [
            r'\bwho\b', r'\bwhat is\b', r'\bwhen\b',
            r'\bwhere\b', r'\btrue\b', r'\bfalse\b',
            r'\bfact\b', r'\bcapital of\b', r'\bhow many\b',
            r'\bis it true\b',
        ]
        fact_hits = sum(1 for p in fact_patterns if re.search(p, text_lower))
        fact_score = min(fact_hits / 3.0, 1.0)

        return [code_score, math_score, conv_score, fact_score]

    def extract(self, prompt: str, tokenizer=None) -> Tensor:
        """Extract feature vector from a single prompt.

        Args:
            prompt: The input prompt string
            tokenizer: Optional tokenizer for token-level features

        Returns:
            Feature tensor of shape (total_dim,)
        """
        # Sentence embedding
        embedding = self.embedding_model.encode(
            prompt, convert_to_tensor=True, device=self.device
        )
        if embedding.dim() > 1:
            embedding = embedding.squeeze(0)

        # Prompt length (normalized by dividing by 2048)
        prompt_len = torch.tensor([len(prompt) / 2048.0], device=self.device)

        # Domain hints
        domain_hints = torch.tensor(
            self._compute_domain_hints(prompt), device=self.device
        )

        # Token ID stats
        if tokenizer is not None:
            token_ids = tokenizer.encode(prompt)
            token_ids_t = torch.tensor(token_ids, dtype=torch.float32)
            vocab_size = max(tokenizer.vocab_size, 1)
            token_mean = torch.tensor([token_ids_t.mean().item() / vocab_size], device=self.device)
            token_std = torch.tensor([token_ids_t.std().item() / vocab_size], device=self.device)
        else:
            token_mean = torch.tensor([0.0], device=self.device)
            token_std = torch.tensor([0.0], device=self.device)

        features = torch.cat([
            embedding.to(self.device),
            prompt_len,
            domain_hints,
            token_mean,
            token_std,
        ])

        return features

    def extract_batch(self, prompts: list[str], tokenizer=None) -> Tensor:
        """Extract feature vectors for a batch of prompts.

        Args:
            prompts: List of prompt strings
            tokenizer: Optional tokenizer for token-level features

        Returns:
            Feature tensor of shape (batch_size, total_dim)
        """
        # Batch sentence embeddings
        embeddings = self.embedding_model.encode(
            prompts, convert_to_tensor=True, device=self.device, show_progress_bar=False
        )

        batch_features = []
        for i, prompt in enumerate(prompts):
            emb = embeddings[i]

            prompt_len = torch.tensor([len(prompt) / 2048.0], device=self.device)
            domain_hints = torch.tensor(
                self._compute_domain_hints(prompt), device=self.device
            )

            if tokenizer is not None:
                token_ids = tokenizer.encode(prompt)
                token_ids_t = torch.tensor(token_ids, dtype=torch.float32)
                vocab_size = max(tokenizer.vocab_size, 1)
                token_mean = torch.tensor([token_ids_t.mean().item() / vocab_size], device=self.device)
                token_std = torch.tensor([token_ids_t.std().item() / vocab_size], device=self.device)
            else:
                token_mean = torch.tensor([0.0], device=self.device)
                token_std = torch.tensor([0.0], device=self.device)

            features = torch.cat([emb, prompt_len, domain_hints, token_mean, token_std])
            batch_features.append(features)

        return torch.stack(batch_features)
