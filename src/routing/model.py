"""Router MLP classifier for selecting the best draft model."""

import torch
import torch.nn as nn
from torch import Tensor


class RouterMLP(nn.Module):
    """Simple MLP that classifies prompts into draft model categories.

    Architecture: input_dim → hidden_dim (ReLU, dropout) → num_drafts
    """

    def __init__(
        self,
        input_dim: int = 775,
        hidden_dim: int = 256,
        num_drafts: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_drafts = num_drafts
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_drafts),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Logits of shape (batch, num_drafts)
        """
        return self.net(x)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted draft index (argmax).

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Predicted indices of shape (batch,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def predict_with_confidence(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return predicted draft index and confidence score.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            (indices, confidences) where indices is (batch,) and
            confidences is (batch,) with softmax probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            confidences, indices = probs.max(dim=-1)
            return indices, confidences

    def save(self, path: str):
        """Save model weights and config to a checkpoint file."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_drafts": self.num_drafts,
                "dropout": self.dropout_rate,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RouterMLP":
        """Load a RouterMLP from a checkpoint file.

        Args:
            path: Path to the checkpoint file
            device: Device to load the model onto

        Returns:
            Loaded RouterMLP instance
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_drafts=config["num_drafts"],
            dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
