"""Router MLP training loop."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.routing.data_collection import RoutingSample, load_router_data
from src.routing.model import RouterMLP
from src.utils.config import ExperimentConfig
from src.utils.logging import get_logger

logger = get_logger()


class RouterTrainer:
    """Trains the RouterMLP classifier on collected routing data."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(
        self,
        samples: list[RoutingSample] | None = None,
        data_path: str | None = None,
    ) -> RouterMLP:
        """Train the router MLP.

        Args:
            samples: Pre-loaded routing samples (takes priority)
            data_path: Path to JSON data file (used if samples is None)

        Returns:
            Trained RouterMLP model
        """
        if samples is None:
            if data_path is None:
                data_path = self.config.router.training_data_path
            samples = load_router_data(data_path)

        if not samples:
            raise ValueError("No training samples provided")

        # Build tensors from samples
        features = torch.tensor(
            [s.features for s in samples], dtype=torch.float32
        )
        labels = torch.tensor(
            [s.best_draft_id for s in samples], dtype=torch.long
        )

        input_dim = features.shape[1]
        num_drafts = len(set(s.best_draft_id for s in samples))

        # 80/20 train/val split
        n_total = len(features)
        n_train = int(0.8 * n_total)
        indices = torch.randperm(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_dataset = TensorDataset(features[train_idx], labels[train_idx])
        val_dataset = TensorDataset(features[val_idx], labels[val_idx])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.router.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.router.batch_size,
            shuffle=False,
        )

        # Create model
        model = RouterMLP(
            input_dim=input_dim,
            hidden_dim=self.config.router.hidden_dim,
            num_drafts=num_drafts,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.router.learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.config.router.num_epochs):
            # Training
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            val_acc = self._evaluate(model, val_loader)

            logger.info(
                "Epoch %d/%d - loss: %.4f, val_acc: %.4f",
                epoch + 1, self.config.router.num_epochs, avg_loss, val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save checkpoint
        checkpoint_path = self.config.router.router_checkpoint
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_path)
        logger.info(
            "Router training complete. Best val accuracy: %.4f. Saved to %s",
            best_val_acc, checkpoint_path,
        )

        model.eval()
        return model

    def _evaluate(self, model: RouterMLP, dataloader: DataLoader) -> float:
        """Evaluate model accuracy on a dataloader.

        Args:
            model: The RouterMLP model
            dataloader: Validation dataloader

        Returns:
            Accuracy as a float in [0, 1]
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                predictions = model.predict(batch_features)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)

        model.train()
        return correct / max(total, 1)
