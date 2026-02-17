"""Tests for DraftModelTrainer using small models and synthetic data."""

import pytest
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from src.utils.config import ExperimentConfig, TrainingConfig
from src.draft_models.trainer import DraftModelTrainer
from src.draft_models.data import InMemoryDataset


@pytest.fixture(scope="module")
def small_model_and_tokenizer():
    """Load distilgpt2 for testing."""
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture
def synthetic_dataloaders():
    """Create tiny synthetic dataloaders."""
    seq_length = 64
    num_train = 20
    num_val = 5
    vocab_size = 50257  # GPT-2 vocab

    torch.manual_seed(42)
    train_data = [torch.randint(0, vocab_size, (seq_length,)) for _ in range(num_train)]
    val_data = [torch.randint(0, vocab_size, (seq_length,)) for _ in range(num_val)]

    train_dl = DataLoader(InMemoryDataset(train_data), batch_size=4, shuffle=True)
    val_dl = DataLoader(InMemoryDataset(val_data), batch_size=4)
    return train_dl, val_dl


@pytest.fixture
def training_config(tmp_path):
    """Create a minimal training config."""
    config = ExperimentConfig()
    config.training = TrainingConfig(
        output_dir=str(tmp_path / "checkpoints"),
        num_train_steps=10,
        per_device_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dtype="float32",
        save_steps=5,
        eval_steps=5,
        log_steps=5,
    )
    return config


class TestDraftModelTrainer:
    """Tests for the DraftModelTrainer."""

    def test_loss_decreases(self, small_model_and_tokenizer, synthetic_dataloaders, training_config):
        """Training should decrease loss over 10 steps."""
        _, tokenizer = small_model_and_tokenizer
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        # Record initial loss
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_dl))
            initial_loss = model(input_ids=batch, labels=batch).loss.item()

        # Train
        model.train()
        trainer = DraftModelTrainer(
            config=training_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        trainer.train()

        # Check loss decreased
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped.eval()
        device = next(unwrapped.parameters()).device
        with torch.no_grad():
            batch = next(iter(val_dl)).to(device)
            final_loss = unwrapped(input_ids=batch, labels=batch).loss.item()

        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"

    def test_checkpoint_saved(self, small_model_and_tokenizer, synthetic_dataloaders, training_config):
        """Trainer should save checkpoints."""
        model, tokenizer = small_model_and_tokenizer
        # Reset model for clean test
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        trainer = DraftModelTrainer(
            config=training_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        trainer.train()

        output_dir = Path(training_config.training.output_dir)
        # Should have final checkpoint
        assert (output_dir / "final").exists()
        assert (output_dir / "final" / "config.json").exists()

    def test_checkpoint_loadable(self, small_model_and_tokenizer, synthetic_dataloaders, training_config):
        """Saved checkpoint should be loadable by AutoModelForCausalLM."""
        model, tokenizer = small_model_and_tokenizer
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        trainer = DraftModelTrainer(
            config=training_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        trainer.train()

        checkpoint_path = Path(training_config.training.output_dir) / "final"
        loaded = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        assert loaded is not None
        assert loaded.config.vocab_size == 50257  # GPT-2 vocab

    def test_evaluate_returns_finite(self, small_model_and_tokenizer, synthetic_dataloaders, training_config):
        """Evaluation should return a finite loss value."""
        model, tokenizer = small_model_and_tokenizer
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        trainer = DraftModelTrainer(
            config=training_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        val_loss = trainer._evaluate()
        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert not torch.isnan(torch.tensor(val_loss))
        assert not torch.isinf(torch.tensor(val_loss))
