"""Tests for DistillationTrainer using small models and synthetic data."""

import pytest
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from src.utils.config import ExperimentConfig, TrainingConfig, DistillationConfig
from src.draft_models.distiller import DistillationTrainer
from src.draft_models.data import InMemoryDataset


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Load GPT-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def synthetic_dataloaders():
    """Create tiny synthetic dataloaders."""
    seq_length = 64
    num_train = 20
    num_val = 5
    vocab_size = 50257

    torch.manual_seed(42)
    train_data = [torch.randint(0, vocab_size, (seq_length,)) for _ in range(num_train)]
    val_data = [torch.randint(0, vocab_size, (seq_length,)) for _ in range(num_val)]

    train_dl = DataLoader(InMemoryDataset(train_data), batch_size=4, shuffle=True)
    val_dl = DataLoader(InMemoryDataset(val_data), batch_size=4)
    return train_dl, val_dl


@pytest.fixture
def distillation_config(tmp_path):
    """Create a minimal distillation config."""
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
    config.distillation = DistillationConfig(
        enabled=True,
        teacher_model="distilgpt2",
        teacher_dtype="float32",
        alpha=0.5,
        temperature=2.0,
    )
    return config


class TestDistillationTrainer:
    """Tests for the DistillationTrainer."""

    def test_teacher_frozen(self, gpt2_tokenizer, synthetic_dataloaders, distillation_config):
        """Teacher model parameters should not require gradients."""
        student = AutoModelForCausalLM.from_pretrained("distilgpt2")
        teacher = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        trainer = DistillationTrainer(
            config=distillation_config,
            student_model=student,
            teacher_model=teacher,
            tokenizer=gpt2_tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )

        for param in trainer.teacher_model.parameters():
            assert not param.requires_grad

    def test_student_updates(self, gpt2_tokenizer, synthetic_dataloaders, distillation_config):
        """Student model should update during training."""
        student = AutoModelForCausalLM.from_pretrained("distilgpt2")
        teacher = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        # Capture initial weights
        initial_param = next(student.parameters()).clone().detach()

        trainer = DistillationTrainer(
            config=distillation_config,
            student_model=student,
            teacher_model=teacher,
            tokenizer=gpt2_tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        trainer.train()

        # Check weights changed
        final_param = next(
            trainer.accelerator.unwrap_model(trainer.student_model).parameters()
        ).clone().detach().cpu()
        assert not torch.allclose(initial_param, final_param), "Student weights didn't update"

    def test_identical_models_low_kd_loss(self, gpt2_tokenizer, synthetic_dataloaders, distillation_config):
        """When teacher == student, KD loss should be very low."""
        student = AutoModelForCausalLM.from_pretrained("distilgpt2")
        teacher = AutoModelForCausalLM.from_pretrained("distilgpt2")
        # Copy student weights to teacher exactly
        teacher.load_state_dict(student.state_dict())
        train_dl, val_dl = synthetic_dataloaders

        trainer = DistillationTrainer(
            config=distillation_config,
            student_model=student,
            teacher_model=teacher,
            tokenizer=gpt2_tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )

        # Compute loss on first batch (get from prepared dataloader for correct device)
        batch = next(iter(trainer.train_dataloader))
        trainer.student_model.eval()
        with torch.no_grad():
            _, kd_loss, _ = trainer.compute_loss(batch)
        # KD loss between identical models should be near zero
        assert kd_loss < 0.1, f"KD loss too high for identical models: {kd_loss}"

    def test_checkpoint_saved_and_loadable(
        self, gpt2_tokenizer, synthetic_dataloaders, distillation_config
    ):
        """Distillation should save loadable checkpoints."""
        student = AutoModelForCausalLM.from_pretrained("distilgpt2")
        teacher = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        trainer = DistillationTrainer(
            config=distillation_config,
            student_model=student,
            teacher_model=teacher,
            tokenizer=gpt2_tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        trainer.train()

        checkpoint_path = Path(distillation_config.training.output_dir) / "final"
        assert checkpoint_path.exists()
        loaded = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        assert loaded is not None
        assert loaded.config.vocab_size == 50257

    def test_evaluate_returns_finite(
        self, gpt2_tokenizer, synthetic_dataloaders, distillation_config
    ):
        """Evaluation should return a finite loss value."""
        student = AutoModelForCausalLM.from_pretrained("distilgpt2")
        teacher = AutoModelForCausalLM.from_pretrained("distilgpt2")
        train_dl, val_dl = synthetic_dataloaders

        trainer = DistillationTrainer(
            config=distillation_config,
            student_model=student,
            teacher_model=teacher,
            tokenizer=gpt2_tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
        )
        val_loss = trainer._evaluate()
        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert not torch.isnan(torch.tensor(val_loss))
        assert not torch.isinf(torch.tensor(val_loss))
