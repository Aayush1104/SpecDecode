"""Draft model fine-tuning trainer with accelerate/DeepSpeed support."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.utils.config import ExperimentConfig
from src.utils.logging import get_logger, log_metrics
from src.draft_models.metrics import TrainingMetrics


class DraftModelTrainer:
    """Fine-tuning trainer for draft models using accelerate."""

    def __init__(
        self,
        config: ExperimentConfig,
        model,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = get_logger()

        tc = config.training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            mixed_precision="bf16" if tc.dtype == "bfloat16" else "fp16",
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        num_training_steps = tc.num_train_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=tc.warmup_steps,
            num_training_steps=num_training_steps,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        self.best_val_loss = float("inf")

    def train(self):
        """Run step-based training loop."""
        tc = self.config.training
        self.model.train()
        step = 0
        total_loss = 0.0
        tokens_seen = 0
        start_time = time.time()
        train_iter = iter(self.train_dataloader)

        while step < tc.num_train_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            with self.accelerator.accumulate(self.model):
                loss = self._compute_loss(batch)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), tc.max_grad_norm
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                step += 1
                total_loss += loss.detach().item()
                tokens_seen += batch.numel()

                if step % tc.log_steps == 0 and self.accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / tc.log_steps
                    metrics = TrainingMetrics(
                        step=step,
                        train_loss=avg_loss,
                        learning_rate=self.scheduler.get_last_lr()[0],
                        tokens_per_second=tokens_seen / elapsed if elapsed > 0 else 0,
                        gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6
                        if torch.cuda.is_available()
                        else 0,
                    )
                    log_metrics(metrics.to_dict(), step=step)
                    total_loss = 0.0
                    tokens_seen = 0
                    start_time = time.time()

                if step % tc.eval_steps == 0:
                    val_loss = self._evaluate()
                    if self.accelerator.is_main_process:
                        self.logger.info(
                            "Step %d: val_loss=%.4f perplexity=%.2f",
                            step,
                            val_loss,
                            math.exp(min(val_loss, 20)),
                        )
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                            self._save_checkpoint(step, is_best=True)
                    self.model.train()

                if step % tc.save_steps == 0:
                    self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(step, is_final=True)
        if self.accelerator.is_main_process:
            self.logger.info("Training complete at step %d", step)

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss."""
        input_ids = batch
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        return outputs.loss

    def _evaluate(self) -> float:
        """Evaluate on validation set, return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = 100

        with torch.no_grad():
            for batch in self.val_dataloader:
                if num_batches >= max_eval_batches:
                    break
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs.loss
                total_loss += self.accelerator.gather(loss).mean().item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.training.output_dir)
        unwrapped = self.accelerator.unwrap_model(self.model)

        if is_best:
            save_path = output_dir / "best"
            save_path.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.logger.info("Saved best checkpoint to %s", save_path)

        if is_final:
            save_path = output_dir / "final"
            save_path.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.logger.info("Saved final checkpoint to %s", save_path)
        elif not is_best:
            save_path = output_dir / f"step-{step}"
            save_path.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.logger.info("Saved checkpoint to %s", save_path)
