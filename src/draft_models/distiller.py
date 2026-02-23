"""Knowledge distillation trainer for draft models."""

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.utils.config import ExperimentConfig
from src.utils.logging import get_logger, log_metrics
from src.draft_models.metrics import TrainingMetrics


class DistillationTrainer:
    """Trainer that distills knowledge from a teacher (target) to a student (draft)."""

    def __init__(
        self,
        config: ExperimentConfig,
        student_model,
        teacher_model,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = get_logger()

        tc = config.training
        dc = config.distillation

        self.accelerator = Accelerator(
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            mixed_precision="bf16" if tc.dtype == "bfloat16" else "fp16",
        )

        # Teacher is frozen, not wrapped by accelerator
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        # Move teacher to same device as accelerator (unless using device_map)
        if not getattr(self.teacher_model, "hf_device_map", None):
            self.teacher_model = self.teacher_model.to(self.accelerator.device)

        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=tc.warmup_steps,
            num_training_steps=tc.num_train_steps,
        )

        (
            self.student_model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            student_model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        self.temperature = dc.temperature
        self.alpha = dc.alpha
        self.best_val_loss = float("inf")

    def compute_loss(self, batch: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Compute combined KD + LM loss.

        Returns:
            (total_loss, kd_loss_value, lm_loss_value)
        """
        input_ids = batch

        # Student forward (with LM loss)
        student_out = self.student_model(input_ids=input_ids, labels=input_ids)
        lm_loss = student_out.loss
        student_logits = student_out.logits

        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_out = self.teacher_model(input_ids=input_ids)
            teacher_logits = teacher_out.logits

        # Align vocab sizes (e.g., Qwen2.5-7B has 152064, 1.5B has 151936)
        min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        # KL divergence loss
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        total_loss = self.alpha * kd_loss + (1 - self.alpha) * lm_loss
        return total_loss, kd_loss.detach().item(), lm_loss.detach().item()

    def train(self):
        """Run step-based distillation training loop."""
        tc = self.config.training
        self.student_model.train()
        step = 0
        total_loss = 0.0
        total_kd_loss = 0.0
        total_lm_loss = 0.0
        tokens_seen = 0
        start_time = time.time()
        train_iter = iter(self.train_dataloader)

        while step < tc.num_train_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            with self.accelerator.accumulate(self.student_model):
                loss, kd_loss_val, lm_loss_val = self.compute_loss(batch)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.student_model.parameters(), tc.max_grad_norm
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                step += 1
                total_loss += loss.detach().item()
                total_kd_loss += kd_loss_val
                total_lm_loss += lm_loss_val
                tokens_seen += batch.numel()

                if step % tc.log_steps == 0 and self.accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    n = tc.log_steps
                    metrics = TrainingMetrics(
                        step=step,
                        train_loss=total_loss / n,
                        kd_loss=total_kd_loss / n,
                        lm_loss=total_lm_loss / n,
                        learning_rate=self.scheduler.get_last_lr()[0],
                        tokens_per_second=tokens_seen / elapsed if elapsed > 0 else 0,
                        gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6
                        if torch.cuda.is_available()
                        else 0,
                    )
                    log_metrics(metrics.to_dict(), step=step)
                    total_loss = 0.0
                    total_kd_loss = 0.0
                    total_lm_loss = 0.0
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
                    self.student_model.train()

                if step % tc.save_steps == 0:
                    self._save_checkpoint(step)

        self._save_checkpoint(step, is_final=True)
        if self.accelerator.is_main_process:
            self.logger.info("Distillation training complete at step %d", step)

    def _evaluate(self) -> float:
        """Evaluate student on validation set."""
        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = 100

        with torch.no_grad():
            for batch in self.val_dataloader:
                if num_batches >= max_eval_batches:
                    break
                outputs = self.student_model(input_ids=batch, labels=batch)
                loss = outputs.loss
                total_loss += self.accelerator.gather(loss).mean().item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save student model checkpoint."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.training.output_dir)
        unwrapped = self.accelerator.unwrap_model(self.student_model)

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
