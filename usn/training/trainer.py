"""USNTrainer: core training loop with mixed precision and gradient accumulation.

Handles:
- forward → loss → backward → grad_clip → optimizer_step → scheduler_step
- Mixed precision via torch.amp.autocast (bf16 or fp16), GradScaler only for fp16
- Gradient accumulation over configurable micro-batches
- Teacher forcing (ground-truth tokens as input at each step)
- Logging at configurable intervals (loss, lr, grad_norm, tokens/sec)
- DataLoader creation from dataset
- Evaluation (validation loss + perplexity)
- Checkpointing (save/load/resume full training state)
- Early stopping with configurable patience
"""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from usn.config.training_config import USNTrainingConfig
from usn.core.interfaces import SchedulerInterface, TokenizerInterface
from usn.datasets.collate import usn_collate_fn
from usn.losses.cross_entropy import USNCrossEntropyLoss, compute_perplexity
from usn.optim.factory import OptimizerFactory
from usn.optim.schedulers import create_scheduler

logger = logging.getLogger(__name__)


class USNTrainer:
    """Core trainer for USN models.

    Manages the training loop including mixed precision, gradient
    accumulation, gradient clipping, and metric logging. Uses teacher
    forcing (ground-truth tokens as decoder input at each step).

    Args:
        model: USNModel instance to train.
        train_dataset: Training dataset returning dicts with
            'input_ids', 'targets', 'padding_mask'.
        config: USNTrainingConfig with training hyperparameters.
        val_dataset: Optional validation dataset.
        tokenizer: Optional tokenizer (used for logging/decoding).
        optimizer: Optional pre-built optimizer. If None, one is
            created via OptimizerFactory.
        scheduler: Optional pre-built scheduler. If None, one is
            created via create_scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: USNTrainingConfig,
        val_dataset: Dataset | None = None,
        tokenizer: TokenizerInterface | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: SchedulerInterface | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        # Resolve device from model parameters
        self.device = next(model.parameters()).device

        # Create optimizer if not provided
        self.optimizer = optimizer or OptimizerFactory.create(model, config)

        # Create scheduler if not provided
        self.scheduler = scheduler or create_scheduler(config)

        # Loss function (teacher forcing cross-entropy)
        self.loss_fn = USNCrossEntropyLoss(label_smoothing=0.0, ignore_index=-100)

        # Mixed precision setup
        self._setup_mixed_precision()

        # Training state
        self.global_step: int = 0
        self.epoch: int = 0
        self.tokens_seen: int = 0
        self.loss_history: list[float] = []

        # Evaluation and early stopping state
        self.best_val_loss: float = float("inf")
        self._patience_counter: int = 0
        self._stopped_early: bool = False

    def _setup_mixed_precision(self) -> None:
        """Configure autocast dtype and GradScaler based on config."""
        mp = self.config.mixed_precision
        if mp == "bf16":
            self.amp_dtype = torch.bfloat16
            self.use_amp = True
            # GradScaler not needed for bf16 (no underflow risk)
            self.grad_scaler: GradScaler | None = None
        elif mp == "fp16":
            self.amp_dtype = torch.float16
            self.use_amp = True
            # GradScaler only useful on CUDA devices
            if self.device.type == "cuda":
                self.grad_scaler = GradScaler("cuda")
            else:
                self.grad_scaler = None
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
            self.grad_scaler = None

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from a dataset.

        Args:
            dataset: Dataset to wrap.
            shuffle: Whether to shuffle data each epoch.

        Returns:
            Configured DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=usn_collate_fn,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Execute a single training step (forward + backward + update).

        Handles mixed precision autocast, loss scaling, gradient
        accumulation, gradient clipping, and optimizer/scheduler stepping.

        Args:
            batch: Dict with 'input_ids', 'targets', 'padding_mask' tensors.

        Returns:
            Dict with metrics: 'loss', 'grad_norm', 'lr', 'tokens'.
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        padding_mask = batch["padding_mask"].to(self.device)

        # Count tokens in this step
        num_tokens = int(padding_mask.sum().item())

        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits, _ = self.model(input_ids)
                loss = self.loss_fn(logits, targets, mask=padding_mask)
        else:
            logits, _ = self.model(input_ids)
            loss = self.loss_fn(logits, targets, mask=padding_mask)

        # Scale loss by accumulation steps
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Only update weights every gradient_accumulation_steps
        metrics: dict[str, float] = {
            "loss": loss.item(),
            "tokens": float(num_tokens),
        }

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping (fp16 only)
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            metrics["grad_norm"] = grad_norm.item()

            # Optimizer step
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            # Scheduler step
            lr = self.scheduler.get_lr(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            metrics["lr"] = lr
        else:
            # Not an update step — report current lr
            metrics["lr"] = self.scheduler.get_lr(self.global_step)
            metrics["grad_norm"] = 0.0

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run validation and compute loss + perplexity.

        Iterates over the full validation dataset (one pass) with no
        gradient computation. Uses the same mixed-precision autocast
        setting as training.

        Returns:
            Dict with 'val_loss' and 'val_perplexity'. Returns zeros
            if no validation dataset is configured.
        """
        if self.val_dataset is None:
            logger.warning("evaluate() called but no val_dataset provided.")
            return {"val_loss": 0.0, "val_perplexity": 1.0}

        self.model.eval()
        val_loader = self._create_dataloader(self.val_dataset, shuffle=False)

        total_loss = 0.0
        total_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)

            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits, _ = self.model(input_ids)
                    loss = self.loss_fn(logits, targets, mask=padding_mask)
            else:
                logits, _ = self.model(input_ids)
                loss = self.loss_fn(logits, targets, mask=padding_mask)

            total_loss += loss.item()
            total_batches += 1

        self.model.train()

        avg_loss = total_loss / max(total_batches, 1)
        perplexity = math.exp(min(avg_loss, 100.0))  # cap to avoid overflow

        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> None:
        """Save complete training state to a checkpoint file.

        Saves model weights, optimizer state, scheduler state, training
        progress, loss history, random states, and optional extra state.

        Args:
            path: File path to write the checkpoint.
            extra_state: Optional dict of additional state to persist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "tokens_seen": self.tokens_seen,
            "loss_history": self.loss_history,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self._patience_counter,
            # Random states for reproducibility
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.random.get_rng_state(),
        }

        if torch.cuda.is_available():
            checkpoint["rng_cuda"] = torch.cuda.get_rng_state_all()

        if self.grad_scaler is not None:
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        # Save config for reference
        checkpoint["config"] = {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "max_steps": self.config.max_steps,
            "mixed_precision": self.config.mixed_precision,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path} (step {self.global_step})")

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore all training state from a checkpoint file.

        Loads model weights, optimizer state, scheduler state, training
        progress, loss history, and random states.

        Args:
            path: File path of the checkpoint to load.

        Returns:
            The full checkpoint dict (includes extra_state if present).
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint.get("epoch", 0)
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        self.loss_history = checkpoint.get("loss_history", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self._patience_counter = checkpoint.get("patience_counter", 0)

        # Restore random states
        if "rng_python" in checkpoint:
            random.setstate(checkpoint["rng_python"])
        if "rng_numpy" in checkpoint:
            np.random.set_state(checkpoint["rng_numpy"])
        if "rng_torch" in checkpoint:
            torch.random.set_rng_state(checkpoint["rng_torch"])
        if "rng_cuda" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])

        if self.grad_scaler is not None and "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        logger.info(
            f"Checkpoint loaded: {path} (step {self.global_step}, "
            f"best_val_loss={self.best_val_loss:.4f})"
        )
        return checkpoint

    def resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Restore training state from checkpoint and continue training.

        This is a convenience method that loads the checkpoint and then
        calls train() to continue from where it left off.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Training results from the resumed train() call.
        """
        self.load_checkpoint(checkpoint_path)
        logger.info(
            f"Resuming training from step {self.global_step} (target: {self.config.max_steps})"
        )
        return self.train()

    # ------------------------------------------------------------------
    # Early stopping helpers
    # ------------------------------------------------------------------

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early based on validation loss.

        Updates best_val_loss and patience counter. Returns True if
        early stopping patience has been exceeded.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.config.early_stopping_patience <= 0:
            return False

        min_delta = self.config.early_stopping_min_delta
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self._patience_counter = 0
            return False
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at step {self.global_step} "
                    f"(patience={self.config.early_stopping_patience}, "
                    f"best_val_loss={self.best_val_loss:.4f})"
                )
                return True
            return False

    # ------------------------------------------------------------------
    # Checkpoint management helpers
    # ------------------------------------------------------------------

    def _manage_checkpoints(self, checkpoint_dir: Path) -> None:
        """Keep only the N most recent checkpoints.

        Removes older checkpoints to respect max_checkpoints config.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
        """
        max_ckpts = self.config.max_checkpoints
        if max_ckpts <= 0:
            return

        ckpt_files = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Keep the most recent max_checkpoints files
        while len(ckpt_files) > max_ckpts:
            oldest = ckpt_files.pop(0)
            oldest.unlink()
            logger.debug(f"Removed old checkpoint: {oldest}")

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Iterates over the training dataset for max_steps, calling
        train_step for each micro-batch. Logs metrics at configured
        intervals. Evaluates on validation set at eval_interval,
        saves checkpoints at checkpoint_interval, and applies early
        stopping when configured.

        Returns:
            Dict with final training stats: 'final_loss', 'total_steps',
            'total_tokens', 'loss_history', 'best_val_loss',
            'stopped_early'.
        """
        logger.info(
            f"Starting training: max_steps={self.config.max_steps}, "
            f"batch_size={self.config.batch_size}, "
            f"grad_accum={self.config.gradient_accumulation_steps}, "
            f"mixed_precision={self.config.mixed_precision}"
        )

        dataloader = self._create_dataloader(self.train_dataset)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_tokens = 0
        step_start_time = time.time()
        log_start_time = time.time()
        self._stopped_early = False

        while self.global_step < self.config.max_steps:
            self.epoch += 1
            for batch in dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                metrics = self.train_step(batch)

                # Accumulate metrics
                running_loss += metrics["loss"]
                running_tokens += int(metrics["tokens"])
                self.tokens_seen += int(metrics["tokens"])
                self.global_step += 1

                # Log at configured intervals
                if self.global_step % self.config.log_interval == 0:
                    elapsed = time.time() - log_start_time
                    avg_loss = running_loss / self.config.log_interval
                    tokens_per_sec = running_tokens / max(elapsed, 1e-6)

                    self.loss_history.append(avg_loss)

                    logger.info(
                        f"step={self.global_step} | "
                        f"loss={avg_loss:.4f} | "
                        f"lr={metrics['lr']:.2e} | "
                        f"grad_norm={metrics['grad_norm']:.3f} | "
                        f"tok/s={tokens_per_sec:.0f}"
                    )

                    # Reset running metrics
                    running_loss = 0.0
                    running_tokens = 0
                    log_start_time = time.time()

                # Evaluate at eval_interval
                if (
                    self.config.eval_interval > 0
                    and self.global_step % self.config.eval_interval == 0
                    and self.val_dataset is not None
                ):
                    eval_metrics = self.evaluate()
                    logger.info(
                        f"[Eval] step={self.global_step} | "
                        f"val_loss={eval_metrics['val_loss']:.4f} | "
                        f"val_ppl={eval_metrics['val_perplexity']:.2f}"
                    )

                    # Track best validation loss
                    if eval_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val_loss"]

                    # Early stopping check
                    if self._check_early_stopping(eval_metrics["val_loss"]):
                        self._stopped_early = True
                        break

                # Save checkpoint at checkpoint_interval
                if (
                    self.config.checkpoint_interval > 0
                    and self.global_step % self.config.checkpoint_interval == 0
                ):
                    ckpt_dir = Path("checkpoints")
                    ckpt_path = ckpt_dir / f"checkpoint_step_{self.global_step}.pt"
                    self.save_checkpoint(ckpt_path)
                    self._manage_checkpoints(ckpt_dir)

                if self.global_step >= self.config.max_steps:
                    break

            if self._stopped_early:
                break

        total_time = time.time() - step_start_time
        logger.info(
            f"Training complete: {self.global_step} steps, "
            f"{self.tokens_seen:,} tokens in {total_time:.1f}s"
            + (" (early stopped)" if self._stopped_early else "")
        )

        return {
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "total_steps": self.global_step,
            "total_tokens": self.tokens_seen,
            "loss_history": self.loss_history,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self._stopped_early,
        }
