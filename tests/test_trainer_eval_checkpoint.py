"""Tests for USNTrainer evaluate, checkpointing, and early stopping methods."""

import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from usn.config.training_config import USNTrainingConfig
from usn.training.trainer import USNTrainer

# ---------------------------------------------------------------------------
# Minimal model and dataset for testing
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model that mimics USNModel interface (forward → logits, state)."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)
        logits = self.proj(h)
        return logits, None  # (logits, state)


class _TinyDataset(Dataset):
    """Deterministic dataset for testing."""

    def __init__(self, size: int = 64, seq_len: int = 8, vocab_size: int = 32):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        tokens = self.data[idx]
        return {
            "input_ids": tokens[:-1],
            "targets": tokens[1:],
            "padding_mask": torch.ones(self.seq_len - 1, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Test evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _make_trainer(self, val_dataset=None):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=10,
            warmup_steps=0,
            mixed_precision="none",
            eval_interval=5,
            checkpoint_interval=0,
            log_interval=5,
        )
        return USNTrainer(
            model=model,
            train_dataset=train_ds,
            config=config,
            val_dataset=val_dataset,
        )

    def test_evaluate_returns_metrics(self):
        val_ds = _TinyDataset(size=16)
        trainer = self._make_trainer(val_dataset=val_ds)
        result = trainer.evaluate()
        assert "val_loss" in result
        assert "val_perplexity" in result
        assert result["val_loss"] > 0
        assert result["val_perplexity"] > 1.0
        # perplexity == exp(loss)
        assert abs(result["val_perplexity"] - math.exp(result["val_loss"])) < 0.1

    def test_evaluate_no_val_dataset(self):
        trainer = self._make_trainer(val_dataset=None)
        result = trainer.evaluate()
        assert result["val_loss"] == 0.0
        assert result["val_perplexity"] == 1.0


# ---------------------------------------------------------------------------
# Test checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def _make_trainer(self):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=5,
            warmup_steps=0,
            mixed_precision="none",
            checkpoint_interval=0,
            eval_interval=0,
            log_interval=5,
        )
        return USNTrainer(model=model, train_dataset=train_ds, config=config)

    def test_save_and_load_checkpoint(self):
        trainer = self._make_trainer()
        # Simulate some training state
        trainer.global_step = 42
        trainer.epoch = 3
        trainer.tokens_seen = 1000
        trainer.loss_history = [2.5, 2.3, 2.1]
        trainer.best_val_loss = 2.0
        trainer._patience_counter = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            trainer.save_checkpoint(ckpt_path, extra_state={"foo": "bar"})
            assert ckpt_path.exists()

            # Create a fresh trainer and load
            trainer2 = self._make_trainer()
            checkpoint = trainer2.load_checkpoint(ckpt_path)

            assert trainer2.global_step == 42
            assert trainer2.epoch == 3
            assert trainer2.tokens_seen == 1000
            assert trainer2.loss_history == [2.5, 2.3, 2.1]
            assert trainer2.best_val_loss == 2.0
            assert trainer2._patience_counter == 1
            assert checkpoint["extra_state"] == {"foo": "bar"}

    def test_checkpoint_saves_model_weights(self):
        trainer = self._make_trainer()

        # Mutate weight
        trainer.model.proj.weight.data.fill_(99.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)

            # Reset weights
            trainer.model.proj.weight.data.fill_(0.0)
            trainer.load_checkpoint(ckpt_path)

            # Should be restored to 99.0
            assert torch.allclose(
                trainer.model.proj.weight.data,
                torch.full_like(trainer.model.proj.weight.data, 99.0),
            )


# ---------------------------------------------------------------------------
# Test early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_no_early_stopping_when_disabled(self):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=5,
            warmup_steps=0,
            mixed_precision="none",
            early_stopping_patience=0,  # disabled
        )
        trainer = USNTrainer(model=model, train_dataset=train_ds, config=config)
        # Should never return True
        assert not trainer._check_early_stopping(10.0)
        assert not trainer._check_early_stopping(10.0)
        assert not trainer._check_early_stopping(10.0)

    def test_early_stopping_triggers_after_patience(self):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=100,
            warmup_steps=0,
            mixed_precision="none",
            early_stopping_patience=3,
            early_stopping_min_delta=0.01,
        )
        trainer = USNTrainer(model=model, train_dataset=train_ds, config=config)

        # First call sets best
        assert not trainer._check_early_stopping(2.0)
        assert trainer.best_val_loss == 2.0

        # Improvement resets counter
        assert not trainer._check_early_stopping(1.5)
        assert trainer.best_val_loss == 1.5
        assert trainer._patience_counter == 0

        # No improvement: patience increments
        assert not trainer._check_early_stopping(1.5)
        assert trainer._patience_counter == 1

        assert not trainer._check_early_stopping(1.5)
        assert trainer._patience_counter == 2

        # Third no-improvement triggers stopping
        assert trainer._check_early_stopping(1.5)
        assert trainer._patience_counter == 3

    def test_early_stopping_respects_min_delta(self):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=100,
            warmup_steps=0,
            mixed_precision="none",
            early_stopping_patience=2,
            early_stopping_min_delta=0.1,
        )
        trainer = USNTrainer(model=model, train_dataset=train_ds, config=config)

        assert not trainer._check_early_stopping(2.0)
        # Tiny improvement (less than min_delta) should NOT reset counter
        assert not trainer._check_early_stopping(1.95)  # only 0.05 improvement
        assert trainer._patience_counter == 1
        # Still best_val_loss should be 2.0 since 1.95 isn't enough better
        assert trainer.best_val_loss == 2.0


# ---------------------------------------------------------------------------
# Test resume
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_continues_from_checkpoint(self):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=10,
            warmup_steps=0,
            mixed_precision="none",
            checkpoint_interval=0,
            eval_interval=0,
            log_interval=5,
        )
        trainer = USNTrainer(model=model, train_dataset=train_ds, config=config)

        # Train partway
        trainer.config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=5,
            warmup_steps=0,
            mixed_precision="none",
            checkpoint_interval=0,
            eval_interval=0,
            log_interval=5,
        )
        trainer.train()
        assert trainer.global_step == 5

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "resume_ckpt.pt"
            trainer.save_checkpoint(ckpt_path)

            # Create new trainer targeting 10 steps
            model2 = _TinyModel()
            trainer2 = USNTrainer(
                model=model2,
                train_dataset=train_ds,
                config=USNTrainingConfig(
                    learning_rate=1e-3,
                    batch_size=8,
                    max_steps=10,
                    warmup_steps=0,
                    mixed_precision="none",
                    checkpoint_interval=0,
                    eval_interval=0,
                    log_interval=5,
                ),
            )
            result = trainer2.resume(ckpt_path)
            assert trainer2.global_step == 10
            assert result["total_steps"] == 10


# ---------------------------------------------------------------------------
# Test train() integration with eval and early stopping
# ---------------------------------------------------------------------------


class TestTrainIntegration:
    def test_train_calls_evaluate_at_interval(self):
        model = _TinyModel()
        train_ds = _TinyDataset(size=32)
        val_ds = _TinyDataset(size=16)
        config = USNTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=10,
            warmup_steps=0,
            mixed_precision="none",
            eval_interval=5,
            checkpoint_interval=0,
            early_stopping_patience=0,
            log_interval=5,
        )
        trainer = USNTrainer(model=model, train_dataset=train_ds, config=config, val_dataset=val_ds)
        result = trainer.train()
        # Should complete without error; best_val_loss updated
        assert result["total_steps"] == 10
        assert result["best_val_loss"] < float("inf")

    def test_train_early_stops(self):
        """Train with early stopping on a constant-loss val set."""
        model = _TinyModel()
        train_ds = _TinyDataset(size=64)
        val_ds = _TinyDataset(size=16)
        config = USNTrainingConfig(
            learning_rate=1e-6,  # tiny lr so loss barely changes
            batch_size=8,
            max_steps=1000,
            warmup_steps=0,
            mixed_precision="none",
            eval_interval=5,
            checkpoint_interval=0,
            early_stopping_patience=3,
            early_stopping_min_delta=1e-6,
            min_lr=1e-7,
            log_interval=5,
        )
        trainer = USNTrainer(model=model, train_dataset=train_ds, config=config, val_dataset=val_ds)
        result = trainer.train()
        # Should stop well before 1000 steps due to early stopping
        assert result["stopped_early"] is True
        assert result["total_steps"] < 1000
