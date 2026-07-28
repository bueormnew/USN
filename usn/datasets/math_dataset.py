"""Synthetic arithmetic dataset for validation.

Generates simple math problems (addition, subtraction, multiplication)
for testing that the USN architecture can learn autoregressive tasks.
"""

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from usn.tokenizers.char_tokenizer import CharTokenizer


class MathDataset(Dataset[dict[str, Tensor]]):
    """Synthetic arithmetic dataset for validation.

    Generates problems like "5+3=8", "12*7=84", "100-42=58".
    Uses a character-level tokenizer over the math character set.

    Args:
        num_samples: Number of problems to generate.
        max_digits: Maximum digits in operands (1, 2, or 3).
        operations: List of operations to include ("+", "-", "*").
        split: Data split ("train", "val", "test").
        seed: Random seed for reproducibility.
    """

    CHARS = "0123456789+-*= "

    def __init__(
        self,
        num_samples: int = 10000,
        max_digits: int = 2,
        operations: list[str] | None = None,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        if operations is None:
            operations = ["+", "-", "*"]

        self.operations = operations
        self.max_digits = max_digits
        self._tokenizer = CharTokenizer(self.CHARS)

        # Generate problems with split-specific seed for no overlap
        split_offset = {"train": 0, "val": 1, "test": 2}
        rng = random.Random(seed + split_offset.get(split, 0))

        self.problems: list[str] = []
        seen: set[str] = set()
        max_val = 10**max_digits - 1

        while len(self.problems) < num_samples:
            op = rng.choice(operations)
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)

            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            else:
                continue

            problem = f"{a}{op}{b}={result}"
            if problem not in seen:
                seen.add(problem)
                self.problems.append(problem)

    @property
    def tokenizer(self) -> CharTokenizer:
        """The character-level tokenizer for the math character set."""
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a single training example.

        Returns dict with:
            input_ids: (seq_len,) — input tokens (all but last)
            targets: (seq_len,) — target tokens (shifted by 1)
            padding_mask: (seq_len,) — all True (no padding at sample level)
        """
        text = self.problems[idx]
        tokens = self._tokenizer.encode(text)

        # Causal LM: input = tokens[:-1], target = tokens[1:]
        input_ids = tokens[:-1]
        targets = tokens[1:]
        seq_len = len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "padding_mask": torch.ones(seq_len, dtype=torch.bool),
        }
