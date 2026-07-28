"""USN dataset implementations for causal language modeling.

Provides both indexed (USNDataset) and streaming (StreamingUSNDataset)
datasets that create causal LM pairs: input=tokens[:-1], target=tokens[1:].
"""

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from usn.core.interfaces import TokenizerInterface


class USNDataset(Dataset):
    """Indexed dataset for causal language modeling.

    Tokenizes all data on initialization and stores in memory.
    Creates input/target pairs with padding and masks.

    Args:
        texts: List of text strings to train on.
        tokenizer: Tokenizer implementing TokenizerInterface.
        max_seq_len: Maximum sequence length (truncates longer, pads shorter).
    """

    def __init__(
        self, texts: list[str], tokenizer: TokenizerInterface, max_seq_len: int = 512
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[list[int]] = []

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue  # Need at least 2 tokens for input/target
            # Truncate to max_seq_len + 1 (need one extra for target shift)
            tokens = tokens[: max_seq_len + 1]
            self.samples.append(tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a single training example.

        Returns dict with:
            input_ids: (seq_len,) — input tokens
            targets: (seq_len,) — target tokens (shifted by 1)
            padding_mask: (seq_len,) — True for valid positions
        """
        tokens = self.samples[idx]
        seq_len = min(len(tokens) - 1, self.max_seq_len)

        input_ids = tokens[:seq_len]
        targets = tokens[1 : seq_len + 1]

        # Pad if needed
        pad_len = self.max_seq_len - seq_len
        padding_mask = [True] * seq_len + [False] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        targets = targets + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }


class StreamingUSNDataset(IterableDataset):
    """Streaming dataset for large corpora.

    Tokenizes text on the fly from an iterable source.
    Supports a shuffle buffer for randomization.

    Args:
        data_source: Iterable yielding text strings.
        tokenizer: Tokenizer implementing TokenizerInterface.
        max_seq_len: Maximum sequence length.
        shuffle_buffer: Size of the shuffle buffer (0 = no shuffle).
    """

    def __init__(
        self,
        data_source,
        tokenizer: TokenizerInterface,
        max_seq_len: int = 512,
        shuffle_buffer: int = 10000,
    ) -> None:
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buffer: list[list[int]] = []

        for text in self.data_source:
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[: self.max_seq_len + 1]

            if self.shuffle_buffer > 0:
                buffer.append(tokens)
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for sample in buffer:
                        yield self._make_sample(sample)
                    buffer = []
            else:
                yield self._make_sample(tokens)

        # Flush remaining buffer
        if buffer:
            random.shuffle(buffer)
            for sample in buffer:
                yield self._make_sample(sample)

    def _make_sample(self, tokens: list[int]) -> dict[str, Tensor]:
        """Convert token list to a training sample dict.

        Args:
            tokens: List of token IDs (length >= 2).

        Returns:
            Dict with input_ids, targets, and padding_mask tensors.
        """
        seq_len = min(len(tokens) - 1, self.max_seq_len)
        input_ids = tokens[:seq_len]
        targets = tokens[1 : seq_len + 1]

        pad_len = self.max_seq_len - seq_len
        padding_mask = [True] * seq_len + [False] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        targets = targets + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }
