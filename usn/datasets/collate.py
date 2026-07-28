"""Collate function for variable-length sequence batching."""

import torch
from torch import Tensor


def usn_collate_fn(batch: list[dict[str, Tensor]], pad_token_id: int = 0) -> dict[str, Tensor]:
    """Pad and collate variable-length sequences into a batch.

    Pads all sequences to the length of the longest sequence in the batch,
    creates proper padding masks indicating valid (non-padded) positions.

    Args:
        batch: List of sample dicts from USNDataset/MathDataset.
            Each dict must contain 'input_ids', 'targets', and 'padding_mask'.
        pad_token_id: Token ID to use for padding (default: 0).

    Returns:
        Batched dict with padded tensors:
            input_ids: (batch_size, max_len) — padded input token IDs
            targets: (batch_size, max_len) — padded target token IDs
            padding_mask: (batch_size, max_len) — True for valid positions
    """
    max_len = max(sample["input_ids"].size(0) for sample in batch)

    input_ids_list = []
    targets_list = []
    masks_list = []

    for sample in batch:
        seq_len = sample["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([sample["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        targets_list.append(
            torch.cat([sample["targets"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        masks_list.append(
            torch.cat([sample["padding_mask"], torch.zeros(pad_len, dtype=torch.bool)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "targets": torch.stack(targets_list),
        "padding_mask": torch.stack(masks_list),
    }
