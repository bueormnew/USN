"""BPE tokenizer wrapping the HuggingFace tokenizers library.

Provides a thin wrapper around HuggingFace's fast tokenizers with
graceful handling when the ``tokenizers`` package is not installed.
"""

from __future__ import annotations

from usn.core.interfaces import TokenizerInterface

# Attempt to import optional dependency
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing

    _HAS_TOKENIZERS = True
except ImportError:
    _HAS_TOKENIZERS = False


def _check_tokenizers_installed() -> None:
    """Raise ImportError with helpful message if tokenizers not available."""
    if not _HAS_TOKENIZERS:
        raise ImportError(
            "The 'tokenizers' package is required for BPETokenizer. "
            "Install it with: pip install tokenizers"
        )


class BPETokenizer(TokenizerInterface):
    """BPE tokenizer via HuggingFace tokenizers library.

    Wraps HuggingFace's fast tokenizer implementation. Supports
    loading pretrained tokenizers, training new ones, and
    saving/loading from disk.

    Special tokens:
        PAD=0, BOS=1, EOS=2, UNK=3

    Raises ImportError if ``tokenizers`` package is not installed.

    Example:
        >>> tok = BPETokenizer.from_pretrained("gpt2")
        >>> ids = tok.encode("Hello world")
        >>> tok.decode(ids)
        'Hello world'
    """

    # Fixed special token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    _SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        """Initialize with an optional HuggingFace Tokenizer instance.

        Args:
            tokenizer: A pre-configured HuggingFace ``Tokenizer`` object.
                       If None, the tokenizer must be set via train/load/from_pretrained.
        """
        _check_tokenizers_installed()
        self._tokenizer: Tokenizer | None = tokenizer

    def _ensure_ready(self) -> Tokenizer:
        """Ensure the internal tokenizer is initialized."""
        if self._tokenizer is None:
            raise RuntimeError(
                "BPETokenizer is not initialized. Use from_pretrained(), "
                "train(), or load() to set up the tokenizer."
            )
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        tok = self._ensure_ready()
        encoding = tok.encode(text)
        return encoding.ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        tok = self._ensure_ready()
        return tok.decode(token_ids, skip_special_tokens=True)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of texts to token IDs.

        Args:
            texts: List of input text strings.

        Returns:
            List of token ID lists.
        """
        tok = self._ensure_ready()
        encodings = tok.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def batch_decode(self, batch_ids: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists back to text.

        Args:
            batch_ids: List of token ID lists.

        Returns:
            List of decoded text strings.
        """
        tok = self._ensure_ready()
        return [tok.decode(ids, skip_special_tokens=True) for ids in batch_ids]

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        tok = self._ensure_ready()
        return tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """ID of the padding token."""
        return self.PAD_ID

    @property
    def bos_token_id(self) -> int:
        """ID of the beginning-of-sequence token."""
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        """ID of the end-of-sequence token."""
        return self.EOS_ID

    @property
    def unk_token_id(self) -> int:
        """ID of the unknown token."""
        return self.UNK_ID

    def save(self, path: str) -> None:
        """Save the tokenizer to a file.

        Args:
            path: File path to save the tokenizer JSON.
        """
        tok = self._ensure_ready()
        tok.save(path)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Load a tokenizer from a saved file.

        Args:
            path: Path to the tokenizer JSON file.

        Returns:
            BPETokenizer instance.
        """
        _check_tokenizers_installed()
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, name: str) -> BPETokenizer:
        """Load a pretrained tokenizer from HuggingFace hub.

        Args:
            name: Model name or path on HuggingFace hub (e.g., "gpt2").

        Returns:
            BPETokenizer instance.
        """
        _check_tokenizers_installed()
        tokenizer = Tokenizer.from_pretrained(name)
        return cls(tokenizer=tokenizer)

    @classmethod
    def train(
        cls,
        corpus: list[str],
        vocab_size: int = 30000,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Train a new BPE tokenizer on a corpus.

        Args:
            corpus: List of text strings to train on.
            vocab_size: Target vocabulary size.
            min_frequency: Minimum frequency for a merge to be applied.

        Returns:
            Trained BPETokenizer instance.
        """
        _check_tokenizers_installed()

        # Initialize a BPE model with UNK token
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Set up trainer with special tokens
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls._SPECIAL_TOKENS,
        )

        # Train from the corpus
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Set up post-processor for special tokens
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")

        if bos_id is not None and eos_id is not None:
            tokenizer.post_processor = TemplateProcessing(
                single="<bos> $A <eos>",
                pair="<bos> $A <eos> <bos> $B <eos>",
                special_tokens=[
                    ("<bos>", bos_id),
                    ("<eos>", eos_id),
                ],
            )

        return cls(tokenizer=tokenizer)

    def __repr__(self) -> str:
        if self._tokenizer is None:
            return "BPETokenizer(uninitialized)"
        return f"BPETokenizer(vocab_size={self.vocab_size})"
