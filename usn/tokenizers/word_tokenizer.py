"""Word-level tokenizer with configurable vocabulary.

Implements TokenizerInterface with whitespace-based splitting
and a user-provided or auto-built word vocabulary.
"""

from usn.core.interfaces import TokenizerInterface


class WordTokenizer(TokenizerInterface):
    """Word-level tokenizer splitting on whitespace.

    Special tokens occupy IDs 0-3:
        PAD=0, BOS=1, EOS=2, UNK=3

    The vocabulary maps words to IDs starting at offset 4.

    Example:
        >>> tok = WordTokenizer(["hello", "world", "foo"])
        >>> tok.encode("hello world")
        [4, 5]
        >>> tok.decode([4, 5])
        'hello world'
    """

    # Fixed special token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    _SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
    _NUM_SPECIAL = 4

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        """Initialize with an optional list of words as vocabulary.

        Args:
            vocabulary: Ordered list of words. Duplicates are removed
                        while preserving order. If None or empty, use
                        ``from_text`` to build vocabulary.
        """
        if vocabulary is None:
            vocabulary = []

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_words: list[str] = []
        for w in vocabulary:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        self._words = unique_words
        self._word_to_id: dict[str, int] = {
            w: i + self._NUM_SPECIAL for i, w in enumerate(unique_words)
        }
        self._id_to_word: dict[int, str] = {
            i + self._NUM_SPECIAL: w for i, w in enumerate(unique_words)
        }

    @classmethod
    def from_text(cls, text: str, min_freq: int = 1) -> "WordTokenizer":
        """Build a WordTokenizer from text, using whitespace splitting.

        Args:
            text: Text to extract words from.
            min_freq: Minimum frequency for a word to be included
                      in the vocabulary.

        Returns:
            WordTokenizer with vocabulary derived from the text.
        """
        word_counts: dict[str, int] = {}
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

        # Preserve first-appearance order, filter by min_freq
        seen: set[str] = set()
        words: list[str] = []
        for word in text.split():
            if word not in seen and word_counts[word] >= min_freq:
                seen.add(word)
                words.append(word)

        return cls(words)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs by splitting on whitespace.

        Words not in vocabulary are mapped to UNK_ID.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        if not text.strip():
            return []
        return [self._word_to_id.get(word, self.UNK_ID) for word in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text, joining words with spaces.

        Special tokens (PAD, BOS, EOS, UNK) are skipped in output.
        Unknown IDs are replaced with '<?>'.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        words: list[str] = []
        for tid in token_ids:
            # Skip special tokens
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            word = self._id_to_word.get(tid)
            if word is not None:
                words.append(word)
            else:
                words.append("<?>")  # Placeholder for invalid IDs
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._words) + self._NUM_SPECIAL

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

    @property
    def words(self) -> list[str]:
        """List of words in the vocabulary (excluding special tokens)."""
        return list(self._words)

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={self.vocab_size})"
