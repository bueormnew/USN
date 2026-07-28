"""Character-level tokenizer for simple experiments.

Implements TokenizerInterface with character-level encoding/decoding.
Useful for synthetic tasks (e.g., math dataset) where each character
is a token.
"""

from usn.core.interfaces import TokenizerInterface


class CharTokenizer(TokenizerInterface):
    """Character-level tokenizer for testing and experiments.

    Special tokens occupy IDs 0-3:
        PAD=0, BOS=1, EOS=2, UNK=3

    The vocabulary is built from a provided string of characters,
    or auto-built from text via ``from_text``.

    Example:
        >>> tok = CharTokenizer("0123456789+-=* ")
        >>> tok.encode("3+5=8")
        [7, 14, 9, 16, 12]
        >>> tok.decode([7, 14, 9, 16, 12])
        '3+5=8'
    """

    # Fixed special token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    _SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
    _NUM_SPECIAL = 4

    def __init__(self, chars: str = "") -> None:
        """Initialize with a string of unique characters as vocabulary.

        Args:
            chars: String of characters to include in vocabulary.
                   Duplicates are removed while preserving order.
                   If empty, use ``from_text`` to build vocabulary later.
        """
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_chars: list[str] = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        self._chars = unique_chars
        # Map char -> token_id (offset by special tokens)
        self._char_to_id: dict[str, int] = {
            c: i + self._NUM_SPECIAL for i, c in enumerate(unique_chars)
        }
        # Map token_id -> char
        self._id_to_char: dict[int, str] = {
            i + self._NUM_SPECIAL: c for i, c in enumerate(unique_chars)
        }

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a CharTokenizer from text, using all unique characters found.

        Args:
            text: Text to extract characters from.

        Returns:
            CharTokenizer with vocabulary derived from the text.
        """
        # Collect unique characters in order of first appearance
        seen: set[str] = set()
        chars: list[str] = []
        for c in text:
            if c not in seen:
                seen.add(c)
                chars.append(c)
        return cls("".join(chars))

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs.

        Characters not in vocabulary are mapped to UNK_ID.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        return [self._char_to_id.get(c, self.UNK_ID) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Special tokens (PAD, BOS, EOS, UNK) are skipped in output.
        Unknown IDs are replaced with the Unicode replacement character.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        chars: list[str] = []
        for tid in token_ids:
            # Skip special tokens
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            char = self._id_to_char.get(tid)
            if char is not None:
                chars.append(char)
            else:
                chars.append("\ufffd")  # Replacement character for invalid IDs
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._chars) + self._NUM_SPECIAL

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
    def characters(self) -> list[str]:
        """List of characters in the vocabulary (excluding special tokens)."""
        return list(self._chars)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size}, chars={''.join(self._chars[:20])}{'...' if len(self._chars) > 20 else ''})"
