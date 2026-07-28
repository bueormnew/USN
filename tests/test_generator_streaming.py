"""Tests for USNGenerator streaming and async streaming methods."""

import asyncio

import pytest
import torch
import torch.nn as nn

from usn.config.generation_config import USNGenerationConfig
from usn.core.interfaces import TokenizerInterface
from usn.core.types import ModelState, UnifiedState
from usn.inference.generator import USNGenerator


class MockTokenizer(TokenizerInterface):
    """Simple tokenizer for testing: maps chars to ints."""

    def __init__(self, vocab_size: int = 32) -> None:
        self._vocab_size = vocab_size
        # Special tokens
        self._pad = 0
        self._bos = 1
        self._eos = 2

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), self._vocab_size - 1) for c in text] or [self._bos]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(min(t, 127)) for t in token_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad

    @property
    def bos_token_id(self) -> int:
        return self._bos

    @property
    def eos_token_id(self) -> int:
        return self._eos


class MockModel(nn.Module):
    """Mock USN model that generates deterministic token sequences."""

    def __init__(
        self, vocab_size: int = 32, d_model: int = 16, d_s: int = 8, k: int = 4, eos_after: int = 5
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_s = d_s
        self.k = k
        self.eos_after = eos_after
        self._step = 0
        # Need at least one parameter so next(model.parameters()) works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, initial_state: ModelState | None = None):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Generate logits that strongly prefer token at position (step + 5)
        logits = torch.full((batch_size, seq_len, self.vocab_size), -10.0)
        # Make a specific token dominant
        target_token = (self._step + 5) % self.vocab_size
        # If we've generated enough tokens, emit EOS (token 2)
        if self._step >= self.eos_after:
            target_token = 2  # EOS
        logits[:, :, target_token] = 10.0
        self._step += 1

        # Return dummy state
        state = self._make_state(batch_size)
        return logits, state

    def _make_state(self, batch_size: int) -> ModelState:
        s = torch.zeros(batch_size, self.d_s)
        r = torch.zeros(batch_size, self.k, self.k)
        return ModelState(layers=(UnifiedState(semantic=s, relational=r),))

    def eval(self):
        return self


@pytest.fixture
def generator():
    """Create a USNGenerator with mock model and tokenizer."""
    model = MockModel(vocab_size=32, eos_after=5)
    tokenizer = MockTokenizer(vocab_size=32)
    config = USNGenerationConfig(temperature=0, max_new_tokens=10)
    return USNGenerator(model=model, tokenizer=tokenizer, config=config)


class TestStream:
    """Tests for the stream() method."""

    def test_stream_yields_tuples(self, generator):
        """stream() should yield (token_text, token_id, log_prob) tuples."""
        results = list(generator.stream("hi"))
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 3
            token_text, token_id, log_prob = item
            assert isinstance(token_text, str)
            assert isinstance(token_id, int)
            assert isinstance(log_prob, float)

    def test_stream_stops_on_eos(self, generator):
        """stream() should stop when EOS token is generated."""
        results = list(generator.stream("hi"))
        # Mock model emits EOS after 5 steps, so we get at most 5 tokens
        assert len(results) <= 5

    def test_stream_stops_on_max_tokens(self):
        """stream() respects max_new_tokens limit."""
        model = MockModel(vocab_size=32, eos_after=100)  # EOS far away
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=0, max_new_tokens=3)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)
        results = list(gen.stream("hi"))
        assert len(results) <= 3

    def test_stream_log_probs_are_negative(self, generator):
        """Log probabilities should be <= 0."""
        results = list(generator.stream("hi"))
        for _, _, log_prob in results:
            assert log_prob <= 0.0

    def test_stream_token_ids_are_valid(self, generator):
        """Token IDs should be valid (non-negative, within vocab)."""
        results = list(generator.stream("hi"))
        for _, token_id, _ in results:
            assert 0 <= token_id < 32

    def test_stream_no_eos_in_output(self, generator):
        """EOS token should not appear in yielded tokens."""
        results = list(generator.stream("hi"))
        for _, token_id, _ in results:
            assert token_id != generator.tokenizer.eos_token_id

    def test_stream_override_max_new_tokens(self):
        """max_new_tokens parameter overrides config."""
        model = MockModel(vocab_size=32, eos_after=100)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=0, max_new_tokens=50)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)
        results = list(gen.stream("hi", max_new_tokens=2))
        assert len(results) <= 2

    def test_stream_is_iterator(self, generator):
        """stream() should return an iterator (generator)."""
        result = generator.stream("hi")
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


class TestAstream:
    """Tests for the astream() async streaming method."""

    def test_astream_yields_tuples(self, generator):
        """astream() should yield same structure as stream()."""

        async def collect():
            results = []
            async for item in generator.astream("hi"):
                results.append(item)
            return results

        results = asyncio.run(collect())
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 3
            token_text, token_id, log_prob = item
            assert isinstance(token_text, str)
            assert isinstance(token_id, int)
            assert isinstance(log_prob, float)

    def test_astream_matches_stream(self):
        """astream() should produce same output as stream() for same input."""
        model = MockModel(vocab_size=32, eos_after=5)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        sync_results = list(gen.stream("hello"))

        # Reset model state counter for fair comparison
        model._step = 0

        async def collect():
            results = []
            async for item in gen.astream("hello"):
                results.append(item)
            return results

        async_results = asyncio.run(collect())
        assert sync_results == async_results

    def test_astream_stops_on_eos(self, generator):
        """astream() should stop on EOS."""

        async def collect():
            results = []
            async for item in generator.astream("hi"):
                results.append(item)
            return results

        results = asyncio.run(collect())
        assert len(results) <= 5

    def test_astream_is_async_iterator(self, generator):
        """astream() should return an async iterator."""

        async def check():
            ait = generator.astream("hi")
            assert hasattr(ait, "__aiter__")
            assert hasattr(ait, "__anext__")

        asyncio.run(check())
