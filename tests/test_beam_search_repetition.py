"""Tests for USNGenerator beam search and repetition penalty methods."""

import pytest
import torch
import torch.nn as nn

from usn.config.generation_config import USNGenerationConfig
from usn.core.interfaces import TokenizerInterface
from usn.core.types import ModelState, UnifiedState
from usn.inference.generator import USNGenerator


class MockTokenizer(TokenizerInterface):
    """Simple tokenizer for testing."""

    def __init__(self, vocab_size: int = 32) -> None:
        self._vocab_size = vocab_size
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
    """Mock USN model for testing beam search and repetition penalty."""

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
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, initial_state: ModelState | None = None):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Generate logits with a spread of token probabilities
        logits = torch.randn(batch_size, seq_len, self.vocab_size) * 0.5
        # Make specific tokens have higher logits
        target_token = (self._step + 5) % self.vocab_size
        if self._step >= self.eos_after:
            target_token = 2  # EOS
        logits[:, :, target_token] = 5.0
        # Give second-best token a moderate score for beam diversity
        second_token = (self._step + 7) % self.vocab_size
        if second_token != target_token:
            logits[:, :, second_token] = 3.0
        self._step += 1

        state = self._make_state(batch_size)
        return logits, state

    def _make_state(self, batch_size: int) -> ModelState:
        s = torch.zeros(batch_size, self.d_s)
        r = torch.zeros(batch_size, self.k, self.k)
        return ModelState(layers=(UnifiedState(semantic=s, relational=r),))

    def eval(self):
        return self


class TestApplyRepetitionPenalty:
    """Tests for _apply_repetition_penalty method."""

    @pytest.fixture
    def generator(self):
        model = MockModel(vocab_size=32, eos_after=10)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        return USNGenerator(model=model, tokenizer=tokenizer, config=config)

    def test_positive_logits_are_divided(self, generator):
        """Positive logits of repeated tokens should be divided by penalty."""
        logits = torch.tensor([5.0, 3.0, -2.0, 1.0, 0.5])
        generated_ids = torch.tensor([0, 1])  # Tokens 0 and 1 were generated
        penalty = 2.0

        result = generator._apply_repetition_penalty(logits, generated_ids, penalty)

        # Token 0 (positive): 5.0 / 2.0 = 2.5
        assert torch.isclose(result[0], torch.tensor(2.5))
        # Token 1 (positive): 3.0 / 2.0 = 1.5
        assert torch.isclose(result[1], torch.tensor(1.5))
        # Tokens 2, 3, 4 not affected
        assert torch.isclose(result[3], torch.tensor(1.0))
        assert torch.isclose(result[4], torch.tensor(0.5))

    def test_negative_logits_are_multiplied(self, generator):
        """Negative logits of repeated tokens should be multiplied by penalty."""
        logits = torch.tensor([-3.0, 2.0, -1.0, 4.0, -0.5])
        generated_ids = torch.tensor([0, 2, 4])
        penalty = 1.5

        result = generator._apply_repetition_penalty(logits, generated_ids, penalty)

        # Token 0 (negative): -3.0 * 1.5 = -4.5
        assert torch.isclose(result[0], torch.tensor(-4.5))
        # Token 2 (negative): -1.0 * 1.5 = -1.5
        assert torch.isclose(result[2], torch.tensor(-1.5))
        # Token 4 (negative): -0.5 * 1.5 = -0.75
        assert torch.isclose(result[4], torch.tensor(-0.75))
        # Tokens 1, 3 not affected
        assert torch.isclose(result[1], torch.tensor(2.0))
        assert torch.isclose(result[3], torch.tensor(4.0))

    def test_penalty_1_is_noop(self, generator):
        """Penalty of 1.0 should not change logits (edge case)."""
        logits = torch.tensor([3.0, -2.0, 1.0])
        generated_ids = torch.tensor([0, 1, 2])
        penalty = 1.0

        result = generator._apply_repetition_penalty(logits, generated_ids, penalty)

        assert torch.allclose(result, logits)

    def test_does_not_modify_ungenerated_tokens(self, generator):
        """Tokens not in generated_ids should remain unchanged."""
        logits = torch.tensor([5.0, 3.0, -2.0, 1.0, -1.0])
        generated_ids = torch.tensor([0])  # Only token 0
        penalty = 2.0

        result = generator._apply_repetition_penalty(logits, generated_ids, penalty)

        # Unaffected tokens
        assert torch.isclose(result[1], torch.tensor(3.0))
        assert torch.isclose(result[2], torch.tensor(-2.0))
        assert torch.isclose(result[3], torch.tensor(1.0))
        assert torch.isclose(result[4], torch.tensor(-1.0))

    def test_duplicate_ids_handled(self, generator):
        """Duplicate token IDs in generated_ids should not cause issues."""
        logits = torch.tensor([5.0, 3.0, -2.0])
        generated_ids = torch.tensor([0, 0, 1, 1, 0])  # Duplicates
        penalty = 2.0

        result = generator._apply_repetition_penalty(logits, generated_ids, penalty)

        # Token 0 should be penalized once (not multiple times)
        assert torch.isclose(result[0], torch.tensor(2.5))
        assert torch.isclose(result[1], torch.tensor(1.5))
        # Token 2 unaffected
        assert torch.isclose(result[2], torch.tensor(-2.0))

    def test_zero_logits_unchanged(self, generator):
        """Zero logits should remain zero regardless of penalty."""
        logits = torch.tensor([0.0, 5.0, -3.0])
        generated_ids = torch.tensor([0])
        penalty = 2.0

        result = generator._apply_repetition_penalty(logits, generated_ids, penalty)

        # 0.0 is not > 0, so it goes to the multiply path: 0.0 * 2.0 = 0.0
        assert torch.isclose(result[0], torch.tensor(0.0))


class TestBeamSearch:
    """Tests for _beam_search method."""

    def test_returns_list_of_dicts(self):
        """_beam_search should return a list of hypothesis dicts."""
        model = MockModel(vocab_size=32, eos_after=3)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=3, max_tokens=10, length_penalty=1.0)

        assert isinstance(results, list)
        assert len(results) > 0
        for hyp in results:
            assert "tokens" in hyp
            assert "log_prob" in hyp
            assert "score" in hyp
            assert isinstance(hyp["tokens"], list)
            assert isinstance(hyp["log_prob"], float)
            assert isinstance(hyp["score"], float)

    def test_beam_width_limits_results(self):
        """Number of results should not exceed beam_width."""
        model = MockModel(vocab_size=32, eos_after=3)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=2, max_tokens=10, length_penalty=1.0)

        assert len(results) <= 2

    def test_results_sorted_by_score_descending(self):
        """Results should be sorted by score (best first)."""
        model = MockModel(vocab_size=32, eos_after=4)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=3, max_tokens=10, length_penalty=1.0)

        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_log_probs_are_negative(self):
        """All log probabilities should be negative (or zero)."""
        model = MockModel(vocab_size=32, eos_after=3)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=3, max_tokens=10, length_penalty=1.0)

        for hyp in results:
            assert hyp["log_prob"] <= 0.0

    def test_early_stopping_on_eos(self):
        """Beams should stop when EOS is generated."""
        model = MockModel(vocab_size=32, eos_after=2)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=20)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=2, max_tokens=20, length_penalty=1.0)

        # With EOS after 2 steps, sequences shouldn't be very long
        for hyp in results:
            # EOS token (2) should not appear in tokens (it stops the beam)
            # but lengths should be short since model emits EOS early
            assert len(hyp["tokens"]) <= 10

    def test_max_tokens_respected(self):
        """Generation should not exceed max_tokens."""
        model = MockModel(vocab_size=32, eos_after=100)  # EOS far away
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=256)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=2, max_tokens=5, length_penalty=1.0)

        for hyp in results:
            assert len(hyp["tokens"]) <= 5

    def test_length_penalty_zero_no_normalization(self):
        """With length_penalty=0, score should equal log_prob."""
        model = MockModel(vocab_size=32, eos_after=3)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=2, max_tokens=10, length_penalty=0.0)

        for hyp in results:
            # length^0 = 1, so score = log_prob / 1 = log_prob
            assert abs(hyp["score"] - hyp["log_prob"]) < 1e-6

    def test_beam_width_1_greedy(self):
        """beam_width=1 should behave like greedy decoding."""
        model = MockModel(vocab_size=32, eos_after=3)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        prompt_ids = tokenizer.encode("hi")
        results = gen._beam_search(prompt_ids, beam_width=1, max_tokens=10, length_penalty=1.0)

        assert len(results) == 1
        assert len(results[0]["tokens"]) > 0


class TestRepetitionPenaltyIntegration:
    """Tests that repetition penalty is integrated into generate()."""

    def test_generate_with_repetition_penalty(self):
        """generate() should apply repetition penalty when configured."""
        model = MockModel(vocab_size=32, eos_after=8)
        tokenizer = MockTokenizer(vocab_size=32)
        # High repetition penalty
        config = USNGenerationConfig(temperature=0, max_new_tokens=8, repetition_penalty=2.0)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        output = gen.generate("hi")

        # Should produce output without errors
        assert output.token_ids.shape[0] == 1
        assert output.token_ids.shape[1] >= 0

    def test_generate_no_penalty_when_1(self):
        """generate() should not apply penalty when repetition_penalty=1.0."""
        model = MockModel(vocab_size=32, eos_after=5)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=0, max_new_tokens=10, repetition_penalty=1.0)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        output = gen.generate("hi")
        assert output.token_ids.shape[0] == 1

    def test_generate_penalty_via_kwargs(self):
        """repetition_penalty can be overridden via kwargs."""
        model = MockModel(vocab_size=32, eos_after=8)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=0, max_new_tokens=8, repetition_penalty=1.0)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        # Override penalty via kwargs
        output = gen.generate("hi", repetition_penalty=1.5)
        assert output.token_ids.shape[0] == 1


class TestCloneState:
    """Tests for _clone_state helper."""

    def test_clone_produces_independent_copy(self):
        """Cloned state should be independent of original."""
        model = MockModel(vocab_size=32, eos_after=5)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        s = torch.randn(1, 8)
        r = torch.randn(1, 4, 4)
        original = ModelState(layers=(UnifiedState(semantic=s, relational=r),))

        cloned = gen._clone_state(original)

        # Modify original
        original.layers[0].semantic.fill_(999.0)

        # Clone should be unaffected
        assert not torch.allclose(cloned.layers[0].semantic, torch.tensor(999.0))

    def test_clone_preserves_values(self):
        """Cloned state should have same values as original."""
        model = MockModel(vocab_size=32, eos_after=5)
        tokenizer = MockTokenizer(vocab_size=32)
        config = USNGenerationConfig(temperature=1.0, max_new_tokens=10)
        gen = USNGenerator(model=model, tokenizer=tokenizer, config=config)

        s = torch.randn(1, 8)
        r = torch.randn(1, 4, 4)
        original = ModelState(layers=(UnifiedState(semantic=s.clone(), relational=r.clone()),))

        cloned = gen._clone_state(original)

        assert torch.allclose(cloned.layers[0].semantic, original.layers[0].semantic)
        assert torch.allclose(cloned.layers[0].relational, original.layers[0].relational)
