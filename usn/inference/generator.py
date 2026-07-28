"""USN Generator - Autoregressive text generation with O(1) memory.

Generates text token-by-token using the persistent state mechanism.
Memory usage is constant regardless of generated sequence length.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from usn.config.generation_config import USNGenerationConfig
from usn.core.interfaces import TokenizerInterface
from usn.core.types import GenerationOutput, ModelState, UnifiedState

logger = logging.getLogger(__name__)


class USNGenerator:
    """Autoregressive text generator for USN models.

    Operates with O(1) memory w.r.t. generated length — only the
    fixed-size state is maintained, not full context history.

    Args:
        model: Trained USN model (must support forward with state).
        tokenizer: Tokenizer for encode/decode operations.
        config: Generation configuration (temperature, top-k, etc.).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: TokenizerInterface,
        config: USNGenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or USNGenerationConfig()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate text from prompt(s).

        Args:
            prompt: Input text string or list of strings (batch).
            max_new_tokens: Override config.max_new_tokens if provided.
            **kwargs: Override generation config parameters.

        Returns:
            GenerationOutput with token_ids, log_probs, and final_state.
        """
        self.model.eval()
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = kwargs.get("temperature", self.config.temperature)
        top_k = kwargs.get("top_k", self.config.top_k)
        top_p = kwargs.get("top_p", self.config.top_p)
        repetition_penalty = kwargs.get("repetition_penalty", self.config.repetition_penalty)

        # Handle batch
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Encode prompts
        batch_ids = [self.tokenizer.encode(p) for p in prompts]

        # Prefill: process prompt tokens to build initial state
        state = self._prefill(batch_ids)

        # Generation loop
        generated: list[list[int]] = [[] for _ in range(len(prompts))]
        all_log_probs: list[list[float]] = [[] for _ in range(len(prompts))]
        last_token = torch.tensor(
            [[p[-1]] for p in batch_ids], dtype=torch.long, device=self.device
        )

        stop_tokens = (
            set(self.config.stop_tokens)
            if self.config.stop_tokens
            else {self.tokenizer.eos_token_id}
        )

        # Track which sequences have finished
        finished = [False] * len(prompts)

        for step in range(max_tokens):
            # Single-step forward: O(1) memory per token
            logits, state = self.model(last_token, initial_state=state)
            logits = logits[:, -1, :]  # (batch, vocab)

            # Apply repetition penalty before decode strategy
            if repetition_penalty > 1.0:
                for i in range(len(prompts)):
                    if not finished[i] and generated[i]:
                        gen_ids = torch.tensor(generated[i], dtype=torch.long, device=self.device)
                        logits[i] = self._apply_repetition_penalty(
                            logits[i], gen_ids, repetition_penalty
                        )

            # Decode strategy: apply temperature, top-k, top-p
            logits = self._apply_decode_strategy(logits, temperature, top_k, top_p)

            # Sample or greedy decode
            probs = F.softmax(logits, dim=-1)
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, next_token)

            # Append tokens for unfinished sequences
            for i in range(len(prompts)):
                if finished[i]:
                    continue
                tok = int(next_token[i].item())
                if tok in stop_tokens:
                    finished[i] = True
                else:
                    generated[i].append(tok)
                    all_log_probs[i].append(token_log_probs[i].item())

            # Stop if all sequences finished
            if all(finished):
                break

            last_token = next_token

        return self._build_output(generated, all_log_probs, state, len(prompts))

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[tuple[str, int, float]]:
        """Stream generated tokens one at a time.

        Python generator yielding (token_text, token_id, log_prob) as each
        token is produced. No buffering — yields immediately. Uses O(1)
        memory per generated token (only the fixed-size state is kept).

        Args:
            prompt: Input text string (single prompt only).
            max_new_tokens: Override config.max_new_tokens if provided.
            **kwargs: Override generation config parameters
                (temperature, top_k, top_p).

        Yields:
            Tuple of (token_text, token_id, log_prob) for each generated token.
        """
        self.model.eval()
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = kwargs.get("temperature", self.config.temperature)
        top_k = kwargs.get("top_k", self.config.top_k)
        top_p = kwargs.get("top_p", self.config.top_p)

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)

        # Prefill: process prompt tokens to build initial state
        state = self._prefill([prompt_ids])

        # Prepare last token from prompt
        last_token = torch.tensor([[prompt_ids[-1]]], dtype=torch.long, device=self.device)

        stop_tokens = (
            set(self.config.stop_tokens)
            if self.config.stop_tokens
            else {self.tokenizer.eos_token_id}
        )

        for _step in range(max_tokens):
            # Single-step forward: O(1) memory per token
            logits, state = self.model(last_token, initial_state=state)
            logits = logits[:, -1, :]  # (1, vocab)

            # Apply decode strategy
            logits = self._apply_decode_strategy(logits, temperature, top_k, top_p)

            # Sample or greedy decode
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Compute log probability
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_prob = log_probs.gather(1, next_token).item()

            tok_id = int(next_token.item())

            # Check stop condition
            if tok_id in stop_tokens:
                return

            # Decode token text
            token_text = self.tokenizer.decode([tok_id])

            # Yield immediately — no buffering
            yield (token_text, tok_id, float(token_log_prob))

            last_token = next_token

    @torch.no_grad()
    async def astream(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, int, float]]:
        """Async stream generated tokens one at a time.

        Async version of stream() for use with async web frameworks
        (e.g., FastAPI, aiohttp). Same behavior as stream() but yields
        asynchronously.

        Args:
            prompt: Input text string (single prompt only).
            max_new_tokens: Override config.max_new_tokens if provided.
            **kwargs: Override generation config parameters
                (temperature, top_k, top_p).

        Yields:
            Tuple of (token_text, token_id, log_prob) for each generated token.
        """
        for token_text, token_id, log_prob in self.stream(prompt, max_new_tokens, **kwargs):
            yield (token_text, token_id, log_prob)

    def _prefill(self, batch_ids: list[list[int]]) -> ModelState:
        """Process prompt tokens to populate initial state.

        Processes tokens one at a time to build state correctly,
        maintaining exact consistency with the sequential recurrence
        used during training. This ensures the write gate g_t sees
        the correct S_{t-1} at each step.

        Args:
            batch_ids: List of encoded prompt token ID sequences.

        Returns:
            ModelState after processing all prompt tokens.
        """
        # Process token by token for correct state building
        # (batch_ids may have different lengths — handle first sequence for now)
        state = None
        for batch_tokens in batch_ids:
            state = None
            for tok in batch_tokens:
                input_t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
                _, state = self.model(input_t, initial_state=state)
        return state  # type: ignore[return-value]

    def _apply_decode_strategy(
        self, logits: Tensor, temperature: float, top_k: int, top_p: float
    ) -> Tensor:
        """Apply decoding strategy to logits.

        Strategies applied in order:
        1. Temperature scaling (if temperature > 0 and != 1.0)
        2. Top-k filtering (if top_k > 0)
        3. Top-p / nucleus filtering (if top_p < 1.0)

        Args:
            logits: Raw logits (batch, vocab_size).
            temperature: Sampling temperature (0 = greedy).
            top_k: Number of top tokens to keep (0 = disabled).
            top_p: Cumulative probability threshold (1.0 = disabled).

        Returns:
            Filtered logits ready for sampling.
        """
        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            logits = self._top_k_filter(logits, top_k)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)

        return logits

    def _top_k_filter(self, logits: Tensor, k: int) -> Tensor:
        """Zero out all logits below the k-th highest value.

        Args:
            logits: (batch, vocab_size) logits tensor.
            k: Number of top values to retain.

        Returns:
            Filtered logits with non-top-k values set to -inf.
        """
        top_k_values, _ = logits.topk(k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        return logits

    def _top_p_filter(self, logits: Tensor, p: float) -> Tensor:
        """Nucleus sampling: keep smallest set with cumulative prob >= p.

        Args:
            logits: (batch, vocab_size) logits tensor.
            p: Cumulative probability threshold.

        Returns:
            Filtered logits with low-probability tokens set to -inf.
        """
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Remove tokens with cumulative probability above p
        mask = cumulative_probs - sorted_probs > p
        sorted_logits[mask] = float("-inf")

        # Scatter back to original positions
        logits = logits.scatter(1, sorted_indices, sorted_logits)
        return logits

    def _build_output(
        self,
        generated: list[list[int]],
        all_log_probs: list[list[float]],
        state: ModelState,
        batch_size: int,
    ) -> GenerationOutput:
        """Build GenerationOutput from collected tokens and log probs.

        Args:
            generated: List of generated token ID lists per sequence.
            all_log_probs: List of log probability lists per sequence.
            state: Final model state after generation.
            batch_size: Number of sequences in the batch.

        Returns:
            GenerationOutput named tuple.
        """
        max_gen_len = max((len(g) for g in generated), default=0)

        if max_gen_len == 0:
            token_ids_tensor = torch.zeros(batch_size, 0, dtype=torch.long, device=self.device)
            log_probs_tensor = torch.zeros(batch_size, 0, device=self.device)
        else:
            token_ids_tensor = torch.full(
                (batch_size, max_gen_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            log_probs_tensor = torch.zeros(batch_size, max_gen_len, device=self.device)

            for i, (g, lp) in enumerate(zip(generated, all_log_probs)):
                if g:
                    token_ids_tensor[i, : len(g)] = torch.tensor(
                        g, dtype=torch.long, device=self.device
                    )
                    log_probs_tensor[i, : len(lp)] = torch.tensor(lp, device=self.device)

        return GenerationOutput(
            token_ids=token_ids_tensor,
            log_probs=log_probs_tensor,
            final_state=state,
        )

    def _apply_repetition_penalty(
        self, logits: Tensor, generated_ids: Tensor, penalty: float
    ) -> Tensor:
        """Apply repetition penalty to logits of previously generated tokens.

        Reduces the probability of tokens that have already been generated.
        For positive logits, divides by the penalty factor (>1.0).
        For negative logits, multiplies by the penalty factor (same effect
        of reducing probability since dividing a negative number makes it
        more negative, but multiplying achieves the same directional effect).

        Args:
            logits: (vocab_size,) logits tensor for a single sequence.
            generated_ids: 1-D tensor of previously generated token IDs.
            penalty: Penalty factor (>1.0). Higher values penalize more.

        Returns:
            Modified logits tensor with repetition penalty applied.
        """
        # Get unique token IDs that have been generated
        unique_ids = generated_ids.unique()

        # Gather logits for previously generated tokens
        score = logits[unique_ids]

        # Apply penalty: divide positive logits, multiply negative logits
        # This ensures the probability of repeated tokens always decreases
        score = torch.where(
            score > 0,
            score / penalty,
            score * penalty,
        )

        # Scatter modified scores back
        logits = logits.clone()
        logits[unique_ids] = score
        return logits

    @torch.no_grad()
    def _beam_search(
        self,
        prompt_ids: list[int],
        beam_width: int,
        max_tokens: int,
        length_penalty: float,
    ) -> list[dict]:
        """Beam search decoding maintaining multiple hypotheses.

        Maintains beam_width active hypotheses, expanding each at every
        step and keeping only the top-scoring candidates. Uses length
        penalty to normalize scores across different sequence lengths.

        Score = log_prob / (length ^ length_penalty)

        Args:
            prompt_ids: Encoded prompt token IDs.
            beam_width: Number of active hypotheses to maintain.
            max_tokens: Maximum number of tokens to generate.
            length_penalty: Exponent for length normalization.
                1.0 = linear length normalization.
                0.0 = no length normalization.
                >1.0 = favor longer sequences.

        Returns:
            List of hypothesis dicts sorted by score (best first), each
            containing:
                - tokens: list[int] of generated token IDs
                - log_prob: float total log probability
                - score: float length-normalized score
        """
        self.model.eval()

        # Prefill state from prompt
        state = self._prefill([prompt_ids])

        stop_tokens = (
            set(self.config.stop_tokens)
            if self.config.stop_tokens
            else {self.tokenizer.eos_token_id}
        )

        # Initialize beams
        # Each beam: {tokens, log_prob, state, finished}
        beams: list[dict] = [
            {
                "tokens": [],
                "log_prob": 0.0,
                "state": state,
                "finished": False,
            }
        ]

        # Last token from prompt to seed generation
        last_tokens = [prompt_ids[-1]]

        for step in range(max_tokens):
            all_candidates: list[dict] = []

            for beam_idx, beam in enumerate(beams):
                if beam["finished"]:
                    # Finished beams are carried forward as-is
                    all_candidates.append(beam)
                    continue

                # Forward pass for this beam
                token_input = torch.tensor(
                    [[last_tokens[beam_idx]]],
                    dtype=torch.long,
                    device=self.device,
                )
                logits, new_state = self.model(token_input, initial_state=beam["state"])
                logits = logits[:, -1, :]  # (1, vocab)

                # Apply repetition penalty if configured
                if self.config.repetition_penalty > 1.0 and beam["tokens"]:
                    gen_ids = torch.tensor(beam["tokens"], dtype=torch.long, device=self.device)
                    logits[0] = self._apply_repetition_penalty(
                        logits[0], gen_ids, self.config.repetition_penalty
                    )

                # Get log probabilities
                log_probs = F.log_softmax(logits[0], dim=-1)

                # Get top-k candidates for expansion
                topk_log_probs, topk_ids = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    tok_id = topk_ids[k].item()
                    tok_log_prob = topk_log_probs[k].item()

                    new_tokens = beam["tokens"] + [tok_id]
                    new_log_prob = beam["log_prob"] + tok_log_prob
                    is_finished = tok_id in stop_tokens

                    # Clone state for each new candidate
                    cloned_state = self._clone_state(new_state)

                    candidate = {
                        "tokens": new_tokens,
                        "log_prob": new_log_prob,
                        "state": cloned_state,
                        "finished": is_finished,
                    }
                    all_candidates.append(candidate)

            # Score candidates with length penalty
            def _score(candidate: dict) -> float:
                length = max(len(candidate["tokens"]), 1)
                return candidate["log_prob"] / (length**length_penalty)

            # Sort by score and keep top beam_width
            all_candidates.sort(key=_score, reverse=True)
            beams = all_candidates[:beam_width]

            # Early stopping: all beams finished
            if all(b["finished"] for b in beams):
                break

            # Prepare last tokens for next step
            last_tokens = []
            for beam in beams:
                if beam["finished"] or not beam["tokens"]:
                    last_tokens.append(prompt_ids[-1])
                else:
                    last_tokens.append(beam["tokens"][-1])

        # Build final results sorted by score
        results: list[dict] = []
        for beam in beams:
            length = max(len(beam["tokens"]), 1)
            score = beam["log_prob"] / (length**length_penalty)
            results.append(
                {
                    "tokens": beam["tokens"],
                    "log_prob": beam["log_prob"],
                    "score": score,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _clone_state(self, state: ModelState) -> ModelState:
        """Deep-clone a ModelState for beam expansion.

        Each beam needs its own copy of the state to avoid mutations
        between beams affecting each other.

        Args:
            state: ModelState to clone.

        Returns:
            A new ModelState with cloned tensors.
        """
        cloned_layers: list[UnifiedState] = []
        for layer_state in state.layers:
            cloned_layers.append(
                UnifiedState(
                    semantic=layer_state.semantic.clone(),
                    relational=layer_state.relational.clone(),
                )
            )
        return ModelState(layers=tuple(cloned_layers))
