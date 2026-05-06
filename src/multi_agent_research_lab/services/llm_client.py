"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# gpt-4o-mini pricing (USD per 1M tokens, as of 2025)
_INPUT_COST_PER_1M = 0.15
_OUTPUT_COST_PER_1M = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class TokenUsageAccumulator:
    """Accumulates token usage across multiple LLM calls."""

    total_input_tokens: int = field(default=0)
    total_output_tokens: int = field(default=0)
    total_cost_usd: float = field(default=0.0)

    def add(self, response: LLMResponse) -> None:
        if response.input_tokens:
            self.total_input_tokens += response.input_tokens
        if response.output_tokens:
            self.total_output_tokens += response.output_tokens
        if response.cost_usd:
            self.total_cost_usd += response.cost_usd


class LLMClient:
    """Provider-agnostic LLM client — backed by OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment / .env")

        from openai import OpenAI  # imported lazily to keep startup fast

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with token usage tracked."""
        logger.debug("LLMClient.complete | model=%s | system_len=%d | user_len=%d",
                     self._model, len(system_prompt), len(user_prompt))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        choice = response.choices[0]
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        cost_usd: float | None = None
        if input_tokens is not None and output_tokens is not None:
            cost_usd = (
                input_tokens / 1_000_000 * _INPUT_COST_PER_1M
                + output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M
            )

        result = LLMResponse(
            content=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        logger.debug("LLMClient.complete | tokens_in=%s tokens_out=%s cost=$%.6f",
                     input_tokens, output_tokens, cost_usd or 0)
        return result
