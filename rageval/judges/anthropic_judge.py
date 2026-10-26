"""Judge backed by the Anthropic Messages API."""

import json

from rageval.config import JudgeConfig
from rageval.judges.base import JudgeError, LLMJudge, RawCompletion
from rageval.judges.cache import JudgeCache


class AnthropicJudge(LLMJudge):
    provider = "anthropic"

    def __init__(self, config: JudgeConfig, cache: JudgeCache | None = None, client=None):
        super().__init__(config, cache)
        import anthropic

        # The SDK already retries 429/5xx a couple of times; these extra retries with
        # longer backoff cover rate-limit windows and outages that outlast those.
        self.client = client or anthropic.AsyncAnthropic()
        self.retryable_exceptions = (anthropic.APIConnectionError, anthropic.RateLimitError,
                                     anthropic.InternalServerError)

    async def _complete(self, system: str, user: str, schema: dict) -> RawCompletion:
        response = await self.client.beta.messages.create(
            model=self.config.resolved_model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise JudgeError(f"judge request was declined (category: {category})")
        if response.stop_reason == "max_tokens":
            raise JudgeError("judge response hit max_tokens before finishing")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise JudgeError("judge response contained no text block")
        # Validate early so a schema mismatch surfaces as a JudgeError, not a KeyError later.
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"judge returned invalid JSON: {exc.msg}") from exc
        return RawCompletion(text=text, input_tokens=response.usage.input_tokens,
                             output_tokens=response.usage.output_tokens)

    async def aclose(self) -> None:
        await super().aclose()
        await self.client.close()
