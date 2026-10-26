"""Judge backed by the OpenAI Chat Completions API."""

from rageval.config import JudgeConfig
from rageval.judges.base import JudgeError, LLMJudge, RawCompletion
from rageval.judges.cache import JudgeCache


class OpenAIJudge(LLMJudge):
    provider = "openai"

    def __init__(self, config: JudgeConfig, cache: JudgeCache | None = None, client=None):
        super().__init__(config, cache)
        import openai

        self.client = client or openai.AsyncOpenAI()
        self.retryable_exceptions = (openai.APIConnectionError, openai.RateLimitError,
                                     openai.InternalServerError)

    async def _complete(self, system: str, user: str, schema: dict) -> RawCompletion:
        response = await self.client.chat.completions.create(
            model=self.config.resolved_model,
            temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "judgement", "schema": schema, "strict": True},
            },
        )
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise JudgeError(f"judge request was declined: {choice.message.refusal}")
        if choice.message.content is None:
            raise JudgeError("judge response contained no content")
        usage = response.usage
        return RawCompletion(
            text=choice.message.content,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    async def aclose(self) -> None:
        await super().aclose()
        await self.client.close()
