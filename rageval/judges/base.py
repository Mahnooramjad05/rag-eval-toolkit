"""Judge interface plus the shared machinery for LLM-backed judges.

A judge answers four narrow questions that the generation metrics are built on:
extract claims, verify claims against context, rate answer relevance, and rate
whether a retrieved chunk is relevant. Keeping them narrow makes each LLM call
cheap, cacheable and easy to inspect in the report.
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel

from rageval.config import JudgeConfig
from rageval.judges import prompts
from rageval.judges.cache import JudgeCache
from rageval.schemas import JudgeUsage


class JudgeError(RuntimeError):
    """Raised when a judge call fails after retries or returns unusable output."""


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    reason: str


class RelevanceVerdict(BaseModel):
    score: int  # 1-5
    reason: str

    @property
    def normalized(self) -> float:
        return (self.score - 1) / 4


class ChunkVerdict(BaseModel):
    relevant: bool
    reason: str


@dataclass
class UsageTracker:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def record(self, *, input_tokens: int = 0, output_tokens: int = 0,
                     latency_s: float = 0.0, cache_hit: bool = False) -> None:
        async with self._lock:
            if cache_hit:
                self.cache_hits += 1
                return
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.latency_s += latency_s

    def snapshot(self, config: JudgeConfig | None = None) -> JudgeUsage:
        cost = None
        if config and config.input_cost_per_mtok is not None and config.output_cost_per_mtok is not None:
            cost = round(
                self.input_tokens / 1e6 * config.input_cost_per_mtok
                + self.output_tokens / 1e6 * config.output_cost_per_mtok,
                6,
            )
        return JudgeUsage(
            calls=self.calls,
            cache_hits=self.cache_hits,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_s=round(self.latency_s, 3),
            est_cost_usd=cost,
        )


class Judge(ABC):
    """Interface every judge implements."""

    provider: str = "base"

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.usage = UsageTracker()

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.config.resolved_model}"

    @abstractmethod
    async def extract_claims(self, text: str) -> list[str]: ...

    @abstractmethod
    async def verify_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimVerdict]: ...

    @abstractmethod
    async def rate_answer_relevance(self, question: str, answer: str) -> RelevanceVerdict: ...

    @abstractmethod
    async def rate_chunk_relevance(self, question: str, chunk: str,
                                   reference: str | None = None) -> ChunkVerdict: ...

    async def aclose(self) -> None:
        """Release network clients and caches."""


@dataclass
class RawCompletion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class RateLimiter:
    """Spaces request starts evenly to stay under a requests-per-minute budget."""

    def __init__(self, requests_per_minute: int | None):
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise JudgeError(f"judge did not return JSON: {text[:200]!r}") from None
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise JudgeError(f"judge returned malformed JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise JudgeError("judge returned JSON that is not an object")
    return value


class LLMJudge(Judge):
    """Base for judges backed by a hosted model.

    Subclasses implement `_complete` (one request, no retries) and list which
    exceptions are worth retrying. Caching, concurrency limits, rate limiting,
    retries and usage tracking live here so every provider behaves the same.
    """

    retryable_exceptions: tuple[type[BaseException], ...] = ()
    base_retry_delay: float = 1.0

    def __init__(self, config: JudgeConfig, cache: JudgeCache | None = None):
        super().__init__(config)
        self.cache = cache
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._limiter = RateLimiter(config.requests_per_minute)

    @abstractmethod
    async def _complete(self, system: str, user: str, schema: dict) -> RawCompletion: ...

    async def complete_json(self, user: str, schema: dict) -> dict:
        key = None
        if self.cache is not None:
            key = JudgeCache.make_key(self.provider, self.config.resolved_model,
                                      prompts.SYSTEM, user, schema)
            cached = self.cache.get(key)
            if cached is not None:
                await self.usage.record(cache_hit=True)
                return cached

        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    await self._limiter.wait()
                    started = time.monotonic()
                    raw = await self._complete(prompts.SYSTEM, user, schema)
                    elapsed = time.monotonic() - started
                break
            except self.retryable_exceptions as exc:
                attempt += 1
                if attempt > self.config.max_retries:
                    raise JudgeError(f"{self.name} failed after {attempt} attempts: {exc}") from exc
                await asyncio.sleep(self.base_retry_delay * 2 ** (attempt - 1))

        await self.usage.record(input_tokens=raw.input_tokens,
                                output_tokens=raw.output_tokens, latency_s=elapsed)
        data = parse_json_object(raw.text)
        if self.cache is not None and key is not None:
            self.cache.set(key, data)
        return data

    async def extract_claims(self, text: str) -> list[str]:
        data = await self.complete_json(prompts.EXTRACT_CLAIMS.format(text=text),
                                        prompts.EXTRACT_CLAIMS_SCHEMA)
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            raise JudgeError("judge returned 'claims' that is not a list")
        return [str(c).strip() for c in claims if str(c).strip()]

    async def verify_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimVerdict]:
        if not claims:
            return []
        data = await self.complete_json(
            prompts.VERIFY_CLAIMS.format(context=prompts.format_contexts(contexts),
                                         claims=prompts.format_claims(claims)),
            prompts.VERIFY_CLAIMS_SCHEMA,
        )
        by_index = {}
        for item in data.get("verdicts", []):
            if isinstance(item, dict) and isinstance(item.get("index"), int):
                by_index[item["index"]] = item
        verdicts = []
        for i, claim in enumerate(claims):
            item = by_index.get(i)
            if item is None:
                verdicts.append(ClaimVerdict(claim=claim, supported=False,
                                             reason="Judge returned no verdict for this claim."))
            else:
                verdicts.append(ClaimVerdict(claim=claim, supported=bool(item.get("supported")),
                                             reason=str(item.get("reason", ""))))
        return verdicts

    async def rate_answer_relevance(self, question: str, answer: str) -> RelevanceVerdict:
        data = await self.complete_json(
            prompts.ANSWER_RELEVANCE.format(question=question, answer=answer),
            prompts.ANSWER_RELEVANCE_SCHEMA,
        )
        try:
            score = min(5, max(1, int(data["score"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeError("judge returned no usable relevance score") from exc
        return RelevanceVerdict(score=score, reason=str(data.get("reason", "")))

    async def rate_chunk_relevance(self, question: str, chunk: str,
                                   reference: str | None = None) -> ChunkVerdict:
        reference_block = (f"\nREFERENCE ANSWER (for guidance):\n{reference}\n" if reference else "")
        data = await self.complete_json(
            prompts.CHUNK_RELEVANCE.format(question=question, chunk=chunk,
                                           reference_block=reference_block),
            prompts.CHUNK_RELEVANCE_SCHEMA,
        )
        return ChunkVerdict(relevant=bool(data.get("relevant")), reason=str(data.get("reason", "")))

    async def aclose(self) -> None:
        if self.cache is not None:
            self.cache.close()
