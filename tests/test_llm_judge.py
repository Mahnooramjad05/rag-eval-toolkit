"""LLM judge plumbing: parsing, caching, retries and the provider request shapes.

Provider clients are replaced with fakes, so these tests make no network calls.
"""

import json
from types import SimpleNamespace

import pytest

from rageval.config import JudgeConfig
from rageval.judges.anthropic_judge import AnthropicJudge
from rageval.judges.base import JudgeError, LLMJudge, RawCompletion, parse_json_object
from rageval.judges.cache import JudgeCache
from rageval.judges.openai_judge import OpenAIJudge


class Flaky(Exception):
    pass


class FakeJudge(LLMJudge):
    provider = "fake"
    retryable_exceptions = (Flaky,)
    base_retry_delay = 0.0

    def __init__(self, responses, config=None, cache=None):
        super().__init__(config or JudgeConfig(provider="mock", model="fake-model", max_retries=2), cache)
        self.responses = list(responses)
        self.prompts = []

    async def _complete(self, system, user, schema):
        self.prompts.append(user)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return RawCompletion(text=item, input_tokens=100, output_tokens=20)


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 2} hope it helps') == {"a": 2}
    with pytest.raises(JudgeError):
        parse_json_object("no json here")
    with pytest.raises(JudgeError):
        parse_json_object("[1, 2]")


async def test_verify_claims_maps_by_index_and_flags_missing():
    judge = FakeJudge([json.dumps({"verdicts": [
        {"index": 1, "supported": True, "reason": "stated"},
    ]})])
    verdicts = await judge.verify_claims(["claim zero", "claim one"], ["ctx"])
    assert [v.supported for v in verdicts] == [False, True]
    assert "no verdict" in verdicts[0].reason
    assert "0. claim zero" in judge.prompts[0]


async def test_relevance_score_is_clamped():
    judge = FakeJudge(['{"score": 9, "reason": "x"}'])
    verdict = await judge.rate_answer_relevance("q", "a")
    assert verdict.score == 5 and verdict.normalized == 1.0


async def test_cache_avoids_second_call(tmp_path):
    cache = JudgeCache(tmp_path / "cache.sqlite")
    judge = FakeJudge(['{"claims": ["one fact"]}'], cache=cache)
    assert await judge.extract_claims("text") == ["one fact"]
    assert await judge.extract_claims("text") == ["one fact"]  # would pop an empty list if not cached
    usage = judge.usage.snapshot()
    assert (usage.calls, usage.cache_hits, usage.input_tokens) == (1, 1, 100)
    cache.close()


async def test_retries_then_succeeds():
    judge = FakeJudge([Flaky("429"), Flaky("503"), '{"claims": []}'])
    assert await judge.extract_claims("text") == []
    assert judge.usage.calls == 1


async def test_gives_up_after_max_retries():
    judge = FakeJudge([Flaky("1"), Flaky("2"), Flaky("3")])
    with pytest.raises(JudgeError, match="after 3 attempts"):
        await judge.extract_claims("text")


def test_cost_estimate_uses_configured_prices():
    judge = FakeJudge([], config=JudgeConfig(provider="mock", input_cost_per_mtok=2.0,
                                             output_cost_per_mtok=10.0))
    judge.usage.input_tokens, judge.usage.output_tokens = 1_000_000, 100_000
    assert judge.usage.snapshot(judge.config).est_cost_usd == pytest.approx(3.0)
    assert judge.usage.snapshot(None).est_cost_usd is None


# --- Anthropic request shape -------------------------------------------------

class FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def anthropic_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=SimpleNamespace(category="cyber"),
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=50, output_tokens=10),
    )


def make_anthropic(response):
    messages = FakeAnthropicMessages(response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    judge = AnthropicJudge(JudgeConfig(provider="anthropic"), client=client)
    return judge, messages


async def test_anthropic_request_uses_structured_output_and_fallbacks():
    judge, messages = make_anthropic(anthropic_response('{"claims": ["a"]}'))
    assert await judge.extract_claims("text") == ["a"]
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kw
    assert judge.usage.input_tokens == 50


async def test_anthropic_refusal_raises():
    judge, _ = make_anthropic(anthropic_response("", stop_reason="refusal"))
    with pytest.raises(JudgeError, match="declined"):
        await judge.extract_claims("text")


# --- OpenAI request shape ----------------------------------------------------

async def test_openai_request_uses_json_schema():
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content='{"relevant": true, "reason": "ok"}', refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    judge = OpenAIJudge(JudgeConfig(provider="openai"), client=client)
    verdict = await judge.rate_chunk_relevance("q", "chunk", reference="ref")
    assert verdict.relevant is True
    assert captured["model"] == "gpt-4o-mini"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert "REFERENCE ANSWER" in captured["messages"][1]["content"]
