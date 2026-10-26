import pytest

from rageval.metrics.answer_relevance import AnswerRelevance
from rageval.metrics.context_precision import ContextPrecision, average_precision
from rageval.metrics.context_recall import ContextRecall
from rageval.metrics.faithfulness import Faithfulness
from rageval.schemas import EvalSample

TEAM = ("The Team plan costs 99 dollars per month and includes 15 user seats, "
        "unlimited dashboards and scheduled email reports.")
STARTER = "The Starter plan costs 29 dollars per month and includes 3 user seats and 10 dashboards."
OFF_TOPIC = "All customer data is encrypted at rest with AES-256 and in transit with TLS 1.2 or higher."


def sample(**kwargs) -> EvalSample:
    base = dict(id="s1", question="How much does the Team plan cost per month?",
                ground_truth="The Team plan costs 99 dollars per month.", contexts=[TEAM])
    return EvalSample(**{**base, **kwargs})


def test_average_precision():
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, True]) == pytest.approx(0.5)
    assert average_precision([False, False]) == 0.0


async def test_faithfulness_supported_answer(judge):
    [result] = await Faithfulness().compute(
        sample(answer="The Team plan costs 99 dollars per month."), judge)
    assert result.score == 1.0
    assert result.details["claims"][0]["supported"] is True


async def test_faithfulness_catches_wrong_number(judge):
    [result] = await Faithfulness().compute(
        sample(answer="The Team plan costs 79 dollars per month."), judge)
    assert result.score == 0.0
    assert "79" in result.details["claims"][0]["reason"]


async def test_faithfulness_partial(judge):
    answer = ("The Team plan costs 99 dollars per month. "
              "It also includes a dedicated account manager and priority phone support.")
    [result] = await Faithfulness().compute(sample(answer=answer), judge)
    assert result.score == 0.5


@pytest.mark.parametrize("kwargs, reason", [
    ({"answer": None}, "no answer"),
    ({"answer": "The Team plan costs 99 dollars.", "contexts": []}, "no contexts"),
])
async def test_faithfulness_skips(judge, kwargs, reason):
    [result] = await Faithfulness().compute(sample(**kwargs), judge)
    assert result.score is None
    assert result.details["skipped"] == reason


async def test_answer_relevance_ranks_on_topic_higher(judge):
    [on_topic] = await AnswerRelevance().compute(
        sample(answer="The Team plan costs 99 dollars per month."), judge)
    [off_topic] = await AnswerRelevance().compute(
        sample(answer="Data is encrypted with AES-256."), judge)
    assert on_topic.score > off_topic.score
    assert 0.0 <= off_topic.score <= 1.0


async def test_context_precision_rewards_relevant_chunk_first(judge):
    [good] = await ContextPrecision().compute(sample(contexts=[TEAM, OFF_TOPIC]), judge)
    [bad] = await ContextPrecision().compute(sample(contexts=[OFF_TOPIC, TEAM]), judge)
    assert good.score == 1.0
    assert bad.score == pytest.approx(0.5)
    assert [c["rank"] for c in bad.details["chunks"]] == [1, 2]


async def test_context_recall(judge):
    [covered] = await ContextRecall().compute(sample(), judge)
    [missing] = await ContextRecall().compute(sample(contexts=[STARTER]), judge)
    assert covered.score == 1.0
    assert missing.score == 0.0


async def test_context_recall_skips_without_ground_truth(judge):
    [result] = await ContextRecall().compute(sample(ground_truth=None), judge)
    assert result.score is None
