import math

import pytest

from rageval.metrics.retrieval import (
    RetrievalMetrics,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rageval.schemas import EvalSample

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "x"}  # "x" was never retrieved


def test_hit_rate():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 3) == 1.0


def test_recall():
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_precision():
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_precision_uses_retrieved_count_when_shorter_than_k():
    assert precision_at_k(["b"], RELEVANT, 5) == 1.0
    assert precision_at_k([], RELEVANT, 5) == 0.0


def test_mrr():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
    assert reciprocal_rank(["a", "c"], RELEVANT) == 0.0


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)          # hits at ranks 2 and 5
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant docs
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["b", "e", "x"], RELEVANT, 3) == pytest.approx(1.0)


async def test_metric_outputs_all_names(judge):
    metric = RetrievalMetrics([1, 3])
    sample = EvalSample(id="s", question="q", relevant_ids=["b"], retrieved_ids=["a", "b"])
    results = {r.name: r.score for r in await metric.compute(sample, judge)}
    assert set(results) == set(metric.names())
    assert results["mrr"] == 0.5
    assert results["hit_rate@1"] == 0.0
    assert results["recall@3"] == 1.0


async def test_metric_skips_without_labels(judge):
    metric = RetrievalMetrics([3])
    sample = EvalSample(id="s", question="q", retrieved_ids=["a"])
    results = await metric.compute(sample, judge)
    assert all(r.score is None for r in results)
    assert results[0].details["skipped"] == "no relevant_ids labels"
