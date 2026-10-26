"""Rank-based retrieval metrics. Deterministic, no LLM involved.

All metrics use binary relevance: a retrieved id is relevant if it appears in
the sample's `relevant_ids`.
"""

import math

from rageval.judges import Judge
from rageval.metrics.base import Metric
from rageval.schemas import EvalSample, MetricResult


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision over the top k. When fewer than k items were retrieved, the
    denominator is the number actually retrieved, so short lists aren't penalized twice."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(doc in relevant for doc in top) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, doc in enumerate(retrieved[:k], start=1) if doc in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


class RetrievalMetrics(Metric):
    name = "retrieval"

    def __init__(self, k_values: list[int]):
        self.k_values = k_values

    def names(self) -> list[str]:
        out = ["mrr"]
        for k in self.k_values:
            out += [f"hit_rate@{k}", f"recall@{k}", f"precision@{k}", f"ndcg@{k}"]
        return out

    async def compute(self, sample: EvalSample, judge: Judge) -> list[MetricResult]:
        if not sample.relevant_ids:
            return [MetricResult(name=n, score=None, details={"skipped": "no relevant_ids labels"})
                    for n in self.names()]
        if not sample.retrieved_ids:
            return [MetricResult(name=n, score=None, details={"skipped": "no retrieved_ids"})
                    for n in self.names()]

        retrieved, relevant = sample.retrieved_ids, set(sample.relevant_ids)
        details = {"retrieved": retrieved, "relevant": sorted(relevant)}
        results = [MetricResult(name="mrr", score=reciprocal_rank(retrieved, relevant), details=details)]
        for k in self.k_values:
            results += [
                MetricResult(name=f"hit_rate@{k}", score=hit_rate_at_k(retrieved, relevant, k)),
                MetricResult(name=f"recall@{k}", score=recall_at_k(retrieved, relevant, k)),
                MetricResult(name=f"precision@{k}", score=precision_at_k(retrieved, relevant, k)),
                MetricResult(name=f"ndcg@{k}", score=ndcg_at_k(retrieved, relevant, k)),
            ]
        return results
