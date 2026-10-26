"""Context precision: are the relevant chunks ranked near the top of the retrieved context.

Computed as average precision over the judge's per-chunk relevance verdicts,
so a relevant chunk at rank 1 counts more than the same chunk at rank 5.
"""

import asyncio

from rageval.judges import Judge
from rageval.metrics.base import Metric
from rageval.schemas import EvalSample, MetricResult


def average_precision(relevance: list[bool]) -> float:
    hits, total = 0, 0.0
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


class ContextPrecision(Metric):
    name = "context_precision"

    async def compute(self, sample: EvalSample, judge: Judge) -> list[MetricResult]:
        if not sample.contexts:
            return self.skip("no contexts")
        verdicts = await asyncio.gather(*(
            judge.rate_chunk_relevance(sample.question, chunk, sample.ground_truth)
            for chunk in sample.contexts
        ))
        relevance = [v.relevant for v in verdicts]
        return [MetricResult(
            name=self.name,
            score=average_precision(relevance),
            details={"chunks": [{"rank": i + 1, **v.model_dump()} for i, v in enumerate(verdicts)]},
        )]
