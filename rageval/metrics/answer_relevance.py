"""Answer relevance: does the answer address the question (regardless of correctness)."""

from rageval.judges import Judge
from rageval.metrics.base import Metric
from rageval.schemas import EvalSample, MetricResult


class AnswerRelevance(Metric):
    name = "answer_relevance"

    async def compute(self, sample: EvalSample, judge: Judge) -> list[MetricResult]:
        if not sample.answer:
            return self.skip("no answer")
        verdict = await judge.rate_answer_relevance(sample.question, sample.answer)
        return [MetricResult(
            name=self.name,
            score=verdict.normalized,
            details={"rating": verdict.score, "reason": verdict.reason},
        )]
