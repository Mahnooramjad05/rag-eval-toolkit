"""Faithfulness: what share of the answer's claims are supported by the retrieved context."""

from rageval.judges import Judge
from rageval.metrics.base import Metric
from rageval.schemas import EvalSample, MetricResult


class Faithfulness(Metric):
    name = "faithfulness"

    async def compute(self, sample: EvalSample, judge: Judge) -> list[MetricResult]:
        if not sample.answer:
            return self.skip("no answer")
        if not sample.contexts:
            return self.skip("no contexts")

        claims = await judge.extract_claims(sample.answer)
        if not claims:
            return self.skip("answer contains no factual claims")
        verdicts = await judge.verify_claims(claims, sample.contexts)
        supported = sum(v.supported for v in verdicts)
        return [MetricResult(
            name=self.name,
            score=supported / len(verdicts),
            details={"claims": [v.model_dump() for v in verdicts]},
        )]
