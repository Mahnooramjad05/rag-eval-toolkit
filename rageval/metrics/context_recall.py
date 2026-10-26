"""Context recall: how much of the ground-truth answer the retrieved context could support."""

from rageval.judges import Judge
from rageval.metrics.base import Metric
from rageval.schemas import EvalSample, MetricResult


class ContextRecall(Metric):
    name = "context_recall"

    async def compute(self, sample: EvalSample, judge: Judge) -> list[MetricResult]:
        if not sample.ground_truth:
            return self.skip("no ground_truth")
        if not sample.contexts:
            return self.skip("no contexts")

        claims = await judge.extract_claims(sample.ground_truth)
        if not claims:
            return self.skip("ground_truth contains no factual claims")
        verdicts = await judge.verify_claims(claims, sample.contexts)
        covered = sum(v.supported for v in verdicts)
        return [MetricResult(
            name=self.name,
            score=covered / len(verdicts),
            details={"ground_truth_claims": [v.model_dump() for v in verdicts]},
        )]
