"""Metric interface."""

from abc import ABC, abstractmethod

from rageval.judges import Judge
from rageval.schemas import EvalSample, MetricResult


class Metric(ABC):
    name: str

    @abstractmethod
    async def compute(self, sample: EvalSample, judge: Judge) -> list[MetricResult]:
        """Return one or more results. Use score=None when the metric doesn't apply."""

    def skip(self, reason: str, name: str | None = None) -> list[MetricResult]:
        return [MetricResult(name=name or self.name, score=None, details={"skipped": reason})]
