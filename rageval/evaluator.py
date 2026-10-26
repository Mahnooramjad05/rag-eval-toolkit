"""Runs every configured metric over every sample and aggregates the results."""

import asyncio
import logging
import secrets
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rageval.config import EvalConfig
from rageval.judges import Judge, JudgeError
from rageval.metrics import Metric, build_metrics
from rageval.schemas import EvalSample, MetricResult, RunResult, SampleResult

logger = logging.getLogger(__name__)


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


class Evaluator:
    def __init__(self, config: EvalConfig, judge: Judge, metrics: list[Metric] | None = None):
        self.config = config
        self.judge = judge
        self.metrics = metrics if metrics is not None else build_metrics(config)

    async def evaluate(self, samples: list[EvalSample], dataset_name: str = "inline") -> RunResult:
        semaphore = asyncio.Semaphore(self.config.max_concurrent_samples)

        async def run_one(sample: EvalSample) -> SampleResult:
            async with semaphore:
                return await self._evaluate_sample(sample)

        sample_results = await asyncio.gather(*(run_one(s) for s in samples))
        summary, coverage = aggregate(sample_results)
        return RunResult(
            run_id=new_run_id(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            dataset=dataset_name,
            judge=self.judge.name,
            config=self.config.model_dump(mode="json"),
            summary=summary,
            coverage=coverage,
            usage=self.judge.usage.snapshot(self.config.judge),
            samples=list(sample_results),
        )

    async def _evaluate_sample(self, sample: EvalSample) -> SampleResult:
        per_metric = await asyncio.gather(*(self._safe_compute(m, sample) for m in self.metrics))
        metrics: dict[str, MetricResult] = {}
        for results in per_metric:
            for result in results:
                metrics[result.name] = result
        return SampleResult(sample_id=sample.id, question=sample.question,
                            answer=sample.answer, metrics=metrics)

    async def _safe_compute(self, metric: Metric, sample: EvalSample) -> list[MetricResult]:
        """One failing judge call marks that metric as errored instead of aborting the run."""
        try:
            return await metric.compute(sample, self.judge)
        except JudgeError as exc:
            logger.warning("metric %s failed on sample %s: %s", metric.name, sample.id, exc)
            return [MetricResult(name=metric.name, score=None, details={"error": str(exc)})]


def aggregate(sample_results: list[SampleResult]) -> tuple[dict[str, float], dict[str, int]]:
    """Mean score per metric over the samples where the metric applied."""
    scores: dict[str, list[float]] = defaultdict(list)
    for sample in sample_results:
        for name, result in sample.metrics.items():
            if result.score is not None:
                scores[name].append(result.score)
    summary = {name: round(sum(vals) / len(vals), 4) for name, vals in sorted(scores.items())}
    coverage = {name: len(vals) for name, vals in sorted(scores.items())}
    return summary, coverage


def save_run(run: RunResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(path: str | Path) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text(encoding="utf-8"))
