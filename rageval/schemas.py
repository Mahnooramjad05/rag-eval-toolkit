"""Core data models shared by the loader, metrics, evaluator and reports."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvalSample(BaseModel):
    """One question from an evaluation dataset.

    `answer`, `contexts` and `retrieved_ids` can be left empty when an adapter
    is used to call the RAG pipeline under test and fill them in.
    """

    id: str
    question: str
    ground_truth: str | None = None
    relevant_ids: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    answer: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class MetricResult(BaseModel):
    """Score for one metric on one sample.

    `score` is None when the metric does not apply to the sample, for example
    retrieval metrics on a sample without relevance labels.
    """

    name: str
    score: float | None
    details: dict[str, Any] = Field(default_factory=dict)


class SampleResult(BaseModel):
    sample_id: str
    question: str
    answer: str | None
    metrics: dict[str, MetricResult]


class JudgeUsage(BaseModel):
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    est_cost_usd: float | None = None


class RunResult(BaseModel):
    run_id: str
    created_at: str
    dataset: str
    judge: str
    config: dict[str, Any]
    summary: dict[str, float]
    coverage: dict[str, int]
    usage: JudgeUsage
    samples: list[SampleResult]
