"""Evaluation settings, loaded from YAML and overridable from environment variables."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Provider = Literal["mock", "openai", "anthropic"]

DEFAULT_MODELS: dict[str, str] = {
    "mock": "heuristic-v1",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5-5",
}

ALL_METRICS = [
    "retrieval",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
]


class JudgeConfig(BaseModel):
    provider: Provider = "mock"
    model: str | None = None
    # Anthropic only: judge tasks are classification-like, so low effort is a sensible default.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_concurrency: int = Field(default=4, ge=1)
    requests_per_minute: int | None = Field(default=None, ge=1)
    max_retries: int = Field(default=3, ge=0)
    cache_path: str | None = ".rageval_cache.sqlite"
    # Prices are left to the user so the toolkit never reports stale numbers.
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]


class MockJudgeConfig(BaseModel):
    support_threshold: float = Field(default=0.6, ge=0, le=1)
    chunk_relevance_threshold: float = Field(default=0.35, ge=0, le=1)


class EvalConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: list(ALL_METRICS))
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    mock: MockJudgeConfig = Field(default_factory=MockJudgeConfig)
    max_concurrent_samples: int = Field(default=8, ge=1)
    output_dir: str = "runs"

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(ALL_METRICS))
        if unknown:
            raise ValueError(f"unknown metrics {unknown}; choose from {ALL_METRICS}")
        return value

    @field_validator("k_values")
    @classmethod
    def _positive_k(cls, value: list[int]) -> list[int]:
        if not value or any(k < 1 for k in value):
            raise ValueError("k_values must be a non-empty list of positive integers")
        return sorted(set(value))


def load_config(path: str | Path | None = None) -> EvalConfig:
    """Load a YAML config (or defaults), then apply environment overrides.

    LLM_PROVIDER and JUDGE_MODEL override the file so CI can switch judges
    without editing configs.
    """
    data: dict = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    config = EvalConfig.model_validate(data)

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        config.judge.provider = provider  # type: ignore[assignment]
        config.judge = JudgeConfig.model_validate(config.judge.model_dump())
    model = os.getenv("JUDGE_MODEL", "").strip()
    if model:
        config.judge.model = model
    return config
