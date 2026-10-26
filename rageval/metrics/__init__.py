"""Metric registry."""

from rageval.config import EvalConfig
from rageval.metrics.answer_relevance import AnswerRelevance
from rageval.metrics.base import Metric
from rageval.metrics.context_precision import ContextPrecision
from rageval.metrics.context_recall import ContextRecall
from rageval.metrics.faithfulness import Faithfulness
from rageval.metrics.retrieval import RetrievalMetrics

__all__ = ["Metric", "build_metrics"]


def build_metrics(config: EvalConfig) -> list[Metric]:
    factories = {
        "retrieval": lambda: RetrievalMetrics(config.k_values),
        "faithfulness": Faithfulness,
        "answer_relevance": AnswerRelevance,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }
    return [factories[name]() for name in config.metrics]
