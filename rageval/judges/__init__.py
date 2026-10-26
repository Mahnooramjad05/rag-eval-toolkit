"""Judge construction: picks a provider and falls back to the mock judge when no key is set."""

import logging
import os

from rageval.config import EvalConfig
from rageval.judges.base import ChunkVerdict, ClaimVerdict, Judge, JudgeError, RelevanceVerdict
from rageval.judges.cache import JudgeCache
from rageval.judges.mock_judge import MockJudge

__all__ = ["ChunkVerdict", "ClaimVerdict", "Judge", "JudgeError", "RelevanceVerdict", "build_judge"]

logger = logging.getLogger(__name__)

_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def build_judge(config: EvalConfig) -> Judge:
    provider = config.judge.provider
    if provider != "mock" and not os.getenv(_KEY_ENV[provider]):
        logger.warning("%s is not set; falling back to the offline mock judge "
                       "(heuristic scores, for demos only).", _KEY_ENV[provider])
        config.judge.provider = "mock"
        config.judge.model = None
        provider = "mock"

    if provider == "mock":
        return MockJudge(config.judge, config.mock)

    cache = JudgeCache(config.judge.cache_path) if config.judge.cache_path else None
    if provider == "openai":
        from rageval.judges.openai_judge import OpenAIJudge
        return OpenAIJudge(config.judge, cache)
    from rageval.judges.anthropic_judge import AnthropicJudge
    return AnthropicJudge(config.judge, cache)
