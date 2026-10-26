from pathlib import Path

import pytest

from rageval.config import EvalConfig, JudgeConfig
from rageval.judges.mock_judge import MockJudge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test uses the mock judge, makes no network calls and writes only to tmp_path."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RAGEVAL_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def config() -> EvalConfig:
    return EvalConfig(judge=JudgeConfig(provider="mock", cache_path=None))


@pytest.fixture
def judge(config) -> MockJudge:
    return MockJudge(config.judge, config.mock)
