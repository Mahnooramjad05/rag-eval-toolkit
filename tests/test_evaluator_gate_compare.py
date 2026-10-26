import pytest

from conftest import DATA

from rageval.config import load_config
from rageval.evaluator import Evaluator, aggregate, load_run, save_run
from rageval.gate import GateConfigError, check_gate, load_thresholds, parse_thresholds
from rageval.judges import JudgeError
from rageval.loader import load_dataset
from rageval.metrics.base import Metric
from rageval.reporting import compare_runs, render_report, to_markdown
from rageval.schemas import EvalSample, MetricResult, SampleResult


async def make_run(config, judge, path="sample_with_answers.jsonl"):
    return await Evaluator(config, judge).evaluate(load_dataset(DATA / path), dataset_name=path)


async def test_run_summary_and_roundtrip(config, judge, tmp_path):
    run = await make_run(config, judge)
    assert {"faithfulness", "answer_relevance", "context_precision",
            "context_recall", "mrr", "recall@5"} <= set(run.summary)
    assert run.coverage["faithfulness"] == 4
    assert run.judge == "mock:heuristic-v1"
    path = save_run(run, tmp_path / "runs")
    assert load_run(path) == run


def test_aggregate_ignores_skipped():
    results = [
        SampleResult(sample_id="a", question="q", answer=None,
                     metrics={"m": MetricResult(name="m", score=1.0)}),
        SampleResult(sample_id="b", question="q", answer=None,
                     metrics={"m": MetricResult(name="m", score=None)}),
        SampleResult(sample_id="c", question="q", answer=None,
                     metrics={"m": MetricResult(name="m", score=0.5)}),
    ]
    summary, coverage = aggregate(results)
    assert summary == {"m": 0.75} and coverage == {"m": 2}


class Exploding(Metric):
    name = "exploding"

    async def compute(self, sample, judge):
        raise JudgeError("provider down")


async def test_judge_error_is_recorded_not_fatal(config, judge):
    run = await Evaluator(config, judge, metrics=[Exploding()]).evaluate(
        [EvalSample(id="x", question="q")])
    result = run.samples[0].metrics["exploding"]
    assert result.score is None and result.details["error"] == "provider down"
    assert run.summary == {}


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "my-judge")
    cfg = load_config()
    assert cfg.judge.provider == "openai" and cfg.judge.resolved_model == "my-judge"


def test_config_rejects_unknown_metric(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("metrics: [faithfulness, vibes]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vibes"):
        load_config(path)


# --- gate --------------------------------------------------------------------

def test_parse_thresholds():
    assert parse_thresholds(["faithfulness=0.8", "recall@5 = 0.7"]) == {
        "faithfulness": 0.8, "recall@5": 0.7}
    for bad in ["faithfulness", "=0.5", "faithfulness=high", "faithfulness=1.5"]:
        with pytest.raises(GateConfigError):
            parse_thresholds([bad])


def test_load_thresholds_file(tmp_path):
    path = tmp_path / "gate.yaml"
    path.write_text("thresholds:\n  faithfulness: 0.9\n", encoding="utf-8")
    assert load_thresholds(path) == {"faithfulness": 0.9}
    path.write_text("other: 1\n", encoding="utf-8")
    with pytest.raises(GateConfigError):
        load_thresholds(path)


async def test_check_gate(config, judge):
    run = await make_run(config, judge)
    checks = {c.metric: c for c in check_gate(run, {"mrr": 0.0, "faithfulness": 0.99, "missing": 0.1})}
    assert checks["mrr"].passed
    assert not checks["faithfulness"].passed  # a02 hallucinates, so the mean is below 0.99
    assert not checks["missing"].passed and "not computed" in checks["missing"].message


# --- compare & report ----------------------------------------------------------

async def test_compare_finds_regressions(config, judge):
    samples = load_dataset(DATA / "sample_with_answers.jsonl")
    baseline = await Evaluator(config, judge).evaluate(samples)
    worse = [s.model_copy(update={"answer": "The Team plan costs 79 dollars per month."})
             if s.id == "a01" else s for s in samples]
    candidate = await Evaluator(config, judge).evaluate(worse)

    comparison = compare_runs(baseline, candidate)
    faith = next(m for m in comparison.metrics if m.name == "faithfulness")
    assert faith.delta < 0
    assert any(r.sample_id == "a01" and r.metric == "faithfulness" for r in comparison.regressions)
    md = to_markdown(comparison)
    assert "Questions that got worse" in md and "| faithfulness |" in md


async def test_report_escapes_html_and_flags_mock(config, judge):
    run = await Evaluator(config, judge).evaluate([EvalSample(
        id="x", question="<b>q</b>", answer="<script>alert(1)</script> is the answer here.",
        contexts=["ctx"])])
    html = render_report(run)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "offline mock judge" in html
