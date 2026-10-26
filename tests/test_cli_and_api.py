import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from conftest import DATA, ROOT

from api.main import app as api_app
from rageval.cli import app as cli_app

runner = CliRunner()


def run_cli(*args):
    return runner.invoke(cli_app, [str(a) for a in args])


def test_cli_run_with_adapter_then_report_compare_gate(tmp_path):
    out = tmp_path / "runs"
    for adapter in ("answer_top3", "answer_top5"):
        result = run_cli("run", "--dataset", DATA / "sample_qa.jsonl",
                         "--config", ROOT / "configs" / "default.yaml",
                         "--adapter", f"python:examples.baseline_rag:{adapter}",
                         "--output-dir", out / adapter)
        assert result.exit_code == 0, result.output
        assert "faithfulness" in result.output

    top3 = next((out / "answer_top3").glob("*.json"))
    top5 = next((out / "answer_top5").glob("*.json"))
    assert top3.with_suffix(".html").exists()

    data = json.loads(top3.read_text(encoding="utf-8"))
    assert len(data["samples"]) == 12
    assert data["samples"][0]["answer"]  # filled by the adapter

    result = run_cli("report", top3, "--output", tmp_path / "r.html")
    assert result.exit_code == 0 and (tmp_path / "r.html").exists()

    md = tmp_path / "cmp.md"
    result = run_cli("compare", top3, top5, "--markdown", md)
    assert result.exit_code == 0 and "| Metric |" in md.read_text(encoding="utf-8")

    assert run_cli("gate", top3, "--min", "mrr=0.5").exit_code == 0
    failed = run_cli("gate", top3, "--min", "faithfulness=1.0", "--min", "precision@5=1.0")
    assert failed.exit_code == 1


def test_cli_errors_are_clean(tmp_path):
    result = run_cli("run", "--dataset", tmp_path / "missing.jsonl")
    assert result.exit_code == 2 and "dataset not found" in result.output

    result = run_cli("run", "--dataset", DATA / "sample_qa.jsonl", "--adapter", "ftp://nope")
    assert result.exit_code == 2 and "unknown adapter" in result.output

    result = run_cli("run", "--dataset", DATA / "sample_qa.jsonl", "--provider", "bard")
    assert result.exit_code == 2 and "unknown provider" in result.output


def test_cli_falls_back_to_mock_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    result = run_cli("run", "--dataset", DATA / "sample_with_answers.jsonl",
                     "--output-dir", tmp_path / "r", "--no-report")
    assert result.exit_code == 0, result.output
    assert "judge mock:heuristic-v1" in result.output


def test_api_evaluate_and_fetch():
    client = TestClient(api_app)
    assert client.get("/health").json()["judge_provider"] == "mock"

    samples = [json.loads(line) for line in
               (DATA / "sample_with_answers.jsonl").read_text(encoding="utf-8").splitlines()]
    response = client.post("/evaluate", json={"samples": samples,
                                              "metrics": ["faithfulness", "retrieval"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "faithfulness" in body["summary"] and "answer_relevance" not in body["summary"]

    run = client.get(f"/runs/{body['run_id']}")
    assert run.status_code == 200 and len(run.json()["samples"]) == 4
    report = client.get(body["report_url"])
    assert report.status_code == 200 and "RAG evaluation report" in report.text


def test_api_validation():
    client = TestClient(api_app)
    assert client.get("/runs/..%2F..%2Fsecrets").status_code in (400, 404)
    assert client.get("/runs/not-a-run").status_code == 400
    assert client.get("/runs/20260101-000000-abcdef").status_code == 404
    assert client.post("/evaluate", json={"samples": []}).status_code == 422
    bad = client.post("/evaluate", json={"samples": [{"id": "a", "question": "q"}],
                                         "metrics": ["vibes"]})
    assert bad.status_code == 422
