"""HTTP service for running evaluations and fetching saved runs and reports."""

import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rageval import __version__
from rageval.config import load_config
from rageval.evaluator import Evaluator, load_run, save_run
from rageval.judges import build_judge
from rageval.reporting import render_report
from rageval.schemas import EvalSample, RunResult

MAX_SAMPLES = 500
_RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")

app = FastAPI(title="rageval", version=__version__,
              description="Score RAG pipelines on retrieval quality, faithfulness and answer quality.")


class EvaluateRequest(BaseModel):
    samples: list[EvalSample] = Field(min_length=1, max_length=MAX_SAMPLES)
    dataset_name: str = "api"
    metrics: list[str] | None = None


class EvaluateResponse(BaseModel):
    run_id: str
    judge: str
    summary: dict[str, float]
    coverage: dict[str, int]
    report_url: str


def _config():
    """Config file comes from RAGEVAL_CONFIG; provider and model can still be set by env vars."""
    return load_config(os.getenv("RAGEVAL_CONFIG") or None)


def _output_dir() -> Path:
    return Path(_config().output_dir)


def _run_path(run_id: str) -> Path:
    # Strict id format so a request can never point outside the runs directory.
    if not _RUN_ID.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    path = _output_dir() / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return path


@app.get("/health")
def health() -> dict:
    config = _config()
    return {"status": "ok", "version": __version__, "judge_provider": config.judge.provider}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    config = _config()
    if request.metrics is not None:
        try:
            config = config.model_validate({**config.model_dump(), "metrics": request.metrics})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    judge = build_judge(config)
    try:
        run = await Evaluator(config, judge).evaluate(request.samples, request.dataset_name)
    finally:
        await judge.aclose()
    save_run(run, config.output_dir)
    return EvaluateResponse(run_id=run.run_id, judge=run.judge, summary=run.summary,
                            coverage=run.coverage, report_url=f"/runs/{run.run_id}/report")


@app.get("/runs/{run_id}", response_model=RunResult)
def get_run(run_id: str) -> RunResult:
    return load_run(_run_path(run_id))


@app.get("/runs/{run_id}/report", response_class=HTMLResponse)
def get_report(run_id: str) -> str:
    return render_report(load_run(_run_path(run_id)))
