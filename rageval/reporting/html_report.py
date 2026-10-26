"""Render a run as a single self-contained HTML file (no external scripts or fonts)."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rageval.schemas import RunResult

TEMPLATES = Path(__file__).parent / "templates"

HEADLINE_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "mrr"]

_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))


def score_band(score: float | None) -> str:
    if score is None:
        return "na"
    if score >= 0.8:
        return "good"
    if score >= 0.5:
        return "warn"
    return "bad"


def _fmt(score: float | None) -> str:
    return "–" if score is None else f"{score:.2f}"


_env.filters["band"] = score_band
_env.filters["fmt"] = _fmt


def render_report(run: RunResult) -> str:
    headline = [(name, run.summary.get(name)) for name in HEADLINE_METRICS if name in run.summary]
    metric_names = list(run.summary)
    # Retrieval @k metrics make the per-sample table too wide; show the headline ones there.
    table_metrics = [n for n in HEADLINE_METRICS if n in run.summary]
    return _env.get_template("report.html.j2").render(
        run=run,
        headline=headline,
        metric_names=metric_names,
        table_metrics=table_metrics,
        is_mock=run.judge.startswith("mock:"),
    )


def write_report(run: RunResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(run), encoding="utf-8")
    return path
