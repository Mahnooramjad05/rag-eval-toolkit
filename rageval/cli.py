"""Command-line interface: `rageval run | report | compare | gate`."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from rageval.adapters import AdapterError, build_adapter, fill_samples
from rageval.config import load_config
from rageval.evaluator import Evaluator, load_run, save_run
from rageval.gate import GateConfigError, check_gate, load_thresholds, parse_thresholds
from rageval.judges import build_judge
from rageval.loader import DatasetError, load_dataset
from rageval.reporting import compare_runs, to_markdown, write_report

app = typer.Typer(help="Evaluate RAG pipelines: retrieval quality, faithfulness and answer quality.",
                  no_args_is_help=True, add_completion=False)


def _fail(message: str) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")


@app.command()
def run(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="JSONL or CSV dataset.")],
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML config file.")] = None,
    adapter: Annotated[Optional[str], typer.Option(
        help="Fill missing answers by calling your RAG: an http(s) URL or python:module:function.")] = None,
    provider: Annotated[Optional[str], typer.Option(help="Judge provider: mock, openai or anthropic.")] = None,
    output_dir: Annotated[Optional[Path], typer.Option(help="Where to write run files.")] = None,
    report: Annotated[bool, typer.Option(help="Also write an HTML report.")] = True,
):
    """Evaluate a dataset and save the run (JSON, plus an HTML report)."""
    try:
        cfg = load_config(config)
        if provider:
            if provider not in ("mock", "openai", "anthropic"):
                _fail(f"unknown provider '{provider}' (use mock, openai or anthropic)")
            cfg.judge.provider = provider  # type: ignore[assignment]
            cfg.judge.model = None
        samples = load_dataset(dataset)
    except (DatasetError, ValueError, OSError) as exc:
        _fail(str(exc))
        return

    async def _go():
        nonlocal samples
        if adapter:
            samples = await fill_samples(samples, build_adapter(adapter), cfg.max_concurrent_samples)
        judge = build_judge(cfg)
        try:
            return await Evaluator(cfg, judge).evaluate(samples, dataset_name=dataset.name)
        finally:
            await judge.aclose()

    try:
        result = asyncio.run(_go())
    except AdapterError as exc:
        _fail(str(exc))
        return

    out = output_dir or Path(cfg.output_dir)
    run_path = save_run(result, out)
    typer.echo(f"Run {result.run_id}  ({len(result.samples)} samples, judge {result.judge})")
    for name, score in result.summary.items():
        typer.echo(f"  {name:<20} {score:.3f}   ({result.coverage[name]} scored)")
    typer.echo(f"Saved {run_path}")
    if report:
        report_path = write_report(result, run_path.with_suffix(".html"))
        typer.echo(f"Report {report_path}")


@app.command()
def report(run_file: Annotated[Path, typer.Argument(help="Run JSON written by `rageval run`.")],
           output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None):
    """Render an HTML report for a saved run."""
    try:
        result = load_run(run_file)
    except (OSError, ValueError) as exc:
        _fail(f"cannot read run file: {exc}")
        return
    path = write_report(result, output or run_file.with_suffix(".html"))
    typer.echo(f"Report {path}")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline run JSON.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run JSON.")],
    markdown: Annotated[Optional[Path], typer.Option(help="Also write the comparison as Markdown.")] = None,
    min_change: Annotated[float, typer.Option(help="Per-question change that counts as a regression.")] = 0.1,
):
    """Diff two runs metric by metric and list questions that got worse."""
    try:
        comparison = compare_runs(load_run(baseline), load_run(candidate), min_change)
    except (OSError, ValueError) as exc:
        _fail(f"cannot read run file: {exc}")
        return
    text = to_markdown(comparison)
    typer.echo(text)
    if markdown:
        markdown.write_text(text, encoding="utf-8")


@app.command()
def gate(
    run_file: Annotated[Path, typer.Argument(help="Run JSON to check.")],
    min_: Annotated[Optional[list[str]], typer.Option(
        "--min", help="Threshold like faithfulness=0.8 (repeatable).")] = None,
    thresholds_file: Annotated[Optional[Path], typer.Option(
        "--thresholds", help="YAML file with a 'thresholds' mapping.")] = None,
):
    """Exit with code 1 if any metric is below its threshold."""
    try:
        thresholds = load_thresholds(thresholds_file) if thresholds_file else {}
        thresholds.update(parse_thresholds(min_ or []))
        result = load_run(run_file)
    except (GateConfigError, OSError, ValueError) as exc:
        _fail(str(exc))
        return
    if not thresholds:
        _fail("no thresholds given; use --min metric=value or --thresholds file.yaml")

    checks = check_gate(result, thresholds)
    for check in checks:
        typer.secho(check.message, fg=typer.colors.GREEN if check.passed else typer.colors.RED)
    if all(c.passed for c in checks):
        typer.secho("Quality gate passed.", fg=typer.colors.GREEN)
    else:
        typer.secho("Quality gate failed.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
