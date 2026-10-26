"""Compare two runs metric by metric and list the questions that got worse."""

from dataclasses import dataclass, field

from rageval.schemas import RunResult


@dataclass
class MetricDelta:
    name: str
    baseline: float | None
    candidate: float | None

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


@dataclass
class SampleChange:
    sample_id: str
    question: str
    metric: str
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


@dataclass
class Comparison:
    baseline_id: str
    candidate_id: str
    metrics: list[MetricDelta]
    regressions: list[SampleChange] = field(default_factory=list)
    improvements: list[SampleChange] = field(default_factory=list)


def compare_runs(baseline: RunResult, candidate: RunResult, min_change: float = 0.1) -> Comparison:
    names = sorted(set(baseline.summary) | set(candidate.summary))
    metrics = [MetricDelta(n, baseline.summary.get(n), candidate.summary.get(n)) for n in names]

    base_samples = {s.sample_id: s for s in baseline.samples}
    regressions, improvements = [], []
    for cand in candidate.samples:
        base = base_samples.get(cand.sample_id)
        if base is None:
            continue
        for name, result in cand.metrics.items():
            before = base.metrics.get(name)
            if result.score is None or before is None or before.score is None:
                continue
            change = SampleChange(cand.sample_id, cand.question, name, before.score, result.score)
            if change.delta <= -min_change:
                regressions.append(change)
            elif change.delta >= min_change:
                improvements.append(change)

    regressions.sort(key=lambda c: c.delta)
    improvements.sort(key=lambda c: -c.delta)
    return Comparison(baseline.run_id, candidate.run_id, metrics, regressions, improvements)


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def to_markdown(comparison: Comparison, max_rows: int = 15) -> str:
    """Markdown summary, suitable for a pull-request comment."""
    lines = [
        f"### RAG eval: `{comparison.candidate_id}` vs baseline `{comparison.baseline_id}`",
        "",
        "| Metric | Baseline | Candidate | Change |",
        "|---|---:|---:|---:|",
    ]
    for m in comparison.metrics:
        marker = ""
        if m.delta is not None and m.delta <= -0.01:
            marker = " (worse)"
        elif m.delta is not None and m.delta >= 0.01:
            marker = " (better)"
        lines.append(f"| {m.name} | {_fmt(m.baseline)} | {_fmt(m.candidate)} | "
                     f"{_fmt(m.delta, signed=True)}{marker} |")

    for title, changes in (("Questions that got worse", comparison.regressions),
                           ("Questions that improved", comparison.improvements)):
        if not changes:
            continue
        lines += ["", f"**{title}** ({len(changes)})", "",
                  "| Sample | Metric | Before | After |", "|---|---|---:|---:|"]
        for c in changes[:max_rows]:
            lines.append(f"| {c.sample_id} | {c.metric} | {c.baseline:.2f} | {c.candidate:.2f} |")
        if len(changes) > max_rows:
            lines.append(f"| ... | {len(changes) - max_rows} more | | |")
    return "\n".join(lines) + "\n"
