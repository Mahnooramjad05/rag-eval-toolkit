"""Quality gate: fail CI when a run's scores drop below agreed thresholds."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from rageval.schemas import RunResult


class GateConfigError(ValueError):
    pass


@dataclass
class GateCheck:
    metric: str
    threshold: float
    actual: float | None

    @property
    def passed(self) -> bool:
        return self.actual is not None and self.actual >= self.threshold

    @property
    def message(self) -> str:
        if self.actual is None:
            return f"{self.metric}: not computed in this run (needs >= {self.threshold:.2f})"
        status = "ok" if self.passed else "FAIL"
        return f"{self.metric}: {self.actual:.3f} (needs >= {self.threshold:.2f}) {status}"


def parse_thresholds(items: list[str]) -> dict[str, float]:
    """Parse CLI items like 'faithfulness=0.8'."""
    thresholds = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise GateConfigError(f"threshold '{item}' must look like metric=0.8")
        try:
            thresholds[name.strip()] = _check_range(name, float(value))
        except ValueError as exc:
            raise GateConfigError(f"threshold '{item}': {exc}") from exc
    return thresholds


def load_thresholds(path: str | Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get("thresholds")
    if not isinstance(raw, dict) or not raw:
        raise GateConfigError(f"{path} must contain a non-empty 'thresholds' mapping")
    try:
        return {str(k): _check_range(str(k), float(v)) for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise GateConfigError(f"{path}: {exc}") from exc


def _check_range(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} threshold must be between 0 and 1")
    return value


def check_gate(run: RunResult, thresholds: dict[str, float]) -> list[GateCheck]:
    return [GateCheck(metric=name, threshold=t, actual=run.summary.get(name))
            for name, t in thresholds.items()]
