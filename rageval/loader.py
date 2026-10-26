"""Load evaluation datasets from JSONL or CSV files."""

import csv
import json
from pathlib import Path

from pydantic import ValidationError

from rageval.schemas import EvalSample

LIST_FIELDS = ("relevant_ids", "retrieved_ids", "contexts")


class DatasetError(ValueError):
    """Raised when a dataset file cannot be parsed or fails validation."""


def load_dataset(path: str | Path) -> list[EvalSample]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = _read_jsonl(path)
    elif suffix == ".csv":
        records = _read_csv(path)
    else:
        raise DatasetError(f"unsupported dataset format '{suffix}' (use .jsonl or .csv)")

    samples: list[EvalSample] = []
    seen: set[str] = set()
    for line_no, record in records:
        try:
            sample = EvalSample.model_validate(record)
        except ValidationError as exc:
            raise DatasetError(f"{path.name}, record {line_no}: {exc.errors()[0]['msg']} "
                               f"({'.'.join(map(str, exc.errors()[0]['loc']))})") from exc
        if sample.id in seen:
            raise DatasetError(f"{path.name}, record {line_no}: duplicate id '{sample.id}'")
        seen.add(sample.id)
        samples.append(sample)

    if not samples:
        raise DatasetError(f"{path.name} contains no samples")
    return samples


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}, line {line_no}: invalid JSON ({exc.msg})") from exc
    return records


def _read_csv(path: Path) -> list[tuple[int, dict]]:
    """CSV list columns hold JSON arrays, e.g. relevant_ids = ["doc1#0", "doc2#3"]."""
    records = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row_no, row in enumerate(csv.DictReader(fh), start=2):
            record: dict = {k: v for k, v in row.items() if v not in (None, "")}
            for field in LIST_FIELDS:
                if field in record:
                    try:
                        record[field] = json.loads(record[field])
                    except json.JSONDecodeError as exc:
                        raise DatasetError(
                            f"{path.name}, row {row_no}: column '{field}' must be a JSON array"
                        ) from exc
            records.append((row_no, record))
    return records
