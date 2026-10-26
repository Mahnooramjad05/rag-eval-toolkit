"""Adapters call the RAG pipeline under test to fill in answers, contexts and retrieved ids.

Both adapters expect the pipeline to return a dict like:
    {"answer": str, "contexts": [str, ...], "retrieved_ids": [str, ...]}
`contexts` and `retrieved_ids` must be in rank order and the same length.
"""

import asyncio
from typing import Protocol

from rageval.schemas import EvalSample


class AdapterError(RuntimeError):
    pass


class Adapter(Protocol):
    async def query(self, question: str) -> dict: ...


def apply_response(sample: EvalSample, response: dict) -> EvalSample:
    if not isinstance(response, dict) or "answer" not in response:
        raise AdapterError(f"pipeline response for sample '{sample.id}' has no 'answer' field")
    contexts = list(response.get("contexts") or [])
    retrieved_ids = list(response.get("retrieved_ids") or [])
    if retrieved_ids and len(retrieved_ids) != len(contexts):
        raise AdapterError(f"sample '{sample.id}': contexts and retrieved_ids differ in length")
    return sample.model_copy(update={
        "answer": str(response["answer"]),
        "contexts": contexts,
        "retrieved_ids": retrieved_ids,
    })


async def fill_samples(samples: list[EvalSample], adapter: Adapter,
                       max_concurrency: int = 8) -> list[EvalSample]:
    """Query the pipeline for every sample that has no answer yet."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def fill(sample: EvalSample) -> EvalSample:
        if sample.answer is not None:
            return sample
        async with semaphore:
            return apply_response(sample, await adapter.query(sample.question))

    return list(await asyncio.gather(*(fill(s) for s in samples)))


def build_adapter(spec: str) -> Adapter:
    """Build an adapter from a CLI spec: 'http://...' / 'https://...' or 'python:module:function'."""
    if spec.startswith(("http://", "https://")):
        from rageval.adapters.http_adapter import HTTPAdapter
        return HTTPAdapter(spec)
    if spec.startswith("python:"):
        from rageval.adapters.python_adapter import PythonAdapter
        return PythonAdapter(spec.removeprefix("python:"))
    raise AdapterError(f"unknown adapter '{spec}' (use an http(s) URL or python:module:function)")
