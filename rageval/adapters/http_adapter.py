"""Query a RAG pipeline exposed over HTTP: POST {"question": ...} -> JSON response."""

import httpx

from rageval.adapters import AdapterError


class HTTPAdapter:
    def __init__(self, url: str, timeout: float = 60.0, headers: dict | None = None):
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}

    async def query(self, question: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, json={"question": question},
                                             headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise AdapterError(f"RAG endpoint {self.url} failed: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"RAG endpoint {self.url} did not return JSON") from exc
