"""A deliberately small RAG pipeline to evaluate in the demo.

Retrieval is BM25 over paragraph chunks of data/sample_docs. "Generation" is
extractive (the best-matching sentences from the retrieved chunks), so the demo
runs offline and deterministically. Swap `answer_top3` for your own pipeline,
or point `rageval run --adapter` at your RAG's HTTP endpoint.
"""

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rageval.text import content_tokens, split_sentences, tokenize

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    """Split each markdown file into paragraphs, skipping headings. Ids look like 'pricing.md#2'."""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]
        body = [p for p in paragraphs if p and not p.startswith("#")]
        chunks += [(f"{path.name}#{i}", text) for i, text in enumerate(body)]
    return chunks


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.avg_len = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(tok for doc in self.docs for tok in set(doc))
        n = len(self.docs)
        self.idf = {tok: math.log(1 + (n - f + 0.5) / (f + 0.5)) for tok, f in df.items()}
        self.tf = [Counter(doc) for doc in self.docs]

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.docs):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
            out.append(sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                           for t in terms if t in tf))
        return out


@lru_cache(maxsize=1)
def _index() -> tuple[list[tuple[str, str]], BM25]:
    chunks = load_chunks()
    return chunks, BM25([text for _, text in chunks])


def retrieve(question: str, k: int) -> list[tuple[str, str]]:
    chunks, bm25 = _index()
    ranked = sorted(zip(bm25.scores(question), range(len(chunks))), key=lambda x: (-x[0], x[1]))
    return [chunks[i] for _, i in ranked[:k]]


def answer(question: str, k: int = 3, max_sentences: int = 2) -> dict:
    retrieved = retrieve(question, k)
    q_terms = content_tokens(question)
    candidates = []
    for rank, (_, text) in enumerate(retrieved):
        for sentence in split_sentences(text):
            score = len(q_terms & content_tokens(sentence))
            candidates.append((score, -rank, sentence))
    candidates.sort(reverse=True)
    best = [s for score, _, s in candidates[:max_sentences] if score > 0]
    return {
        "answer": " ".join(best) if best else "I could not find this in the documentation.",
        "contexts": [text for _, text in retrieved],
        "retrieved_ids": [chunk_id for chunk_id, _ in retrieved],
    }


def answer_top3(question: str) -> dict:
    return answer(question, k=3)


def answer_top5(question: str) -> dict:
    return answer(question, k=5)


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(answer(" ".join(sys.argv[1:]) or "How much does the Team plan cost?"), indent=2))
