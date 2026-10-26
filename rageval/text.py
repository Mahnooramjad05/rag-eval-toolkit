"""Small text helpers used by the mock judge and the example RAG pipeline."""

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may might more most must no not of on or our should so such than that the their
    them then there these they this those to was we were what when where which while who why will
    with would you your""".split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Lowercased tokens without stopwords, with a crude plural fold (days -> day)."""
    tokens = set()
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.add(tok)
    return tokens


def numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    sentences = []
    for part in _SENTENCE_SPLIT.split(text.strip()):
        part = part.strip()
        if len(part.split()) >= min_words:
            sentences.append(part)
    return sentences


def overlap(needle: set[str], haystack: set[str]) -> float:
    """Fraction of `needle` found in `haystack` (1.0 when needle is empty)."""
    if not needle:
        return 1.0
    return len(needle & haystack) / len(needle)
