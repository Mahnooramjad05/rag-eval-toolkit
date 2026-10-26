"""Offline heuristic judge.

FOR DEMOS AND TESTS ONLY. It approximates the LLM judge with word overlap so the
toolkit runs without an API key. It cannot understand paraphrase or negation, so
its scores are not a substitute for a real judge.
"""

from rageval.config import JudgeConfig, MockJudgeConfig
from rageval.judges.base import ChunkVerdict, ClaimVerdict, Judge, RelevanceVerdict
from rageval.text import content_tokens, numbers, overlap, split_sentences


class MockJudge(Judge):
    provider = "mock"

    def __init__(self, config: JudgeConfig, mock_config: MockJudgeConfig | None = None):
        super().__init__(config)
        self.mock = mock_config or MockJudgeConfig()

    async def extract_claims(self, text: str) -> list[str]:
        await self.usage.record()
        return split_sentences(text)

    async def verify_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimVerdict]:
        await self.usage.record()
        context_text = " ".join(contexts)
        context_tokens = content_tokens(context_text)
        context_numbers = numbers(context_text)
        verdicts = []
        for claim in claims:
            missing_numbers = numbers(claim) - context_numbers
            score = overlap(content_tokens(claim), context_tokens)
            if missing_numbers:
                verdicts.append(ClaimVerdict(
                    claim=claim, supported=False,
                    reason=f"Numbers not found in context: {', '.join(sorted(missing_numbers))}."))
            elif score >= self.mock.support_threshold:
                verdicts.append(ClaimVerdict(
                    claim=claim, supported=True,
                    reason=f"{score:.0%} of the claim's key terms appear in the context."))
            else:
                verdicts.append(ClaimVerdict(
                    claim=claim, supported=False,
                    reason=f"Only {score:.0%} of the claim's key terms appear in the context."))
        return verdicts

    async def rate_answer_relevance(self, question: str, answer: str) -> RelevanceVerdict:
        await self.usage.record()
        score = overlap(content_tokens(question), content_tokens(answer))
        rating = 1 + round(4 * score)
        return RelevanceVerdict(
            score=rating, reason=f"The answer covers {score:.0%} of the question's key terms.")

    async def rate_chunk_relevance(self, question: str, chunk: str,
                                   reference: str | None = None) -> ChunkVerdict:
        await self.usage.record()
        target = content_tokens(question) | content_tokens(reference or "")
        score = overlap(target, content_tokens(chunk))
        relevant = score >= self.mock.chunk_relevance_threshold
        return ChunkVerdict(
            relevant=relevant,
            reason=f"{score:.0%} of the question and reference terms appear in the chunk.")
