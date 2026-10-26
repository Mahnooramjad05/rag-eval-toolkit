"""Prompt templates and JSON schemas for the LLM judge.

Each task returns JSON matching its schema, so parsing never depends on free text.
"""

SYSTEM = (
    "You are a strict, impartial evaluator of retrieval-augmented generation (RAG) systems. "
    "Judge only from the material provided. Do not use outside knowledge to decide whether "
    "a statement is supported. Respond with JSON that matches the requested schema."
)

EXTRACT_CLAIMS = """Break the text below into short, self-contained factual claims.
Each claim must be understandable on its own (resolve pronouns) and state one fact.
Ignore greetings, hedges and questions. Return an empty list if there are no factual claims.

TEXT:
{text}"""

EXTRACT_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
    "additionalProperties": False,
}

VERIFY_CLAIMS = """For each numbered claim, decide whether the CONTEXT directly supports it.
A claim is supported only if the context states it or it follows unambiguously from the context.
Numbers, dates and names must match exactly. If the context is silent or contradicts the claim,
it is not supported. Give a one-sentence reason for each verdict.

CONTEXT:
{context}

CLAIMS:
{claims}"""

VERIFY_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "supported", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

ANSWER_RELEVANCE = """Rate how directly the ANSWER addresses the QUESTION, ignoring whether it is correct.
5 = fully and directly answers it; 3 = partially answers or adds a lot of irrelevant material;
1 = does not address the question at all (including refusals and "I don't know").

QUESTION:
{question}

ANSWER:
{answer}"""

ANSWER_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

CHUNK_RELEVANCE = """Decide whether the CHUNK contains information that helps answer the QUESTION.
{reference_block}
QUESTION:
{question}

CHUNK:
{chunk}"""

CHUNK_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["relevant", "reason"],
    "additionalProperties": False,
}


def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(contexts))


def format_claims(claims: list[str]) -> str:
    return "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims))
