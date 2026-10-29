"""Grounded Q&A over retrieved chunks, with structured (schema-validated) output."""

import re
from collections import Counter
from typing import Literal

import tiktoken
from pydantic import BaseModel, Field

from app.edgar.chunking import ENCODING
from app.edgar.hybrid import Hit
from app.edgar.llm import ChatModel, Usage
from app.edgar.prompts import load_prompt
from app.edgar.vectorstore import chunk_id

CITATION = re.compile(r"\[(\d+)\]")
Status = Literal["answered", "partial", "not_in_passages", "off_topic"]
GROUNDED = ("answered", "partial")  # statuses whose answers must cite passages


class ModelClaim(BaseModel):
    text: str = Field(description="One-sentence finding")
    passages: list[int] = Field(default_factory=list, description="Numbers of the supporting passages")


class ModelAnswer(BaseModel):
    """What the LLM must return: judgements and passage numbers only, never passage content."""

    status: Status = Field(
        description="answered | partial (passages cover part of it) | not_in_passages | off_topic (not about the filings)"
    )
    answer: str = Field(description="Short direct answer with [n] passage citations")
    claims: list[ModelClaim] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    chunk_ids: list[str]
    tickers: list[str]  # derived from the cited passages, not from the model


class Source(BaseModel):
    """One passage the model saw. `ref` is its [n] label in the answer; look the text up by `chunk_id`."""

    ref: int
    chunk_id: str
    ticker: str
    company: str
    filing_type: str
    fiscal_year: int
    filing_date: str
    source_file: str
    chunk_index: int


class QAResult(BaseModel):
    """What callers get: the answer, claims resolved to chunk ids, the passages it saw and any problems found."""

    question: str
    model: str
    status: Status
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    sources: list[Source]
    filters: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    usage: Usage | None = None  # tokens, latency and cost of the LLM call; None when the model was not called


def _tokens(hit: Hit, encoder) -> int:
    count = hit.doc.metadata.get("token_count") or len(encoder.encode(hit.doc.page_content, disallowed_special=()))
    return count + 30  # passage header


def passage_tokens(hit: Hit) -> int:
    """What one passage costs against the context budget (text + header)."""
    return _tokens(hit, tiktoken.get_encoding(ENCODING))


def select_hits(hits: list[Hit], max_tokens: int) -> list[Hit]:
    """Fit hits into a token budget, taking each company's best passages first so none is starved.

    Returns the kept hits in their original order.
    """
    encoder = tiktoken.get_encoding(ENCODING)
    seen: Counter[str] = Counter()
    order = []
    for i, hit in enumerate(hits):
        ticker = hit.doc.metadata["ticker"]
        order.append((seen[ticker], i))
        seen[ticker] += 1
    kept, used = set(), 0
    for _, i in sorted(order):
        cost = _tokens(hits[i], encoder)
        if used + cost <= max_tokens or not kept:
            kept.add(i)
            used += cost
    return [hit for i, hit in enumerate(hits) if i in kept]


def build_context(hits: list[Hit]) -> tuple[str, list[Source]]:
    """Number the passages 1..n and render them for the prompt."""
    sources, blocks = [], []
    for n, hit in enumerate(hits, 1):
        m = hit.doc.metadata
        sources.append(
            Source(
                ref=n,
                chunk_id=chunk_id(m),
                ticker=m["ticker"],
                company=m["company"],
                filing_type=m["filing_type"],
                fiscal_year=m["fiscal_year"],
                filing_date=m["filing_date"],
                source_file=m["source_file"],
                chunk_index=m["chunk_index"],
            )
        )
        blocks.append(f"[{n}] {m['ticker']} · {m['filing_type']} · FY{m['fiscal_year']} · filed {m['filing_date']}\n{hit.doc.page_content}")
    return "\n\n".join(blocks), sources


def validate(answer: ModelAnswer, n_sources: int) -> list[str]:
    """Drop citations to passages that do not exist (in place) and report grounding problems."""
    warnings = []
    valid = set(range(1, n_sources + 1))
    grounded = answer.status in GROUNDED
    for claim in answer.claims:
        bad = [i for i in claim.passages if i not in valid]
        if bad:
            warnings.append(f"removed invalid passage ids {bad} from a claim")
            claim.passages = [i for i in claim.passages if i in valid]
        if not claim.passages and grounded:
            warnings.append(f"uncited claim: {claim.text[:80]}")
    cited_in_text = {int(i) for i in CITATION.findall(answer.answer)}
    if cited_in_text - valid:
        warnings.append(f"answer text cites unknown passages {sorted(cited_in_text - valid)}")
    if grounded and not cited_in_text & valid and not any(c.passages for c in answer.claims):
        warnings.append("answer has no citations")
    if answer.status == "off_topic" and (answer.claims or cited_in_text):
        warnings.append("off-topic answer should not make claims or cite passages")
    if answer.status == "not_in_passages" and answer.claims:
        warnings.append("not_in_passages answer still makes claims")
    return warnings


def resolve_claims(answer: ModelAnswer, sources: list[Source]) -> list[Claim]:
    """Map each claim's passage numbers to chunk ids, and its tickers to the companies of those passages."""
    by_ref = {s.ref: s for s in sources}
    claims = []
    for claim in answer.claims:
        cited = [by_ref[i] for i in claim.passages if i in by_ref]
        claims.append(
            Claim(
                text=claim.text,
                chunk_ids=list(dict.fromkeys(s.chunk_id for s in cited)),
                tickers=list(dict.fromkeys(s.ticker for s in cited)),
            )
        )
    return claims


def answer_question(
    question: str, hits: list[Hit], llm: ChatModel, max_context_tokens: int = 8000, filters: dict | None = None
) -> QAResult:
    """Answer `question` from retrieved `hits` only. No hits -> not_in_passages, no model call."""
    model = getattr(llm, "model", type(llm).__name__)
    filters = filters or {}
    if not hits:
        return QAResult(
            question=question,
            model=model,
            status="not_in_passages",
            answer="No relevant passages were retrieved.",
            sources=[],
            filters=filters,
            warnings=["no passages to answer from"],
        )

    kept = select_hits(hits, max_context_tokens)
    context, sources = build_context(kept)
    answer = llm.generate(
        system=load_prompt("qa_system"),
        user=load_prompt("qa_user", question=question, context=context),
        schema=ModelAnswer,
    )
    warnings = validate(answer, len(sources))
    if len(kept) < len(hits):
        warnings.append(f"{len(hits) - len(kept)} lower-ranked passages dropped to fit the context budget")
    return QAResult(
        question=question,
        model=model,
        status=answer.status,
        answer=answer.answer,
        claims=resolve_claims(answer, sources),
        caveats=answer.caveats,
        sources=sources,
        filters=filters,
        warnings=warnings,
        usage=llm.usage.model_copy(),
    )
