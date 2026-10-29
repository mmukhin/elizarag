"""Retrieval quality on the live index: do the expected chunks make the top k?"""

import pytest

from app.edgar.query_filters import load_companies, resolve_filters
from app.edgar.vectorstore import Filters, chunk_id
from tests.evals.harness import failures, load_cases, record
from tests.evals.scorers import Check

pytestmark = pytest.mark.eval


@pytest.mark.parametrize("suite, case", load_cases("retrieval"))
def test_retrieval(retriever, suite, case):
    tickers = case.get("tickers") or []
    if missing := set(tickers) - set(retriever.tickers):
        pytest.skip(f"{sorted(missing)} not in the collection")
    hits = retriever.search(
        case["question"],
        k=case.get("k", 5),
        filters=Filters(tickers, case.get("fiscal_years")),
        per_ticker=len(tickers) > 1,
    )
    ranked = [chunk_id(h.doc.metadata) for h in hits]
    expected = case["expected_chunk_ids"]
    found = [i for i in expected if i in ranked]
    first = min((ranked.index(i) + 1 for i in found), default=None)
    checks = [Check("hit_at_k", bool(found), f"top-k {ranked}")]
    if len(tickers) > 1:  # balanced comparison: every company's expected chunk should be retrieved
        checks.append(Check("recall_at_k", len(found) == len(expected), f"found {found} of {expected}"))
    record(suite, case, checks, first_rank=first, recall=round(len(found) / len(expected), 2))
    assert all(c.passed for c in checks), failures(checks)


TEMPLATES = [
    "How has {name}'s revenue and growth outlook changed over the last two years?",
    "What are the main risk factors {name} describes?",
]


@pytest.mark.parametrize("company", load_companies(), ids=lambda c: c.ticker)
def test_company_questions_retrieve_only_that_company(retriever, company):
    """End to end on the live index, the way the UI does it: question -> resolve_filters -> hybrid search.

    Every passage must belong to the company the question names. Unfiltered purity is recorded too,
    to show how much the filter matters.
    """
    if company.ticker not in retriever.tickers:
        pytest.skip(f"{company.ticker} not in the collection")
    name = (company.aliases or company.ambiguous_aliases or (company.name,))[0]
    checks, unfiltered = [], []
    for template in TEMPLATES:
        question = template.format(name=name)
        resolved = resolve_filters(question, load_companies(), retriever.years_by_ticker)
        hits = retriever.search(question, k=5, filters=resolved.filters, per_ticker=True)
        others = sorted({h.doc.metadata["ticker"] for h in hits} - {company.ticker})
        checks.append(Check(f"pure: {template[:30]}", bool(hits) and not others, f"{len(hits)} hits, others {others}"))
        raw = retriever.search(question, k=5)
        unfiltered.append(sum(h.doc.metadata["ticker"] == company.ticker for h in raw) / max(len(raw), 1))
    record("company_purity", {"id": company.ticker}, checks, unfiltered_purity=round(sum(unfiltered) / len(unfiltered), 2))
    assert all(c.passed for c in checks), failures(checks)
