"""Retrieval quality on the live index: do the expected chunks make the top k?"""

import pytest

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
