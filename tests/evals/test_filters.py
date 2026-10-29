"""Filters parsed from realistic questions. Deterministic today (no model), so it runs without Ollama."""

import pytest

from app.edgar.query_filters import load_companies, parse_filters
from tests.evals.harness import failures, load_cases, record
from tests.evals.scorers import Check

pytestmark = pytest.mark.eval

# years in the corpus per ticker (10-K + 10-Q), fixed so the suite does not depend on what is indexed
YEARS = {t: [2022, 2023, 2024, 2025] for t in ("AAPL", "NVDA", "TSLA", "JPM", "KO", "PEP", "GOOG", "XOM", "MSFT", "BAC", "WMT")}


@pytest.mark.parametrize("suite, case", load_cases("filters"))
def test_filters(suite, case):
    parsed = parse_filters(case["question"], load_companies(), YEARS).as_dict()
    expected = case["expected"]
    checks = [
        Check(field, parsed.get(field, []) == expected.get(field, []), f"got {parsed.get(field)}, want {expected.get(field)}")
        for field in ("tickers", "fiscal_years", "filing_types", "quarters")
    ]
    record(suite, case, checks, parsed=parsed)
    assert all(c.passed for c in checks), failures(checks)
