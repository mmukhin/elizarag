"""The eval scorers themselves, on hand-made results, so a broken scorer cannot pass a bad answer."""

import json

import pytest

from app.edgar import get_chunk
from app.edgar.qa import Claim, QAResult
from tests.evals.harness import CASES, load_cases, pinned_hits
from tests.evals.scorers import (
    check_cites_gold,
    check_claims_cited,
    check_mentions,
    check_names_missing,
    check_no_claims,
    check_not_mentions,
    check_numbers_grounded,
    check_tickers_covered,
    numbers,
    score,
)
from tests.helpers import needs_corpus

AAPL = "AAPL_10K_2025-10-31_full.txt#30"
TSLA = "TSLA_10K_2026-01-29_full.txt#18"


def result(status="answered", answer="Net sales were $416.2 billion [1].", claims=None, **extra) -> QAResult:
    claims = [Claim(text="Net sales rose 6%.", chunk_ids=[AAPL], tickers=["AAPL"])] if claims is None else claims
    return QAResult(question="q", model="m", status=status, answer=answer, claims=claims, sources=[], **extra)


def test_numbers_skips_citations_years_and_single_digits():
    assert numbers("In fiscal 2025 sales were 416,161 [3], up 6% from 2 segments; margin 46.9%") == [416161, 46.9]
    assert numbers("2025 and 7", skip_trivial=False) == [2025, 7]


@pytest.mark.parametrize(
    "answer, ok",
    [
        ("Total net sales were $416,161 million [1].", True),
        ("Total net sales were $416.2 billion [1].", True),  # millions -> billions, rounded
        ("Total net sales were $512.3 billion [1].", False),  # invented
        ("Sales grew 6% to 416,161 in fiscal 2025.", True),
    ],
)
def test_numbers_grounded(answer, ok):
    passage = "Total net sales | $ | 416,161 | 6 | % | 391,035"
    assert check_numbers_grounded(result(answer=answer, claims=[]), [passage]).passed is ok


def test_claims_cited_requires_citations_only_for_grounded_statuses():
    assert check_claims_cited(result()).passed
    uncited = result(answer="No cites.", claims=[Claim(text="x", chunk_ids=[], tickers=[])])
    assert not check_claims_cited(uncited).passed
    assert check_claims_cited(result(status="off_topic", answer="No.", claims=[])).passed


def test_tickers_covered():
    assert check_tickers_covered(result(), ["AAPL"]).passed
    check = check_tickers_covered(result(), ["AAPL", "TSLA"])
    assert not check.passed and "TSLA" in check.detail


def test_mentions_alternatives_and_case():
    r = result(answer="Driven by the App Store [1].")
    assert check_mentions(r, ["app store|advertising"]).passed
    assert not check_mentions(r, ["cloud"]).passed
    assert not check_not_mentions(r, ["APP STORE"]).passed


def test_cites_gold_and_no_claims():
    assert check_cites_gold(result(), [AAPL, TSLA]).passed
    assert not check_cites_gold(result(), [TSLA]).passed
    assert check_no_claims(result(status="off_topic", answer="I only answer filing questions.", claims=[])).passed
    assert not check_no_claims(result(status="off_topic", answer="Mix flour [1].", claims=[])).passed


def test_score_selects_checks_from_expect():
    expect = {"status": ["off_topic"], "no_claims": True, "must_not_mention": ["flour"]}
    good = score(result(status="off_topic", answer="I only answer questions about SEC filings.", claims=[]), expect, [])
    assert [c.name for c in good] == ["status", "citations_valid", "numbers_grounded", "must_not_mention", "no_claims"]
    assert all(c.passed for c in good)
    bad = score(result(answer="Use 2 cups of flour [1].", claims=[]), expect, [])
    assert {c.name for c in bad if not c.passed} == {"status", "must_not_mention", "no_claims"}


@needs_corpus
@pytest.mark.parametrize("suite", sorted(p.stem for p in CASES.glob("*.json")))
def test_case_files_are_well_formed(suite):
    cases = json.loads((CASES / f"{suite}.json").read_text())
    assert len({c["id"] for c in cases}) == len(cases) == len(load_cases(suite))
    for case in cases:
        for chunk_id in case.get("chunk_ids", []) + case.get("expected_chunk_ids", []):
            get_chunk(chunk_id)  # raises if the id does not exist in the corpus
        if "chunk_ids" in case:
            assert set(case["expect"].get("gold_chunk_ids", [])) <= set(case["chunk_ids"])


@needs_corpus
def test_pinned_hits_rebuild_the_indexed_chunk():
    [hit] = pinned_hits([AAPL])
    assert hit.doc.metadata["ticker"] == "AAPL" and hit.doc.metadata["chunk_index"] == 30
    assert "416,161" in hit.doc.page_content


def test_names_missing_needs_the_missing_period_in_caveats_or_answer():
    stale = result(status="partial", answer="Revenue rose in FY2024 and FY2025 [1].", caveats=["No FY2026 filing in the passages."])
    assert check_names_missing(stale, ["2026"]).passed
    vague = result(status="partial", answer="Revenue rose [1].", caveats=["Some data may be missing."])
    check = check_names_missing(vague, ["2026"])
    assert not check.passed and "2026" in check.detail
