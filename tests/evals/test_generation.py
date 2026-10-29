"""Answer quality on pinned passages: answerable, off-topic and not-in-passages questions."""

import time
from datetime import date

import pytest

from app.edgar.qa import answer_question
from tests.evals.harness import failures, load_cases, pinned_hits, record
from tests.evals.scorers import score
from tests.helpers import needs_corpus

pytestmark = [pytest.mark.eval, needs_corpus]

CASES = load_cases("answerable") + load_cases("off_topic") + load_cases("not_in_passages")
TODAY = date(2026, 10, 29)  # fixed, so results don't drift with the calendar; a case can set its own "today"


@pytest.mark.parametrize("suite, case", CASES)
def test_generation(llm, suite, case):
    hits = pinned_hits(case["chunk_ids"])
    started = time.monotonic()
    today = date.fromisoformat(case["today"]) if case.get("today") else TODAY
    result = answer_question(case["question"], hits, llm, today=today)
    checks = score(result, case["expect"], [h.doc.page_content for h in hits])
    record(
        suite,
        case,
        checks,
        seconds=round(time.monotonic() - started, 1),
        status=result.status,
        answer=result.answer,
        warnings=result.warnings,
    )
    assert all(c.passed for c in checks), failures(checks)
