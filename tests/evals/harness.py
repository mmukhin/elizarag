"""Load eval cases, rebuild their passages from the corpus, and collect results for the run report."""

import json
from pathlib import Path

import pytest

from app.edgar import get_chunk
from app.edgar.hybrid import Hit

CASES = Path(__file__).resolve().parent / "cases"
RESULTS: list[dict] = []  # one row per case, written to results/ at session end (see conftest.py)


def load_cases(suite: str) -> list:
    """Cases from `cases/<suite>.json` as pytest params; `"xfail": "reason"` marks a known failure."""
    params = []
    for case in json.loads((CASES / f"{suite}.json").read_text()):
        marks = [pytest.mark.xfail(reason=case["xfail"], strict=False)] if case.get("xfail") else []
        params.append(pytest.param(suite, case, id=f"{suite}/{case['id']}", marks=marks))
    return params


def pinned_hits(chunk_ids: list[str]) -> list[Hit]:
    """The case's passages, in order, rebuilt locally (generation evals do not depend on retrieval)."""
    return [Hit(get_chunk(i), 1.0 / n) for n, i in enumerate(chunk_ids, 1)]


def record(suite: str, case: dict, checks: list, **extra) -> None:
    RESULTS.append(
        {"suite": suite, "id": case["id"], "passed": all(c.passed for c in checks)}
        | extra
        | {"checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks}}
    )


def failures(checks: list) -> str:
    return "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed)
